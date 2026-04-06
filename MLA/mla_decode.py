#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X

# gpumode leaderboard reference
"""
Reference implementation for MLA (Multi-head Latent Attention) decode kernel.

Uses aiter MLA kernels (mla_decode_fwd) as the reference.
DeepSeek R1 forward_absorb MLA: absorbed q (576), compressed kv_buffer (576),
output v_head_dim = kv_lora_rank = 512.

The input provides:
  q:       (total_q, 16, 576) bfloat16 — absorbed query
  kv_data: dict with KV cache in three formats:
    "bf16":  Tensor  (total_kv, 1, 576)  bfloat16          — highest precision
    "fp8":   (Tensor, Tensor)  kv_buffer fp8 + scalar scale — per-tensor quantized
    "mxfp4": (Tensor, Tensor)  kv_buffer fp4x2 + fp8_e8m0  — block-32 quantized
  The reference quantizes Q to fp8 on-the-fly inside ref_kernel.

The reference kernel quantizes Q to fp8 on-the-fly and uses fp8 KV (a8w8 kernel),
which is ~2-3x faster than bf16 on MI355X with negligible accuracy loss.

Decode only — persistent mode with get_mla_metadata_v1.
"""

import torch
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference

import triton
import triton.language as tl

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import (
    dynamic_mxfp4_quant,
    mxfp4_to_f32,
    e8m0_to_f32,
)

# ---------------------------------------------------------------------------
# DeepSeek R1 latent MQA constants (forward_absorb path)
# https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/config.json
# ---------------------------------------------------------------------------
NUM_HEADS = 16
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576
V_HEAD_DIM = KV_LORA_RANK                        # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

PAGE_SIZE = 1
NUM_KV_SPLITS = 32

# FP8 dtype (platform-specific via aiter)
FP8_DTYPE = aiter_dtypes.fp8

# Query dtype for the reference kernel: "fp8" or "bf16"
Q_DTYPE = "fp8"

# KV cache dtype for the reference kernel: "fp8" or "bf16"
KV_DTYPE = "fp8"


# ---------------------------------------------------------------------------
# FP8 quantization (sglang style: dynamic per-tensor)
# ---------------------------------------------------------------------------
def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-tensor FP8 quantization (following sglang scaled_fp8_quant).

    Args:
        tensor: bf16 tensor to quantize

    Returns:
        (fp8_tensor, scale) where scale is a scalar float32 tensor.
        Dequantize: fp8_tensor.to(bf16) * scale
    """
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# MXFP4 quantization (aiter native: block-32, fp4x2 + fp8_e8m0 dtypes)
# Uses aiter.utility.fp4_utils.dynamic_mxfp4_quant
# ---------------------------------------------------------------------------

def quantize_mxfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MXFP4 block-wise quantization using aiter's dynamic_mxfp4_quant.

    Block size = 32. Each block gets an E8M0 scale factor.
    Two FP4 E2M1 values are packed per byte.

    Args:
        tensor: bf16 tensor of shape [B, M, N] (N must be divisible by 32)

    Returns:
        (fp4_data, scale_e8m0)
        - fp4_data:   shape [B, M, N//2] in aiter_dtypes.fp4x2
        - scale_e8m0: shape [B*M, ceil(N/32)] padded, in aiter_dtypes.fp8_e8m0
    """
    orig_shape = tensor.shape  # (B, M, N)
    B, M, N = orig_shape

    # dynamic_mxfp4_quant expects 2D: (B*M, N)
    tensor_2d = tensor.reshape(B * M, N)
    fp4_data_2d, scale_e8m0 = dynamic_mxfp4_quant(tensor_2d)

    # Reshape fp4_data back to 3D: (B, M, N//2)
    fp4_data = fp4_data_2d.view(B, M, N // 2)

    return fp4_data, scale_e8m0


def dequantize_mxfp4(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize MXFP4 tensor using aiter utilities.

    Note: dynamic_mxfp4_quant may pad both row and block dimensions in scale_e8m0.
    We trim scales to match the actual data dimensions.

    Args:
        fp4_data:   packed FP4 data, shape [B, M, N//2] in fp4x2 or uint8
        scale_e8m0: E8M0 block scale factors (possibly padded) in fp8_e8m0
        orig_shape: original (B, M, N) for reshaping
        dtype:      output dtype

    Returns:
        Dequantized tensor of shape orig_shape.
    """
    B, M, N = orig_shape
    num_rows = B * M
    block_size = 32
    num_blocks = N // block_size  # actual blocks needed (e.g. 576/32 = 18)

    # Unpack FP4 to float32: mxfp4_to_f32 expects (..., N//2) -> (..., N)
    fp4_data_2d = fp4_data.reshape(num_rows, N // 2)
    float_vals = mxfp4_to_f32(fp4_data_2d)  # (num_rows, N)

    # Convert E8M0 scales to float32 and trim padded dimensions
    scale_f32 = e8m0_to_f32(scale_e8m0)  # (padded_rows, padded_blocks)
    scale_f32 = scale_f32[:num_rows, :num_blocks]  # (num_rows, num_blocks)

    # Apply block scales
    float_vals_blocked = float_vals.view(num_rows, num_blocks, block_size)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(B, M, N).to(dtype)


# ---------------------------------------------------------------------------
# Persistent mode metadata helpers
# ---------------------------------------------------------------------------

def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int = NUM_KV_SPLITS,
):
    """Allocate and populate work buffers for persistent mla_decode_fwd."""
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    # Populate the metadata buffers
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,   # num_heads_per_head_k
        nhead_kv,            # num_heads_k
        True,                # is_causal
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Aiter reference kernel (decode only)
# ---------------------------------------------------------------------------

def _aiter_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MLA decode attention using aiter persistent-mode kernel.

    Supports multiple Q/KV dtype combinations:
      - Q_DTYPE="fp8":  fp8 Q + fp8 KV (a8w8) — fastest on MI355X
      - Q_DTYPE="bf16": bf16 Q + bf16 KV (a16w16) — highest precision

    q:          (total_q, num_heads, 576)  fp8 or bf16
    kv_buffer:  (total_kv, 1, 576)         fp8 or bf16
    q_scale:    scalar float32 (required for fp8 Q, None for bf16)
    kv_scale:   scalar float32 (required for fp8 KV, None for bf16)
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    total_kv_len = int(kv_indptr[-1].item())

    # Reshape kv_buffer to 4D for aiter: (total_kv, page_size, nhead_kv, dim)
    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])

    max_q_len = q_seq_len
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q.dtype, kv_buffer.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=NUM_KV_SPLITS,
    )

    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=NUM_KV_SPLITS,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o


@triton.jit
def mla_decode_fwd_kernel(
    q_ptr, kv_ptr, o_ptr,
    qt_stride, qh_stride, qd_stride,
    kvt_stride, kvh_stride, kvd_stride,
    ot_stride, oh_stride, od_stride,
    qo_indptr, kv_indptr,
    sm_scale,
    Q_TILE_SIZE: tl.constexpr, #TODO can we load query as tile
    KV_TILE_SIZE: tl.constexpr,
    q_total: tl.constexpr, n_heads: tl.constexpr, d_Q: tl.constexpr,
    kv_total: tl.constexpr, d_KV: tl.constexpr, d_O: tl.constexpr
):
    seq_idx = tl.program_id(0)    
    head_idxs = tl.program_id(1)  
    
    #d_Q_padded = 1024    # next power of 2 instead of 576
    
    o_block_ptr = tl.make_block_ptr(o_ptr,
                                    shape=(q_total*n_heads,d_O),
                                    strides=(oh_stride,od_stride),
                                    offsets=(seq_idx * n_heads + head_idxs*Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE,d_O),
                                    order=(1,0))  
    
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,  
        shape=(q_total * n_heads, d_Q), # flattened into 2d
        strides=(qh_stride, qd_stride),  
        offsets=(seq_idx * n_heads + head_idxs*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1024),
        order=(1, 0)
    )
    
    query_tile = tl.load(q_block_ptr, boundary_check=(1,))
    
    # kv cache for variable length sequences
    seq_start_idx = tl.load(kv_indptr + seq_idx)
    seq_end_idx = tl.load(kv_indptr + seq_idx + 1)
    seq_len = seq_end_idx - seq_start_idx
    
    # online softmax accumulators
    max_el = tl.full((Q_TILE_SIZE,), value=float("-inf"), dtype=tl.float32)
    norm_factor = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    
    #tile output
    tile_output = tl.zeros((Q_TILE_SIZE,d_O), dtype=tl.float32)
    
    for i in range(0, seq_len, KV_TILE_SIZE):
        kv_block_ptr = tl.make_block_ptr(kv_ptr,
                                         shape=(kv_total,d_KV),
                                         strides=(kvt_stride,kvd_stride),
                                         offsets=(seq_start_idx + i, 0),
                                         block_shape=(KV_TILE_SIZE,1024),
                                         order=(1,0)
                                         )
        
        v_block_ptr = tl.make_block_ptr(
            base=kv_ptr, 
            shape=(kv_total, d_KV), 
            strides=(kvt_stride, kvd_stride), # KV cache strides
            offsets=(seq_start_idx + i, 0),  
            block_shape=(KV_TILE_SIZE, 512), 
            order=(1, 0)
        )
        
        kv_tile = tl.load(kv_block_ptr, boundary_check=(1,))
        v_tile = tl.load(v_block_ptr, boundary_check=(1,))
        qk = tl.dot(query_tile,tl.trans(kv_tile))    # Q.K^T
        qk *= sm_scale
        
        row_max = tl.max(qk, axis=1)
        last_max_el = max_el
        max_el = tl.maximum(last_max_el,row_max)
        
        p = tl.exp(qk-max_el[:,None])
        
        rescale_factor = tl.exp(last_max_el - max_el)
        norm_factor =  rescale_factor*norm_factor + tl.sum(p, axis=1)
        tile_output = tile_output*rescale_factor[:,None]
        tile_output = tl.dot(p.to(tl.bfloat16), v_tile.to(tl.bfloat16), acc=tile_output)        
        
    tile_output = (tile_output/norm_factor[:,None]).to(tl.bfloat16)    
    tl.store(o_block_ptr, tile_output, boundary_check=(0,1))
    
    
def custom_kernel(data):
    """
    Launcher for mla_decode_fwd_kernel.
    Input data: (q, kv_data, qo_indptr, kv_indptr, config)
    """
    # 1. Unpack the input tuple
    q, kv_data, qo_indptr, kv_indptr, config = data
    
    # 2. Extract metadata from config and tensors
    # total_q is the first dimension of q: (total_q, 16, 576)
    total_q = q.shape[0]
    batch_size = config["batch_size"]
    n_heads = config.get("num_heads", 16)
    
    # Dimensions for DeepSeek R1 MLA
    d_Q = 576  # Latent QK dim
    d_KV = 576 # Latent KV dim
    d_O = 512  # Output V dim
    
    sm_scale = config["sm_scale"]
    
    # 3. Select the BF16 KV cache from the dictionary
    # kv_data["bf16"] shape: (total_kv, 1, 576)
    kv_ptr = kv_data["bf16"]
    kv_total = kv_ptr.shape[0]
    
    # 4. Allocate Output Tensor (total_q, 16, 512)
    output = torch.empty((total_q, n_heads, d_O), device=q.device, dtype=torch.bfloat16)

    # 5. Define Tiling Constants
    # Q_TILE_SIZE: How many heads a single program handles
    # KV_TILE_SIZE: How many KV tokens to load in the inner loop
    Q_TILE_SIZE = 1 
    KV_TILE_SIZE = 32
    
    # 6. Grid Calculation
    # program_id(0) -> seq_idx (Batch/Sequence)
    # program_id(1) -> head_idxs (Groups of heads)
    grid = (batch_size, n_heads // Q_TILE_SIZE)

    # 7. Launch Your Kernel
    mla_decode_fwd_kernel[grid](
        q_ptr=q, 
        kv_ptr=kv_ptr, 
        o_ptr=output,
        # Query Strides
        qt_stride=q.stride(0), 
        qh_stride=q.stride(1), 
        qd_stride=q.stride(2),
        # KV Strides
        kvt_stride=kv_ptr.stride(0), 
        kvh_stride=kv_ptr.stride(1), 
        kvd_stride=kv_ptr.stride(2),
        # Output Strides
        ot_stride=output.stride(0), 
        oh_stride=output.stride(1), 
        od_stride=output.stride(2),
        # Metadata pointers
        qo_indptr=qo_indptr, 
        kv_indptr=kv_indptr,
        # Scaler
        sm_scale=sm_scale,
        # Constexprs
        Q_TILE_SIZE=Q_TILE_SIZE, 
        KV_TILE_SIZE=KV_TILE_SIZE,
        q_total=total_q, 
        n_heads=n_heads, 
        d_Q=d_Q,
        kv_total=kv_total, 
        d_KV=d_KV, 
        d_O=d_O,
        # Optimization hints for MI355X
        #num_warps=4,
        #num_stages=3
    )

    return output

check_implementation = make_match_reference(custom_kernel, rtol=1e-01, atol=1e-01)
