# amd-mixed-mla (GPU MODE x AMD HACKATHON)

This is the inner attention kernel from DeepSeek R1's forward_absorb MLA path. The absorbed query and compressed KV cache are provided directly — you only need to implement the attention computation with variable-length batching (indptr).

The reference uses aiter a8w8 MLA decode kernel (mla_decode_fwd, fp8 Q + fp8 KV, persistent mode), which is ~2-3x faster than bf16 on MI355X.

## DeepSeek R1 forward_absorb MLA config:

 - num_heads = 16 (query heads, after TP split)
 - num_kv_heads = 1 (shared latent KV head)
 - kv_lora_rank = 512
 - qk_rope_head_dim = 64
 - qk_head_dim = 576 (kv_lora_rank + qk_rope_head_dim, absorbed q/k dim)
 - v_head_dim = 512 (= kv_lora_rank, output dim)
 - sm_scale = 1/sqrt(576)
 - dtype: q=bfloat16
 - decode only (q_seq_len=1, kv_seq_len up to 8k)
 - KV buffer format (forward_absorb):

Full 576 dims are used as keys (for Q@K^T score computation)

First 512 dims (kv_lora_rank) are used as values (for output computation)

### Input tuple: (q, kv_data, qo_indptr, kv_indptr, config)

q: (total_q, 16, 576) bfloat16 — absorbed query

kv_data: dict with three KV cache formats: kv_data["bf16"] — Tensor (total_kv, 1, 576) bfloat16 kv_data["fp8"] — (Tensor, Tensor): kv_buffer fp8 + scalar scale kv_data["mxfp4"] — (Tensor, Tensor): kv_buffer fp4x2 + fp8_e8m0 scale

qo_indptr: (batch_size+1,) int32 — query segment pointers

kv_indptr: (batch_size+1,) int32 — KV segment pointers

config: dict with MLA parameters

### Return:

attention output: (total_q, 16, 512) bfloat16

## Key optimization opportunities:

 - Use mxfp4 KV cache for even lower memory bandwidth (4x savings over bf16)
 - Fuse dequantization with attention to skip bf16 materialization
 - Custom kernel with tighter memory access patterns
 - MQA: 1 KV head shared across 16 query heads — minimize redundant memory loads
 - Decode: q_seq_len=1, kv_seq_len up to 8k — memory-bound workload
 - Variable-length batching: indptr-based segmented attention
 - Split K/V from buffer: full 576 dims for keys, first 512 dims for values


## GPU MODE

All these details about kernel are directly ported from GPU MODE's reference provided with hackathon:

https://www.gpumode.com/home