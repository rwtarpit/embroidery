#!POPCORN leaderboard matmul_v2
#!POPCORN gpu A100

from task import input_t, output_t
from utils import make_match_reference

import triton
import triton.language as tl


# KxM @ MxN == KxN

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
            ax_stride, ay_stride,
            bx_stride, by_stride,
            cx_stride, cy_stride,
            K: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
            K_TILE_SIZE: tl.constexpr, 
            N_TILE_SIZE: tl.constexpr, 
            M_TILE_SIZE: tl.constexpr 
            ):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(K, M),
        strides=(ax_stride, ay_stride),
        offsets=(pid_k * K_TILE_SIZE, 0), 
        block_shape=(K_TILE_SIZE, M_TILE_SIZE),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(M, N),
        strides=(bx_stride, by_stride),
        offsets=(0, pid_n * N_TILE_SIZE), 
        block_shape=(M_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0)
    )

    accum = tl.zeros((K_TILE_SIZE, N_TILE_SIZE), dtype=tl.float32)

    for i in range(0, tl.cdiv(M, M_TILE_SIZE)):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        accum = tl.dot(a_tile, b_tile, acc=accum)
        
        a_block_ptr = tl.advance(a_block_ptr, (0, M_TILE_SIZE))
        b_block_ptr = tl.advance(b_block_ptr, (M_TILE_SIZE, 0))

    # write the C tile back to global memory
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(K, N),
        strides=(cx_stride, cy_stride),
        offsets=(pid_k * K_TILE_SIZE, pid_n * N_TILE_SIZE),
        block_shape=(K_TILE_SIZE, N_TILE_SIZE),
        order=(1, 0)
    )
    tl.store(c_block_ptr, accum.to(tl.float16), boundary_check=(0, 1))
    
    
def custom_kernel(data: input_t) -> output_t:
    a,b,c = data
    K,M = a.shape
    N = b.shape[-1]
    
    K_TILE_SIZE, N_TILE_SIZE, M_TILE_SIZE = 128, 128, 32
    
    grid = (triton.cdiv(K, K_TILE_SIZE), triton.cdiv(N, N_TILE_SIZE))
    
    matmul_kernel[grid](
        a,b,c,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        K, M, N,
        K_TILE_SIZE, N_TILE_SIZE, M_TILE_SIZE,
        num_warps = 4,
        num_stages = 4
    )
    return c

check_implementation = make_match_reference(custom_kernel)