/*
kernel 2
each block computes a tile of output tile(BMxBN)
each warp in block computes (BM/TILE_SIZE_M, BN/TILE_SIZE_N)
for tf32, fragment size = (16,16,8)
so each warp loops over 2 subtiles of A (using another subloop) with inner loop over all tiles of B (with subloop)
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <mma.h>
using namespace nvcuda;

#define TILE_SIZE_M 16
#define TILE_SIZE_N 8
#define TILE_SIZE_K 8
#define WARP_SIZE 32

// swizzling logic
template<int ROWS, int COLS>
__device__ __forceinline__ int swizzle(int row, int col) {
    static_assert((COLS % 4) == 0, "COLS must be divisible by 4");
    static_assert((COLS/4 & (COLS/4 - 1)) == 0, "COLS/4 must be power of 2");
    
    constexpr int chunks_per_row = COLS / 4;
    constexpr int chunk_mask     = chunks_per_row - 1;  // bits to XOR with
    
    int col_chunk       = col >> 2;                      // which float4 chunk
    int swizzled_chunk  = col_chunk ^ (row & chunk_mask);
    int col_swizzled    = (swizzled_chunk << 2) | (col & 3);
    
    return row * COLS + col_swizzled;
}
// swizzle(innerColA*4 + 0, innerRowA + offset, BM)


namespace wt {
    template <const int BM, const int BN, const int BK, const int rowStrideA,
            const int rowStrideB, typename T>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                float *As, float *Bs, int innerRowA, int innerColA,
                                int innerRowB, int innerColB, T &barrier) {

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        cuda::memcpy_async(&As[swizzle<BM,BK>((innerRowA+offset), innerColA*4)],
            &A[(innerRowA + offset) * K + innerColA * 4],
            cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            barrier);

    }
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        cuda::memcpy_async(&Bs[swizzle<BK,BN>((innerRowB+offset), innerColB*4)],
            &B[(innerRowB+offset)* N + innerColB*4],
            cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            barrier);
        }
    }
}

template<uint BM, uint BN, uint BK, uint NUM_THREADS, uint ACCUM_SIZE, uint NUM_STAGES>
__global__ void GEMM_tc(float* A, float*B, float*C, int N, int M, int K){

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope_block> barriers[NUM_STAGES];
    //cuda::barrier<cuda::thread_scope_block>::arrival_token tokens[NUM_STAGES];

    if (block.thread_rank() == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init(&barriers[i], block.size());
        }
    }
    __syncthreads();

    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    __shared__ alignas(128) float As[NUM_STAGES][BM*BK];
    __shared__ alignas(128) float Bs[NUM_STAGES][BK*BN];

    int warp_id = threadIdx.x / WARP_SIZE;  // 0-3
    int lane_id = threadIdx.x % 32;
    int group_id = lane_id >> 2;     // lane_id/4
    int thread_in_group = lane_id % 4;
    constexpr uint total_warps = NUM_THREADS / WARP_SIZE;  // 4
    constexpr uint TILES_PER_WARP_M = (BM / total_warps) / TILE_SIZE_M;     // 2 16x16 tiles per warp
    constexpr uint TILES_PER_WARP_N = BN / TILE_SIZE_N;   // 16 tiles (8x8)
    
    A += K * BM * tile_row;
    B += BN * tile_col;
    //C += K*BM*tile_row + tile_col*BN + warp_id*TILE_SIZE_N;

    const uint innerRowA = threadIdx.x / (BK / 4);   // which row
    const uint innerColA = threadIdx.x % (BK / 4);   // which float4 chunk
    const uint rowStrideA = NUM_THREADS / (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint rowStrideB = NUM_THREADS / (BN / 4);
    const uint NUM_TILES = K / BK;

    float accum[ACCUM_SIZE]= {0.0f}; 

    // prologue
    int fetch=0;
    for(int i=0; i<NUM_STAGES-1 && i<NUM_TILES; ++i, ++fetch){
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + fetch * BK, B + (size_t)fetch * BK * N, As[i], Bs[i], innerRowA, innerColA, innerRowB, innerColB, barriers[i]);
        //tokens[i] = barriers[i].arrive();
    }
    //auto token = barriers[0].arrive(); 
    //barriers[0].wait(std::move(token));


    // outer-most loop over block tiles
    for (uint tile = 0; tile < NUM_TILES; ++tile) {
        int cur_stage = tile % NUM_STAGES;
        int next_fetch = fetch % NUM_STAGES;

        barriers[cur_stage].wait(barriers[cur_stage].arrive());

        if(fetch < NUM_TILES){
            wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + fetch * BK, B + (size_t)fetch * BK * N, As[next_fetch], Bs[next_fetch], innerRowA, innerColA, innerRowB, innerColB, barriers[next_fetch]);
            //tokens[next_fetch] = barriers[next_fetch].arrive();
            ++fetch;
            //__syncthreads();
        }
        

        for(int warp_tile_A=0; warp_tile_A<TILES_PER_WARP_M; ++warp_tile_A){
            for(int inner_tile=0; inner_tile<BK/TILE_SIZE_K; ++inner_tile){
                //int offset_As = (warp_id * TILES_PER_WARP_M + warp_tile_A) * TILE_SIZE_M * BK + inner_tile * TILE_SIZE_K;
                //float* tile_A_ptr = &As[offset_As];
                int warp_row_base = (warp_id * TILES_PER_WARP_M + warp_tile_A) * TILE_SIZE_M;
                // load sub tile A in register fragments (swizzled)
                float fa0 = As[cur_stage][swizzle<BM,BK>(warp_row_base + group_id,     inner_tile * TILE_SIZE_K + thread_in_group)];
                float fa2 = As[cur_stage][swizzle<BM,BK>(warp_row_base + group_id,     inner_tile * TILE_SIZE_K + thread_in_group + 4)];
                float fa1 = As[cur_stage][swizzle<BM,BK>(warp_row_base + group_id + 8, inner_tile * TILE_SIZE_K + thread_in_group)];
                float fa3 = As[cur_stage][swizzle<BM,BK>(warp_row_base + group_id + 8, inner_tile * TILE_SIZE_K + thread_in_group + 4)];

                uint32_t a0, a1, a2, a3;
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(a0) : "f"(fa0));
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(a1) : "f"(fa1));
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(a2) : "f"(fa2));
                asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(a3) : "f"(fa3));
                
                for(int warp_tile_B=0; warp_tile_B<TILES_PER_WARP_N; ++warp_tile_B){
                    int b_row_base = inner_tile * TILE_SIZE_K;  // which K-subtile (0 or 8)
                    int b_col_base = warp_tile_B * TILE_SIZE_N; // which N-tile

                    // B fragment: row=K dim, col=N dim
                    // fb0: row=thread_in_group,   col=group_id  (within the 8×8 subtile)
                    // fb1: row=thread_in_group+4, col=group_id
                    float fb0 = Bs[cur_stage][swizzle<BK,BN>(b_row_base + thread_in_group,     b_col_base + group_id)];
                    float fb1 = Bs[cur_stage][swizzle<BK,BN>(b_row_base + thread_in_group + 4, b_col_base + group_id)];

                    uint32_t b0, b1;
                    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b0) : "f"(fb0));
                    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(b1) : "f"(fb1));

                    // compute using mma ptx
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(accum[(warp_tile_A * TILES_PER_WARP_N + warp_tile_B) * 4 + 0]), "+f"(accum[(warp_tile_A * TILES_PER_WARP_N + warp_tile_B) * 4 + 1]),
                        "+f"(accum[(warp_tile_A * TILES_PER_WARP_N + warp_tile_B) * 4 + 2]), "+f"(accum[(warp_tile_A * TILES_PER_WARP_N + warp_tile_B) * 4 + 3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                        "r"(b0), "r"(b1)
                    );
                }
            }
        }    
        //barriers[next_fetch].wait(std::move(token));
        //__syncthreads();
    }

    // Store
    for(int tile_a = 0; tile_a < TILES_PER_WARP_M; ++tile_a){
        for(int tile_b = 0; tile_b < TILES_PER_WARP_N; ++tile_b){

            int warp_row_offset = (warp_id * TILES_PER_WARP_M + tile_a) * TILE_SIZE_M;
            int warp_col_offset = tile_b * TILE_SIZE_N;

            int base = (tile_a * TILES_PER_WARP_N + tile_b) * 4;

            int row_top = tile_row*BM + warp_row_offset + group_id;
            int row_bot = tile_row*BM + warp_row_offset + group_id + 8;
            int col_0   = tile_col*BN + warp_col_offset + thread_in_group*2;
            int col_1   = tile_col*BN + warp_col_offset + thread_in_group*2 + 1;

            C[row_top * N + col_0] = accum[base + 0];  // d0
            C[row_top * N + col_1] = accum[base + 1];  // d1
            C[row_bot * N + col_0] = accum[base + 2];  // d2
            C[row_bot * N + col_1] = accum[base + 3];  // d3
        }
    }
}