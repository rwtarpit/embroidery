/*
each block computes a tile of output tile(BMxBN)
each warp in block computes (BM/TILE_SIZE_M, BN/TILE_SIZE_N)
for tf32, fragment size = (16,16,8)
so each warp loops over 2 subtiles of A (using another subloop) with inner loop over all tiles of B (with subloop)
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/pipeline>   // replaces cuda/barrier
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

/*
128x32
cpr = 8
cm = 8-1 = 7
cc = col >> 2
sc = cc ^ (row & 7)
cs = (sc << 2) | (col & 3)
== row*32 + cs

32x128
cpr = 32
cm = 32-1 = 31
cc = col >> 2
sc = cc ^ (row & 31)
cs = (sc << 2) | (col & 3)
== row*128 + cs
*/

__device__ __forceinline__ int SWIZZLE_A(int row, int col) {
    return (row * 32) + (col ^ ((row & 7) << 2));
}

// Mathematically identical to your original SWIZZLE_B, but compressed
__device__ __forceinline__ int SWIZZLE_B(int row, int col) {
    return (row * 128) + (col ^ ((row & 31) << 2));
}



namespace wt {
    template <const int BM, const int BN, const int BK, const int rowStrideA,
              const int rowStrideB>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 float *As, float *Bs, int innerRowA, int innerColA,
                                 int innerRowB, int innerColB) {

        // producer_acquire must be called before submitting async copies
        // (caller is responsible — see main loop)

        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            int swizzled_offset = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[SWIZZLE_A(innerRowA + offset, innerColA * 4)]));
            //cuda::memcpy_async(
            //    &As[swizzled_offset],
            //    &A[(innerRowA + offset) * K + innerColA * 4],
            //    cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            //    pipe);
            
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                ::"r"(swizzled_offset) "l"(&A[(innerRowA + offset) * K + innerColA * 4])
            );
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            uint32_t swizzled_offset = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[SWIZZLE_B(innerRowB + offset, innerColB * 4)]));
            
            //cuda::memcpy_async(
            //    &Bs[swizzled_offset],
            //    &B[(innerRowB + offset) * N + innerColB * 4],
            //    cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
            //    pipe);

            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                ::"r"(swizzled_offset) "l"(&B[(innerRowB + offset) * N + innerColB * 4])
            );
        }
    }
}

template<uint BM, uint BN, uint BK, uint NUM_THREADS, uint ACCUM_SIZE, uint NUM_STAGES>
__global__ void __launch_bounds__(NUM_THREADS) GEMM_tc(float* A, float* B, float* C, long long* dbg, int N, int M, int K) {
    //bool profiler = (threadIdx.x == 0);
    //long long computeTime = 0, b_time = 0, a_time = 0, mma_time = 0;
    /*__shared__ long long kernel_start;

    if(profiler)
        kernel_start = clock64();

    __syncthreads(); */
    //long long start;
    //long long time; 

    //auto block = cooperative_groups::this_thread_block();

    // pipeline shared state 
    //__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> pipe_state;
    //auto pipe = cuda::make_pipeline(block, &pipe_state);
        // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int tile_col = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int tile_row = (linear_id / SWIZZLE_W) % gridDim.y;

    //int tile_col = blockIdx.x;
    //int tile_row = blockIdx.y;

    __shared__ alignas(128) float As[NUM_STAGES][BM * BK];
    __shared__ alignas(128) float Bs[NUM_STAGES][BK * BN];

    int warp_id = threadIdx.x / WARP_SIZE;  // 0-3
    int lane_id = threadIdx.x % 32;
    int group_id        = lane_id >> 2;     // lane_id/4
    int thread_in_group = lane_id % 4;
    constexpr uint TILES_PER_WARP_M = (BM / 4) / TILE_SIZE_M;  // (128/4)/16 = 2
    constexpr uint TILES_PER_WARP_N = (BN / 2) / TILE_SIZE_N;  // (128/2)/8  = 8
    const uint warp_m = warp_id % 4;
    const uint warp_n = warp_id / 4;

    A += K * BM * tile_row;
    B += BN * tile_col;

    const uint innerRowA  = threadIdx.x / (BK / 4);
    const uint innerColA  = threadIdx.x % (BK / 4);
    const uint rowStrideA = NUM_THREADS / (BK / 4);
    const uint innerRowB  = threadIdx.x / (BN / 4);
    const uint innerColB  = threadIdx.x % (BN / 4);
    const uint rowStrideB = NUM_THREADS / (BN / 4);
    const uint NUM_TILES  = K / BK;

    float accum[ACCUM_SIZE] = {0.0f};

    // Prologue: fill NUM_STAGES-1 buffers without waiting
    // producer_acquire + memcpy_async + producer_commit for each stage
    int fetch = 0;
    for (int i = 0; i < NUM_STAGES - 1 && i < (int)NUM_TILES; ++i, ++fetch) {
        //pipe.producer_acquire();                       // claim stage slot
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K,
            A + fetch * BK, B + (size_t)fetch * BK * N,
            As[fetch % NUM_STAGES], Bs[fetch % NUM_STAGES],
            innerRowA, innerColA, innerRowB, innerColB
        );
        //pipe.producer_commit();                        // signal copies submitted
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // outer-most loop over block tiles
    for (uint tile = 0; tile < NUM_TILES; ++tile) {
        int cur_stage  = tile  % NUM_STAGES;
        int next_stage = fetch % NUM_STAGES;   // where the next fetch lands

        // --- overlap: kick off next async fetch BEFORE waiting on cur_stage ---
        
        if (fetch < (int)NUM_TILES) {
            //if(profiler){
            //    start = clock64();
            //}
            //pipe.producer_acquire();
            wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
                N, K,
                A + fetch * BK, B + (size_t)fetch * BK * N,
                As[next_stage], Bs[next_stage],
                innerRowA, innerColA, innerRowB, innerColB
            );
            //pipe.producer_commit();
            asm volatile("cp.async.commit_group;\n" ::);
            ++fetch;
            //if(profiler){
            //   produceTime += clock64() - start;
            //}
        }
        

        uint32_t frag_A[TILES_PER_WARP_M][4];
        uint32_t frag_B[TILES_PER_WARP_N][2];
        // wait until cur_stage data is fully in smem, then release the slot
        //if (profiler){
        //    start = clock64();
        //}
        //pipe.consumer_wait();
        //if(profiler){
        //   waitTime += clock64() - start;
        //}
        // compute on cur_stage smem tile
        //if (profiler){
        //    start = clock64();
        //}

        //__syncthreads();
        

        for (int inner_tile = 0; inner_tile < BK / TILE_SIZE_K; ++inner_tile) {

            int b_row_base = inner_tile * TILE_SIZE_K;  // which K-subtile (0 or 8)
            int b_base = warp_n*(BN/2);

            int base0 = SWIZZLE_B(b_row_base + thread_in_group, b_base + group_id);
            int base1 = SWIZZLE_B(b_row_base + thread_in_group + 4, b_base + group_id);

            int base0_prefix = base0 & ~((BN/4 - 1) << 2);
            int base1_prefix = base1 & ~((BN/4 - 1) << 2);

            int base0_chunk = base0 >> 2;
            int base1_chunk = base1 >> 2;

            for (int j = 0; j < TILES_PER_WARP_N; ++j) {
                int b_col_base = b_base + (j * TILE_SIZE_N); // which N-tile
                
                //int swizzled_offset_B0 = SWIZZLE_B(b_row_base + thread_in_group, b_col_base + group_id);
                //int swizzled_offset_B1 = SWIZZLE_B(b_row_base + thread_in_group + 4, b_col_base + group_id);

                int chunk_step = (j * TILE_SIZE_N) >> 2;
                int swizzled_offset_B0 = base0_prefix | ((base0_chunk ^ chunk_step) << 2);
                int swizzled_offset_B1 = base1_prefix | ((base1_chunk ^ chunk_step) << 2);
                
                
                
                frag_B[j][0] = __float_as_uint(Bs[cur_stage][swizzled_offset_B0]);
                frag_B[j][1] = __float_as_uint(Bs[cur_stage][swizzled_offset_B1]);
                
                
            }
            
            //if(profiler){
            //    time = clock64();
            //}
            int row_in_tile = lane_id % 16;   // 0-15: natural row within the 16-row tile
            int k_half      = lane_id / 16;   // 0 or 1: selects K-cols 0-3 or 4-7
            int a_col = inner_tile * TILE_SIZE_K + k_half * 4;

            for (int i = 0; i < TILES_PER_WARP_M; ++i) {
            
                int warp_row_base = (warp_m * (BM/4) +  i*TILE_SIZE_M);
                // Each thread provides address of its row within its assigned sub-tile
                
                int a_row = warp_row_base + row_in_tile;
                
                int swizzled_offset_A = SWIZZLE_A(a_row, a_col);
                uint32_t As_ptr = __cvta_generic_to_shared(&As[cur_stage][swizzled_offset_A]);

                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(frag_A[i][0]), "=r"(frag_A[i][1]), "=r"(frag_A[i][2]), "=r"(frag_A[i][3]) // Outputs
                    : "r"(As_ptr)                                                          // Input
                );
            }    
            
            //frag_A[0] = 0, frag_A[1] = 0, frag_A[2] = 0, frag_A[3] = 0; 
            for(int i = 0; i < TILES_PER_WARP_M; ++i){
                
                for (int j = 0; j < TILES_PER_WARP_N; ++j) { 
                    
                    
                    // compute using mma ptx
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};"
                        : "+f"(accum[(i * TILES_PER_WARP_N + j) * 4 + 0]),
                          "+f"(accum[(i * TILES_PER_WARP_N + j) * 4 + 1]),
                          "+f"(accum[(i * TILES_PER_WARP_N + j) * 4 + 2]),
                          "+f"(accum[(i * TILES_PER_WARP_N + j) * 4 + 3])
                        : "r"(frag_A[i][0]), "r"(frag_A[i][1]), "r"(frag_A[i][2]), "r"(frag_A[i][3]),
                          "r"(frag_B[j][0]), "r"(frag_B[j][1])
                    );

                    
                }
            }
            //if(profiler){
            //    b_time += clock64() - time;
            //}
        }
        asm volatile("cp.async.wait_group 0;\n" ::);

        __syncthreads();
        //if(profiler){
        //computeTime += clock64() - start;
        //}
        //pipe.consumer_release();   // free cur_stage slot for reuse by producer
    
    }
    //if (profiler){
    //    start = clock64();
    //}

    // Store

    for (int tile_a = 0; tile_a < TILES_PER_WARP_M; ++tile_a) {
        for (int tile_b = 0; tile_b < TILES_PER_WARP_N; ++tile_b) {

            int acc_base = (tile_a * TILES_PER_WARP_N + tile_b) * 4;

            int row_base = tile_row * BM
                        + warp_m * (BM / 4)           // which M-partition this warp owns
                        + tile_a * TILE_SIZE_M;       // which 16-row tile within partition

            int col_base = tile_col * BN
                        + warp_n * (BN / 2)           // which N-partition this warp owns
                        + tile_b * TILE_SIZE_N;       // which 8-col tile within partition

            int row_top = row_base + group_id;
            int row_bot = row_base + group_id + 8;
            
            // Since we are using float2, we index by the pair of floats.
            // col_0 must be naturally aligned to 8 bytes (i.e., even column index).
            int col_vec = (col_base + thread_in_group * 2) >> 1; 

            // Cast global memory pointer to float2 for vectorized stores
            float2* C_f2 = (float2*)C;

            // Vectorized store for the top row (d0 and d1)
            C_f2[row_top * (N / 2) + col_vec] = make_float2(accum[acc_base + 0], accum[acc_base + 1]);

            // Vectorized store for the bottom row (d2 and d3)
            C_f2[row_bot * (N / 2) + col_vec] = make_float2(accum[acc_base + 2], accum[acc_base + 3]);
        }
    }
    //if(profiler){
    //    storeTime += clock64() - start;
    //}
    //__syncthreads();
    //if(profiler){
    //kernelTime += clock64() - kernel_start;
    //}
    /*
    if (threadIdx.x == 0) {
        int b = blockIdx.y * gridDim.x + blockIdx.x;
        //dbg[6*b + 0] = waitTime;
        dbg[5*b + 0] = computeTime;
        dbg[5*b + 1] = (long long)NUM_TILES;
        dbg[5*b + 2] = b_time;
        dbg[5*b + 3] = a_time;
        dbg[5*b + 4] = mma_time;

    } */
}