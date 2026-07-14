// 95% on 256x32x128 tile size. simplifies main loop for exact 2 stages pipeline using xor bit flip instead
// of tracking no. of tiles consumed

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>
using namespace nvcuda;

#define TILE_SIZE_M 16
#define TILE_SIZE_N 8
#define TILE_SIZE_K 8
#define WARP_SIZE 32
//#define NUM_STAGES 2   // hardcoded: this version assumes a double-buffered pipeline

__device__ __forceinline__ int SWIZZLE_A(int row, int col) {
    return (row * 32) + (col ^ ((row & 7) << 2));
}

__device__ __forceinline__ int SWIZZLE_B(int row, int col) {
    return (row * 128) + (col ^ ((row & 31) << 2));
}

namespace wt {
    template <const int BM, const int BN, const int BK, const int rowStrideA,
              const int rowStrideB>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 float *As, float *Bs, int innerRowA, int innerColA,
                                 int innerRowB, int innerColB) {

#pragma unroll
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            uint32_t swizzled_offset = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[SWIZZLE_A(innerRowA + offset, innerColA * 4)]));

            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(swizzled_offset), "l"(&A[(innerRowA + offset) * K + innerColA * 4])
            );
        }
#pragma unroll
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            uint32_t swizzled_offset = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[SWIZZLE_B(innerRowB + offset, innerColB * 4)]));

            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(swizzled_offset), "l"(&B[(innerRowB + offset) * N + innerColB * 4])
            );
        }
    }
}

template<uint BM, uint BN, uint BK, uint NUM_THREADS, uint ACCUM_SIZE, uint NUM_STAGES>
__global__ void __launch_bounds__(NUM_THREADS) GEMM_tc(float* A, float* B, float* C, long long* dbg, int N, int M, int K) {

    (void)dbg;

    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int tile_col = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int tile_row = (linear_id / SWIZZLE_W) % gridDim.y;

    // --- shared memory: only the double-buffered A/B tiles now.
    // No more Cs aliasing — the epilogue writes straight to global memory. ---
    struct SharedTiles {
        float As[NUM_STAGES][BM * BK];
        float Bs[NUM_STAGES][BK * BN];
    };
    __shared__ alignas(128) SharedTiles smem;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % 32;
    int group_id        = lane_id >> 2;
    int thread_in_group = lane_id % 4;
    constexpr uint TILES_PER_WARP_M = (BM / 4) / TILE_SIZE_M;
    constexpr uint TILES_PER_WARP_N = (BN / 2) / TILE_SIZE_N;
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

    // Warp/lane-derived offsets used by the compute body — constant across
    // all K-tiles, so hoisted out here instead of being recomputed per tile.
    int b_base = warp_n * (BN / 2);
    int row_in_tile = lane_id % 16;
    int k_half      = lane_id / 16;
    const uint warp_row_partition = warp_m * (BM / 4);

    // ---------------- Prologue: load tile 0 into stage 0 ----------------
    int stage = 0;
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, smem.As[0], smem.Bs[0],
        innerRowA, innerColA, innerRowB, innerColB);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    // ldmatrix + mma pass over one already-resident stage. Shared across the
    // main loop and the pulled-out tail so there's no duplicated asm.
    auto computeStage = [&](int st) {
        uint32_t frag_A[TILES_PER_WARP_M][4];
        uint32_t frag_B[TILES_PER_WARP_N][2];
        int a_prefix[TILES_PER_WARP_M];
        int a_chunk[TILES_PER_WARP_M];

#pragma unroll
        for (int i = 0; i < (int)TILES_PER_WARP_M; ++i) {
            int warp_row_base = warp_row_partition + i * TILE_SIZE_M;
            int a_row = warp_row_base + row_in_tile;
            int base = SWIZZLE_A(a_row, k_half * 4);
            a_prefix[i] = base & ~(0x7 << 2);
            a_chunk[i]  = base >> 2;
        }

#pragma unroll
        for (int inner_tile = 0; inner_tile < (int)(BK / TILE_SIZE_K); ++inner_tile) {
            int b_row_base = inner_tile * TILE_SIZE_K;

            int base0 = SWIZZLE_B(b_row_base + thread_in_group, b_base + group_id);
            int base1 = SWIZZLE_B(b_row_base + thread_in_group + 4, b_base + group_id);

            int base0_prefix = base0 & ~((BN / 4 - 1) << 2);
            int base1_prefix = base1 & ~((BN / 4 - 1) << 2);
            int base0_chunk  = base0 >> 2;
            int base1_chunk  = base1 >> 2;

#pragma unroll
            for (int j = 0; j < (int)TILES_PER_WARP_N; ++j) {
                int chunk_step = (j * TILE_SIZE_N) >> 2;
                int off0 = base0_prefix | ((base0_chunk ^ chunk_step) << 2);
                int off1 = base1_prefix | ((base1_chunk ^ chunk_step) << 2);
                frag_B[j][0] = __float_as_uint(smem.Bs[st][off0]);
                frag_B[j][1] = __float_as_uint(smem.Bs[st][off1]);
            }

            int chunk_step_a = inner_tile * 2;

#pragma unroll
            for (int i = 0; i < (int)TILES_PER_WARP_M; ++i) {
                int off = a_prefix[i] | ((a_chunk[i] ^ chunk_step_a) << 2);
                uint32_t As_ptr = __cvta_generic_to_shared(&smem.As[st][off]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(frag_A[i][0]), "=r"(frag_A[i][1]), "=r"(frag_A[i][2]), "=r"(frag_A[i][3])
                    : "r"(As_ptr)
                );
            }

#pragma unroll
            for (int i = 0; i < (int)TILES_PER_WARP_M; ++i) {
#pragma unroll
                for (int j = 0; j < (int)TILES_PER_WARP_N; ++j) {
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
        }
    };

    // ---------------- Main loop: NUM_TILES - 1 steady-state iterations ----------------
    // Each iteration prefetches tile+1 into the *other* buffer, computes on
    // the current buffer while that copy is in flight, then waits for it
    // before flipping the stage bit.
    for (uint tile = 0; tile + 1 < NUM_TILES; ++tile) {
        int cur = stage;
        int nxt = cur ^ 1;   // XOR toggle — no fetch counter, no modulo

        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K,
            A + (tile + 1) * BK, B + (size_t)(tile + 1) * BK * N,
            smem.As[nxt], smem.Bs[nxt],
            innerRowA, innerColA, innerRowB, innerColB
        );
        asm volatile("cp.async.commit_group;\n" ::);

        computeStage(cur);

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        stage = nxt;
    }

    // ---------------- Tail: final tile, no prefetch/wait needed ----------------
    computeStage(stage);

    // ---------------- Epilogue: direct float2 stores to global memory ----------------
    // No shared-memory staging, no swizzle, no barriers — every thread's
    // accum values are already final and private, so they can go straight
    // to C.
#pragma unroll
    for (int tile_a = 0; tile_a < (int)TILES_PER_WARP_M; ++tile_a) {
#pragma unroll
        for (int tile_b = 0; tile_b < (int)TILES_PER_WARP_N; ++tile_b) {
            int acc_base = (tile_a * TILES_PER_WARP_N + tile_b) * 4;

            int row_base = warp_m * (BM / 4) + tile_a * TILE_SIZE_M;
            int col_base = warp_n * (BN / 2) + tile_b * TILE_SIZE_N;

            int row_top = row_base + group_id;
            int row_bot = row_top + 8;
            int col     = col_base + thread_in_group * 2;

            int global_row_top = tile_row * BM + row_top;
            int global_row_bot = tile_row * BM + row_bot;
            int global_col     = tile_col * BN + col;

            *reinterpret_cast<float2*>(&C[(size_t)global_row_top * N + global_col]) =
                make_float2(accum[acc_base + 0], accum[acc_base + 1]);
            *reinterpret_cast<float2*>(&C[(size_t)global_row_bot * N + global_col]) =
                make_float2(accum[acc_base + 2], accum[acc_base + 3]);
        }
    }
}
