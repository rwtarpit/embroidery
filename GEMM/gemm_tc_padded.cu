/*
each block computes a tile of output tile(BMxBN)
each warp in block computes (BM/TILE_SIZE_M, BN/TILE_SIZE_N)
for tf32, fragment size = (16,16,8)
so each warp loops over 2 tiles of A (using another subloop) with inner loop over all tiles of B (with subloop)

=== BANK CONFLICT ANALYSIS & FIXES ===

Shared memory has 32 banks, each 4 bytes wide.
Bank of a float element = (byte_offset / 4) % 32.
A conflict occurs when ≥2 threads in a warp hit different addresses in the same bank.

--- As (col-major): As[k_col * BM + m_row] ---
  wmma loads with stride = BM floats.
  Stride in bytes = BM * 4.
  Example BM=64: stride = 256B → 256/4 = 64 banks → 64 % 32 = 0.
  Every row in the fragment starts at the SAME bank. → 4-way conflict (16 rows / 8-bank cycle).

  Fix: pad each "column" (the BM-length dimension) by PAD_A floats.
  Padded stride = (BM + PAD_A).
  Choose PAD_A so (BM + PAD_A) % 32 ≠ 0 and not a small divisor of 32.
  BM=64: PAD_A=4 → stride=68 floats, 272B → 272/4=68 banks, 68%32=4. 
  Rows land on banks 0,4,8,12,16,20,24,28,0,4,... → conflict-free within 8-row groups 
  (Each successive row shifts by 4 banks; 8 shifts × 4 = 32 → full cycle = no alias in 8 rows,
   and since BM tile rows = 16, the pattern repeats cleanly.)

--- Bs (row-major): Bs[k_row * BN + n_col] ---
  wmma loads with stride = BN floats.
  Stride in bytes = BN * 4.
  Example BN=128: stride = 512B → 512/4=128 banks → 128%32=0. → 8-way conflict.
  Example BN=64:  stride = 256B → 256/4=64  banks → 64%32=0.  → 4-way conflict.

  Fix: pad each row of Bs by PAD_B floats.
  Padded stride = (BN + PAD_B).
  BN=128: PAD_B=4 → stride=132, 528B → 528/4=132 banks, 132%32=4. Conflict-free 
  BN=64:  PAD_B=4 → stride=68,  272B → 272/4=68  banks, 68%32=4.  Conflict-free 

  PAD_B=4 works for all BN that are multiples of 32 (which they must be for wmma).

Summary of changes:
  1. As declared as As[BM * BK + PAD_A * BK]  — pad each of BK columns by PAD_A
       → actual col stride = BM + PAD_A
  2. Bs declared as Bs[BK * (BN + PAD_B)]     — pad each of BK rows by PAD_B
       → actual row stride = BN + PAD_B
  3. loadFromGmem writes use padded strides
  4. wmma::load_matrix_sync uses padded strides as ldm
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <mma.h>
using namespace nvcuda;

#define TILE_SIZE_M 16
#define TILE_SIZE_N 16
#define TILE_SIZE_K 8
#define WARP_SIZE   32

// Padding constants (in floats).
// PAD_A: appended to the M-dimension of each K-slice of As (col-major stride).
// PAD_B: appended to the N-dimension of each K-row of Bs (row-major stride).
// Value 4 ensures the padded stride % 32 == 4, breaking all aliasing for
// BM/BN that are multiples of 32.
#define PAD_A 2     //4
#define PAD_B 8     //8

// pad of 4 for A and B with BM BN BK NUM_THREADS = 128,128,32,128 : 38.19 tflops median tile 4.498ms
// pad of 4-A 8-B with BM BN BK NUM_THREADS = 128,128,32,128 : 39.47 tflops median tile 4.353ms
// pad of 2-A 8-B with BM BN BK NUM_THREADS = 128,128,32,128 : 40.38 tflops median tile 4.255

namespace wt {

// As is col-major: logical shape [BK][BM], stored as col-stride = (BM + PAD_A).
//   As[k][m] = As_ptr[k * (BM + PAD_A) + m]
//
// Bs is row-major: logical shape [BK][BN], stored as row-stride = (BN + PAD_B).
//   Bs[k][n] = Bs_ptr[k * (BN + PAD_B) + n]
//
// The global load of A transposes on the fly (row-major global → col-major smem).
template <const int BM, const int BN, const int BK,
          const int rowStrideA, const int rowStrideB>
__device__ void loadFromGmem(int N, int K,
                              const float *A, const float *B,
                              float *As,      float *Bs,
                              int innerRowA,  int innerColA,
                              int innerRowB,  int innerColB)
{
    // --- Load A: global row-major → shared col-major (with padding) ---
    // Each thread loads 4 floats (float4) from a row of A in global memory,
    // then scatters them into columns of As in shared memory.
    // Padded col-stride in As = BM + PAD_A.
    constexpr int AS_COL_STRIDE = BM + PAD_A;

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const float4 tmp = reinterpret_cast<const float4 *>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];

        // Write transposed: element at global (row, k_col+c) → As[k_col+c][row]
        As[(innerColA * 4 + 0) * AS_COL_STRIDE + (innerRowA + offset)] = tmp.x;
        As[(innerColA * 4 + 1) * AS_COL_STRIDE + (innerRowA + offset)] = tmp.y;
        As[(innerColA * 4 + 2) * AS_COL_STRIDE + (innerRowA + offset)] = tmp.z;
        As[(innerColA * 4 + 3) * AS_COL_STRIDE + (innerRowA + offset)] = tmp.w;
    }

    // --- Load B: global row-major → shared row-major (with padding) ---
    // Padded row-stride in Bs = BN + PAD_B.
    constexpr int BS_ROW_STRIDE = BN + PAD_B;

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        // Destination in Bs uses padded stride; source in B uses N (full matrix width).
        const float4 src = reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
        reinterpret_cast<float4 *>(
            &Bs[(innerRowB + offset) * BS_ROW_STRIDE + innerColB * 4])[0] = src;
    }
}

} // namespace wt


template<uint BM, uint BN, uint BK, uint NUM_THREADS>
__global__ void GEMM_tc(float* A, float* B, float* C, int N, int M, int K)
{
    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    // Padded shared memory declarations.
    //
    // As: col-major [BK columns, each BM+PAD_A tall]
    //   Total floats = BK * (BM + PAD_A)
    //   Bank analysis: col stride = (BM+PAD_A)*4 bytes
    //     BM=64: (64+4)*4 = 272B → bank offset per col = 272/4 % 32 = 68%32 = 4 
    //
    // Bs: row-major [BK rows, each BN+PAD_B wide]
    //   Total floats = BK * (BN + PAD_B)
    //   Bank analysis: row stride = (BN+PAD_B)*4 bytes
    //     BN=128: (128+4)*4 = 528B → bank offset per row = 528/4 % 32 = 132%32 = 4 
    //     BN=64:  (64+4)*4  = 272B → bank offset per row = 272/4 % 32 = 68%32  = 4 
    __shared__ float As[BK * (BM + PAD_A)];
    __shared__ float Bs[BK * (BN + PAD_B)];

    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr uint total_warps      = NUM_THREADS / WARP_SIZE;
    constexpr uint TILES_PER_WARP_M = (BM / total_warps) / TILE_SIZE_M;
    constexpr uint TILES_PER_WARP_N = BN / TILE_SIZE_N;
    constexpr uint SUBTILES_PER_TILE = BK / TILE_SIZE_K;

    A += K  * BM * tile_row;
    B += BN * tile_col;

    wmma::fragment<wmma::matrix_a,    TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
                   wmma::precision::tf32, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b,    TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
                   wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
                   float> acc_frag[TILES_PER_WARP_M][TILES_PER_WARP_N];

    for (int i = 0; i < TILES_PER_WARP_M; i++)
        for (int j = 0; j < TILES_PER_WARP_N; j++)
            wmma::fill_fragment(acc_frag[i][j], 0.0f);

    // Global→shared index arithmetic (unchanged from original)
    const uint innerRowA  = threadIdx.x / (BK / 4);
    const uint innerColA  = threadIdx.x % (BK / 4);
    const uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB  = threadIdx.x / (BN / 4);
    const uint innerColB  = threadIdx.x % (BN / 4);
    const uint rowStrideB = NUM_THREADS / (BN / 4);

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs,
            innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();

        for (uint tile_a = 0; tile_a < TILES_PER_WARP_M; ++tile_a) {
            // row_in_M: starting row of this warp's M-tile within the BM block.
            int row_in_M = (warp_id * TILES_PER_WARP_M + tile_a) * TILE_SIZE_M;

            for (uint subtile = 0; subtile < SUBTILES_PER_TILE; ++subtile) {

                // As is col-major with padded stride (BM + PAD_A).
                // Pointer to the start of subtile's K-slice for this warp's M-rows.
                // As[k_start .. k_start+TILE_SIZE_K-1][row_in_M .. row_in_M+15]
                // = &As[subtile * TILE_SIZE_K * (BM + PAD_A) + row_in_M]
                // ldm = BM + PAD_A  (distance in floats between consecutive K-rows in As)
                float* As_ptr = &As[subtile * TILE_SIZE_K * (BM + PAD_A) + row_in_M];
                wmma::load_matrix_sync(a_frag, As_ptr, BM + PAD_A);
                //                                      ^^^^^^^^^^^^
                // FIXED: was BM (unpadded). Must match actual allocation stride
                // so wmma steps correctly between K-rows in shared memory.

                for (uint tile_b = 0; tile_b < TILES_PER_WARP_N; ++tile_b) {
                    // Bs is row-major with padded stride (BN + PAD_B).
                    // Bs[k_start][tile_b * TILE_SIZE_N]
                    // = &Bs[subtile * TILE_SIZE_K * (BN + PAD_B) + tile_b * TILE_SIZE_N]
                    // ldm = BN + PAD_B
                    float* Bs_ptr = &Bs[subtile * TILE_SIZE_K * (BN + PAD_B)
                                        + tile_b * TILE_SIZE_N];
                    wmma::load_matrix_sync(b_frag, Bs_ptr, BN + PAD_B);
                    //                                      ^^^^^^^^^^^^
                    // FIXED: was BN (unpadded).

                    wmma::mma_sync(acc_frag[tile_a][tile_b],
                                   a_frag, b_frag,
                                   acc_frag[tile_a][tile_b]);
                }
            }
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // Store results
    for (uint tile_a = 0; tile_a < TILES_PER_WARP_M; ++tile_a) {
        for (uint tile_b = 0; tile_b < TILES_PER_WARP_N; ++tile_b) {
            int row = tile_row * BM + (warp_id * TILES_PER_WARP_M + tile_a) * TILE_SIZE_M;
            int col = tile_col * BN + tile_b * TILE_SIZE_N;
            wmma::store_matrix_sync(&C[row * N + col],
                                    acc_frag[tile_a][tile_b],
                                    N, wmma::mem_row_major);
        }
    }
}