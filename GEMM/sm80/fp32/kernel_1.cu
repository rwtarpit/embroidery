// reaches ~33-35% of cublas

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WM = 64;
const int WN = 64;
const int WMMA_M = 16;  // only supported tf32 fragment shape
const int WMMA_N = 16;
const int WMMA_K = 8;

namespace wmma = nvcuda::wmma;

namespace wt {
// ---- UNCHANGED: identical to the CUDA-core version ----
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

// computes WMITER x WNITER fragments of shape WMMA_M x WMMA_N per warp-tile.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WMMA_M,
          const int WMMA_N, const int WMMA_K>
__device__ void processFromSmemWMMA(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        accum[WMITER][WNITER],
    const float *As, const float *Bs, const uint warpRow, const uint warpCol) {
  for (uint dotIdx = 0; dotIdx < BK; dotIdx += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                    wmma::precision::tf32, wmma::col_major>
        aFrag[WMITER];
    // Bs is stored row-major with leading dim BN (natural B layout).
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                    wmma::precision::tf32, wmma::row_major>
        bFrag[WNITER];

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      const float *aPtr =
          &As[dotIdx * BM + warpRow * WM + wSubRowIdx * WMMA_M];
      wmma::load_matrix_sync(aFrag[wSubRowIdx], aPtr, BM);
      for (int i = 0; i < aFrag[wSubRowIdx].num_elements; ++i) {
        aFrag[wSubRowIdx].x[i] = wmma::__float_to_tf32(aFrag[wSubRowIdx].x[i]);
      }
    }

    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      const float *bPtr =
          &Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WMMA_N];
      wmma::load_matrix_sync(bFrag[wSubColIdx], bPtr, BN);
      for (int i = 0; i < bFrag[wSubColIdx].num_elements; ++i) {
        bFrag[wSubColIdx].x[i] = wmma::__float_to_tf32(bFrag[wSubColIdx].x[i]);
      }
    }

    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        wmma::mma_sync(accum[wSubRowIdx][wSubColIdx], aFrag[wSubRowIdx],
                       bFrag[wSubColIdx], accum[wSubRowIdx][wSubColIdx]);
      }
    }
  }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMMA_M/WMMA_N/WMMA_K Tensor Core fragment shape (tf32: 16x16x8)
 */
template <const int BM, const int BN, const int BK, const int NUM_THREADS, const int ACCUM_SIZE,
          const int NUM_STAGES>
__global__ void __launch_bounds__(NUM_THREADS)
    GEMM_tc(float *A, float *B, float *C, int M, int N, int K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / 32; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // number of WMMA fragment tiles covering each warp's output tile
  constexpr uint WMITER = WM / WMMA_M;
  constexpr uint WNITER = WN / WMMA_N;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // per-warp accumulator fragments (replaces threadResults[])
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      accum[WMITER][WNITER];
  for (uint i = 0; i < WMITER; ++i)
    for (uint j = 0; j < WNITER; ++j)
      wmma::fill_fragment(accum[i][j], 0.0f);

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmemWMMA<BM, BN, BK, WM, WN, WMITER, WNITER, WMMA_M,
                            WMMA_N, WMMA_K>(accum, As, Bs, warpRow, warpCol);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results: C = alpha * accum + beta * C, fused via a
  // load-modify-store on an accumulator fragment for each WMMA sub-tile.
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim =
          C + (wSubRowIdx * WMMA_M) * N + wSubColIdx * WMMA_N;

      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
      wmma::load_matrix_sync(cFrag, C_interim, N, wmma::mem_row_major);

      for (int i = 0; i < cFrag.num_elements; ++i) {
        cFrag.x[i] = accum[wSubRowIdx][wSubColIdx].x[i];
      }

      wmma::store_matrix_sync(C_interim, cFrag, N, wmma::mem_row_major);
    }
  }
}
