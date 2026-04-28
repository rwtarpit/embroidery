/*
each block computes a tile of output tile(BMxBN)
each warp in block computes (BM/TILE_SIZE_M, BN/TILE_SIZE_N)
for tf32, fragment size = (16,16,8)
so each warp loops over 2 tiles of A (using another subloop) with inner loop over all tiles of B (with subloop)
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include <mma.h>
using namespace nvcuda;

#define TILE_SIZE_M 16
#define TILE_SIZE_N 16
#define TILE_SIZE_K 8
#define WARP_SIZE 32


namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
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
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
}
}

template<uint BM, uint BN, uint BK, uint NUM_THREADS>
__global__ void GEMM_tc(float* A, float*B, float*C, int N, int M, int K){

    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    int warp_id = threadIdx.x / WARP_SIZE;  // 0-3
    constexpr uint total_warps = NUM_THREADS / WARP_SIZE;  // 4
    constexpr uint TILES_PER_WARP_M = (BM / total_warps) / TILE_SIZE_M;     // 2 16x16 tiles per warp
    constexpr uint TILES_PER_WARP_N = BN / TILE_SIZE_N;   // 8 16x16 tiles
    constexpr uint SUBTILES_PER_TILE = BK / TILE_SIZE_K;    // 2
    
    A += K * BM * tile_row;
    B += BN * tile_col;
    //C += K*BM*tile_row + tile_col*BN + warp_id*TILE_SIZE_N;

    wmma::fragment<wmma::matrix_a, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, wmma::precision::tf32, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, float>
    acc_frag[TILES_PER_WARP_M][TILES_PER_WARP_N];   // accums per warp

    // Initialize all of them to zero
    for (int i = 0; i < TILES_PER_WARP_M; i++){
        for (int j = 0; j < TILES_PER_WARP_N; j++){
            wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    const uint rowStrideB = NUM_THREADS / (BN / 4);


    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();

        for(uint tile_a=0; tile_a < TILES_PER_WARP_M; ++tile_a){
            for(uint subtile_a=0; subtile_a<SUBTILES_PER_TILE; ++subtile_a){
                int tile_a_idx = (warp_id + total_warps*tile_a + subtile_a*TILE_SIZE_K) * TILE_SIZE_M;
                float* As_ptr = &As[tile_a_idx];
                wmma::load_matrix_sync(a_frag, As_ptr, BM);
                
                for(uint subtile_b=0; subtile_b<SUBTILES_PER_TILE; ++subtile_b){
                    for(uint tile_b=0; tile_b < TILES_PER_WARP_N; ++tile_b){
                        int tile_b_idx = tile_b * TILE_SIZE_N + TILE_SIZE_K*subtile_b*BN;
                        float* Bs_ptr = &Bs[tile_b_idx];
                        wmma::load_matrix_sync(b_frag, Bs_ptr, BN);

                        wmma::mma_sync(acc_frag[tile_a][tile_b], a_frag, b_frag, acc_frag[tile_a][tile_b]);
                    }
                }
            }
        }
    A += BK;       // advance A pointer along K
    B += BK * N;   // advance B pointer along K
    __syncthreads();
    }
    // store back to GMEM
    for (uint tile_a = 0; tile_a < TILES_PER_WARP_M; ++tile_a) {
        for (uint tile_b = 0; tile_b < TILES_PER_WARP_N; ++tile_b) {
            int row = (tile_row * BM) + (warp_id + total_warps * tile_a) * TILE_SIZE_M;
            int col = (tile_col * BN) + tile_b * TILE_SIZE_N;
            wmma::store_matrix_sync(&C[row * N + col], acc_frag[tile_a][tile_b], N, wmma::mem_row_major);
        }
    }
}