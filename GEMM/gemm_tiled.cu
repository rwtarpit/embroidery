#include <cuda_runtime.h>


// A(KxM) @ B(MxN) = C(KxN)
// {"k":4096,"m":4096,"n":5120}
// tiles A(128x16); B(16x128); C(128x128)
// SMEM = [128*16*2] = 16KB/block for fp32
// each block calculates one output tile
// grid= (32,40); block= (128,)

// acc to my brain, As will have 2 way bank conflict and Bs will have none
__global__
void gemm( float* A, float* B, float* C,int k, int m, int n, int TOTAL_THREADS, float alpha, float beta){
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int WARP_SIZE = 32;
    int TILE_SIZE_N = 128;
    int TILE_SIZE_K = 128;
    int TILE_SIZE_M = 16;   // inner dim

    // warp tiling
    int WARP_TILE_K = 64;
    int WARP_TILE_N = 64;
    // WARP_TILE_M will be 16; inner dim

    // 2D square block of 4 warps
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // thread tiling
    int THREAD_TILE_K = 8;  // single iteration
    int THREAD_TILE_N = 4;
    int THREAD_TILE_N_ITER = 4;

    // thread in warp
    int warpTid = threadIdx.x % WARP_SIZE;  // 0-31
    int warpTidRow = warpTid / THREAD_TILE_N;   // 0-7
    int warpTidCol = warpTid % THREAD_TILE_N;   // 0-3
    int warpsliceN = WARP_TILE_N / THREAD_TILE_N_ITER; // 16

    // registers to accumulate data
    float REG_A[THREAD_TILE_K] = {0.0};
    float REG_B[THREAD_TILE_N*THREAD_TILE_N_ITER] = {0.0};
    float THREAD_OUT[THREAD_TILE_K*THREAD_TILE_N*THREAD_TILE_N_ITER] = {0.0};

    A += tile_x * TILE_SIZE_K * m;
    B += tile_y * TILE_SIZE_N;
    // bring C to start of warp for writing output back
    C+= tile_y*TILE_SIZE_K*m + tile_x*TILE_SIZE_N + warp_row*warpTidRow*m + warp_col*warpTidCol;

    int row_stride_B = TOTAL_THREADS / (TILE_SIZE_N/4);   // each thread skips 4 rows in each iteration
    int innerRowB = threadIdx.x / (TILE_SIZE_N/4);
    int innerColB = threadIdx.x % (TILE_SIZE_N/4);

    int row_stride_A = (TOTAL_THREADS*4) / TILE_SIZE_M; // each thread skips 32 rows in each iteration
    int innerRowA = threadIdx.x / (TILE_SIZE_M/4);
    int innerColA = threadIdx.x % (TILE_SIZE_M/4);

    __shared__ float smem[128*16*2];
    float* smem_A = smem;
    float* smem_B = &smem[16*128];

    float4* vector_A = reinterpret_cast<float4*>(A);
    float4* vector_B = reinterpret_cast<float4*>(B);
    float4* vector_C = reinterpret_cast<float4*>(C);

    // outer load
    for(int blkId=0; blkId<m; blkId+=TILE_SIZE_M){
        // load B's tile from GMEM into SMEM
        for(int i=0; i+row_stride_B<TILE_SIZE_M; i+=row_stride_B){
            float4 elements = vector_B[(innerRowB + i)*n + innerColB];
            smem_B[(i+innerRowB)*TILE_SIZE_N + innerColB + 0] = elements.x;
            smem_B[(i+innerRowB)*TILE_SIZE_N + innerColB + 1] = elements.y;
            smem_B[(i+innerRowB)*TILE_SIZE_N + innerColB + 2] = elements.z;
            smem_B[(i+innerRowB)*TILE_SIZE_N + innerColB + 3] = elements.w;
        }
        // load A's tile from GMEM into SMEM
        for(int i=0; i+row_stride_A<TILE_SIZE_K; i+=row_stride_A){
            float4 elements = vector_A[(i+innerRowA)*m + innerColA];
            smem_A[(innerColA + 0) * TILE_SIZE_K + innerRowA + i] = elements.x;
            smem_A[(innerColA + 1) * TILE_SIZE_K + innerRowA + i] = elements.y;
            smem_A[(innerColA + 2) * TILE_SIZE_K + innerRowA + i] = elements.z;
            smem_A[(innerColA + 3) * TILE_SIZE_K + innerRowA + i] = elements.w;
        }
        __syncthreads();

        // outer loop
        // we iteratively load partial tiles from smem
        for(int dotIdx=0; dotIdx<TILE_SIZE_M; ++dotIdx){

            // load from A
            for(int el=0; el<THREAD_TILE_K; ++el){
                REG_A[el] = smem_A[warp_row*WARP_TILE_K + dotIdx*TILE_SIZE_K + warpTidRow*THREAD_TILE_K + el];
            }
            // load from B
            for(int i=0; i<THREAD_TILE_N_ITER; ++i){
                for(int el=0; el<THREAD_TILE_N; ++el){
                    REG_B[i*THREAD_TILE_N + el] = smem_B[warp_col*WARP_TILE_N + dotIdx*TILE_SIZE_N + i*warpsliceN + warpTidCol*THREAD_TILE_N + el];
                }
            }
            // compute
            for(int wCol=0; wCol<THREAD_TILE_N_ITER; ++wCol){
                for(int resIdxK=0; resIdxK<THREAD_TILE_K; ++resIdxK){
                    for(int resIdxN=0; resIdxN<THREAD_TILE_N; ++resIdxN){
                        THREAD_OUT[resIdxK*(THREAD_TILE_N*THREAD_TILE_N_ITER) + wCol*THREAD_TILE_N + resIdxN] +=
                            REG_A[resIdxK] * REG_B[wCol*THREAD_TILE_N + resIdxN];
                    }
                }
            }
        }

        B += TILE_SIZE_M * n;
        A += TILE_SIZE_M;
        __syncthreads();
    }

    // store back to GMEM
    for(int wCol=0; wCol<THREAD_TILE_N_ITER; ++wCol){
        float* C_interim = C + wCol*warpsliceN;
        for(int resIdxK=0; resIdxK<THREAD_TILE_K; ++resIdxK){
            for(int resIdxN=0; resIdxN<THREAD_TILE_N; resIdxN+=4){  // for float4
                float4 tmp = reinterpret_cast<float4*>(& C_interim[warpTidRow*THREAD_TILE_K*n + resIdxK*n + resIdxN])[0];
                const int i = (resIdxK) * (TILE_SIZE_N * THREAD_TILE_N) +
                        wCol * THREAD_TILE_N + resIdxN;
                tmp.x = alpha * THREAD_OUT[i + 0] + beta * tmp.x;
                tmp.y = alpha * THREAD_OUT[i + 1] + beta * tmp.y;
                tmp.z = alpha * THREAD_OUT[i + 2] + beta * tmp.z;
                tmp.w = alpha * THREAD_OUT[i + 3] + beta * tmp.w;
                // write back
                reinterpret_cast<float4*>(& C_interim[warpTidRow*THREAD_TILE_K*n + resIdxK*n + resIdxN])[0] = tmp;
            }
        }
    }

}