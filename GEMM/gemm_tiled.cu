#include <cuda_runtime.h>


// A(KxM) @ B(MxN) = C(KxN)
// {"k":4096,"m":4096,"n":5120}
// tiles A(128x16); B(16x128); C(128x128)
// SMEM = [128*16*2] = 16KB/block for fp32
// each block calculates one output tile
// grid= (32,40); block= (128,)
__global__
void gemm( float* A, float* B, float* C,int k, int m, int n, int TOTAL_THREADS){
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int WARP_SIZE = 32;
    int warp_id = threadIdx.x / WARP_SIZE;
    int TILE_SIZE_N = 128;
    int TILE_SIZE_K = 128;
    int TILE_SIZE_M = 16;   // inner dim
    
    A += tile_x * TILE_SIZE_K * m;
    B += tile_y * TILE_SIZE_N;

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

    float REG_A[4];
    float REG_B[4];

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

        // load B's tile from SMEM into registers (need 16 elements)
        for(int i=0; i+row_stride_B<TILE_SIZE_M; i+=row_stride_B){
            for(int j=0; j<4; ++j){
                REG_B[j] = smem_B[(innerRowB+i)*n + innerColB + (WARP_SIZE*j)];
            }
        }
        // load A's tile from SMEM into registers


        B += TILE_SIZE_M * n;
        A += TILE_SIZE_M;
    }

}