#include <cuda_runtime.h>
// #include <torch/extension.h>

#define K_TILE_SIZE 128
#define N_TILE_SIZE 128

// A(KxM) @ B(MxN) = C(KxN)
// tiles A(128x16); B(16x128); C(128x128)
// grid= (K/128, N/128); block= (128,)
template <const int TILE_SIZE_N, const int TILE_SIZE_K, const int TILE_SIZE_M,
          const int WARP_TILE_K, const int WARP_TILE_N,
          const int THREAD_TILE_N_ITER, const int THREAD_TILE_K,
          const int THREAD_TILE_N, const int WARP_SIZE>
__global__ void gemm(float *A, float *B, float *C, int k, int m, int n,
                     int TOTAL_THREADS, float alpha, float beta) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;

    // 2x2 warp grid in a block
    int warp_id  = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // thread position inside its warp
    int warpTid    = threadIdx.x % WARP_SIZE;
    int warpTidRow = warpTid / THREAD_TILE_N;   // 0-7
    int warpTidCol = warpTid % THREAD_TILE_N;   // 0-3

    // N-slice each warp iterates over
    int warpsliceN = WARP_TILE_N / THREAD_TILE_N_ITER;

    float REG_A[THREAD_TILE_K] = {0.0f};
    float REG_B[THREAD_TILE_N_ITER * THREAD_TILE_N] = {0.0f};
    float THREAD_OUT[THREAD_TILE_K * THREAD_TILE_N_ITER * THREAD_TILE_N] = {0.0f};

    A += tile_x * TILE_SIZE_K * m;
    B += tile_y * TILE_SIZE_N;
    C += tile_x * TILE_SIZE_K * n
       + tile_y * TILE_SIZE_N
       + warp_row * WARP_TILE_K * n
       + warp_col * WARP_TILE_N;

    // GMEM load indices
    // B: each thread loads a float4, row stride covers TILE_SIZE_M rows
    int row_stride_B = TOTAL_THREADS / (TILE_SIZE_N / 4);
    int innerRowB = threadIdx.x / (TILE_SIZE_N / 4);
    int innerColB = threadIdx.x % (TILE_SIZE_N / 4);

    // A: each thread loads a float4, row stride covers TILE_SIZE_K rows
    int row_stride_A = (TOTAL_THREADS * 4) / TILE_SIZE_M;
    int innerRowA = threadIdx.x / (TILE_SIZE_M / 4);
    int innerColA = threadIdx.x % (TILE_SIZE_M / 4);

    __shared__ float smem[TILE_SIZE_M * TILE_SIZE_K + TILE_SIZE_M * TILE_SIZE_N];
    float *smem_A = smem;
    float *smem_B = smem + TILE_SIZE_M * TILE_SIZE_K;

    // outer loop over M dimension in chunks of TILE_SIZE_M
    for (int blkId = 0; blkId < m; blkId += TILE_SIZE_M) {

        // load B tile (TILE_SIZE_M x TILE_SIZE_N) from GMEM into SMEM
        for (int i = 0; i + row_stride_B <= TILE_SIZE_M; i += row_stride_B) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &B[(innerRowB + i) * n + innerColB * 4])[0];
            smem_B[(innerRowB + i) * TILE_SIZE_N + innerColB * 4 + 0] = tmp.x;
            smem_B[(innerRowB + i) * TILE_SIZE_N + innerColB * 4 + 1] = tmp.y;
            smem_B[(innerRowB + i) * TILE_SIZE_N + innerColB * 4 + 2] = tmp.z;
            smem_B[(innerRowB + i) * TILE_SIZE_N + innerColB * 4 + 3] = tmp.w;
        }

        // load A tile (TILE_SIZE_K x TILE_SIZE_M) transposed into SMEM
        // stored as (TILE_SIZE_M x TILE_SIZE_K) so inner-dim access is contiguous
        for (int i = 0; i + row_stride_A <= TILE_SIZE_K; i += row_stride_A) {
            float4 tmp = reinterpret_cast<const float4 *>(
                &A[(innerRowA + i) * m + innerColA * 4])[0];
            smem_A[(innerColA * 4 + 0) * TILE_SIZE_K + innerRowA + i] = tmp.x;
            smem_A[(innerColA * 4 + 1) * TILE_SIZE_K + innerRowA + i] = tmp.y;
            smem_A[(innerColA * 4 + 2) * TILE_SIZE_K + innerRowA + i] = tmp.z;
            smem_A[(innerColA * 4 + 3) * TILE_SIZE_K + innerRowA + i] = tmp.w;
        }
        __syncthreads();

        // compute: iterate over the TILE_SIZE_M inner dimension
        for (int dotIdx = 0; dotIdx < TILE_SIZE_M; ++dotIdx) {

            // load this thread's slice of A from smem into registers
            for (int el = 0; el < THREAD_TILE_K; ++el) {
                REG_A[el] = smem_A[dotIdx * TILE_SIZE_K
                                   + warp_row * WARP_TILE_K
                                   + warpTidRow * THREAD_TILE_K
                                   + el];
            }

            // load this thread's slice of B from smem into registers
            for (int i = 0; i < THREAD_TILE_N_ITER; ++i) {
                for (int el = 0; el < THREAD_TILE_N; ++el) {
                    REG_B[i * THREAD_TILE_N + el] =
                        smem_B[dotIdx * TILE_SIZE_N
                               + warp_col * WARP_TILE_N
                               + i * warpsliceN
                               + warpTidCol * THREAD_TILE_N
                               + el];
                }
            }

            // outer product accumulate
            for (int resIdxK = 0; resIdxK < THREAD_TILE_K; ++resIdxK) {
                for (int wCol = 0; wCol < THREAD_TILE_N_ITER; ++wCol) {
                    for (int resIdxN = 0; resIdxN < THREAD_TILE_N; ++resIdxN) {
                        THREAD_OUT[resIdxK * (THREAD_TILE_N_ITER * THREAD_TILE_N)
                                   + wCol * THREAD_TILE_N
                                   + resIdxN] +=
                            REG_A[resIdxK] * REG_B[wCol * THREAD_TILE_N + resIdxN];
                    }
                }
            }
        }

        // advance to next TILE_SIZE_M chunk
        A += TILE_SIZE_M;
        B += TILE_SIZE_M * n;
        __syncthreads();
    }

    // write THREAD_OUT back to GMEM via C
    for (int wCol = 0; wCol < THREAD_TILE_N_ITER; ++wCol) {
        float *C_interim = C + wCol * warpsliceN;
        for (int resIdxK = 0; resIdxK < THREAD_TILE_K; ++resIdxK) {
            for (int resIdxN = 0; resIdxN < THREAD_TILE_N; resIdxN += 4) {
                // accumulator index
                const int i = resIdxK * (THREAD_TILE_N_ITER * THREAD_TILE_N)
                            + wCol * THREAD_TILE_N
                            + resIdxN;
                // C row = warpTidRow*THREAD_TILE_K + resIdxK
                // C col = warpTidCol*THREAD_TILE_N  + resIdxN  (via C_interim offset)
                float4 tmp = reinterpret_cast<float4 *>(
                    &C_interim[(warpTidRow * THREAD_TILE_K + resIdxK) * n
                               + warpTidCol * THREAD_TILE_N
                               + resIdxN])[0];
                tmp.x = alpha * THREAD_OUT[i + 0] + beta * tmp.x;
                tmp.y = alpha * THREAD_OUT[i + 1] + beta * tmp.y;
                tmp.z = alpha * THREAD_OUT[i + 2] + beta * tmp.z;
                tmp.w = alpha * THREAD_OUT[i + 3] + beta * tmp.w;
                reinterpret_cast<float4 *>(
                    &C_interim[(warpTidRow * THREAD_TILE_K + resIdxK) * n
                               + warpTidCol * THREAD_TILE_N
                               + resIdxN])[0] = tmp;
            }
        }
    }
}

/*
// benchmark against pytorch
torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B) {
    auto k = A.size(0);
    auto m = A.size(1);
    auto n = B.size(1);
    auto C = torch::zeros({k, n}, A.options()); // zeros so beta*C is correct

    const int thread_block_size = 128;
    const dim3 blocks((k + K_TILE_SIZE - 1) / K_TILE_SIZE,
                      (n + N_TILE_SIZE - 1) / N_TILE_SIZE);

    gemm<128, 128, 16, 64, 64, 4, 8, 4, 32>
        <<<blocks, thread_block_size>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
            k, m, n, thread_block_size, 1.0f, 0.0f);

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_gemm, "My GEMM kernel forward");
}
*/