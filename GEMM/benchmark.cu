// bench.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "gemm_tiled.cu"

// ── error checking macros ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                   \
                    __FILE__, __LINE__, st);                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)


// ── timing helpers ───────────────────────────────────────────────────────────
struct Timer {
    cudaEvent_t start, stop;
    Timer()  { CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));  }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end()  {                          // returns ms
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// run fn() WARMUP times, then ITERS times, return {median_ms, min_ms, max_ms}
template<typename Fn>
struct Stats { float median, min, max; };

template<typename Fn>
Stats<Fn> measure(Fn fn, int warmup = 10, int iters = 100) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    float *times = new float[iters];
    Timer t;
    for (int i = 0; i < iters; i++) {
        t.begin();
        fn();
        times[i] = t.end();
    }

    // sort for median
    for (int i = 0; i < iters-1; i++)
        for (int j = i+1; j < iters; j++)
            if (times[i] > times[j]) { float tmp=times[i]; times[i]=times[j]; times[j]=tmp; }

    Stats<Fn> s;
    s.median = times[iters/2];
    s.min    = times[0];
    s.max    = times[iters-1];
    delete[] times;
    return s;
}

// ── correctness check ────────────────────────────────────────────────────────
void check_correctness(float *d_A, float *d_B, float *d_C_ref, float *d_C_custom,
                       int k, int m, int n, cublasHandle_t handle)
{
    // cuBLAS: note column-major convention — computes B^T * A^T = C^T
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, k, m,
        &alpha,
        d_B, n,
        d_A, m,
        &beta,
        d_C_ref, n));

    const int thread_block_size = 128;
    const dim3 blocks((k+K_TILE_SIZE-1)/K_TILE_SIZE, (n+N_TILE_SIZE-1)/N_TILE_SIZE);
    gemm<128,128,16,64,64,4,8,4,32><<<blocks, thread_block_size>>>(
        d_A, d_B, d_C_custom, k, m, n, thread_block_size, 1.0f, 0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy both to host and compare
    float *h_ref     = new float[k*n];
    float *h_custom  = new float[k*n];
    CUDA_CHECK(cudaMemcpy(h_ref,    d_C_ref,    k*n*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_custom, d_C_custom, k*n*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (int i = 0; i < k*n; i++)
        max_err = fmaxf(max_err, fabsf(h_ref[i] - h_custom[i]));

    printf("Correctness check: max_err = %.2e  %s\n",
           max_err, max_err < 1e-2f ? "✓ PASS" : "✗ FAIL");

    delete[] h_ref;
    delete[] h_custom;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    const int k = 4096, m = 4096, n = 5120;
    const double flops = 2.0 * k * m * n;

    printf("GEMM  A(%dx%d) @ B(%dx%d) = C(%dx%d)\n\n", k, m, m, n, k, n);

    // allocate + init device memory
    float *d_A, *d_B, *d_C_ref, *d_C_custom;
    CUDA_CHECK(cudaMalloc(&d_A,        k*m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,        m*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref,    k*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, k*n*sizeof(float)));

    // fill with random data on host, copy to device
    float *h_A = new float[k*m];
    float *h_B = new float[m*n];
    for (int i = 0; i < k*m; i++) h_A[i] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < m*n; i++) h_B[i] = (float)rand()/RAND_MAX - 0.5f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, k*m*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, m*n*sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_A;
    delete[] h_B;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ── correctness ──────────────────────────────────────────────────────────
    check_correctness(d_A, d_B, d_C_ref, d_C_custom, k, m, n, handle);
    printf("\n");

    // ── benchmark cuBLAS ─────────────────────────────────────────────────────
    float alpha = 1.0f, beta = 0.0f;
    auto cublas_fn = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, k, m, &alpha, d_B, n, d_A, m, &beta, d_C_ref, n));
    };
    auto s_cublas = measure(cublas_fn);
    printf("cuBLAS        median %7.3f ms  min %7.3f ms  max %7.3f ms  |  %6.2f TFLOPS\n",
           s_cublas.median, s_cublas.min, s_cublas.max,
           flops / s_cublas.median / 1e9);

    // ── benchmark custom kernel ───────────────────────────────────────────────
    const int thread_block_size = 128;
    const dim3 blocks((k+K_TILE_SIZE-1)/K_TILE_SIZE, (n+N_TILE_SIZE-1)/N_TILE_SIZE);
    auto custom_fn = [&]() {
        gemm<128,128,16,64,64,4,8,4,32><<<blocks, thread_block_size>>>(
            d_A, d_B, d_C_custom, k, m, n, thread_block_size, 1.0f, 0.0f);
    };
    auto s_custom = measure(custom_fn);
    printf("Custom kernel median %7.3f ms  min %7.3f ms  max %7.3f ms  |  %6.2f TFLOPS\n",
           s_custom.median, s_custom.min, s_custom.max,
           flops / s_custom.median / 1e9);

    printf("\nSpeedup vs cuBLAS: %.2fx\n", s_cublas.median / s_custom.median);

    // ── cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_ref); cudaFree(d_C_custom);
    cublasDestroy(handle);
    return 0;
}