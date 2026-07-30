// bench.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "kerenl_7.cu"   

// ── macro helpers ─────────────────────────────────────────────────────────────
#define CDIV(M, N) (((M) + (N) - 1) / (N))

// ── error checking macros ────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                    \
                    __FILE__, __LINE__, st);                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)


// ── benchmark knobs ──────────────────────────────────────────────────────────
static constexpr int WARMUP_ITERS = 10;
static constexpr int BENCH_ITERS  = 100;
static constexpr int NUM_BUFFERS  = 8; // Number of distinct (A, B, C) matrix sets


// ── timing helpers ───────────────────────────────────────────────────────────
struct Stats { float median, min, max, avg; };

template<typename Fn>
Stats measure_circular(Fn fn, int num_buffers, int warmup = WARMUP_ITERS, int iters = BENCH_ITERS) {
    // 1. Warmup cycling through buffers
    for (int i = 0; i < warmup; i++) {
        fn(i % num_buffers);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Allocate asynchronous events for per-iteration stats
    cudaEvent_t *starts = new cudaEvent_t[iters];
    cudaEvent_t *stops  = new cudaEvent_t[iters];
    float *times        = new float[iters];

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventCreate(&starts[i]));
        CUDA_CHECK(cudaEventCreate(&stops[i]));
    }

    // 3. Launch asynchronously cycling through distinct matrix buffers
    for (int i = 0; i < iters; i++) {
        int buf_idx = i % num_buffers; // Cycle through input buffers to force L2 evictions
        CUDA_CHECK(cudaEventRecord(starts[i]));
        fn(buf_idx);
        CUDA_CHECK(cudaEventRecord(stops[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Gather timing data
    double sum = 0.0;
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventElapsedTime(&times[i], starts[i], stops[i]));
        sum += times[i];
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(stops[i]);
    }

    // Sort to extract percentiles
    for (int i = 0; i < iters - 1; i++) {
        for (int j = i + 1; j < iters; j++) {
            if (times[i] > times[j]) {
                float tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
        }
    }

    Stats s;
    s.min    = times[0];
    s.max    = times[iters - 1];
    s.median = times[iters / 2];
    s.avg    = (float)(sum / iters);

    delete[] starts;
    delete[] stops;
    delete[] times;
    return s;
}


// ── kernel configs ────────────────────────────────────────────────────────────

// GEMM_tc<BM,BN,BK,NUM_THREADS>
static constexpr uint TC_BM      = 256;
static constexpr uint TC_BN      = 128;
static constexpr uint TC_BK      = 32;
static constexpr uint TC_THREADS = 256; 
static constexpr uint NUM_STAGES = 2;
static constexpr uint ACCUM_SIZE = (TC_BM * TC_BN) / TC_THREADS;


// ── helpers ───────────────────────────────────────────────────────────────────
static void cublas_ref(cublasHandle_t handle,
                       float *d_A, float *d_B, float *d_C,
                       int M, int N, int K,
                       float alpha = 1.f, float beta = 0.f)
{
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
}

static void launch_tc(float *d_A, float *d_B, float *d_C,
                      int M, int N, int K)
{
    const dim3 grid(CDIV(N, TC_BN), CDIV(M, TC_BM));
    GEMM_tc<TC_BM, TC_BN, TC_BK, TC_THREADS, ACCUM_SIZE, NUM_STAGES><<<grid, TC_THREADS>>>(
        d_A, d_B, d_C, N, M, K);
}


// ── correctness helpers ───────────────────────────────────────────────────────
static void diff(const char *name, float *d_ref, float *d_custom, int M, int N)
{
    float *h_ref    = new float[M*N];
    float *h_custom = new float[M*N];
    CUDA_CHECK(cudaMemcpy(h_ref,    d_ref,    M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_custom, d_custom, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.f;
    for (int i = 0; i < M*N; i++)
        max_err = fmaxf(max_err, fabsf(h_ref[i] - h_custom[i]));

    printf("Correctness %-20s max_err = %.2e  %s\n",
           name, max_err, max_err < 5e-2f ? "PASS" : "FAIL");
    delete[] h_ref;
    delete[] h_custom;
}

// ── reporting helper ─────────────────────────────────────────────────────────
static void report(const char *name, const Stats &s, double flops)
{
    printf("%-16s median %7.3f ms  avg %7.3f ms  min %7.3f ms  max %7.3f ms  |  avg %6.2f TFLOPS  median %6.2f TFLOPS\n",
           name, s.median, s.avg, s.min, s.max,
           flops / s.avg / 1e9,
           flops / s.median / 1e9);
}


// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    const int M = 4096, K = 4096, N = 4096;
    const double flops = 2.0 * M * K * N;

    printf("GEMM  A(%dx%d) @ B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("warmup=%d  iters=%d  buffer_pool=%d sets\n\n", WARMUP_ITERS, BENCH_ITERS, NUM_BUFFERS);

    // ── Allocate arrays of device pointers ────────────────────────────────────
    float **d_A     = new float*[NUM_BUFFERS];
    float **d_B     = new float*[NUM_BUFFERS];
    float **d_C_ref = new float*[NUM_BUFFERS];
    float **d_C_k   = new float*[NUM_BUFFERS];

    float *h_A = new float[M*K];
    float *h_B = new float[K*N];

    for (int b = 0; b < NUM_BUFFERS; b++) {
        CUDA_CHECK(cudaMalloc(&d_A[b],     M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[b],     K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_ref[b], M*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_k[b],   M*N*sizeof(float)));

        // Initialize each matrix set with distinct random values
        for (int i = 0; i < M*K; i++) h_A[i] = (float)rand()/RAND_MAX - 0.5f;
        for (int i = 0; i < K*N; i++) h_B[i] = (float)rand()/RAND_MAX - 0.5f;

        CUDA_CHECK(cudaMemcpy(d_A[b], h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B[b], h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));
    }

    delete[] h_A;
    delete[] h_B;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // ── correctness check on buffer 0 ─────────────────────────────────────────
    cublas_ref(handle, d_A[0], d_B[0], d_C_ref[0], M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_tc(d_A[0], d_B[0], d_C_k[0], M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    diff("(kernel)", d_C_ref[0], d_C_k[0], M, N);

    printf("\n");

    // ── benchmark cuBLAS ─────────────────────────────────────────────────────
    auto cublas_fn = [&](int idx) { 
        cublas_ref(handle, d_A[idx], d_B[idx], d_C_ref[idx], M, N, K); 
    };
    auto s_cublas = measure_circular(cublas_fn, NUM_BUFFERS);
    report("cuBLAS", s_cublas, flops);

    // ── benchmark kernel ──────────────────────────────────────────────────────
    auto kernel_fn = [&](int idx) { 
        launch_tc(d_A[idx], d_B[idx], d_C_k[idx], M, N, K); 
    };
    auto s_kernel = measure_circular(kernel_fn, NUM_BUFFERS);
    report("kernel", s_kernel, flops);

    // ── summary ───────────────────────────────────────────────────────────────
    printf("\nSpeedup vs cuBLAS (median):  kernel %.2fx\n",
           s_cublas.median / s_kernel.median);
    printf("Speedup vs cuBLAS (avg):     kernel %.2fx\n",
           s_cublas.avg / s_kernel.avg);

    // ── cleanup ───────────────────────────────────────────────────────────────
    for (int b = 0; b < NUM_BUFFERS; b++) {
        cudaFree(d_A[b]);
        cudaFree(d_B[b]);
        cudaFree(d_C_ref[b]);
        cudaFree(d_C_k[b]);
    }
    delete[] d_A;
    delete[] d_B;
    delete[] d_C_ref;
    delete[] d_C_k;

    cublasDestroy(handle);
    return 0;
}