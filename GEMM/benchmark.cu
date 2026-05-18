// bench.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "gemm_ref.cu"  // runSgemmDoubleBuffering2
#include "gemm_swizzle.cu"   // GEMM_tc

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
    float end()  {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

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


// ── kernel configs ────────────────────────────────────────────────────────────

// runSgemmDoubleBuffering2<BM,BN,BK,WM,WN,WNITER,TM,TN,NUM_THREADS>
//   signature : (M, N, K, alpha, A, B, beta, C)
//   grid      : (CEIL_DIV(N,BN), CEIL_DIV(M,BM))   blockIdx.x=cCol, blockIdx.y=cRow
static constexpr int REF_BM      = 128;
static constexpr int REF_BN      = 128;
static constexpr int REF_BK      = 16;
static constexpr int REF_WM      = 64;
static constexpr int REF_WN      = 64;
static constexpr int REF_WNITER  = 4;
static constexpr int REF_TM      = 4;
static constexpr int REF_TN      = 4;
static constexpr int REF_THREADS = 128;

// GEMM_tc<BM,BN,BK,NUM_THREADS>
//   signature : (A, B, C, N, M, K)
//   grid      : (CEIL_DIV(M,BM), CEIL_DIV(N,BN))   blockIdx.x=tile_col(N), blockIdx.y=tile_row(M)
//
// In GEMM_tc:
//   tile_col = blockIdx.x  → strides along N (cols of C)
//   tile_row = blockIdx.y  → strides along M (rows of C)
//   A offset: K * BM * tile_row   — BM rows of A, each row is K wide
//   B offset: BN * tile_col       — BN cols of B
//   C store : row = tile_row*BM + ...,  col = tile_col*BN + ...
//
// So the grid must be: x = CEIL_DIV(N, BN), y = CEIL_DIV(M, BM)
static constexpr uint TC_BM      = 128; //128
static constexpr uint TC_BN      = 64; //128
static constexpr uint TC_BK      = 64;  //16
static constexpr uint TC_THREADS = 128; //256
static constexpr uint NUM_STAGES = 2;
static constexpr uint ACCUM_SIZE = (TC_BM * TC_BN) / TC_THREADS;


// ── helpers ───────────────────────────────────────────────────────────────────
static void cublas_ref(cublasHandle_t handle,
                       float *d_A, float *d_B, float *d_C,
                       int M, int N, int K,
                       float alpha = 1.f, float beta = 0.f)
{
    // cuBLAS is column-major: cublasSgemm(N,M,K, B,N, A,K) computes row-major A*B
    CUBLAS_CHECK(cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
}

static void launch_ref(float *d_A, float *d_B, float *d_C,
                       int M, int N, int K,
                       float alpha = 1.f, float beta = 0.f)
{
    const dim3 grid(CEIL_DIV(N, REF_BN), CEIL_DIV(M, REF_BM));
    runSgemmDoubleBuffering2<REF_BM, REF_BN, REF_BK,
                             REF_WM, REF_WN, REF_WNITER,
                             REF_TM, REF_TN, REF_THREADS>
        <<<grid, REF_THREADS>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

static void launch_tc(float *d_A, float *d_B, float *d_C, long long* dbg,
                      int M, int N, int K)
{
    // x covers N-dimension (tile_col), y covers M-dimension (tile_row)
    const dim3 grid(CEIL_DIV(N, TC_BN), CEIL_DIV(M, TC_BM));
    GEMM_tc<TC_BM, TC_BN, TC_BK, TC_THREADS, ACCUM_SIZE, NUM_STAGES><<<grid, TC_THREADS>>>(
        d_A, d_B, d_C, dbg, N, M, K);
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


// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    // A: M x K,  B: K x N,  C: M x N
    const int M = 4096, K = 4096, N = 5120;
    const double flops = 2.0 * M * K * N;

    printf("GEMM  A(%dx%d) @ B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);

    float *d_A, *d_B, *d_C_ref, *d_C_custom, *d_C_tc;
    long long* dbg;
    int nblocks = CEIL_DIV(N,TC_BN) * CEIL_DIV(M,TC_BM);
    CUDA_CHECK(cudaMalloc(&dbg, 3 * nblocks * sizeof(long long)));
    CUDA_CHECK(cudaMemset(dbg, 0, 3 * nblocks * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_A,        M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B,        K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_ref,    M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_custom, M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_tc,     M*N*sizeof(float)));

    float *h_A = new float[M*K];
    float *h_B = new float[K*N];
    for (int i = 0; i < M*K; i++) h_A[i] = (float)rand()/RAND_MAX - 0.5f;
    for (int i = 0; i < K*N; i++) h_B[i] = (float)rand()/RAND_MAX - 0.5f;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_A;
    delete[] h_B;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // ── correctness ──────────────────────────────────────────────────────────
    cublas_ref(handle, d_A, d_B, d_C_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_ref(d_A, d_B, d_C_custom, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    diff("(DoubleBuffering2)", d_C_ref, d_C_custom, M, N);

    launch_tc(d_A, d_B, d_C_tc, dbg, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    diff("(GEMM_tc)", d_C_ref, d_C_tc, M, N);

    printf("\n");


    // ── benchmark cuBLAS ─────────────────────────────────────────────────────
    float alpha = 1.f, beta = 0.f;
    auto cublas_fn = [&]() { cublas_ref(handle, d_A, d_B, d_C_ref, M, N, K); };
    auto s_cublas  = measure(cublas_fn);
    printf("cuBLAS           median %7.3f ms  min %7.3f ms  max %7.3f ms  |  %6.2f TFLOPS\n",
           s_cublas.median, s_cublas.min, s_cublas.max,
           flops / s_cublas.median / 1e9);

    // ── benchmark DoubleBuffering2 ────────────────────────────────────────────
    auto ref_fn   = [&]() { launch_ref(d_A, d_B, d_C_custom, M, N, K); };
    auto s_ref    = measure(ref_fn);
    printf("DoubleBuffering2 median %7.3f ms  min %7.3f ms  max %7.3f ms  |  %6.2f TFLOPS\n",
           s_ref.median, s_ref.min, s_ref.max,
           flops / s_ref.median / 1e9);

    // ── benchmark GEMM_tc ─────────────────────────────────────────────────────
    auto tc_fn    = [&]() { launch_tc(d_A, d_B, d_C_tc, dbg, M, N, K); };
    auto s_tc     = measure(tc_fn);
    printf("GEMM_tc          median %7.3f ms  min %7.3f ms  max %7.3f ms  |  %6.2f TFLOPS\n",
           s_tc.median, s_tc.min, s_tc.max,
           flops / s_tc.median / 1e9);
    
    long long h_dbg[3 * nblocks];
    cudaMemcpy(h_dbg, dbg, 3*nblocks*sizeof(long long), cudaMemcpyDeviceToHost);

    // average across blocks
    long long avgWait = 0, avgCompute = 0;
    for (int b = 0; b < nblocks; b++) {
        avgWait    += h_dbg[3*b + 0];
        avgCompute += h_dbg[3*b + 1];
    }
    printf("avg wait cycles per block:    %lld\n", avgWait / nblocks);
    printf("avg compute cycles per block: %lld\n", avgCompute / nblocks);
    printf("wait / compute ratio:         %.3f\n", 
        (float)avgWait / (float)avgCompute);
        

    // ── summary ───────────────────────────────────────────────────────────────
    printf("\nSpeedup vs cuBLAS:  DoubleBuffering2 %.2fx  |  GEMM_tc %.2fx\n",
           s_cublas.median / s_ref.median,
           s_cublas.median / s_tc.median);

    // ── cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C_ref); cudaFree(d_C_custom); cudaFree(d_C_tc);
    cublasDestroy(handle);
    return 0;
}