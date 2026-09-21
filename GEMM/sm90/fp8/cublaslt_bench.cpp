#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "UNKNOWN_CUBLAS_ERROR";
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ \
                      << " - Code (" << err << "): " << cublasGetStatusString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int NUM_BUFFERS = 16;
constexpr int WARMUP_ITERS = 5000;
constexpr int TIMING_ITERS = 2000;
constexpr size_t WORKSPACE_SIZE = 64 * 1024 * 1024; // 64 MB workspace

// TOGGLE FAST ACCUMULATION HERE:
constexpr bool USE_FAST_ACCUM = false;

int main() {
    // 1. Force Device Context Initialization
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(0));

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    std::cout << "========================================================\n";
    std::cout << " SM90 FP8 (E4M3) -> FP16 MatMul Benchmark\n";
    std::cout << " Target Dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << " Fast Accumulation: " << (USE_FAST_ACCUM ? "ENABLED (1)" : "DISABLED (0)") << "\n";
    std::cout << " Circular Buffers: " << NUM_BUFFERS << " (Cold L2 Testing)\n";
    std::cout << " Warmup: " << WARMUP_ITERS << " | Iterations: " << TIMING_ITERS << "\n";
    std::cout << "========================================================\n\n";

    cublasLtHandle_t ltHandle;
    cudaStream_t stream;
    CHECK_CUBLAS(cublasLtCreate(&ltHandle));
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 2. Matrix Layout Descriptors
    cublasLtMatrixLayout_t A_layout, B_layout, C_layout;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A_layout, CUDA_R_8F_E4M3, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B_layout, CUDA_R_8F_E4M3, K, N, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C_layout, CUDA_R_16F,     M, N, M));

    // 3. Operation Descriptor Creation
    cublasLtMatmulDesc_t operationDesc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // 4. Set Fast Accumulation Attribute via int8_t
    int8_t fast_accum = USE_FAST_ACCUM ? 1 : 0;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, 
        CUBLASLT_MATMUL_DESC_FAST_ACCUM, 
        &fast_accum, 
        sizeof(fast_accum)
    ));

    // 5. Scale Factor Allocations (FP8)
    float h_scaleA = 1.0f, h_scaleB = 1.0f;
    float *d_scaleA = nullptr, *d_scaleB = nullptr;
    CHECK_CUDA(cudaMalloc(&d_scaleA, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scaleB, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scaleA, &h_scaleA, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scaleB, &h_scaleB, sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, 
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
        &d_scaleA, 
        sizeof(d_scaleA)
    ));

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc, 
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
        &d_scaleB, 
        sizeof(d_scaleB)
    ));

    float alpha = 1.0f, beta = 0.0f;

    // 6. Memory Buffer Allocation
    size_t sizeA = (size_t)M * K * sizeof(__nv_fp8_e4m3);
    size_t sizeB = (size_t)N * K * sizeof(__nv_fp8_e4m3);
    size_t sizeC = (size_t)M * N * sizeof(__half);

    void* d_A[NUM_BUFFERS];
    void* d_B[NUM_BUFFERS];
    void* d_C[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        CHECK_CUDA(cudaMalloc(&d_A[i], sizeA));
        CHECK_CUDA(cudaMalloc(&d_B[i], sizeB));
        CHECK_CUDA(cudaMalloc(&d_C[i], sizeC));
        CHECK_CUDA(cudaMemset(d_A[i], 0x3C, sizeA));
        CHECK_CUDA(cudaMemset(d_B[i], 0x3C, sizeB));
        CHECK_CUDA(cudaMemset(d_C[i], 0, sizeC));
    }

    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, WORKSPACE_SIZE));

    // 7. Heuristics Auto-Tuning Setup
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WORKSPACE_SIZE, sizeof(WORKSPACE_SIZE)));

    constexpr int MAX_RESULTS = 64;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResults[MAX_RESULTS];

    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, A_layout, B_layout, C_layout, C_layout,
        preference, MAX_RESULTS, heuristicResults, &returnedResults));

    if (returnedResults == 0) {
        std::cerr << "Error: No valid cuBLASLt algorithms found!" << std::endl;
        return -1;
    }

    // 8. Auto-Tune Search Loop
    float bestTimeMs = 1e9f;
    cublasLtMatmulAlgo_t bestAlgo;
    bool foundValidAlgo = false;

    for (int i = 0; i < returnedResults; ++i) {
        cublasStatus_t status = cublasLtMatmul(
            ltHandle, operationDesc, &alpha,
            d_A[0], A_layout, d_B[0], B_layout, &beta,
            d_C[0], C_layout, d_C[0], C_layout,
            &heuristicResults[i].algo, workspace, WORKSPACE_SIZE, stream);

        if (status != CUBLAS_STATUS_SUCCESS) continue;
        CHECK_CUDA(cudaStreamSynchronize(stream));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int iter = 0; iter < 100; ++iter) {
            int buf_idx = iter % NUM_BUFFERS;
            cublasLtMatmul(ltHandle, operationDesc, &alpha,
                           d_A[buf_idx], A_layout, d_B[buf_idx], B_layout, &beta,
                           d_C[buf_idx], C_layout, d_C[buf_idx], C_layout,
                           &heuristicResults[i].algo, workspace, WORKSPACE_SIZE, stream);
        }
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float elapsedMs = 0;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
        float avgMs = elapsedMs / 100.0f;

        if (avgMs < bestTimeMs) {
            bestTimeMs = avgMs;
            bestAlgo = heuristicResults[i].algo;
            foundValidAlgo = true;
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    if (!foundValidAlgo) {
        std::cerr << "Error: All heuristic algorithms failed during benchmarking!" << std::endl;
        return -1;
    }

    // Destroy preference right after heuristic search is done
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));

    // 9. Warmup Execution
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        int buf_idx = iter % NUM_BUFFERS;
        cublasLtMatmul(
            ltHandle, operationDesc, &alpha,
            d_A[buf_idx], A_layout, d_B[buf_idx], B_layout, &beta,
            d_C[buf_idx], C_layout, d_C[buf_idx], C_layout,
            &bestAlgo, workspace, WORKSPACE_SIZE, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 10. Timed Benchmark Run
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    CHECK_CUDA(cudaEventRecord(startEvent, stream));
    for (int iter = 0; iter < TIMING_ITERS; ++iter) {
        int buf_idx = iter % NUM_BUFFERS;
        cublasLtMatmul(
            ltHandle, operationDesc, &alpha,
            d_A[buf_idx], A_layout, d_B[buf_idx], B_layout, &beta,
            d_C[buf_idx], C_layout, d_C[buf_idx], C_layout,
            &bestAlgo, workspace, WORKSPACE_SIZE, stream);
    }
    CHECK_CUDA(cudaEventRecord(stopEvent, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float totalTimeMs = 0;
    CHECK_CUDA(cudaEventElapsedTime(&totalTimeMs, startEvent, stopEvent));

    // Convert total milliseconds to average microseconds (us) per iteration
    double avgTimeUs = (double(totalTimeMs) * 1000.0) / double(TIMING_ITERS);
    double tflops = (2.0 * double(M) * double(N) * double(K)) / (avgTimeUs * 1e-6) / 1e12;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "================ Benchmark Results ================\n";
    std::cout << " Average Time per Iteration : " << avgTimeUs << " us\n";
    std::cout << " Throughput                 : " << tflops << " FP8 TFLOPS\n";
    std::cout << "===================================================\n";

    // 11. Clean up resources safely
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream));

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_scaleA));
    CHECK_CUDA(cudaFree(d_scaleB));

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A_layout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B_layout));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(C_layout));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtDestroy(ltHandle));

    return 0;
}