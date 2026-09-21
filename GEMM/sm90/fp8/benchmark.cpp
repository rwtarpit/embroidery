#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include "cute_fp8_gemm.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int NUM_BUFFERS = 16;
constexpr int WARMUP_ITERS = 5000;
constexpr int TIMING_ITERS = 2000;

int main() {
    const int M = 4096, N = 4096, K = 4096;

    // 1. Force Device 0 Active Context Initialization
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    std::cout << "========================================================\n";
    std::cout << " Native C++ Benchmark (AOT CuTe DSL FP8 Kernel)\n";
    std::cout << " Cold L2 Buffers: " << NUM_BUFFERS << " | Shape: " << M << "x" << N << "x" << K << "\n";
    std::cout << "========================================================\n\n";

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 2. Allocate FP8 Scale Factors on Device
    float h_scaleA = 1.0f, h_scaleB = 1.0f;
    float *d_scaleA, *d_scaleB;
    CUDA_CHECK(cudaMalloc(&d_scaleA, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scaleB, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_scaleA, &h_scaleA, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scaleB, &h_scaleB, sizeof(float), cudaMemcpyHostToDevice));

    // 3. Allocate Circular Memory Buffers
    size_t sizeA = (size_t)M * K * sizeof(__nv_fp8_e4m3);
    size_t sizeB = (size_t)N * K * sizeof(__nv_fp8_e4m3);
    size_t sizeC = (size_t)M * N * sizeof(__half);

    void *d_A[NUM_BUFFERS], *d_B[NUM_BUFFERS], *d_C[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        CUDA_CHECK(cudaMalloc(&d_A[i], sizeA));
        CUDA_CHECK(cudaMalloc(&d_B[i], sizeB));
        CUDA_CHECK(cudaMalloc(&d_C[i], sizeC));
        CUDA_CHECK(cudaMemset(d_A[i], 0x3C, sizeA)); // Valid FP8 values
        CUDA_CHECK(cudaMemset(d_B[i], 0x3C, sizeB));
        CUDA_CHECK(cudaMemset(d_C[i], 0, sizeC));
    }

    // 4. Construct C-ABI Scale Tensors (dynamic_shapes has 1 element)
    cute_fp8_gemm_Tensor_scale_a_t scale_a_tensor{}; 
    scale_a_tensor.data = d_scaleA;
    scale_a_tensor.dynamic_shapes[0] = 1;

    cute_fp8_gemm_Tensor_scale_b_t scale_b_tensor{}; 
    scale_b_tensor.data = d_scaleB;
    scale_b_tensor.dynamic_shapes[0] = 1;

    // 5. Construct C-ABI Matrix Tensors (dynamic_shapes[2], dynamic_strides[1])
    cute_fp8_gemm_Tensor_a_t tensor_A[NUM_BUFFERS];
    cute_fp8_gemm_Tensor_b_t tensor_B[NUM_BUFFERS];
    cute_fp8_gemm_Tensor_c_t tensor_C[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        // Tensor A
        tensor_A[i] = {};
        tensor_A[i].data = d_A[i];
        tensor_A[i].dynamic_shapes[0] = M;
        tensor_A[i].dynamic_shapes[1] = K;
        tensor_A[i].dynamic_strides[0] = K;

        // Tensor B
        tensor_B[i] = {};
        tensor_B[i].data = d_B[i];
        tensor_B[i].dynamic_shapes[0] = N;
        tensor_B[i].dynamic_shapes[1] = K;
        tensor_B[i].dynamic_strides[0] = K;

        // Tensor C
        tensor_C[i] = {};
        tensor_C[i].data = d_C[i];
        tensor_C[i].dynamic_shapes[0] = M;
        tensor_C[i].dynamic_shapes[1] = N;
        tensor_C[i].dynamic_strides[0] = N;
    }

    // 6. Load Module into Device 0 Context
    cute_fp8_gemm_Kernel_Module_t module{};
    cute_fp8_gemm_Kernel_Module_Load(&module);

    // 7. Warmup Loop
    std::cout << "Executing Warmup..." << std::endl;
    for (int iter = 0; iter < WARMUP_ITERS; ++iter) {
        int idx = iter % NUM_BUFFERS;
        int status = cute_dsl_cute_fp8_gemm_wrapper(
            &module,
            &tensor_A[idx], &tensor_B[idx], &tensor_C[idx],
            &scale_a_tensor, &scale_b_tensor,
            stream
        );
        if (status != 0) {
            std::cerr << "Wrapper launch failed at warmup iteration " << iter 
                      << " with status code: " << status << std::endl;
            return status;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    std::cout << "Warmup Finished Successfully!\n" << std::endl;

    // 8. Timed Cold-L2 Benchmark Loop
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iter = 0; iter < TIMING_ITERS; ++iter) {
        int idx = iter % NUM_BUFFERS;
        cute_dsl_cute_fp8_gemm_wrapper(
            &module,
            &tensor_A[idx], &tensor_B[idx], &tensor_C[idx],
            &scale_a_tensor, &scale_b_tensor,
            stream
        );
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    // Convert total milliseconds to average microseconds (us) per iteration
    double avgTimeUs = (double(elapsedMs) * 1000.0) / double(TIMING_ITERS);
    double tflops = (2.0 * double(M) * double(N) * double(K)) / (avgTimeUs * 1e-6) / 1e12;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "================ Benchmark Results ================\n";
    std::cout << " Average Time per Iteration : " << avgTimeUs << " us\n";
    std::cout << " Throughput                 : " << tflops << " FP8 TFLOPS\n";
    std::cout << "===================================================\n";

    // 9. Cleanup
    cute_fp8_gemm_Kernel_Module_Unload(&module);

    for (int i = 0; i < NUM_BUFFERS; ++i) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }
    CUDA_CHECK(cudaFree(d_scaleA));
    CUDA_CHECK(cudaFree(d_scaleB));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}