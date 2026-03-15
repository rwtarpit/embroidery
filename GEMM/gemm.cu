// https://siboehm.com/articles/22/CUDA-MMM

/*
(General Matrix Multiplication)
GEMM operation is building block of modern AI systems.
It is the most used operation in transformer based models.

    `C = alpha*(A matmul B) + beta*C`
A = (m x k),
B = (k x n),
C = (m x n),
alpha and beta are scalers(float)
*/


#include <iostream>
#include <cuda_runtime.h>


__global__
void GEMM(float* A, float* B, float* C, int m, int k, int n, float alpha, float beta){
    /* load a row of A in sharad memory and a tile/row of B (n,)
    then we will find partial dot products.
    (k threads per block)
    firstly for bare minimum, we will load one row of B as a tile in the loop
    */
    int row = blockIdx.x;
    
    extern __shared__ float data[]; //(k,)
    float* row_A = (float*)data;
    float* tile_B = &row_A[k];

// assuming small matrix sizes (k,) fits in shared memory, so each thread loads a single element of row in row_A.
    for(int col=threadIdx.x; col<k; col+=blockDim.x){
        row_A[col] = A[row*k + col];
    }
    __syncthreads();

    // either scales initial values in C, or cleans the garbage if beta=0
    for(int col=threadIdx.x; col<n; col+=blockDim.x){
        C[row*n + col] *= beta;
    }

    for(int i=0; i<k; i++){
        for(int col=threadIdx.x; col<n; col+=blockDim.x){
        tile_B[col] = B[i*n + col];
        }
        __syncthreads();

        for(int col=threadIdx.x; col<n; col+=blockDim.x){
            C[row*n + col] += alpha * row_A[i] * tile_B[col];
        }
        __syncthreads();
    }    

}