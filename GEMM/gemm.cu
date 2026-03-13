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
    /*first idea that comes is to load one row of A in each block's shared memory,
    and then keep loading all rows of of B to get the dot product and
    writing it back to C
    */
    int row = blockIdx.x;
    
    extern __shared__ float row_A[]; //(k,)
// assuming small matrix sizes (k,) fits in shared memory, so each thread loads a single element of row in row_A.
    for(int col=threadIdx.x; col<k; col+=blockDim.x){
        row_A[col] = A[row*k + col];
    }
// just realised loading columns of B will thrash memory coalescing badly !?!?

}