#include <cuda_runtime.h>
#include <iostream>


/*
Naive Fused-Softmax kernel

loads input matrix 3 times from global memory:
    - to calculate max element per row for normalizing input
    - to calculate numerator of each element : exp(x-row_max)
    - to calculate final value of each element by division with rowsum(exp(x_i-row_max))

stores data back to global memory twice:
    - saves normalized elements after x_i = exp(x_i-row_max)
    - saves final softmax value of matrix

uses shared memory declared once to store:
    - final row_max for each row/block
    - final normalization factor / denominator for each row/block

uses Parellel or Tree-based Reduction to find:
    - row_max in each block
    - row_sum for normalization factor / denominator in each block
*/
__global__
void softmax(float* matrix, int m, int n){

    unsigned int row = blockIdx.x;

    extern __shared__ float row_data[];
    float local_max = -INFINITY;

    for(int col=threadIdx.x; col<n; col+=blockDim.x){
        local_max = fmaxf(local_max, matrix[row * n + col]);
    }
    row_data[threadIdx.x] = local_max;
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(s > threadIdx.x){
            row_data[threadIdx.x] = fmaxf(row_data[threadIdx.x], row_data[threadIdx.x + s]);
        }
        __syncthreads();
    }

    float final_row_max = row_data[0]; 

    float local_sum = 0;
    for(int col=threadIdx.x; col<n; col+=blockDim.x){
        float col_el = matrix[row*n + col];

        col_el -= final_row_max;
        col_el = expf(col_el);
        local_sum += col_el;

        matrix[row*n + col] = col_el;
    }
    row_data[threadIdx.x] = local_sum;
    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(s > threadIdx.x){
            row_data[threadIdx.x] += row_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scaling_factor = row_data[0];

    for(int col=threadIdx.x; col<n; col+=blockDim.x){
        float col_el = matrix[row*n + col];

        col_el /= scaling_factor;
        matrix[row*n + col] = col_el;
    }
}