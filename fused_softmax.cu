#include <cuda_runtime.h>
#include <iostream>

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