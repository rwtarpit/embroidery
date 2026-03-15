/*
SiLU is an activation function -> f(x) = x/(1 + e^-x)
*/


#include <iostream>
#include <cuda_runtime.h>


__global__
void SiLU( float* matrix, int m, int n){
    // since each element's ops are independent, this is an embarrassingly parellel situation

    unsigned int row = blockIdx.x;
    float* row_ptr = matrix + row * n;

    float4* matrix4 = reinterpret_cast<float4*>(row_ptr);
    
    for(int col=threadIdx.x; col<n/4; col+=blockDim.x){
        float4 elements = matrix4[col];

        matrix4[col] = make_float4(
            elements.x/(1.0f+expf(-elements.x)),
            elements.y/(1.0f+expf(-elements.y)),
            elements.z/(1.0f+expf(-elements.z)),
            elements.w/(1.0f+expf(-elements.w))
        );    
    }
    for(int col=(n/4)*4 + threadIdx.x; col<n; col+=blockDim.x){
            float element = row_ptr[col];
            row_ptr[col] = element/(1.0f+expf(-element));
        }
}


__global__
void SiLU_GridStride(float* matrix, int total_elements) {
    // grid stride loop for bette occupancy
    float4* matrix4 = reinterpret_cast<float4*>(matrix);
    int n4 = total_elements / 4;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n4; i += stride) {
        float4 elements = matrix4[i];

        matrix4[i] = make_float4(
            elements.x / (1.0f + __expf(-elements.x)),
            elements.y / (1.0f + __expf(-elements.y)),
            elements.z / (1.0f + __expf(-elements.z)),
            elements.w / (1.0f + __expf(-elements.w))
        );
    }

    int tail_start = n4 * 4;
    for (int i = tail_start + index; i < total_elements; i += stride) {
        float element = matrix[i];
        matrix[i] = element / (1.0f + __expf(-element));
    }
}