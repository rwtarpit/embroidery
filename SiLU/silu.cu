/*
SiLU is an activation function -> f(x) = x/(1 + e^-x)
*/


#include <iostream>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


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
void SiLU_GridStride(float* __restrict__ matrix, int total_elements) {
    // grid stride loop for better occupancy
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


// latency hiding using cuda pipeline asynchronization

/*
idea is to keep the memory bus busy and keep loading data 
in shared memory without any registers in bwtween, while SMs are busy doing the math for 
previous data tile

this can be more easily done by manual software pipelining and loading directly
into registers as data don't need to be shared among threads
*/
__global__
void SiLU_async(float* __restrict__ matrix, int total_elements, int tile_size){
    extern __shared__ float tile_data[];
    
    float* tiles[2];
    tiles[0] = &tile_data[0];
    tiles[1] = &tile_data[tile_size];

    cg::grid_group grid = cg::this_grid();

    int i = grid.thread_rank();
    int stride = grid.size();

    // depth=2 
    cuda::pipeline<cuda::thread_scope_thread, 2> pipe = cuda::make_pipeline();

    // prefetch tile 0 
    if(i < total_elements){
        cuda::memcpy_async(&tiles[0][threadIdx.x], &matrix[i], sizeof(float), pipe);
    }
    pipe.producer_commit();

    for(; i < total_elements; i += stride){
        int current_buf = (i / stride) % 2;
        int next_buf = 1 - current_buf;

        int next_i = i + stride;

        // issue prefetch for next tile
        // so memory controller works while we compute current tile
        if(next_i < total_elements){
            cuda::memcpy_async(&tiles[next_buf][threadIdx.x], &matrix[next_i], sizeof(float), pipe);
        }
        pipe.producer_commit();  
        pipe.consumer_wait();

        float element = tiles[current_buf][threadIdx.x];
        matrix[i] = element / (1.0f + __expf(-element));

        pipe.consumer_release();
    }

    // drain the last in-flight prefetch
    pipe.consumer_wait();
    pipe.consumer_release();
}