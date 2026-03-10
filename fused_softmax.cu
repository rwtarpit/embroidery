#include <cuda_runtime.h>
#include <iostream>
//TODO: https://siboehm.com/articles/22/CUDA-MMM

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

arithmetic_intensity : 5/20 == 0.25 FLOPS/bytes
*/
__global__
void three_pass_naive_softmax(float* matrix, int m, int n){

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

/*
optimisations over the above kernel:
    - float4 vectorisation for instruction level parellelism
    - __shfl_xor_sync for parellel reduction at warp level
    - loop unrolling

update : this kernel is still memory bound since we are 
        doing too few ops compared to the data we have to
        move back and forth from global memory.
    to further optimise the softmax kernel we need to reduce
    passes to global memory. this can be done by online softmax.
*/
__global__
void three_pass_optimized_softmax(float* matrix, int m, int n){
    unsigned int row = blockIdx.x;
    //shared memory size now should be block_size/32 since we are using __shfl_xor_sync
    extern __shared__ float row_data[];
    float4* matrix4 = reinterpret_cast<float4*>(matrix);

    float local_max = -INFINITY;
    for(int col=threadIdx.x; col<n/4; col+=blockDim.x){
        float4 col_el = matrix4[row*(n/4) + col];
        float local_max4 = fmaxf(fmaxf(col_el.x, col_el.y), fmaxf(col_el.z, col_el.w));
        local_max = fmaxf(local_max, local_max4);
    }
    // warp max using __shfl_xor_sync
    #pragma unroll
    for(int offset=16; offset>0; offset>>=1){
        float remote_local_max = __shfl_xor_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, remote_local_max);
    }
    float warp_max = local_max;

    if(threadIdx.x%32 == 0){
        row_data[threadIdx.x/32] = warp_max;
    }
    __syncthreads();

    for(unsigned int s=(blockDim.x/32)/2; s>0; s>>=1){
        if(threadIdx.x < s){
            row_data[threadIdx.x] = fmaxf(row_data[threadIdx.x], row_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = row_data[0];

    float local_sum = 0.0f;
    for(unsigned int col=threadIdx.x; col<n/4; col+=blockDim.x){
        float4 col_el = matrix4[row*(n/4) + col];
        col_el.x = expf(col_el.x-row_max);
        col_el.y = expf(col_el.y-row_max);
        col_el.z = expf(col_el.z-row_max);
        col_el.w = expf(col_el.w-row_max);

        float local_sum4 = col_el.x + col_el.y + col_el.z + col_el.w;
        local_sum += local_sum4;

        //matrix4[row*(n/4) + col] = col_el;
    }
    
    #pragma unroll
    for(int offset=16; offset>0; offset>>=1){
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    float warp_sum = local_sum;

    if(threadIdx.x%32 ==0){
        row_data[threadIdx.x/32] = warp_sum;
    }
    __syncthreads();

    for(unsigned int s=(blockDim.x/32)/2; s>0; s>>=1){
        if(threadIdx.x < s){
            row_data[threadIdx.x] += row_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scaling_factor = row_data[0];

    for(unsigned int col=threadIdx.x; col<n/4; col+=blockDim.x){
        float4 col_el = matrix4[row*(n/4) + col];
        col_el.x = expf(col_el.x - row_max) / scaling_factor;
        col_el.y = expf(col_el.y - row_max) / scaling_factor;
        col_el.z = expf(col_el.z - row_max) / scaling_factor;
        col_el.w = expf(col_el.w - row_max) / scaling_factor;
        
        matrix4[row*(n/4) + col] =  col_el;
    }    
}


// online softmax kernel
__global__
void online_softmax(float* matrix, int m, int n){
    unsigned int row = blockIdx.x;
    // shared memory now will store both local_max and local_scaling_factor
    // we use float2 for it
    extern __shared__ float2 row_data[];
    float4* matrix4 = reinterpret_cast<float4*>(matrix);

    float local_max = -INFINITY;
    float local_scaling_factor = 0.0f;

    for(int col=threadIdx.x; col<(n/4); col+=blockDim.x){
        float4 col_el = matrix4[row*(n/4)+col];
        float col_els[4] = {col_el.x, col_el.y, col_el.z, col_el.w};

        for(int i=0; i<4; i++) {
        float val = col_els[i];
            if (val > local_max) {
                local_scaling_factor = local_scaling_factor * expf(local_max - val) + 1.0f;
                local_max = val;
            } else {
                local_scaling_factor += expf(val - local_max);
            }
        }
    }
    row_data[threadIdx.x] = make_float2(local_max, local_scaling_factor);
    __syncthreads();

    for(unsigned s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x<s){  
            float2 el1 = row_data[threadIdx.x];
            float2 el2 = row_data[threadIdx.x + s];

            float rel_max = fmaxf(el1.x, el2.x);
            float combined_sum = el1.y*expf(el1.x-rel_max) + el2.y*expf(el2.x-rel_max);

            row_data[threadIdx.x] = make_float2(rel_max, combined_sum);
        }
        __syncthreads();
    }
    float row_max = row_data[0].x;
    float scaling_factor = row_data[0].y;

    for(unsigned int col=threadIdx.x; col<(n/4); col+=blockDim.x){
        float4 col_el = matrix4[row*(n/4) + col];

        col_el.x = expf(col_el.x - row_max) / scaling_factor;
        col_el.y = expf(col_el.y - row_max) / scaling_factor;
        col_el.z = expf(col_el.z - row_max) / scaling_factor;
        col_el.w = expf(col_el.w - row_max) / scaling_factor;
        
        matrix4[row*(n/4) + col] =  col_el;
    }

}