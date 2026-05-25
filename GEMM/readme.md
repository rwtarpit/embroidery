# Benchmarks

## Naive TF32 tiled GEMM kernel (#3c29edef7ad2a2377ab22bce009f5764e192e183)

These initially are quick results without much detailing for baseline purpose. I will be logging more details below when i start to make substantial change or come across a new bottleneck or optimization method.

### GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

BM = 128; BN = 128; BK = 16;  NUM_THREADS = 256;

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS

Correctness (GEMM_tc)            max_err = 2.27e-02  PASS

cuBLAS           median   1.584 ms  min   1.580 ms  max   1.607 ms  |  108.45 TFLOPS

DoubleBuffering2 median  10.804 ms  min  10.761 ms  max  10.911 ms  |   15.90 TFLOPS

GEMM_tc          median   6.568 ms  min   6.562 ms  max   6.577 ms  |   26.16 TFLOPS


Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.24x

## smem padded TF32 tiled GEMM kernel

GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS
Correctness (GEMM_tc)            max_err = 2.27e-02  PASS

cuBLAS           median   1.576 ms  min   1.573 ms  max   1.581 ms  |  109.01 TFLOPS
DoubleBuffering2 median  10.759 ms  min  10.746 ms  max  10.870 ms  |   15.97 TFLOPS
GEMM_tc          median   4.241 ms  min   4.233 ms  max   4.252 ms  |   40.51 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.37x

## Naive GEMM with tensor cores and swizzled A tiles

GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS
Correctness (GEMM_tc)            max_err = 0.00e+00  PASS

cuBLAS           median   1.584 ms  min   1.573 ms  max   1.692 ms  |  108.45 TFLOPS
DoubleBuffering2 median  10.774 ms  min  10.762 ms  max  10.881 ms  |   15.95 TFLOPS
GEMM_tc          median   5.427 ms  min   5.418 ms  max   5.442 ms  |   31.66 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.29x

## GEMM with swizzled A and B tiles and coalesced Global loads

GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS
Correctness (GEMM_tc)            max_err = 0.00e+00  PASS

cuBLAS           median   1.581 ms  min   1.578 ms  max   1.589 ms  |  108.66 TFLOPS
DoubleBuffering2 median  10.757 ms  min  10.743 ms  max  10.893 ms  |   15.97 TFLOPS
GEMM_tc          median   3.670 ms  min   3.658 ms  max   3.717 ms  |   46.81 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.43x

## swizzled GEMM generalised

GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS
Correctness (GEMM_tc)            max_err = 0.00e+00  PASS

cuBLAS           median   1.577 ms  min   1.573 ms  max   1.580 ms  |  108.94 TFLOPS
DoubleBuffering2 median  10.757 ms  min  10.744 ms  max  10.868 ms  |   15.97 TFLOPS
GEMM_tc          median   3.957 ms  min   3.951 ms  max   3.965 ms  |   43.42 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.40x

 TC_BM      = 256; //128
TC_BN      = 128; //128
TC_BK      = 16;  //16
TC_THREADS = 256; //256

## async pipeline + swizzling + tf32 manual smem loads

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS
Correctness (GEMM_tc)            max_err = 0.00e+00  PASS

cuBLAS           median   1.587 ms  min   1.584 ms  max   1.597 ms  |  108.24 TFLOPS
DoubleBuffering2 median  10.806 ms  min  10.754 ms  max  10.936 ms  |   15.90 TFLOPS
GEMM_tc          median   3.190 ms  min   3.176 ms  max   3.195 ms  |   53.86 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.50x

BM: 256, BN: 128, BK: 16, NUM_THREADS: 256


### Profiling Result for this kernel:

We are computing on whole `BK` of tile B in a warp, leading to requirement of excess registers for storing partial accumulation. This is leading to low occupancy. 

Also due to this the kernel is compute bound, not memory latency bound, therefore pipeling isn't increasing performance as kernel is taking excess of time in compute itself

Now, as we reduced BN to 64 from 256 and increased BM to 512, we saw +5 TFLOPS and -0.319ms speed. I think we need to change/reduce per warp allocated work to increase occupancy and reduce register pressure.

## Story till now:

By profiling our loads and compute time, we discovered that surprisingly our kernel is on opposite regime of where it should be. Even after using tensor cores our kernel sits in compute bound regime, and this is the reason why increasing num_stages in pipeline doesn't bring any performance gains. The pipeline works as it should and keeps the smem full by loading next few stages of data from smem, but compute takes so much time that the loaded tiles sit there in smem.

### Profiling result for num stages = 3:

avg wait cycles per block:    6101

avg compute cycles per block: 365427

wait / compute ratio:         0.017

### Profiling result for num stages = 2:

avg wait cycles per block:    6629

avg compute cycles per block: 300933

wait / compute ratio:         0.022

### Profiling result for num stages = 1:

avg wait cycles per block:    84051

avg compute cycles per block: 650799

wait / compute ratio:         0.129

Now we can see for num stages = 2 and 3 the waiting time for next tile to be in smem (by `consumer.wait()`) is exactly the same, and so is compute time. meanwhile when we do `num_stages` = 1, ie no pipelining, wait time for data from smem increases and due to which the compute time also increases as now tensor cores are also stalling due to no data. Therefore this verifies that pipelining is indeed correct but overhead in mma instructions is too much and kernel is compute bound. 

Our next target will be to explore better compute methods for better occupancy, register reuse, etc, and also look at overhead instructions like swizzling and dtype conversions, etc.

## Using ldmatrix with TF32.

Before `gemm_ldmatrix.cu` kernel, I assumed that ldmatrix instruction can't be used for TF32 loads due to different bit packing. But to my surprise this can be done if we manage our layout correctly and precisely follow Nvidia's PTX guide. Thanks to [Gau Nernst](https://github.com/gau-nernst) for pointing and explaining this detail to me. To this,  I had to relay much of our kernel and had to transpose B tiles in SMEM to align with ldmatrix loads. we used ldmatrix.m8n8.{x4/x2} instructions for A tile and B tile respectively.

A = 128x16 B = 16x128 Threads = 128

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS

Correctness (GEMM_tc)            max_err = 1.38e-02  PASS

cuBLAS           median   1.580 ms  min   1.576 ms  max   1.585 ms  |  108.73 TFLOPS

DoubleBuffering2 median  10.768 ms  min  10.744 ms  max  10.964 ms  |   15.95 TFLOPS

GEMM_tc          median   2.887 ms  min   2.882 ms  max   2.890 ms  |   59.51 TFLOPS

avg wait cycles per block:    35230

avg compute cycles per block: 417701

wait / compute ratio:         0.084

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.55x


## SWIZZLE with ldmatrix

This was probably the biggest performance boost till now. To align with ldmatrix, we had to use 128 bit swizzling layout as ldmatrix uses 4 threads to load 128 contiguous bits before distributing them in fragments of different threads.

cuBLAS           median   1.587 ms  min   1.584 ms  max   1.597 ms  |  108.24 TFLOPS

DoubleBuffering2 median  10.806 ms  min  10.754 ms  max  10.936 ms  |   15.90 TFLOPS

GEMM_tc          median   2.790 ms  min   2.706 ms  max   2.992 ms  |   81.86 TFLOPS

`Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.77x` (our best yet so far)


## Investigate

Let us try to investigate and improve based on results / data we have and profiling our kernel again.

### A100 Specs

Max Shared Memory per SM: 164 KB.

Max Shared Memory per Block: 163 KB

L2 Cache Size: 40 MB

Registers per SM: 256 KB (65,536 32-bit registers).

Max Registers per Thread: 255.

Max Threads per SM: 2048

| A100 Variant | Memory Type | Peak Memory Bandwidth | Max Compute Throughput | Arithmetic Intensity Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **A100 80GB SXM** | HBM2e | 2,039 GB/s | 156 TFLOPS | **76.51 FLOPs/byte** |
| **A100 80GB PCIe** | HBM2e | 1,935 GB/s | 156 TFLOPS | **80.62 FLOPs/byte** |
| **A100 40GB SXM / PCIe** | HBM2 | 1,555 GB/s | 156 TFLOPS | **100.32 FLOPs/byte** |

To keep the tensor cores saturated we need to perform ~100 ops per byte of data loaded from GMEM. 

Current Arithmetic Intensity : 128 × 16 × 128 × 2 / (128 × 16 × 2) / 4 = 32(approx)

for our current tile sizes, this means our tensor cores are mostly stalling and kernel is largely memory bound. ie we need to increase arithmetic intensity. we will try 256x256 output tile keeping the TILE_K fixed to 16 to keep SMEM in check as well.

that will give around 62 FLOPS/Bytes

Also we need to rewrite the warp level logic to let a warp calculate more squarish subtile instead.

### Using Square block output per warp over horizontal tiles

From now on we would use 128x16x128 with 256 threads for better performance quantification

`ldmatrix_load.cu` uses our old horizontal strip per warp of output tile:


cuBLAS           median   1.585 ms  min   1.580 ms  max   1.602 ms  |  108.38 TFLOPS

DoubleBuffering2 median  10.777 ms  min  10.756 ms  max  10.956 ms  |   15.94 TFLOPS

GEMM_tc          median   4.385 ms  min   4.382 ms  max   4.390 ms  |   39.18 TFLOPS

avg wait cycles per block:    45381 
avg compute cycles per block: 659671 
wait / compute ratio:         0.069

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.36x

`block_out.cu` uses the squarish block output per warp (32x64 per warp)

cuBLAS           median   1.592 ms  min   1.583 ms  max   1.607 ms  |  107.89 TFLOPS

DoubleBuffering2 median  10.766 ms  min  10.753 ms  max  10.958 ms  |   15.96 TFLOPS

GEMM_tc          median   3.205 ms  min   3.167 ms  max   3.210 ms  |   53.60 TFLOPS

avg wait cycles per block:    52749 
avg compute cycles per block: 381381 
wait / compute ratio:         0.138

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.50x

As expected this works much better. The reasons for this that i came to know is lesser Smem data loading and more reuse (via broadcasting).

 When each warp computes a horizontal row long output tile, for each warp we load more data:

Ex : (32x16 + 16x64) = 1536 floats VS (16x16 + 16x128) = 2304 floats

This means less Arithmetic Intensity per warp (I really don't know if this is a term 🙂). Another factor that I think is related here  is data reuse. In long rect. tile per warp, each warp is loading different subtile of A (no data reuse here) although each warp does use whole B  tile completely, but I dont think there is hardware level multicast feature (like Hopper TMA) in A100 and load instructions are still issued. So if i go with this theory, since we are loading more data in rect tile method, it means more instruction issues for data loading in Smem which is again stalling Tensor Cores. Would love to know concrete reasons for this one too.

### Load first, Compute Altogether

We are looping over a single mma instruction, ie load 2 tiles, perform `mma` and repeat, this likely again is stalling TCs. Can we pipeline this part too?