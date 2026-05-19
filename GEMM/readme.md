# Benchamrks

## Naive TF32 tiled GEMM kernel (#3c29edef7ad2a2377ab22bce009f5764e192e183)

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

Before `gemm_ldmatrix.cu` kernel, I assumed that ldmatrix instruction can't be used for TF32 loads due to different bit packing. But to my surprise this can be done if we manage our layout correctly and precisely follow Nvidia's PTX guide. Thanks to (Gau Nernst) for pointing and explaining this detail to me. To this,  I had to relay much of our kernel and had to transpose B tiles in SMEM to align with ldmatrix loads. we used ldmatrix.m8n8.{x4/x2} instructions for A tile and B tile respectively.


cuBLAS           median   1.597 ms  min   1.589 ms  max   1.710 ms  |  107.55 TFLOPS

DoubleBuffering2 median  10.763 ms  min  10.747 ms  max  10.918 ms  |   15.96 TFLOPS

GEMM_tc          median  4.508 ms  min  4.343 ms  max  4.738 ms  |   39.07 TFLOPS


## SWIZZLE with ldmatrix

This was probably the biggest performance boost till now. To align with ldmatrix, we had to use 128 bit swizzling layout as ldmatrix uses 4 threads to load 128 contiguous bits before distributing them in fragments of different threads.

cuBLAS           median   1.587 ms  min   1.584 ms  max   1.597 ms  |  108.24 TFLOPS

DoubleBuffering2 median  10.806 ms  min  10.754 ms  max  10.936 ms  |   15.90 TFLOPS

GEMM_tc          median   2.790 ms  min   2.706 ms  max   2.992 ms  |   81.86 TFLOPS

Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.77x