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