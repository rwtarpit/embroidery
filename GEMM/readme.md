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