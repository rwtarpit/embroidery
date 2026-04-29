# Benchamrks

## Naive TF32 tiled GEMM kernel (#3453bfc)

### GEMM  A(4096x4096) @ B(4096x5120) = C(4096x5120)

BM = 128; BN = 128; BK = 16;  NUM_THREADS = 256;

Correctness (DoubleBuffering2)   max_err = 7.46e-03  PASS

Correctness (GEMM_tc)            max_err = 2.27e-02  PASS

cuBLAS           median   1.584 ms  min   1.580 ms  max   1.607 ms  |  108.45 TFLOPS

DoubleBuffering2 median  10.804 ms  min  10.761 ms  max  10.911 ms  |   15.90 TFLOPS

GEMM_tc          median   6.568 ms  min   6.562 ms  max   6.577 ms  |   26.16 TFLOPS


Speedup vs cuBLAS:  DoubleBuffering2 0.15x  |  GEMM_tc 0.24x