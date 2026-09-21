# Optimising fp32 GEMM on A100(sm80)

## Benchmarks

| Kernel | Major Change | % of cuBLAS |
| :--- | :--- | :---: |
| [Kernel 1](#kernel-1) | CUDA cores → tensor cores | ~33% |
| [Kernel 2](#kernel-2) | PTX mma + swizzling | ~43% |
| [Kernel 3](#kernel-3) | async loads + pipelining | ~50% |
| [Kernel 4](#kernel-4) | warp remap + ldmatrix | ~79% |
| [Kernel 5](#kernel-5) | grid swizzling | ~85% |
| [Kernel 6](#kernel-6) | float2 GMEM stores | ~90% |
| [Kernel 7](#kernel-7) | loop unrolling + epilogue | ~100% (~105 TFLOPS) |

Measured with 20 warmp up runs and 200 actual iterations over 8 different input matrices for L2 cache flushing between iterations.