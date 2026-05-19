# Embroidery - GPU Kernels Worklog

This repo consists of my kernels written in CUDA, Triton, etc and serve as my own worklog as well look back guide.

## GEMM 

GEMM stands for General Matrix Multiplication i.e `C = A x B` in linear algebra. GEMM kernel is the most important kernel in AI workloads as more than 80% of computation in LLMS and other transformer based models consists of GEMM. Accelerating GEMM is directly equivalent of accelerating naively set AI workloads by more than half alone.

Currently I am working on GEMM from scratch in CUDA for TF32(19 bits for maths) workloads. While working on GEMM in general I found resources for fp16/bf16 and even fp8 workloads to be quite nicely documented for almost all modern generations of GPUS (Ampere, Hopper and Blackwell) but couldn't find much on TF32 and Tensor Cores coupled together. Hence I decided to try myself.

Do check (GEMM)[https://github.com/rwtarpit/embroidery/tree/main/GEMM] to follow along.

## MLA 

In DeepSeek’s models (like DeepSeek-V2, V3, and R1), MLA stands for Multi-Head Latent Attention.  It is a core architectural innovation designed to solve one of the biggest bottlenecks in running large language models: the massive memory footprint required by the KV (Key-Value) Cache.  

THe MLA Decode is a Inference time kernel that fulfills incoming requests by efficiently using page sizes and compressed KV cache (in form of latent vectors).