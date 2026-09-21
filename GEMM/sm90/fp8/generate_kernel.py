import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda_driver

from kernel import GEMM

def main():
    M, N, K = 4096, 4096, 4096
    tile_shape_mn = (128, 256)
    cluster_shape_mn = (2, 1)
    swizzle_size = 8
    raster_along_m = True
    mma_promotion_interval = 128

    a_dtype = cutlass.Float8E4M3FN
    b_dtype = cutlass.Float8E4M3FN
    c_dtype = cutlass.Float16

    # Create dummy tensors to resolve layout, stride, and alignment types
    tmpl_a = (torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    tmpl_b = (torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    tmpl_c = torch.zeros((M, N), dtype=torch.float16, device="cuda")

    scale_a_torch = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    scale_b_torch = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    cute_a, real_a = cutlass_torch.cute_tensor_like(tmpl_a, a_dtype, is_dynamic_layout=True, assumed_align=16)
    cute_b, real_b = cutlass_torch.cute_tensor_like(tmpl_b, b_dtype, is_dynamic_layout=True, assumed_align=16)
    cute_c, real_c = cutlass_torch.cute_tensor_like(tmpl_c, c_dtype, is_dynamic_layout=True, assumed_align=16)
    
    scale_a_cute, _ = cutlass_torch.cute_tensor_like(scale_a_torch, cutlass.Float32, is_dynamic_layout=True, assumed_align=16)
    scale_b_cute, _ = cutlass_torch.cute_tensor_like(scale_b_torch, cutlass.Float32, is_dynamic_layout=True, assumed_align=16)
    
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])

    my_gemm_engine = GEMM(
        tile_shape_mn=tile_shape_mn,
        cluster_shape_mn=cluster_shape_mn,
        swizzle_size=swizzle_size,
        raster_along_m=raster_along_m,
        mma_promotion_interval=mma_promotion_interval,
        max_active_clusters=max_active_clusters
    )

    torch_stream = torch.cuda.current_stream()
    raw_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    print("[1/3] Compiling CuTe-DSL kernel...")
    compiled_gemm = cute.compile(
        my_gemm_engine,
        cute_a, cute_b, cute_c,
        scale_a_cute, scale_b_cute,
        raw_stream,
    )

    print("[2/3] Verifying numerical output against cuBLAS (torch._scaled_mm)...")
    compiled_gemm(
        cute_a, cute_b, cute_c,
        scale_a_cute, scale_b_cute,
        raw_stream,
    )
    b_col_major = real_b.t()
    d_ref_cublas = torch.empty((M, N), dtype=torch.float16, device="cuda")
    torch._scaled_mm(
        real_a, b_col_major,
        scale_a=scale_a_torch, scale_b=scale_b_torch,
        out_dtype=torch.float16, use_fast_accum=False, out=d_ref_cublas,
    )
    torch.cuda.synchronize()
    diff = (real_c.float() - d_ref_cublas.float()).abs().max().item()
    print(f"   -> Max Absolute Difference: {diff:.5e}\n")

    print("[3/3] Performing AOT C-Export...")
    # This generates 'cute_fp8_gemm.h' and 'cute_fp8_gemm.o'
    compiled_gemm.export_to_c(
        file_path="./", 
        file_name="cute_fp8_gemm", 
        function_prefix="cute_fp8_gemm"
    )
    print("AOT Export successful: generated cute_fp8_gemm.h and cute_fp8_gemm.o")

if __name__ == "__main__":
    main()