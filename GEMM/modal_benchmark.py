import modal
import pathlib

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
)

app = modal.App("gemm-benchmark-cu", image=image)

@app.function(gpu="A100", image=image, timeout=600)
def run_benchmark(bench_src: str, gemm_ref: str, gemm_tc: str):
    import subprocess, os, glob

    os.chdir("/root")
    with open("benchmark.cu", "w") as f:
        f.write(bench_src)
    with open("gemm_ref.cu", "w") as f:
        f.write(gemm_ref)
    with open("gemm_swizzle.cu", "w") as f:
        f.write(gemm_tc)

    arch = "sm_80"

    # ── 1. Single binary: -O3 -lineinfo -maxrregcount — exact perf SASS ─────
    # ── 1. Single binary: -O3 -lineinfo -maxrregcount — exact perf SASS ─────
    print("--- Compiling binary ---")
    try:
        compile_result = subprocess.run([
            "nvcc", "-O3", "-lineinfo",
            "-Xptxas=-v",
            "-maxrregcount=250",
            "-arch=" + arch,
            "-o", "bench",
            "benchmark.cu", "-lcublas",
        ], check=True, capture_output=True, text=True)
        
        # If it succeeds, print ptxas info
        print(compile_result.stderr)
        ptxas_info = compile_result.stderr

    except subprocess.CalledProcessError as e:
        print("\n!!! NVCC Compilation Failed !!!")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)  # CUDA C++ compilation errors!
        
        # Gracefully return the compiler error so it writes to your local file
        return f"NVCC Compilation Error:\n{e.stderr}"
    # ── 2. Run benchmark ─────────────────────────────────────────────────────
    print("--- Running Performance Benchmark vs cuBLAS ---")
    subprocess.run(["./bench"], check=True)

    # ── 3. Extract cubins from the same binary ───────────────────────────────
    print("--- Extracting embedded cubins ---")
    os.makedirs("cubins", exist_ok=True)
    os.chdir("cubins")
    subprocess.run(
        ["cuobjdump", "--extract-elf", "all", "../bench"],
        check=True
    )
    os.chdir("/root")

    cubin_files = sorted(glob.glob("cubins/*.cubin"))
    print(f"Found cubins: {cubin_files}")
    if not cubin_files:
        cubin_files = sorted(glob.glob("cubins/*"))
        print(f"Fallback glob found: {cubin_files}")

    # ── 4. nvdisasm -c on each cubin ─────────────────────────────────────────
    sass_combined = ""
    for cubin in cubin_files:
        print(f"--- Disassembling {cubin} ---")

        info = subprocess.run(
            ["nvdisasm", "--print-kernel-header", cubin],
            capture_output=True, text=True
        )
        result = subprocess.run(
            ["nvdisasm", "-c", "-g", cubin],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"  nvdisasm error: {result.stderr[:200]}")
            continue

        sass_combined += f"\n\n{'='*80}\n  CUBIN: {os.path.basename(cubin)}\n{'='*80}\n"
        sass_combined += info.stdout + "\n"
        sass_combined += result.stdout

    if not sass_combined.strip():
        print("--- Fallback: cuobjdump PTX+SASS dump ---")
        for flag in ["--dump-ptx", "--dump-sass"]:
            r = subprocess.run(
                ["cuobjdump", flag, "bench"],
                capture_output=True, text=True
            )
            sass_combined += f"\n\n{'='*80}\n  {flag.upper()}\n{'='*80}\n"
            sass_combined += r.stdout

    # ── 5. Combine ptxas info header + SASS ──────────────────────────────────
    output = f"{'='*80}\n  REGISTER & SMEM USAGE (ptxas -v)\n{'='*80}\n"
    output += ptxas_info + "\n"
    output += sass_combined

    return output


@app.local_entrypoint()
def main():
    bench_src = pathlib.Path("pmpp_v2/matmul_py/benchmark.cu").read_text()
    gemm_ref  = pathlib.Path("pmpp_v2/matmul_py/gemm_ref.cu").read_text()
    gemm_tc   = pathlib.Path("pmpp_v2/matmul_py/gemm_swizzle.cu").read_text()

    output = run_benchmark.remote(bench_src, gemm_ref, gemm_tc)

    pathlib.Path("output_interleaved.sass").write_text(output)
    print("\n[Success] Saved to: output_interleaved.sass")