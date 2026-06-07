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

    # ── 1. Compile performance binary (no -G) and benchmark ─────────────────
    print("--- Compiling performance binary ---")
    subprocess.run([
        "nvcc", "-O3", "-arch=" + arch,
        "-o", "bench_perf",
        "benchmark.cu", "-lcublas",
    ], check=True)

    print("--- Running Performance Benchmark vs cuBLAS ---")
    subprocess.run(["./bench_perf"], check=True)

    # ── 2. Compile benchmark.cu with -G to embed full source debug info ──────
    #    This is the ONLY file that needs to be compiled — the kernel files
    #    are #included into it (they're headers, not standalone TUs).
    print("--- Compiling debug binary (for SASS interleaving) ---")
    subprocess.run([
        "nvcc",
        "-O3",
        "-lineinfo",
        "-arch=" + arch,
        "-o", "bench_debug",
        "benchmark.cu", "-lcublas",
    ], check=True)

    # ── 3. Extract all embedded ELF cubins from the debug binary ────────────
    #    cuobjdump --extract-elf dumps one .cubin per kernel into ./cubins/
    print("--- Extracting embedded cubins ---")
    os.makedirs("cubins", exist_ok=True)
    os.chdir("cubins")
    subprocess.run(
        ["cuobjdump", "--extract-elf", "all", "../bench_debug"],
        check=True
    )
    os.chdir("/root")

    cubin_files = sorted(glob.glob("cubins/*.cubin"))
    print(f"Found cubins: {cubin_files}")

    if not cubin_files:
        # Fallback: some CUDA versions name them differently
        cubin_files = sorted(glob.glob("cubins/*"))
        print(f"Fallback glob found: {cubin_files}")

    # ── 4. Run nvdisasm -c on each cubin ────────────────────────────────────
    combined = ""
    for cubin in cubin_files:
        print(f"--- Disassembling {cubin} ---")

        # First, peek at what kernels are in this cubin
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

        combined += f"\n\n{'='*80}\n  CUBIN: {os.path.basename(cubin)}\n{'='*80}\n"
        combined += info.stdout + "\n"
        combined += result.stdout

    if not combined.strip():
        # ── 5. Nuclear fallback: cuobjdump --dump-sass --dump-ptx ───────────
        #    PTX preserves source line comments even without nvdisasm
        print("--- Fallback: cuobjdump PTX+SASS dump ---")
        for flag, ext in [("--dump-ptx", "ptx"), ("--dump-sass", "sass")]:
            r = subprocess.run(
                ["cuobjdump", flag, "bench_debug"],
                capture_output=True, text=True
            )
            combined += f"\n\n{'='*80}\n  {flag.upper()}\n{'='*80}\n"
            combined += r.stdout

    return combined


@app.local_entrypoint()
def main():
    bench_src = pathlib.Path("pmpp_v2/matmul_py/benchmark.cu").read_text()
    gemm_ref  = pathlib.Path("pmpp_v2/matmul_py/gemm_ref.cu").read_text()
    gemm_tc   = pathlib.Path("pmpp_v2/matmul_py/gemm_swizzle.cu").read_text()

    sass_output = run_benchmark.remote(bench_src, gemm_ref, gemm_tc)

    pathlib.Path("output_interleaved.sass").write_text(sass_output)
    print(f"\n[Success] Interleaved SASS saved to: output_interleaved.sass")