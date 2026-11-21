import pathlib
import modal

image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",
    add_python="3.11",
)

app = modal.App("gemm-benchmark-cu", image=image)


@app.function(gpu="A100", image=image, timeout=600)
def run_job(
    bench_src: str,
    gemm_ref: str,
    gemm_tc: str,
    profile: bool = False,
    ncu_set: str = "basic",
):
    import os
    import subprocess

    os.chdir("/root")
    with open("benchmark.cu", "w") as f:
        f.write(bench_src)
    with open("gemm_ref.cu", "w") as f:
        f.write(gemm_ref)
    with open("gemm_swizzle.cu", "w") as f:
        f.write(gemm_tc)

    arch = "sm_80"

    print("--- Compiling binary ---")
    try:
        compile_result = subprocess.run(
            [
                "nvcc",
                "-O3",
                "-lineinfo",
                "-Xptxas=-v",
                f"-arch={arch}",
                "-o",
                "bench",
                "benchmark.cu",
                "-lcublas",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(compile_result.stderr)
    except subprocess.CalledProcessError as e:
        print("\n!!! NVCC Compilation Failed !!!")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise

    if profile:
        print(f"--- Profiling GEMM_tc with NCU (set={ncu_set}) ---")
        cmd = [
            "ncu",
            "--clock-control", "none",
            "--set", ncu_set,
            "-k", "regex:GEMM_tc",
            "--launch-skip", "1",
            "--launch-count", "1",
            "--kill", "yes",
            "./bench",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)
        except subprocess.CalledProcessError as e:
            print("\n!!! NCU Profiling Failed !!!")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            raise
    else:
        print("--- Running Performance Benchmark vs cuBLAS ---")
        subprocess.run(["./bench"], check=True)


@app.local_entrypoint()
def main(profile: bool = False, ncu_set: str = "basic"):
    bench_src = pathlib.Path("pmpp_v2/matmul_py/benchmark.cu").read_text()
    gemm_ref = pathlib.Path("pmpp_v2/matmul_py/gemm_ref.cu").read_text()
    gemm_tc = pathlib.Path("pmpp_v2/matmul_py/gemm_swizzle.cu").read_text()

    run_job.remote(bench_src, gemm_ref, gemm_tc, profile=profile, ncu_set=ncu_set)