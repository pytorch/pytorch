import itertools
import pandas as pd
import pytest
import torch
import triton

from functools import cache

from torch._inductor.utils import get_num_sms

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@cache
def is_sm100():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


if __name__ == "__main__" and not is_sm100():
    raise RuntimeError("Blackwell NVIDIA GPU required")


def get_grouped_mm_kernel():
    import importlib.util
    import os
    import sys
    import tempfile

    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path

    # Load the template from the same directory as this script
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("grouped_mm_kernel_gluon.jinja")
    constants = {}
    rendered = template.render(**constants)

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
        tmp_file.write(rendered.encode("utf-8"))
        tmp_filename = tmp_file.name

    module_name = os.path.splitext(os.path.basename(tmp_filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, tmp_filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    # fixme: uncomment this!
    # os.remove(tmp_filename)

    return mod.grouped_mm_kernel


grouped_mm_kernel = get_grouped_mm_kernel()


def grouped_mm(
    A,
    B,
    C,
    offs,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_load_buffers,
    num_acc_buffers,
    num_warps,
):
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.bfloat16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    B_BLOCK_SHAPE = [1, BLOCK_N, BLOCK_K]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.bfloat16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.bfloat16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    # This is for non-TMA store.
    assert BLOCK_M % 1 == 0 and BLOCK_N % 32 == 0
    store_layout = gl.BlockedLayout(
        size_per_thread=[BLOCK_M // 1, BLOCK_N // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],  # row-major for coalesced access
    )

    num_sms = get_num_sms()
    NUM_BLOCKS = num_sms
    grouped_mm_kernel[[NUM_BLOCKS]](
        a_desc,
        b_desc,
        c_desc,
        offs,
        C,
        offs.shape[0],
        store_layout,
        NUM_BLOCKS,
        num_load_buffers=num_load_buffers,
        num_acc_buffers=num_acc_buffers,
        num_warps=num_warps,
    )

    # fixme: remove this!
    torch.cuda.synchronize()


@pytest.mark.parametrize("G, M, N, K", [(4, 208, 416, 304), (5, 2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_load_buffers", [1, 2, 3])
@pytest.mark.parametrize("num_acc_buffers", [1, 2, 3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.skipif(not is_sm100(), reason="Requires Hopper or Blackwell")
def test_grouped_mm(
    G, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_acc_buffers, num_warps
):
    A = torch.randn(G * M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn((G, N, K), device="cuda", dtype=torch.bfloat16)
    C = torch.empty(G * M, N, device="cuda", dtype=torch.bfloat16)

    probs = torch.full((G,), 1.0 / G)
    dist = torch.distributions.Multinomial(total_count=G, probs=probs)
    offs = torch.cumsum(dist.sample(), dim=0).to(torch.int32).to("cuda")

    C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
    grouped_mm(
        A,
        B,
        C,
        offs,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_load_buffers,
        num_acc_buffers,
        num_warps,
    )

    torch.testing.assert_close(C_ref, C)


def find_configs_sm100(dtype_AB, dtype_C, dtype_acc, M=None, N=None, K=None):
    dtype_AB_bytes = torch.tensor([], dtype=dtype_AB).element_size()
    dtype_C_bytes = torch.tensor([], dtype=dtype_C).element_size()
    dtype_acc_bytes = torch.tensor([], dtype=dtype_acc).element_size()

    SMEM_PER_SM = 228 * 1024  # 228 KB shared memory
    TMEM_PER_SM = 64 * 1024  # 64K cells of fp32 tensor memory
    REGS_PER_SM = 65536  # 64K registers

    # Reserve SMEM for mbarriers, barriers, and compiler overhead
    SMEM_OVERHEAD = 2048  # Conservative estimate

    configs = []

    if M is not None and N is not None and K is not None:
        if M < 512:
            BLOCK_M_vals = [64]
        else:
            BLOCK_M_vals = [64, 128]

        if N <= 64:
            BLOCK_N_vals = [32, 64]
        elif N <= 512:
            BLOCK_N_vals = [64, 128, 256]
        else:
            BLOCK_N_vals = [128, 256]

        if K < 256:
            BLOCK_K_vals = [64]
        elif K < 1024:
            BLOCK_K_vals = [64, 128]
        else:
            BLOCK_K_vals = [64, 128, 256]
    else:
        BLOCK_M_vals = [64, 128]
        BLOCK_N_vals = [32, 64, 128, 256]
        BLOCK_K_vals = [64, 128, 256]

    # fixme: remove this!
    BLOCK_M_vals = [64, 128]
    BLOCK_N_vals = [128, 256]
    BLOCK_K_vals = [64, 128, 256]

    num_load_buffers_vals = [1, 2, 3, 4]
    num_acc_buffers_vals = [1, 2]
    num_warps_vals = [4, 8]

    for (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_load_buffers,
        num_acc_buffers,
        num_warps,
    ) in itertools.product(
        BLOCK_M_vals,
        BLOCK_N_vals,
        BLOCK_K_vals,
        num_load_buffers_vals,
        num_acc_buffers_vals,
        num_warps_vals,
    ):
        # === SMEM Calculation ===
        # A and B buffers (double-buffered for load/compute overlap)
        a_smem_per_buffer = BLOCK_M * BLOCK_K * dtype_AB_bytes
        b_smem_per_buffer = BLOCK_N * BLOCK_K * dtype_AB_bytes
        load_smem = (a_smem_per_buffer + b_smem_per_buffer) * num_load_buffers

        # C buffer for store (reuses space after load/compute phase)
        c_smem = BLOCK_M * BLOCK_N * dtype_C_bytes

        # Barrier arrays: 2 sets of barriers for load buffers + 2 sets for acc buffers
        barrier_smem = 8 * (num_load_buffers + num_acc_buffers) * 2

        total_smem = load_smem + c_smem + barrier_smem + SMEM_OVERHEAD

        # === TMEM Calculation ===
        # Accumulators are stored in TMEM (fp32)
        tmem_usage = BLOCK_M * BLOCK_N * num_acc_buffers

        # === Register Estimation ===
        # Conservative estimate based on warp specialization
        # Load warp: mostly addresses and loop counters (~32 regs/thread)
        # Compute warp: MMA operands and loop state (~64 regs/thread)
        # Store warp: converted data and addresses (~48 regs/thread per warp)
        load_regs = 1 * 32 * 32  # 1 warp * 32 threads * 32 regs
        compute_regs = 1 * 32 * 64  # 1 warp * 32 threads * 64 regs
        store_regs = (num_warps - 2) * 32 * 48  # Remaining warps * 32 threads * 48 regs
        total_regs = load_regs + compute_regs + store_regs

        # === Occupancy Calculation ===
        # Maximum CTAs per SM limited by each resource
        max_ctas_by_smem = SMEM_PER_SM // total_smem if total_smem > 0 else 0
        max_ctas_by_tmem = TMEM_PER_SM // tmem_usage if tmem_usage > 0 else 0
        max_ctas_by_regs = REGS_PER_SM // total_regs if total_regs > 0 else 0

        # Also limited by max CTAs per SM (hardware limit, typically 32 on recent GPUs)
        MAX_CTAS_PER_SM = 32

        # Actual occupancy is minimum across all resources
        estimated_occupancy = min(
            max_ctas_by_smem, max_ctas_by_tmem, max_ctas_by_regs, MAX_CTAS_PER_SM
        )

        # Filter out invalid configs (occupancy < 1 means doesn't fit)
        if estimated_occupancy < 1:
            continue

        # Optional: Skip very low occupancy configs (uncomment if desired)
        # if estimated_occupancy < 2:
        #     continue

        configs.append(
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_warps,
                estimated_occupancy,
            )
        )

    # Sort by estimated occupancy (higher is generally better)
    # But don't filter - let benchmarking decide
    configs.sort(key=lambda x: x[-1], reverse=True)

    return configs


if __name__ == "__main__":
    torch.manual_seed(0)

    results = []
    for M, G, N, K in [
        # [5, 2, 16, 16],
        # [13, 3, 16, 32],
        # [128, 8, 16, 16],
        # [253, 7, 24, 24],
        # [512, 8, 32, 64],
        # [1024, 16, 256, 1024],
        # [2048, 32, 512, 256],
        # [2048, 32, 512, 2048],
        # [4834, 24, 5120, 1536],
        # [8257, 32, 5120, 1536],
        # [32768, 24, 6144, 2048],
        # [32768, 48, 6144, 2048],
        # [32768, 64, 6144, 2048],
        # [65536, 24, 6144, 2048],
        [65536, 32, 6144, 2048],
        # [65536, 48, 6144, 2048],
        # [65536, 64, 6144, 2048],
        # [131072, 24, 6144, 2048],
        # [131072, 32, 6144, 2048],
        # [131072, 48, 6144, 2048],
        # [131072, 64, 6144, 2048],
    ]:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        offs = torch.arange(M // G, M + 1, M // G, device="cuda", dtype=torch.int32)
        if offs[-1] != M:
            offs[-1] = M

        print("=================================================================")
        print(f"M = {M}, G = {G}, N = {N}, K = {K}")
        print("-----------------------------------------------------------------")

        configs = find_configs_sm100(
            dtype_AB=torch.bfloat16,
            dtype_C=torch.bfloat16,
            dtype_acc=torch.float32,
            M=M,
            N=N,
            K=K,
        )

        flops = 2 * M * N * K

        # Benchmark baseline torch._grouped_mm
        fn = lambda: torch._grouped_mm(A, B.transpose(-2, -1), offs)
        us_cutlass = triton.testing.do_bench(fn, warmup=2, rep=20) * 1e3
        tflops_per_sec = flops * 1e-12 / (us_cutlass * 1e-6)
        print(
            f"{"torch._grouped_mm":<36} {us_cutlass:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
        )

        # Benchmark compiled torch._grouped_mm
        torch._dynamo.reset()
        grouped_mm_triton = torch.compile(
            torch._grouped_mm,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        fn = lambda: grouped_mm_triton(A, B.transpose(-2, -1), offs)
        us_triton = triton.testing.do_bench(fn, warmup=2, rep=20) * 1e3
        tflops_per_sec = flops * 1e-12 / (us_triton * 1e-6)
        print(
            f"{"Triton compiled torch._grouped_mm":<36} {us_triton:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
        )

        # Benchmark compiled torch._grouped_mm
        try:
            torch._dynamo.reset()
            grouped_mm_cute = torch.compile(
                torch._grouped_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTE",
                },
            )
            fn = lambda: grouped_mm_cute(A, B.transpose(-2, -1), offs)
            us_cute = triton.testing.do_bench(fn, warmup=2, rep=20) * 1e3
            tflops_per_sec = flops * 1e-12 / (us_cute * 1e-6)
            print(
                f"{"CuTe compiled torch._grouped_mm":<36} {us_cute:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
            )
        except:
            us_cute = None
            pass
        

        # Autotune and benchmark Gluon grouped MM
        best_ms = float("inf")
        best_config = None
        for config in configs:
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_warps,
                occupancy,
            ) = config

            try:
                # Benchmark
                fn = lambda: grouped_mm(
                    A,
                    B,
                    C,
                    offs,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    num_load_buffers,
                    num_acc_buffers,
                    num_warps,
                )
                ms_curr = triton.testing.do_bench(fn, warmup=2, rep=20)

                if ms_curr < best_ms:
                    best_ms = ms_curr
                    best_config = config

            except Exception as e:
                # Config failed (e.g., resource constraints), skip it
                continue

        us_gluon = float("inf")
        if best_config is not None:
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_warps,
                occupancy,
            ) = best_config

            us_gluon = best_ms * 1e3
            tflops_per_sec = flops * 1e-12 / (us_gluon * 1e-6)
            print(
                f"{"Gluon grouped_mm":<36} {us_gluon:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
            )

            # Print config
            print(
                f"  Best config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                f"num_load_buffers={num_load_buffers}, num_acc_buffers={num_acc_buffers}, num_warps={num_warps}, occupancy={occupancy}"
            )

            # Verify correctness with best config
            try:
                C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
                grouped_mm(
                    A,
                    B,
                    C,
                    offs,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    num_load_buffers,
                    num_acc_buffers,
                    num_warps,
                )
                torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)
                print("  ✓ Correctness check passed")
            except AssertionError:
                print("  ✗ Correctness check FAILED")
            finally:
                if "C_ref" in locals():
                    del C_ref

            # fixme: remove this!
            """
            import triton.profiler as proton

            sid = proton.start(name="grouped_mm", data="trace", context="shadow")
            with proton.scope("CUTLASS grouped_mm"):
                torch._grouped_mm(A, B.transpose(-2, -1), offs)
            with proton.scope("Triton grouped_mm"):
                grouped_mm_triton(A, B.transpose(-2, -1), offs)
            with proton.scope("CuTe grouped_mm"):
                grouped_mm_cute(A, B.transpose(-2, -1), offs)
            with proton.scope("Gluon grouped_mm"):
                grouped_mm(
                    A,
                    B,
                    C,
                    offs,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    num_load_buffers,
                    num_acc_buffers,
                    num_warps,
                )
            proton.finalize(sid)
            """
        else:
            print(f"{"Gluon grouped_mm":<36} No valid config found")

        print()

        # Clean up GPU memory before next iteration
        del A, B, C, offs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        result = {
            "M": M,
            "G": G,
            "N": N,
            "K": K,
            "CUTLASS latency (us)":  us_cutlass,
            }
        if us_cute is not None:
            result["CuTe latency (us)"] = us_cutlass
        result["Triton latency (us)"] = us_triton
        result["Gluon latency (us)"] = us_gluon
        if us_cute is not None:
            result["CuTe speedup"] = us_cutlass / us_cute
        result["Triton speedup"] = us_cutlass / us_triton
        result["Gluon speedup"] = us_cutlass / us_gluon
        results.append(result)

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
