import argparse
import itertools
import pandas as pd
import pytest
import torch
import triton

from functools import cache

from torch._inductor.utils import get_num_sms

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import triton.profiler as proton
import triton.profiler.language as pl


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

    # fixme: remove this!
    # get_tmem_reg_layout and tma.make_tensor_descriptor have to be
    # there, i.e. Triton package is to be built from commit b6201360e
    # or newer, otherwise the kernel won't work
    from triton.experimental.gluon.language.nvidia.blackwell import (
        get_tmem_reg_layout,
    )
    from triton.experimental.gluon.language.nvidia.blackwell.tma import (
        make_tensor_descriptor,
    )

    # Check for features to be used in the kernel
    USE_UPDATE_TENSOR_DESCRIPTOR = False
    try:
        from triton.experimental.gluon.language.nvidia.blackwell.tma import (
            update_tensor_descriptor,
        )

        USE_UPDATE_TENSOR_DESCRIPTOR = True
    except ImportError:
        pass

    # Load the template from the same directory as this script
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("grouped_mm_kernel_gluon.jinja")
    context = {
        "USE_UPDATE_TENSOR_DESCRIPTOR": USE_UPDATE_TENSOR_DESCRIPTOR,
    }
    rendered = template.render(**context)

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
    num_load_warps=1,
    num_compute_warps=1,
    num_store_warps=4,
    maxnreg=128,
):
    device = C.device
    dtype = torch.int8

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device=device, dtype=dtype)

    triton.set_allocator(alloc_fn)

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.bfloat16)
    B_BLOCK_SHAPE = [1, BLOCK_N, BLOCK_K]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.bfloat16)

    M, N = C.shape[0], C.shape[1]
    K = A.shape[1]
    G = offs.shape[0]

    a_stride_0, a_stride_1 = A.stride(0), A.stride(1)
    b_stride_0, b_stride_1, b_stride_2 = B.stride(0), B.stride(1), B.stride(2)
    c_stride_0, c_stride_1 = C.stride(0), C.stride(1)

    assert BLOCK_M % 1 == 0 and BLOCK_N % 32 == 0
    store_layout = gl.BlockedLayout(
        size_per_thread=[BLOCK_M // 1, BLOCK_N // 32],
        threads_per_warp=[1, 32],
        warps_per_cta=[num_store_warps, 1],
        order=[1, 0],
    )

    num_sms = get_num_sms()
    NUM_BLOCKS = num_sms

    grouped_mm_kernel[[NUM_BLOCKS]](
        A,
        B,
        offs,
        C,
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
        dtype=gl.bfloat16,
        M=M,
        N=N,
        K=K,
        a_stride_0=a_stride_0,
        a_stride_1=a_stride_1,
        b_stride_0=b_stride_0,
        b_stride_1=b_stride_1,
        b_stride_2=b_stride_2,
        c_stride_0=c_stride_0,
        c_stride_1=c_stride_1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        G=G,
        store_layout=store_layout,
        NUM_BLOCKS=NUM_BLOCKS,
        NUM_LOAD_BUFFERS=num_load_buffers,
        NUM_ACC_BUFFERS=num_acc_buffers,
        NUM_LOAD_WARPS=num_load_warps,
        NUM_COMPUTE_WARPS=num_compute_warps,
        num_warps=num_store_warps,
        maxnreg=maxnreg,
    )

    torch.cuda.synchronize()


@pytest.mark.parametrize("G, M, N, K", [(4, 208, 416, 304), (5, 2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_load_buffers", [1, 2])
@pytest.mark.parametrize("num_acc_buffers", [1, 2])
@pytest.mark.skipif(not is_sm100(), reason="Requires Hopper or Blackwell")
def test_grouped_mm(
    G, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_load_buffers, num_acc_buffers
):
    device = "cuda"
    dtype = torch.bfloat16
    align = 16 // dtype.itemsize
    N_align = (N + align - 1) // align * align
    K_align = (K + align - 1) // align * align

    A = torch.randn(M, K_align, device=device, dtype=dtype)[:, :K]
    B = torch.randn((G, N, K_align), device=device, dtype=dtype)[:, :, :K]
    C = torch.empty(M, N_align, device=device, dtype=dtype)[:, :N]

    probs = torch.full((G,), 1.0 / G)
    dist = torch.distributions.Multinomial(total_count=M, probs=probs)
    offs = torch.cumsum(dist.sample(), dim=0).to(torch.int32).to(device)

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
    )

    torch.testing.assert_close(C, C_ref)


def find_configs_sm100(dtype_AB, dtype_C, dtype_acc, M=None, N=None, K=None):
    dtype_AB_bytes = torch.tensor([], dtype=dtype_AB).element_size()
    dtype_C_bytes = torch.tensor([], dtype=dtype_C).element_size()
    dtype_acc_bytes = torch.tensor([], dtype=dtype_acc).element_size()

    SMEM_PER_SM = 228 * 1024  # 228 KB shared memory
    TMEM_MAX_COLUMNS = 512  # 512 columns per CTA (Blackwell)
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
    BLOCK_N_vals = [32, 64, 128]
    BLOCK_K_vals = [32, 64, 128, 256]

    NUM_LOAD_BUFFER_vals = [4, 5, 6, 7]
    NUM_ACC_BUFFER_vals = [3, 4, 5, 6]
    NUM_LOAD_WARP_vals = [1, 2]
    NUM_COMPUTE_WARP_vals = [1, 2]
    NUM_STORE_WARP_vals = [4, 8]
    MAXNREG_vals = [128]

    for (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_load_buffers,
        num_acc_buffers,
        num_load_warps,
        num_compute_warps,
        num_store_warps,
        maxnreg,
    ) in itertools.product(
        BLOCK_M_vals,
        BLOCK_N_vals,
        BLOCK_K_vals,
        NUM_LOAD_BUFFER_vals,
        NUM_ACC_BUFFER_vals,
        NUM_LOAD_WARP_vals,
        NUM_COMPUTE_WARP_vals,
        NUM_STORE_WARP_vals,
        MAXNREG_vals,
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
        # TMEM is per-CTA resource measured in "columns"
        # For TensorMemoryLayout with col_stride=1: each [BLOCK_M, BLOCK_N] buffer uses BLOCK_N columns
        # Hardware limit: 512 columns per CTA
        tmem_usage_columns = BLOCK_N * num_acc_buffers

        # === Register Estimation ===
        total_regs = (
            (num_load_warps + num_compute_warps + num_store_warps) * 32 * maxnreg
        )

        # === Occupancy Calculation ===
        # TMEM is a per-CTA hard limit - either fits or doesn't
        if tmem_usage_columns > TMEM_MAX_COLUMNS:
            continue  # Config exceeds TMEM limit, skip it

        # Maximum CTAs per SM limited by SMEM and registers (shared resources)
        max_ctas_by_smem = SMEM_PER_SM // total_smem if total_smem > 0 else 0
        max_ctas_by_regs = REGS_PER_SM // total_regs if total_regs > 0 else 0

        # Also limited by max CTAs per SM (hardware limit, typically 32 on recent GPUs)
        MAX_CTAS_PER_SM = 32

        # Actual occupancy is minimum across shared resources
        estimated_occupancy = min(max_ctas_by_smem, max_ctas_by_regs, MAX_CTAS_PER_SM)

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
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                maxnreg,
                estimated_occupancy,
            )
        )

    # Sort by estimated occupancy (higher is generally better)
    # But don't filter - let benchmarking decide
    configs.sort(key=lambda x: x[-1], reverse=True)

    return configs


def config_helper(description: str):
    # Configure command line arguments for profiling options
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling (default: False)",
    )
    parser.add_argument(
        "--op-measure",
        action="store_true",
        default=False,
        help="Enable operation measurement. Otherwise, we profile timeline trace. (default: False)",
    )
    parser.add_argument(
        "--warp-sampling",
        action="store_true",
        default=False,
        help="Enable warp sampling during profiling (default: False)",
    )
    parser.add_argument(
        "--increase-accuracy",
        action="store_true",
        default=False,
        help="Enable increased-accuracy during profiling (default: False).",
    )
    parser.add_argument(
        "--warp-ids",
        type=str,
        default="0, 2",
        help="Comma-separated list of warp IDs for warp sampling (default: '0, 2')",
    )
    parser.add_argument(
        "--gmem-buffer",
        action="store_true",
        default=False,
        help="Use global memory as the internal buffer during profiling (default: False).",
    )

    args = parser.parse_args()

    # Configure profiling options based on accuracy requirements
    # Default uses clock_64 for long-running kernels with higher overhead
    opts = ""
    # `clock_32` provides lower overhead per record, `time_shift`` post-processes to reduce noise
    if args.increase_accuracy:
        opts = "clock32,time_shift"

    if args.gmem_buffer:
        buf = "global"
    else:
        buf = "shared"

    # Set up profiling mode based on warp sampling preferences
    if args.warp_sampling:
        # Selective warp sampling allows capturing more events within buffer constraints
        # by only profiling specified warps (e.g. "0,1,2,3")
        mode = proton.mode.Default(
            optimizations=opts,
            sampling_strategy="selective",
            sampling_options=args.warp_ids,
            buffer_type=buf,
        )
    else:
        # Profile all warps - provides complete picture but uses more buffer space
        mode = proton.mode.Default(optimizations=opts, buffer_type=buf)

    return args.profile, args.op_measure, mode


if __name__ == "__main__":
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16
    align = 16 // dtype.itemsize

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
        N_align = (N + align - 1) // align * align
        K_align = (K + align - 1) // align * align

        A = torch.randn(M, K_align, device=device, dtype=dtype)[:, :K]
        B = torch.randn(G, N, K_align, device=device, dtype=dtype)[:, :, :K]
        C = torch.empty(M, N_align, device=device, dtype=dtype)[:, :N]
        offs = torch.arange(M // G, M + 1, M // G, device=device, dtype=torch.int32)
        if offs[-1] != M:
            offs[-1] = M

        print("=================================================================")
        print(f"M = {M}, G = {G}, N = {N}, K = {K}")
        print("-----------------------------------------------------------------")

        configs = find_configs_sm100(
            dtype_AB=dtype,
            dtype_C=dtype,
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
        best_fn = None
        for config in configs:
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                maxnreg,
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
                    num_load_warps,
                    num_compute_warps,
                    num_store_warps,
                    maxnreg,
                )
                ms_curr = triton.testing.do_bench(fn, warmup=2, rep=20)

                if ms_curr < best_ms:
                    best_ms = ms_curr
                    best_config = config
                    best_fn = fn
            except Exception as e:
                import traceback

                print("=" * 60)
                print(
                    f"Error with config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}"
                )
                print(
                    f"  num_load_buffers={num_load_buffers}, num_acc_buffers={num_acc_buffers}"
                )
                print(
                    f"  num_load_warps={num_load_warps}, num_compute_warps={num_compute_warps}, num_store_warps={num_store_warps}"
                )
                tmem_calc = BLOCK_N * num_acc_buffers
                print(f"  Calculated TMEM usage: {tmem_calc} columns (limit: 512)")
                traceback.print_exc()
                print("=" * 60)
                continue

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
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                maxnreg,
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
                f"num_load_buffers={num_load_buffers}, num_acc_buffers={num_acc_buffers}, num_load_warps={num_load_warps}, num_compute_warps={num_compute_warps}, num_store_warps={num_store_warps}, maxnreg={maxnreg}, occupancy={occupancy}"
            )

            # Verify correctness with best config
            try:
                C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
                best_fn()
                torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)
                print("  ✓ Correctness check passed")
            except AssertionError:
                print("  ✗ Correctness check FAILED")
            finally:
                if "C_ref" in locals():
                    del C_ref

            description = "Gluon grouped MM with Proton Intra-Kernel Profiling"
            profile, op_measure, mode = config_helper(description)
            if profile:
                pl.enable_semantic("gluon")
                if op_measure:
                    # Operation measurement mode generates scope-level metrics.
                    # View results: proton-viewer -m normalized_cycles grouped_mm.hatchet
                    # Note: cycles are averaged across all warps/CTAs -
                    # adjust for warp specialization
                    sid = proton.start(
                        "grouped_mm",
                        backend="instrumentation",
                        mode=mode,
                    )
                else:
                    # Timeline trace mode generates Chrome trace format
                    # for visualization.
                    # Output file: grouped_mm.chrome_trace
                    sid = proton.start(
                        "grouped_mm",
                        data="trace",
                        backend="instrumentation",
                        mode=mode,
                    )
                # sid = proton.start(
                #    name="grouped_mm", data="tree", context="shadow", backend="cupti"
                # )
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
                        num_load_warps,
                        num_compute_warps,
                        num_store_warps,
                        maxnreg,
                    )
                proton.finalize(sid)
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
            "CUTLASS latency (us)": us_cutlass,
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
