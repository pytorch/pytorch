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


def compute_stage_variants(
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype,
    num_store_warps: int = 4,
    occupancy: int = 1,
    smem_capacity: int = 228 * 1024,  # Blackwell: 228 KB
    tmem_max_columns: int = 512,  # Blackwell: 512 columns per CTA
):
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    # Hardware limit: 232448 bytes (227 KB) as reported in error messages
    # Use this exact limit - no made-up safety factors
    smem_limit = 232448

    # Calculate SMEM usage
    a_bytes_per_stage = BLOCK_M * BLOCK_K * dtype_bytes
    b_bytes_per_stage = BLOCK_N * BLOCK_K * dtype_bytes
    c_bytes_per_stage = BLOCK_M * BLOCK_N * dtype_bytes
    ab_bytes_per_stage = a_bytes_per_stage + b_bytes_per_stage

    # Check if even MINIMUM config fits (1 load buffer, 1 acc buffer)
    # Add small margin for compiler overhead
    min_load_buffers = 1
    min_acc_buffers = 1
    compiler_overhead = 256

    min_smem = (
        ab_bytes_per_stage * min_load_buffers
        + c_bytes_per_stage
        + 8 * min_load_buffers * 2
        + 8 * min_acc_buffers * 2
        + compiler_overhead
    )

    if min_smem > smem_limit:
        # This tile size is too large - even minimal pipelining doesn't fit
        return []

    valid_configs = []

    # Try all combinations of load/acc buffers
    for num_load_buffers in range(8, 0, -1):
        ab_smem = ab_bytes_per_stage * num_load_buffers
        c_smem = c_bytes_per_stage  # 1 epilogue buffer
        load_barrier_smem = 8 * num_load_buffers * 2

        base_smem = ab_smem + c_smem + load_barrier_smem + compiler_overhead

        if base_smem > smem_limit:
            continue  # Too many load buffers

        # Try different acc buffer counts
        max_acc_by_tmem = tmem_max_columns // BLOCK_N
        remaining_smem = smem_limit - base_smem
        max_acc_by_smem = remaining_smem // (8 * 2)  # Each acc buffer needs 2 barriers

        max_acc_buffers = min(max_acc_by_tmem, max_acc_by_smem, 8)

        for num_acc_buffers in range(max_acc_buffers, 0, -1):
            acc_barrier_smem = 8 * num_acc_buffers * 2
            total_smem = base_smem + acc_barrier_smem
            tmem_cols = BLOCK_N * num_acc_buffers

            if total_smem <= smem_limit and tmem_cols <= tmem_max_columns:
                valid_configs.append((num_load_buffers, num_acc_buffers))

    # If no configs found, return empty (will be filtered out by caller)
    return valid_configs


def compute_stages(
    BLOCK_M: int, BLOCK_N: int, BLOCK_K: int, dtype, num_store_warps: int = 4, **kwargs
):
    variants = compute_stage_variants(
        BLOCK_M, BLOCK_N, BLOCK_K, dtype, num_store_warps, **kwargs
    )
    return variants[0] if variants else (1, 1)


def get_grouped_mm_kernel(use_update_tensor_descriptor=False):
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

    # Check if update_tensor_descriptor is available
    has_update_tensor_descriptor = False
    try:
        from triton.experimental.gluon.language.nvidia.blackwell.tma import (
            update_tensor_descriptor,
        )
        has_update_tensor_descriptor = True
    except ImportError:
        pass

    # Use the specified mode if feature is available, otherwise fall
    # back to ragged
    USE_UPDATE_TENSOR_DESCRIPTOR = use_update_tensor_descriptor and has_update_tensor_descriptor

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


grouped_mm_kernel_ragged = get_grouped_mm_kernel(use_update_tensor_descriptor=False)
grouped_mm_kernel_update = get_grouped_mm_kernel(use_update_tensor_descriptor=True)


def grouped_mm(
    A,
    B,
    C,
    offs,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    num_load_buffers=None,
    num_acc_buffers=None,
    num_load_warps=1,
    num_compute_warps=1,
    num_store_warps=4,
    num_load_thread_registers=24,
    num_compute_thread_registers=24,
    maxnreg=128,
    kernel=None,
):
    assert (
        num_load_thread_registers < maxnreg and num_compute_thread_registers < maxnreg
    )

    # Use the specified kernel or default to ragged
    if kernel is None:
        kernel = grouped_mm_kernel_ragged

    if num_load_buffers is None or num_acc_buffers is None:
        computed_load, computed_acc = compute_stages(
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            dtype=torch.bfloat16,
            num_store_warps=num_store_warps,
        )
        if num_load_buffers is None:
            num_load_buffers = computed_load
        if num_acc_buffers is None:
            num_acc_buffers = computed_acc

    device = C.device
    dtype = torch.int8

    def alloc_fn(size: int, alignment: int, stream: int):
        return torch.empty(size, device=device, dtype=dtype)

    triton.set_allocator(alloc_fn)

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.bfloat16)
    B_BLOCK_SHAPE = [1, BLOCK_N, BLOCK_K]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.bfloat16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.bfloat16)
    # 4D layout for ragged descriptor path
    c_layout_ragged = gl.NVMMASharedLayout.get_default_for([1, 1, BLOCK_M, BLOCK_N], gl.bfloat16)

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

    kernel[[NUM_BLOCKS]](
        A,
        B,
        offs,
        C,
        a_layout=a_layout,
        b_layout=b_layout,
        c_layout=c_layout,
        c_layout_ragged=c_layout_ragged,
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
        NUM_LOAD_THREAD_REGISTERS=num_load_thread_registers,
        NUM_COMPUTE_THREAD_REGISTERS=num_compute_thread_registers,
        num_warps=num_store_warps,
        maxnreg=maxnreg,
    )

    torch.cuda.synchronize()


@pytest.mark.parametrize("G, M, N, K", [(4, 208, 416, 304), (5, 2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.skipif(not is_sm100(), reason="Requires Hopper or Blackwell")
def test_grouped_mm(G, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
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

    grouped_mm(A, B, C, offs, BLOCK_M, BLOCK_N, BLOCK_K)

    torch.testing.assert_close(C, C_ref)


def find_configs_sm100(dtype_AB, dtype_C, dtype_acc, M=None, N=None, K=None):
    configs = []

    if M is not None and N is not None and K is not None:
        if M < 256:
            BLOCK_M_vals = [64]
        else:
            BLOCK_M_vals = [64, 128]
        if N <= 64:
            BLOCK_N_vals = [32, 64]
        elif N <= 512:
            BLOCK_N_vals = [64, 128, 256]
        else:
            BLOCK_N_vals = [128, 256]
        if K < 128:
            BLOCK_K_vals = [64]
        elif K < 512:
            BLOCK_K_vals = [64, 128]
        else:
            BLOCK_K_vals = [64, 128, 256]

    NUM_LOAD_WARP_vals = [1, 2]
    NUM_COMPUTE_WARP_vals = [1, 2]
    NUM_STORE_WARP_vals = [4, 8]
    NUM_LOAD_THREAD_REGISTERS_vals = [24]
    NUM_COMPUTE_THREAD_REGISTERS_vals = [24]
    MAXNREG_vals = [128]

    for (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_load_warps,
        num_compute_warps,
        num_store_warps,
        num_load_thread_registers,
        num_compute_thread_registers,
        maxnreg,
    ) in itertools.product(
        BLOCK_M_vals,
        BLOCK_N_vals,
        BLOCK_K_vals,
        NUM_LOAD_WARP_vals,
        NUM_COMPUTE_WARP_vals,
        NUM_STORE_WARP_vals,
        NUM_LOAD_THREAD_REGISTERS_vals,
        NUM_COMPUTE_THREAD_REGISTERS_vals,
        MAXNREG_vals,
    ):
        buffer_variants = compute_stage_variants(
            BLOCK_M, BLOCK_N, BLOCK_K, dtype=dtype_AB, num_store_warps=num_store_warps
        )

        for num_load_buffers, num_acc_buffers in buffer_variants:
            total_regs = (
                (num_load_warps + num_compute_warps + num_store_warps) * 32 * maxnreg
            )
            REGS_PER_SM = 65536
            MAX_CTAS_PER_SM = 32
            estimated_occupancy = min(REGS_PER_SM // total_regs, MAX_CTAS_PER_SM)

            if estimated_occupancy < 1:
                continue

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
                    num_load_thread_registers,
                    num_compute_thread_registers,
                    maxnreg,
                    estimated_occupancy,
                )
            )

    return configs


def autotune_grouped_mm(A, B, C, offs, configs, flops, kernel):
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
            num_load_thread_registers,
            num_compute_thread_registers,
            maxnreg,
            occupancy,
        ) = config

        try:
            def make_fn(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                num_load_thread_registers,
                num_compute_thread_registers,
                maxnreg,
            ):
                return lambda: grouped_mm(
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
                    num_load_thread_registers,
                    num_compute_thread_registers,
                    maxnreg,
                    kernel,
                )

            fn = make_fn(
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                num_load_thread_registers,
                num_compute_thread_registers,
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

    return best_ms, best_config, best_fn


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

    # fixme: uncomment this!
    # os.environ["CUTEDSL_ENABLE_AUTOTUNING"] = "1"
    # os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = "EXHAUSTIVE"

    results = []
    for M, G, N, K in [
        [5, 2, 16, 16],
        [13, 3, 16, 32],
        [128, 8, 16, 16],
        [253, 7, 24, 24],
        [512, 8, 32, 64],
        [1024, 16, 256, 1024],
        [2048, 32, 512, 256],
        [2048, 32, 512, 2048],
        [4834, 24, 5120, 1536],
        [8257, 32, 5120, 1536],
        [32768, 24, 6144, 2048],
        [32768, 48, 6144, 2048],
        [32768, 64, 6144, 2048],
        [65536, 24, 6144, 2048],
        [65536, 32, 6144, 2048],  ##
        [65536, 48, 6144, 2048],
        [65536, 64, 6144, 2048],
        [131072, 24, 6144, 2048],
        [131072, 32, 6144, 2048],
        [131072, 48, 6144, 2048],
        [131072, 64, 6144, 2048],
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
        try:
            fn = lambda: grouped_mm_triton(A, B.transpose(-2, -1), offs)
            us_triton = triton.testing.do_bench(fn, warmup=2, rep=20) * 1e3
            tflops_per_sec = flops * 1e-12 / (us_triton * 1e-6)
            print(
                f"{"Triton compiled torch._grouped_mm":<36} {us_triton:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
            )
        except:
            us_triton = None

        # Benchmark compiled torch._grouped_mm
        try:
            torch._dynamo.reset()
            grouped_mm_cute = torch.compile(
                torch._grouped_mm,
                options={
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTEDSL",
                },
                dynamic=False,
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

        # fixme: remove this!
        us_cute = None

        print("Autotuning Gluon grouped_mm (ragged TMA)...")
        best_ms_ragged, best_config_ragged, best_fn_ragged = autotune_grouped_mm(
            A, B, C, offs, configs, flops, grouped_mm_kernel_ragged
        )

        print("Autotuning Gluon grouped_mm (update_tensor_descriptor)...")
        best_ms_update, best_config_update, best_fn_update = autotune_grouped_mm(
            A, B, C, offs, configs, flops, grouped_mm_kernel_update
        )

        us_gluon_ragged = float("inf")
        us_gluon_update = float("inf")

        if best_config_ragged is not None:
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                num_load_thread_registers,
                num_compute_thread_registers,
                maxnreg,
                occupancy,
            ) = best_config_ragged

            us_gluon_ragged = best_ms_ragged * 1e3
            tflops_per_sec = flops * 1e-12 / (us_gluon_ragged * 1e-6)
            print(
                f"{"Gluon grouped_mm (ragged)":<36} {us_gluon_ragged:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
            )

            # Print config
            print(
                f"  Best config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                f"num_load_buffers={num_load_buffers}, num_acc_buffers={num_acc_buffers}, num_load_warps={num_load_warps}, num_compute_warps={num_compute_warps}, num_store_warps={num_store_warps}, num_load_thread_registers={num_load_thread_registers}, num_compute_thread_registers={num_compute_thread_registers}, maxnreg={maxnreg}, occupancy={occupancy}"
            )

            # Verify correctness with best config
            try:
                C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
                best_fn_ragged()
                torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)
                print("  ✓ Correctness check passed")
            except AssertionError:
                print("  ✗ Correctness check FAILED")
            finally:
                if "C_ref" in locals():
                    del C_ref
        else:
            us_gluon_ragged = None
            print(f"{"Gluon grouped_mm (ragged)":<36} No valid config found")

        if best_config_update is not None:
            (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_load_buffers,
                num_acc_buffers,
                num_load_warps,
                num_compute_warps,
                num_store_warps,
                num_load_thread_registers,
                num_compute_thread_registers,
                maxnreg,
                occupancy,
            ) = best_config_update

            us_gluon_update = best_ms_update * 1e3
            tflops_per_sec = flops * 1e-12 / (us_gluon_update * 1e-6)
            print(
                f"{"Gluon grouped_mm (update_desc)":<36} {us_gluon_update:>9.2f} us {tflops_per_sec:>8.2f} TFLOPS"
            )

            # Print config
            print(
                f"  Best config: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, "
                f"num_load_buffers={num_load_buffers}, num_acc_buffers={num_acc_buffers}, num_load_warps={num_load_warps}, num_compute_warps={num_compute_warps}, num_store_warps={num_store_warps}, num_load_thread_registers={num_load_thread_registers}, num_compute_thread_registers={num_compute_thread_registers}, maxnreg={maxnreg}, occupancy={occupancy}"
            )

            # Verify correctness with best config
            try:
                C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
                best_fn_update()
                torch.testing.assert_close(C, C_ref, rtol=1e-2, atol=1e-2)
                print("  ✓ Correctness check passed")
            except AssertionError:
                print("  ✗ Correctness check FAILED")
            finally:
                if "C_ref" in locals():
                    del C_ref
        else:
            us_gluon_update = None
            print(f"{"Gluon grouped_mm (update_desc)":<36} No valid config found")

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
            result["CuTe latency (us)"] = us_cute
        if us_triton is not None:
            result["Triton latency (us)"] = us_triton
        if us_gluon_ragged is not None:
            result["Gluon (ragged) latency (us)"] = us_gluon_ragged
        if us_gluon_update is not None:
            result["Gluon (update) latency (us)"] = us_gluon_update
        if us_cute is not None:
            result["CuTe speedup"] = us_cutlass / us_cute
        if us_triton is not None:
            result["Triton speedup"] = us_cutlass / us_triton
        if us_gluon_ragged is not None:
            result["Gluon (ragged) speedup"] = us_cutlass / us_gluon_ragged
        if us_gluon_update is not None:
            result["Gluon (update) speedup"] = us_cutlass / us_gluon_update
        results.append(result)

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
