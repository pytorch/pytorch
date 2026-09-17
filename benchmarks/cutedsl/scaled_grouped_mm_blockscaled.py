import argparse
import gc
import importlib
import time
import warnings
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch._native.ops.scaled_grouped_mm._cpp_utils import (
    _call_cpp_scaled_grouped_mm_v2,
)
from torch.nn.functional import scaled_grouped_mm, ScalingType, SwizzleType
from torch.testing._internal.common_quantized import (
    _bfloat16_to_float4_e2m1fn_x2,
    to_blocked,
    to_mxfp,
)


_BENCH_SETTLE_SECONDS = 0.1


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 10 and minor in (0, 3)


def _parse_input_dtype(value):
    value = value.lower()
    if value != "bf16":
        raise argparse.ArgumentTypeError(
            "Only bf16 is supported for --input-dtype for now."
        )
    return value


def _parse_gmnk(value):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Invalid gmnk '{value}'. Expected G,M,N,K.")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected integers."
        ) from exc
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected G,M,N,K >= 1."
        )
    return values


def _parse_mn_pair(value, arg_name: str):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid {arg_name} '{value}'. Expected M,N.")
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} '{value}'. Expected integers."
        ) from exc
    if any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            f"Invalid {arg_name} '{value}'. Expected M,N >= 1."
        )
    return values[0], values[1]


def _parse_mma_tile_mn(value):
    return _parse_mn_pair(value, "--mma-tile-mn")


def _parse_cluster_shape_mn(value):
    return _parse_mn_pair(value, "--cluster-shape-mn")


def _parse_format(value):
    value = value.lower()
    if value not in ("mxfp8", "mxfp4", "nvfp4"):
        raise argparse.ArgumentTypeError(
            f"Unsupported --format '{value}'. Expected mxfp8, mxfp4, or nvfp4."
        )
    return value


_STAT_CHOICES = ("mean", "median", "min", "max", "p10", "p90")


class BenchResult(NamedTuple):
    value: float
    min_us: float
    max_us: float


def _percentile(sorted_samples: list[float], pct: float) -> float:
    idx = (len(sorted_samples) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = idx - lo
    return sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac


def _summarize(samples_us: list[float], stat: str) -> BenchResult:
    sorted_samples = sorted(samples_us)
    if stat == "mean":
        value = sum(samples_us) / len(samples_us)
    elif stat == "median":
        value = _percentile(sorted_samples, 50.0)
    elif stat == "min":
        value = sorted_samples[0]
    elif stat == "max":
        value = sorted_samples[-1]
    elif stat == "p10":
        value = _percentile(sorted_samples, 10.0)
    elif stat == "p90":
        value = _percentile(sorted_samples, 90.0)
    else:
        raise ValueError(f"unknown stat '{stat}', expected one of {_STAT_CHOICES}")
    return BenchResult(value=value, min_us=sorted_samples[0], max_us=sorted_samples[-1])


def _do_bench_cuda(fn, warmup=10, rep=100, stat: str = "median") -> BenchResult:
    for _ in range(warmup):
        time.sleep(_BENCH_SETTLE_SECONDS)
        fn()
    torch.cuda.synchronize()
    samples_us = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        time.sleep(_BENCH_SETTLE_SECONDS)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return _summarize(samples_us, stat)


def _require_cutedsl_scaled_grouped_mm_override():
    from torch._native import cutedsl_utils as cu, registry
    from torch._native.common_utils import check_native_jit_disabled

    if not cu.runtime_available():
        raise RuntimeError(
            "--backend cute requested, but CuTeDSL runtime dependencies "
            "are not available. Install nvidia-cutlass-dsl and "
            "apache-tvm-ffi."
        )
    if check_native_jit_disabled():
        raise RuntimeError(
            "--backend cute requested, but native DSL overrides are disabled "
            "by TORCH_DISABLE_NATIVE_JIT=1."
        )
    if "_scaled_grouped_mm_v2" not in registry.get_dsl_operations("cutedsl"):
        version = cu.runtime_version()
        raise RuntimeError(
            "--backend cute requested, but the CuTeDSL override for "
            "aten::_scaled_grouped_mm_v2 is not registered. Current "
            f"nvidia-cutlass-dsl version: {version}. If this is a newer "
            "local test version, rerun with TORCH_NATIVE_SKIP_VERSION_CHECK=1."
        )


def _maybe_wrap_cuda_graph(fn, label: str, use_cuda_graphs: bool):
    if not use_cuda_graphs:
        return fn

    keep_alive = [None]
    try:
        for _ in range(5):
            keep_alive[0] = fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph):
                keep_alive[0] = fn()
        torch.cuda.current_stream().wait_stream(capture_stream)

        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()

        def _replay():
            graph.replay()

        return _replay
    except Exception as exc:
        warnings.warn(
            (
                f"CUDA graph capture failed for backend '{label}', "
                f"falling back to eager: {exc}"
            ),
            stacklevel=2,
        )
        return fn


def _generate_offsets(total, groups, device, mode="balanced", align=1):
    if total <= 0:
        return torch.zeros(groups, device=device, dtype=torch.int32)
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")
    if mode not in ("balanced", "random"):
        raise ValueError(f"mode must be 'balanced' or 'random', got {mode}")

    if mode == "balanced":
        if align == 1:
            base = total // groups
            remainder = total - base * groups
            if remainder != 0:
                warnings.warn(
                    (
                        f"grouping='balanced' with M={total}, G={groups}: "
                        f"using base size {base} and placing tail "
                        f"{remainder} in the last group"
                    ),
                    stacklevel=2,
                )
            counts = torch.full((groups,), base, device=device, dtype=torch.int64)
            if remainder > 0:
                counts[-1] += remainder
        else:
            units = total // align
            remainder = total - units * align
            base_units = units // groups
            extra_units = units - base_units * groups
            counts = torch.full(
                (groups,), base_units * align, device=device, dtype=torch.int64
            )
            if extra_units > 0:
                counts[-extra_units:] += align
            if remainder != 0 or extra_units != 0:
                warnings.warn(
                    (
                        f"grouping='balanced' with M={total}, G={groups}, "
                        f"align={align}: "
                        f"using aligned base size {base_units * align} "
                        "and placing "
                        f"{extra_units * align + remainder} values in "
                        "the last groups"
                    ),
                    stacklevel=2,
                )
            if remainder > 0:
                counts[-1] += remainder
    elif align == 1:
        probs = torch.full((groups,), 1.0 / groups, device=device)
        counts = torch.distributions.Multinomial(
            total_count=total, probs=probs
        ).sample()
        counts = counts.to(dtype=torch.int64)
    else:
        units = total // align
        remainder = total - units * align
        probs = torch.full((groups,), 1.0 / groups, device=device)
        if units == 0:
            counts = torch.zeros(groups, device=device, dtype=torch.int64)
        else:
            counts = torch.distributions.Multinomial(
                total_count=units, probs=probs
            ).sample()
            counts = counts.to(dtype=torch.int64) * align
        counts[-1] += remainder

    return torch.cumsum(counts, dim=0).to(dtype=torch.int32)


def _convert_to_mxfp8(t, block_size: int):
    t_scale, t_lp = to_mxfp(t, format="mxfp8")
    return t_lp, t_scale


def _unpack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    out = torch.empty(
        (*shape[:-1], shape[-1] * 2), device=uint8_data.device, dtype=torch.uint8
    )
    uint8_data_as_uint8 = uint8_data.view(torch.uint8).contiguous().view(-1)
    out_flat = out.view(-1)
    out_flat[1::2] = uint8_data_as_uint8 >> 4
    out_flat[::2] = uint8_data_as_uint8 & 0xF
    return out_flat.view((*shape[:-1], shape[-1] * 2))


def _convert_to_mxfp4(t, block_size: int):
    t_scale, t_lp = to_mxfp(t, format="mxfp4")
    return t_lp, t_scale


def _convert_to_nvfp4(t, block_size: int):
    f8e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    orig_shape = t.shape
    t_flat = t.reshape(-1, block_size)

    block_max = torch.amax(torch.abs(t_flat), 1) + 1e-12
    global_max = t_flat.abs().max()

    scale_enc = 6.0 * f8e4m3_max / global_max
    scale_dec = 1.0 / scale_enc
    scale_dec_block = block_max / 6.0
    scale_dec_block_f8 = (scale_dec_block * scale_enc).to(torch.float8_e4m3fn)
    scale_enc_block = scale_enc / scale_dec_block_f8.float()

    t_scaled = (scale_enc_block.unsqueeze(1) * t_flat).bfloat16().reshape(orig_shape)
    t_scale = scale_dec_block_f8.reshape(orig_shape[0], -1)
    t_lp = _bfloat16_to_float4_e2m1fn_x2(t_scaled)
    return t_lp, t_scale, scale_dec


def _convert_to_blockscaled(t, block_size: int, format: str):
    if format == "mxfp8":
        t_lp, t_scale = _convert_to_mxfp8(t, block_size)
        return t_lp, t_scale, None
    if format == "mxfp4":
        t_lp, t_scale = _convert_to_mxfp4(t, block_size)
        return t_lp, t_scale, None
    if format == "nvfp4":
        t_lp, t_scale, t_global_scale = _convert_to_nvfp4(t, block_size)
        return t_lp, t_scale, t_global_scale
    raise ValueError(f"Unsupported format: {format}")


def _prepare_inputs(
    g: int,
    m: int,
    n: int,
    k: int,
    dtype,
    seed: int,
    grouping: str = "balanced",
    layout_mode: str = "2d/3d",
    format: str = "mxfp8",
):
    torch.manual_seed(seed)
    device = "cuda"
    block_size = 16 if format == "nvfp4" else 32
    k_quant_multiple = 32 if format == "nvfp4" else block_size
    k_eff = ((k + k_quant_multiple - 1) // k_quant_multiple) * k_quant_multiple
    align = 16 // dtype.itemsize
    k_align = (k_eff + align - 1) // align * align

    if layout_mode == "2d/3d":
        x = torch.randn((m, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.1
        if k_eff > k:
            x[:, k:] = 0
        offs = _generate_offsets(m, g, device, mode=grouping, align=16)

        wq_list = []
        w_blocked_scales = []
        w_global_scales = []
        for i in range(g):
            w_i = (
                torch.randn((n, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.01
            )
            if k_eff > k:
                w_i[:, k:] = 0
            converted = _convert_to_blockscaled(w_i, block_size, format)
            if format == "nvfp4":
                wq, w_scale, w_global = converted
                w_global_scales.append(w_global)
            else:
                wq, w_scale, _ = converted
            w_scale_blocked = to_blocked(w_scale)
            wq_list.append(wq)
            w_blocked_scales.append(w_scale_blocked)
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_blocked_scales = torch.stack(w_blocked_scales, dim=0).contiguous()
        w_global_scales = (
            torch.stack(w_global_scales).to(device=device, dtype=torch.float32)
            if w_global_scales
            else None
        )

        xq_list = []
        x_scale_list = []
        x_global_scales = []
        x_scale_elems = 0
        for i in range(g):
            start = 0 if i == 0 else int(offs[i - 1].item())
            end = int(offs[i].item())
            if end == start:
                if format == "nvfp4":
                    x_global_scales.append(torch.tensor(1.0, device=device))
                continue
            x_slice = x[start:end, :]
            converted = _convert_to_blockscaled(x_slice, block_size, format)
            if format == "nvfp4":
                xq_slice, x_scale, x_global = converted
                x_global_scales.append(x_global)
            else:
                xq_slice, x_scale, _ = converted
            x_scale_blocked = to_blocked(x_scale)
            x_scale_list.append(x_scale_blocked)
            x_scale_elems += x_scale_blocked.numel()
            xq_list.append(xq_slice)
        xq = torch.cat(xq_list, dim=0).contiguous()
        x_scale_flat = torch.empty(
            (x_scale_elems,), device=device, dtype=w_blocked_scales.dtype
        )
        offset = 0
        for t in x_scale_list:
            n = t.numel()
            x_scale_flat[offset : offset + n] = t.view(-1)
            offset += n
        x_blocked_scales = x_scale_flat.reshape(-1, k_eff // block_size).contiguous()
        xq = xq.view(-1, xq.shape[-1])
        x_global_scales = (
            torch.stack(x_global_scales).to(device=device, dtype=torch.float32)
            if x_global_scales
            else None
        )
    else:
        x = torch.randn((m, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.1
        w = torch.randn((n, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.01
        if k_eff > k:
            x[:, k:] = 0
            w[:, k:] = 0
        offs_align = 32 if format in ("mxfp4", "nvfp4") else block_size
        offs = _generate_offsets(k_eff, g, device, mode=grouping, align=offs_align)
        m_rounded = ((m + 127) // 128) * 128
        n_rounded = ((n + 127) // 128) * 128

        xq_list = []
        x_scale_list = []
        x_global_scales = []
        wq_list = []
        w_scale_list = []
        w_global_scales = []
        for i in range(g):
            start = 0 if i == 0 else int(offs[i - 1].item())
            end = int(offs[i].item())
            if end == start:
                if format == "nvfp4":
                    x_global_scales.append(torch.tensor(1.0, device=device))
                    w_global_scales.append(torch.tensor(1.0, device=device))
                continue
            x_slice = x[:, start:end].contiguous()
            w_slice = w[:, start:end].contiguous()
            converted_x = _convert_to_blockscaled(x_slice, block_size, format)
            converted_w = _convert_to_blockscaled(w_slice, block_size, format)
            if format == "nvfp4":
                xq, x_scale, x_global = converted_x
                wq, w_scale, w_global = converted_w
                x_global_scales.append(x_global)
                w_global_scales.append(w_global)
            else:
                xq, x_scale, _ = converted_x
                wq, w_scale, _ = converted_w
            xq_list.append(xq)
            x_scale_list.append(to_blocked(x_scale))
            wq_list.append(wq)
            w_scale_list.append(to_blocked(w_scale))

        xq = torch.cat(xq_list, dim=1).contiguous()
        wq = torch.cat(wq_list, dim=1).contiguous()
        x_blocked_scales = torch.cat(x_scale_list, dim=0).contiguous()
        w_blocked_scales = torch.cat(w_scale_list, dim=0).contiguous()
        x_blocked_scales = x_blocked_scales.reshape(m_rounded, -1).contiguous()
        w_blocked_scales = w_blocked_scales.reshape(n_rounded, -1).contiguous()
        x_global_scales = (
            torch.stack(x_global_scales).to(device=device, dtype=torch.float32)
            if x_global_scales
            else None
        )
        w_global_scales = (
            torch.stack(w_global_scales).to(device=device, dtype=torch.float32)
            if w_global_scales
            else None
        )

    return (
        xq,
        wq,
        x_blocked_scales,
        w_blocked_scales,
        x_global_scales,
        w_global_scales,
        offs,
        k_eff,
    )


def benchmark_scaled_grouped_mm(
    gmnk=None,
    dtype=None,
    seed=0,
    rtol=8e-2,
    atol=8e-2,
    grouping="balanced",
    use_cuda_graphs=False,
    warmup=2,
    rep=20,
    backend="both",
    emit_us_only=False,
    mma_tile_mn: tuple[int, int] | None = None,
    cluster_shape_mn: tuple[int, int] | None = None,
    transpose_ab: bool | None = None,
    layout_mode: str = "2d/3d",
    format: str = "mxfp8",
    stat: str = "median",
):
    if dtype is None:
        dtype = torch.bfloat16
    if backend not in ("both", "cpp", "cute"):
        raise ValueError(f"backend must be one of both/cpp/cute, got {backend}")
    if stat not in _STAT_CHOICES:
        raise ValueError(f"stat must be one of {_STAT_CHOICES}, got {stat}")
    if (mma_tile_mn is None) != (cluster_shape_mn is None):
        raise ValueError(
            "mma_tile_mn and cluster_shape_mn must be provided "
            "together or omitted together"
        )

    if gmnk is None:
        gmnk = [
            [2, 5, 16, 16],
            [3, 13, 16, 32],
            [8, 128, 16, 16],
            [8, 512, 32, 64],
            [16, 1024, 256, 1024],
            [32, 2048, 512, 256],
            [32, 2048, 512, 2048],
            [24, 4834, 5120, 1536],
            [32, 8257, 5120, 1536],
            [24, 32768, 6144, 2048],
            [48, 32768, 6144, 2048],
            [64, 32768, 6144, 2048],
            [24, 65536, 6144, 2048],
            [32, 65536, 6144, 2048],
            [48, 65536, 6144, 2048],
            [64, 65536, 6144, 2048],
            [24, 131072, 6144, 2048],
            [32, 131072, 6144, 2048],
            [48, 131072, 6144, 2048],
            [64, 131072, 6144, 2048],
        ]
        if layout_mode == "2d/2d":
            # Backward-style mapping from 2d/3d forward defaults:
            # (G, M, N, K) -> (G, K, N, M)
            gmnk = [[g, k, n, m] for g, m, n, k in gmnk]
    swizzle = SwizzleType.NO_SWIZZLE
    if torch.version.cuda:
        swizzle = SwizzleType.SWIZZLE_32_4_4
    if format == "nvfp4":
        scale_recipe = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
    else:
        scale_recipe = [ScalingType.BlockWise1x32]

    scale_recipe_int = [recipe.value for recipe in scale_recipe]
    swizzle_int = [swizzle.value]
    run_cpp = backend in ("both", "cpp")
    run_cute = backend in ("both", "cute")
    if run_cute:
        _require_cutedsl_scaled_grouped_mm_override()
    do_correctness = run_cpp and run_cute
    all_valid = True

    results = []

    def _make_cute_call(
        fn: Callable[[], torch.Tensor],
    ) -> Callable[[], torch.Tensor]:
        if mma_tile_mn is None and cluster_shape_mn is None and transpose_ab is None:
            return fn

        sgmm_mod = importlib.import_module(
            "torch._native.ops.scaled_grouped_mm.scaled_grouped_mm_blockscaled"
        )

        def _call_with_override():
            select_name = (
                "_select_kernel_config_fp4"
                if format in ("mxfp4", "nvfp4")
                else "_select_kernel_config_fp8"
            )
            old_select = getattr(sgmm_mod, select_name)

            def _select_override(M, N, K):
                base = old_select(M, N, K)
                return sgmm_mod._KernelConfig(
                    mma_tile_mn if mma_tile_mn is not None else base.mma_tile_mn,
                    (
                        cluster_shape_mn
                        if cluster_shape_mn is not None
                        else base.cluster_shape_mn
                    ),
                    transpose_ab if transpose_ab is not None else base.transpose_ab,
                )

            setattr(sgmm_mod, select_name, _select_override)
            try:
                return fn()
            finally:
                setattr(sgmm_mod, select_name, old_select)

        return _call_with_override

    for g, m, n, k in gmnk:
        if not is_blackwell():
            raise RuntimeError("Benchmark requires Blackwell (sm_100).")

        block_size = 16 if format == "nvfp4" else 32
        k_quant_multiple = 32 if format == "nvfp4" else block_size
        k_eff = ((k + k_quant_multiple - 1) // k_quant_multiple) * k_quant_multiple

        if not emit_us_only:
            print(f"G={g}, M={m}, N={n}, K={k} (eff K={k_eff})")
        prepared_inputs = None
        fn_cpp = None
        fn_cute = None

        def _ensure_bench_fns():
            nonlocal prepared_inputs, fn_cpp, fn_cute
            if fn_cpp is not None and fn_cute is not None:
                return fn_cpp, fn_cute

            prepared_inputs = _prepare_inputs(
                g, m, n, k, dtype, seed, grouping, layout_mode, format
            )
            (
                xq,
                wq,
                x_blocked_scales,
                w_blocked_scales,
                x_global_scales,
                w_global_scales,
                offs,
                _prepared_k_eff,
            ) = prepared_inputs

            fn_cpp = lambda: _call_cpp_scaled_grouped_mm_v2(  # noqa: E731
                xq,
                wq.transpose(-2, -1),
                (
                    [x_blocked_scales, x_global_scales]
                    if format == "nvfp4"
                    else [x_blocked_scales]
                ),
                scale_recipe_int,
                swizzle_int,
                (
                    [w_blocked_scales, w_global_scales]
                    if format == "nvfp4"
                    else [w_blocked_scales]
                ),
                scale_recipe_int,
                swizzle_int,
                offs=offs,
                bias=None,
                out_dtype=torch.bfloat16,
                contraction_dim=[],
                use_fast_accum=False,
            )

            fn_cute_base = lambda: scaled_grouped_mm(  # noqa: E731
                xq,
                wq.transpose(-2, -1),
                (
                    [x_blocked_scales, x_global_scales]
                    if format == "nvfp4"
                    else [x_blocked_scales]
                ),
                scale_recipe,
                (
                    [w_blocked_scales, w_global_scales]
                    if format == "nvfp4"
                    else [w_blocked_scales]
                ),
                scale_recipe,
                swizzle_a=[swizzle],
                swizzle_b=[swizzle],
                offs=offs,
                output_dtype=torch.bfloat16,
            )
            fn_cute = _make_cute_call(fn_cute_base)
            return fn_cpp, fn_cute

        us_cpp = None
        us_cute = None

        fn_cpp, fn_cute = _ensure_bench_fns()
        bench_fn_cpp = (
            _maybe_wrap_cuda_graph(fn_cpp, "cpp", use_cuda_graphs) if run_cpp else None
        )
        bench_fn_cute = (
            _maybe_wrap_cuda_graph(fn_cute, "cute", use_cuda_graphs)
            if run_cute
            else None
        )
        if run_cpp:
            res_cpp = _do_bench_cuda(bench_fn_cpp, warmup=warmup, rep=rep, stat=stat)
            us_cpp = res_cpp.value
        if run_cute:
            if run_cpp:
                # In --backend both, CuTeDSL inherits state left by
                # C++ that inflates CuTeDSL's median/max even though
                # C++ itself is unaffected. Drop two sources:
                # 1. the graph-wrapped C++ bench fn keeps a CUDA graph
                #    (and its private memory pool from
                #    torch.cuda.graph()) alive, doubling GPU memory
                #    pressure during CuTeDSL replays
                # 2. the caching allocator pool shape left by C++
                #    perturbs CuTeDSL's per-iteration output allocs.
                bench_fn_cpp = None
                gc.collect()
                torch.cuda.empty_cache()
            res_cute = _do_bench_cuda(bench_fn_cute, warmup=warmup, rep=rep, stat=stat)
            us_cute = res_cute.value

        speedup = us_cpp / us_cute if us_cpp and us_cute else None

        if emit_us_only:
            print(f"{us_cpp if run_cpp else us_cute}")
            us = us_cpp if run_cpp else us_cute
            return [{"G": g, "M": m, "N": n, "K": k, "us": us}]

        if us_cpp is not None:
            print(
                f"  C++: {us_cpp:.2f} us  ({stat}; "
                f"min={res_cpp.min_us:.2f}, max={res_cpp.max_us:.2f})"
            )
        if us_cute is not None:
            print(
                f"  CuTeDSL: {us_cute:.2f} us  ({stat}; "
                f"min={res_cute.min_us:.2f}, max={res_cute.max_us:.2f})"
            )

        if do_correctness and run_cpp and run_cute:
            fn_cpp, fn_cute = _ensure_bench_fns()
            try:
                out_cpp = fn_cpp()
                out_cute = fn_cute()
                torch.testing.assert_close(out_cpp, out_cute, rtol=rtol, atol=atol)
                print("  ✓ results match")
            except AssertionError:
                print("  ✗ results mismatch")
                all_valid = False

        del prepared_inputs, fn_cpp, fn_cute
        gc.collect()
        torch.cuda.empty_cache()

        results.append(
            {
                "G": g,
                "M": m,
                "N": n,
                "K": k,
                "C++ (us)": us_cpp,
                "CuTeDSL (us)": us_cute,
                "speedup": speedup,
            }
        )
        print()

    import pandas as pd

    df = pd.DataFrame(results)
    for col in ("G", "M", "N", "K"):
        df[col] = df[col].astype("int64")
    for col in df.columns:
        if col not in ("G", "M", "N", "K"):
            fmt_value = lambda x: f"{x:.3f}" if x is not None and pd.notna(x) else "nan"  # noqa: E731
            df[col] = df[col].map(fmt_value)
    print(
        df.to_markdown(
            index=False,
            disable_numparse=True,
            colalign=("right",) * len(df.columns),
        )
    )
    if not all_valid:
        raise RuntimeError("validation failed for one or more shapes")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Benchmark scaled grouped MM (blockscaled) C++ vs CuTeDSL.")
    )
    parser.add_argument(
        "--format",
        type=_parse_format,
        default="mxfp8",
        help="Input/scaling format: mxfp8, mxfp4, or nvfp4.",
    )
    parser.add_argument(
        "--input-dtype",
        dest="input_dtype",
        type=_parse_input_dtype,
        default="bf16",
        help="Input dtype: bf16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for input and offset generation.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=8e-2,
        help="Relative tolerance for correctness checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=8e-2,
        help="Absolute tolerance for correctness checks.",
    )
    parser.add_argument(
        "--gmnk",
        nargs="+",
        type=_parse_gmnk,
        help="Problem sizes as G,M,N,K (space-separated).",
    )
    parser.add_argument(
        "--mma-tile-mn",
        type=_parse_mma_tile_mn,
        default=None,
        help="Override CuTeDSL output MMA tile as M,N (e.g. 256,256).",
    )
    parser.add_argument(
        "--cluster-shape-mn",
        type=_parse_cluster_shape_mn,
        default=None,
        help="Override CuTeDSL cluster shape as M,N (e.g. 2,1).",
    )
    parser.add_argument(
        "--transpose-ab",
        choices=["on", "off"],
        default=None,
        help="Override CuTeDSL transpose_ab tuning knob.",
    )
    parser.add_argument(
        "--grouping",
        choices=["balanced", "random"],
        default="balanced",
        help=("How to split the grouped dimension: M for 2d/3d, K for 2d/2d."),
    )
    parser.add_argument(
        "--layout-mode",
        choices=["2d/3d", "2d/2d"],
        default="2d/3d",
        help="Input layout mode for grouped MM benchmark.",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        default=False,
        help=("Capture backend call in a CUDA graph and benchmark replay()."),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations for timing loops.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Measured iterations for timing loops.",
    )
    parser.add_argument(
        "--backend",
        choices=["both", "cpp", "cute"],
        default="both",
        help="Backend to benchmark: both, cpp, or cute.",
    )
    parser.add_argument(
        "--emit-us-only",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stat",
        choices=list(_STAT_CHOICES),
        default="median",
        help=(
            "Summary statistic to report per backend (default: median). "
            "Min/max are always shown alongside."
        ),
    )
    args = parser.parse_args()
    if (args.mma_tile_mn is None) != (args.cluster_shape_mn is None):
        parser.error("--mma-tile-mn and --cluster-shape-mn must be provided together")
    dtype = torch.bfloat16 if args.input_dtype == "bf16" else torch.float16
    gmnk = args.gmnk if args.gmnk is not None else None
    benchmark_scaled_grouped_mm(
        gmnk=gmnk,
        dtype=dtype,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        grouping=args.grouping,
        use_cuda_graphs=args.use_cuda_graphs,
        warmup=args.warmup,
        rep=args.rep,
        backend=args.backend,
        emit_us_only=args.emit_us_only,
        mma_tile_mn=args.mma_tile_mn,
        cluster_shape_mn=args.cluster_shape_mn,
        transpose_ab=(None if args.transpose_ab is None else args.transpose_ab == "on"),
        layout_mode=args.layout_mode,
        format=args.format,
        stat=args.stat,
    )
