"""Benchmark SM90 DeepSeek FP8 grouped mm."""

import argparse
import gc
import warnings
from dataclasses import dataclass

import torch
import torch._native.ops.scaled_grouped_mm
from torch.nn.functional import scaled_mm, ScalingType, SwizzleType


_STAT_CHOICES = ("median", "mean", "min", "max", "p10", "p90")
_MIN_REF_GROUP_M = 128


@dataclass(frozen=True)
class BenchResult:
    value: float
    min_us: float
    max_us: float


def _require_cutedsl_scaled_grouped_mm_override():
    from torch._native import cutedsl_utils as cu, registry
    from torch._native.common_utils import check_native_jit_disabled

    registry._register_all_overrides()
    if not cu.runtime_available():
        raise RuntimeError(
            "CuTeDSL runtime dependencies are not available. Install "
            "nvidia-cutlass-dsl and apache-tvm-ffi."
        )
    if check_native_jit_disabled():
        raise RuntimeError(
            "native DSL overrides are disabled by TORCH_DISABLE_NATIVE_JIT=1."
        )
    if "_scaled_grouped_mm_v2" not in registry.get_dsl_operations("cutedsl"):
        version = cu.runtime_version()
        raise RuntimeError(
            "CuTeDSL override for aten::_scaled_grouped_mm_v2 is not "
            "registered. Current nvidia-cutlass-dsl version: "
            f"{version}. If this is a newer local test version, rerun "
            "with TORCH_NATIVE_SKIP_VERSION_CHECK=1."
        )


def _pad_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


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


def _quantize_block(x: torch.Tensor, block_outer: int, block_inner: int = 128):
    xb = x.unflatten(1, (-1, block_inner)).unflatten(0, (-1, block_outer))
    amax = xb.abs().amax(dim=[1, 3], keepdim=True).float()
    quant_scale = torch.finfo(torch.float8_e4m3fn).max / amax.clamp_min(1e-12)
    fp8 = (xb * quant_scale).to(torch.float8_e4m3fn)
    fp8 = fp8.flatten(2, 3).flatten(0, 1)
    dequant_scale = quant_scale.reciprocal().flatten(2, 3).flatten(0, 1)
    return fp8, dequant_scale


def _to_ref_scale_1x128(scale_natural: torch.Tensor) -> torch.Tensor:
    return scale_natural.t().contiguous().t()


def _to_op_scale_128x128(scale_natural: torch.Tensor, k: int) -> torch.Tensor:
    # A (1, kb) slice counts as contiguous whatever its strides, so
    # pad/contiguous can no-op and leave a view.
    kb = k // 128
    padded = torch.zeros(
        scale_natural.shape[0],
        _pad_up(kb, 4),
        device=scale_natural.device,
        dtype=scale_natural.dtype,
    )
    padded[:, :kb] = scale_natural[:, :kb]
    return padded.t()


def _make_grouped_scale_b(scales: list[torch.Tensor], K: int, N: int, rhs_block: int):
    G = len(scales)
    if rhs_block == 1:
        K_blocks = K // 128
        grouped = torch.empty_strided(
            (G, N, K_blocks),
            (N * K_blocks, 1, N),
            device=scales[0].device,
            dtype=scales[0].dtype,
        )
    else:
        L4 = _pad_up(K // 128, 4)
        N_blocks = N // 128
        grouped = torch.empty_strided(
            (G, L4, N_blocks),
            (L4 * N_blocks, 1, L4),
            device=scales[0].device,
            dtype=scales[0].dtype,
        )
    for i, scale in enumerate(scales):
        grouped[i].copy_(scale)
    return grouped


def _build_inputs(
    total_m: int,
    g: int,
    K: int,
    N: int,
    lhs_block: int,
    rhs_block: int,
    device: str,
    grouping: str = "balanced",
):
    align = 128 if lhs_block == 128 else 1
    offs = _generate_offsets(total_m, g, device, mode=grouping, align=align)
    a = torch.randn(total_m, K, device=device, dtype=torch.bfloat16)
    a_fp8, a_scale = _quantize_block(a, block_outer=lhs_block)
    if lhs_block == 1:
        a_scale = _to_ref_scale_1x128(a_scale)
    else:
        a_scale = _to_op_scale_128x128(a_scale, K)

    b_fp8_groups, b_scale_op_groups = [], []
    for _ in range(g):
        b_i = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        b_fp8_i, b_scale_i = _quantize_block(b_i, block_outer=rhs_block)
        if rhs_block == 1:
            b_scale_i = _to_ref_scale_1x128(b_scale_i)
        else:
            b_scale_i = _to_op_scale_128x128(b_scale_i, K)
        b_fp8_groups.append(b_fp8_i)
        b_scale_op_groups.append(b_scale_i)

    mat2 = torch.stack(b_fp8_groups, dim=0).transpose(-2, -1)
    scale_b = _make_grouped_scale_b(b_scale_op_groups, K, N, rhs_block)
    return a_fp8, a_scale, mat2, scale_b, b_fp8_groups, offs


def _build_inputs_2d2d(M, g, K, N, lhs_block, rhs_block, device, grouping="balanced"):
    offs = _generate_offsets(K, g, device, mode=grouping, align=128)
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    a_fp8, a_scale_nat = _quantize_block(a, block_outer=lhs_block)
    b = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    b_fp8, b_scale_nat = _quantize_block(b, block_outer=rhs_block)

    def op_layout(nat, block, k):
        return _to_ref_scale_1x128(nat) if block == 1 else _to_op_scale_128x128(nat, k)

    a_scale = op_layout(a_scale_nat, lhs_block, K)
    scale_b = op_layout(b_scale_nat, rhs_block, K)
    return a_fp8, a_scale, b_fp8.t(), scale_b, (a_scale_nat, b_scale_nat, b_fp8), offs


def _make_reference_2d2d(
    a_fp8, nats, lhs_recipe, lhs_block, rhs_recipe, rhs_block, M, N, offs
):
    a_scale_nat, b_scale_nat, b_fp8 = nats
    ends = offs.tolist()

    def op_layout(nat, block, k):
        return _to_ref_scale_1x128(nat) if block == 1 else _to_op_scale_128x128(nat, k)

    slices = []
    start = 0
    for end in ends:
        kb0, kb1 = start // 128, end // 128
        slices.append(
            (
                start,
                end,
                op_layout(a_scale_nat[:, kb0:kb1], lhs_block, end - start),
                op_layout(b_scale_nat[:, kb0:kb1], rhs_block, end - start),
            )
        )
        start = end

    def reference():
        out = torch.zeros(len(ends), M, N, device=a_fp8.device, dtype=torch.bfloat16)
        for i, (ks, ke, a_s, b_s) in enumerate(slices):
            if ke == ks:
                continue
            out[i] = scaled_mm(
                a_fp8[:, ks:ke],
                b_fp8[:, ks:ke].t(),
                a_s,
                lhs_recipe,
                b_s,
                rhs_recipe,
                output_dtype=torch.bfloat16,
            )
        return out

    return reference


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = percentile / 100.0 * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    weight = rank - lo
    return sorted_samples[lo] * (1.0 - weight) + sorted_samples[hi] * weight


def _summarize(samples_us: list[float], stat: str) -> BenchResult:
    sorted_samples = sorted(samples_us)
    if stat == "median":
        value = _percentile(sorted_samples, 50.0)
    elif stat == "mean":
        value = sum(sorted_samples) / len(sorted_samples)
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


def _do_bench_cuda(fn, warmup: int, rep: int, stat: str) -> BenchResult:
    if rep < 1:
        raise ValueError(f"rep must be >= 1, got {rep}")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples_us = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return _summarize(samples_us, stat)


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

        def replay():
            graph.replay()

        return replay
    except Exception as exc:
        warnings.warn(
            (
                f"CUDA graph capture failed for backend '{label}', "
                f"falling back to eager: {exc}"
            ),
            stacklevel=2,
        )
        return fn


def _make_grouped_op(a_fp8, mat2, a_scale, scale_b, lhs_recipe, rhs_recipe, offs):
    from torch._native.ops.scaled_grouped_mm.cutedsl_impl import _cond
    from torch._native.ops.scaled_grouped_mm.group_meta import expected_out_size_stride

    lhs_recipe_int = lhs_recipe.value if hasattr(lhs_recipe, "value") else lhs_recipe
    rhs_recipe_int = rhs_recipe.value if hasattr(rhs_recipe, "value") else rhs_recipe
    swizzle_int = SwizzleType.NO_SWIZZLE.value
    out_size, out_stride = expected_out_size_stride(
        a_fp8, mat2, torch.bfloat16, offs.numel() if mat2.dim() == 2 else None
    )
    out = torch.empty_strided(
        out_size, out_stride, device=a_fp8.device, dtype=torch.bfloat16
    )
    if not _cond(
        a_fp8,
        mat2,
        [a_scale],
        [lhs_recipe_int],
        [swizzle_int],
        [scale_b],
        [rhs_recipe_int],
        [swizzle_int],
        offs=offs,
        out_dtype=torch.bfloat16,
    ):
        raise RuntimeError(
            "CuTeDSL scaled_grouped_mm override rejected benchmark inputs: "
            f"a={tuple(a_fp8.shape)} stride={a_fp8.stride()}, "
            f"b={tuple(mat2.shape)} stride={mat2.stride()}, "
            f"scale_a={tuple(a_scale.shape)} stride={a_scale.stride()}, "
            f"scale_b={tuple(scale_b.shape)} stride={scale_b.stride()}."
        )

    def grouped_op():
        return torch.ops.aten._scaled_grouped_mm_v2(
            a_fp8,
            mat2,
            [a_scale],
            [lhs_recipe_int],
            [swizzle_int],
            [scale_b],
            [rhs_recipe_int],
            [swizzle_int],
            offs=offs,
            out_dtype=torch.bfloat16,
            out=out,
        )

    return grouped_op


def _make_reference(
    a_fp8,
    a_scale,
    scale_b,
    b_fp8_groups,
    lhs_recipe,
    lhs_block,
    rhs_recipe,
    rhs_block,
    N,
    offs,
):
    group_sizes_end = offs.tolist()
    group_count = len(group_sizes_end)

    if group_count == 1:
        a_scale_ref = a_scale
        b_scale_ref = scale_b[0]

        def reference():
            return scaled_mm(
                a_fp8,
                b_fp8_groups[0].t(),
                a_scale_ref,
                lhs_recipe,
                b_scale_ref,
                rhs_recipe,
                output_dtype=torch.bfloat16,
            )

        return reference

    total_m = a_fp8.shape[0]

    def reference():
        out = torch.empty(a_fp8.shape[0], N, device=a_fp8.device, dtype=torch.bfloat16)
        start = 0
        for i, end in enumerate(group_sizes_end):
            if end == start:
                continue
            call_start, call_end = start, end
            group_m = end - start
            padded_m = _pad_up(max(group_m, _MIN_REF_GROUP_M), 4)
            if lhs_block == 1 and padded_m != group_m:
                # cuBLASLt's FP8 heuristic search can fail outright
                # (CUBLAS_STATUS_NOT_SUPPORTED) both for very small M and for
                # M that isn't a multiple of 4, either of which random
                # grouping routinely produces. Widen the call with adjacent
                # rows and keep only the real group's slice of the result.
                call_end = min(total_m, start + padded_m)
                call_start = max(0, call_end - padded_m)
            if lhs_block == 1:
                a_scale_ref_i = a_scale[call_start:call_end, :].t().contiguous().t()
            else:
                a_scale_ref_i = a_scale[:, call_start // 128 : call_end // 128]
            b_scale_ref_i = scale_b[i]
            padded = scaled_mm(
                a_fp8[call_start:call_end, :],
                b_fp8_groups[i].t(),
                a_scale_ref_i,
                lhs_recipe,
                b_scale_ref_i,
                rhs_recipe,
                output_dtype=torch.bfloat16,
            )
            out[start:end, :] = padded[start - call_start : end - call_start, :]
            start = end
        return out

    return reference


def _make_case_fns(
    total_m, g, K, N, lhs_block, rhs_block, grouping="balanced", layout="2d_3d"
):
    device = "cuda"
    lhs_recipe = (
        ScalingType.BlockWise1x128 if lhs_block == 1 else ScalingType.BlockWise128x128
    )
    rhs_recipe = (
        ScalingType.BlockWise1x128 if rhs_block == 1 else ScalingType.BlockWise128x128
    )
    if layout == "2d_2d":
        a_fp8, a_scale, mat2, scale_b, nats, offs = _build_inputs_2d2d(
            total_m, g, K, N, lhs_block, rhs_block, device, grouping=grouping
        )
        return (
            _make_grouped_op(
                a_fp8, mat2, a_scale, scale_b, lhs_recipe, rhs_recipe, offs
            ),
            _make_reference_2d2d(
                a_fp8,
                nats,
                lhs_recipe,
                lhs_block,
                rhs_recipe,
                rhs_block,
                total_m,
                N,
                offs,
            ),
        )
    a_fp8, a_scale, mat2, scale_b, b_fp8_groups, offs = _build_inputs(
        total_m, g, K, N, lhs_block, rhs_block, device, grouping=grouping
    )

    grouped_op = _make_grouped_op(
        a_fp8, mat2, a_scale, scale_b, lhs_recipe, rhs_recipe, offs
    )
    reference = _make_reference(
        a_fp8,
        a_scale,
        scale_b,
        b_fp8_groups,
        lhs_recipe,
        lhs_block,
        rhs_recipe,
        rhs_block,
        N,
        offs,
    )
    return grouped_op, reference


def benchmark_deepseek_scaled_grouped_mm(
    gmnk=None,
    seed=0,
    lhs_block=1,
    rhs_block=128,
    grouping="balanced",
    warmup=10,
    rep=100,
    rtol=7e-2,
    atol=6e-1,
    backend="both",
    emit_us_only=False,
    use_cuda_graphs=False,
    stat="median",
    layout="2d_3d",
):
    if backend not in ("both", "reference", "cute"):
        raise ValueError(f"backend must be one of both/reference/cute, got {backend}")
    if stat not in _STAT_CHOICES:
        raise ValueError(f"stat must be one of {_STAT_CHOICES}, got {stat}")
    if grouping not in ("balanced", "random"):
        raise ValueError(f"grouping must be 'balanced' or 'random', got {grouping}")
    if gmnk is None:
        gmnk = _DEFAULT_GMNK

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise SystemExit("This benchmark requires a Hopper (SM90) GPU.")
    if backend in ("both", "cute"):
        _require_cutedsl_scaled_grouped_mm_override()

    torch.manual_seed(seed)
    run_ref = backend in ("both", "reference")
    run_cute = backend in ("both", "cute")
    do_correctness = run_ref and run_cute
    all_valid = True
    results = []

    for g, m, n, k in gmnk:
        if lhs_block == 128 and rhs_block == 128:
            raise ValueError("DeepSeek scaled_mm does not support 128x128 x 128x128")
        label = (
            "G=1 (direct torch._scaled_mm)"
            if g == 1
            else f"G={g} (per-group torch._scaled_mm loop)"
        )
        if not emit_us_only:
            print(
                f"G={g} M={m} (total) K={k} N={n} "
                f"lhs_block={lhs_block} rhs_block={rhs_block} grouping={grouping} "
                f"[reference: {label}]",
                flush=True,
            )

        fn_cute, fn_ref = _make_case_fns(
            m, g, k, n, lhs_block, rhs_block, grouping=grouping, layout=layout
        )
        bench_ref = (
            _maybe_wrap_cuda_graph(fn_ref, "reference", use_cuda_graphs)
            if run_ref
            else None
        )
        bench_cute = (
            _maybe_wrap_cuda_graph(fn_cute, "cute", use_cuda_graphs)
            if run_cute
            else None
        )

        ref_res = None
        cute_res = None
        if run_ref:
            ref_res = _do_bench_cuda(bench_ref, warmup=warmup, rep=rep, stat=stat)
        if run_cute:
            if run_ref:
                bench_ref = None
                gc.collect()
                torch.cuda.empty_cache()
            cute_res = _do_bench_cuda(bench_cute, warmup=warmup, rep=rep, stat=stat)

        if do_correctness:
            try:
                expected = fn_ref()
                actual = fn_cute()
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                if not emit_us_only:
                    print("  ✓ results match", flush=True)
            except AssertionError:
                if not emit_us_only:
                    print("  ✗ results mismatch", flush=True)
                all_valid = False

        if emit_us_only:
            us = cute_res.value if run_cute else ref_res.value
            print(f"{us}")
            results.append({"G": g, "M": m, "N": n, "K": k, "us": us})
            del fn_cute, fn_ref, bench_ref, bench_cute
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if ref_res is not None:
            print(
                f"  reference: {ref_res.value:.2f} us  ({stat}; "
                f"min={ref_res.min_us:.2f}, max={ref_res.max_us:.2f})"
            )
        if cute_res is not None:
            print(
                f"  CuTeDSL:   {cute_res.value:.2f} us  ({stat}; "
                f"min={cute_res.min_us:.2f}, max={cute_res.max_us:.2f})"
            )
        speedup = (
            ref_res.value / cute_res.value
            if ref_res is not None and cute_res is not None
            else None
        )
        if speedup is not None:
            print(f"  speedup: {speedup:.2f}x")

        results.append(
            {
                "G": g,
                "M": m,
                "N": n,
                "K": k,
                "reference (us)": ref_res.value if ref_res is not None else None,
                "CuTeDSL (us)": cute_res.value if cute_res is not None else None,
                "speedup": speedup,
            }
        )
        print()
        del fn_cute, fn_ref, bench_ref, bench_cute
        gc.collect()
        torch.cuda.empty_cache()

    if not emit_us_only:
        import pandas as pd

        df = pd.DataFrame(results)
        for col in ("G", "M", "N", "K"):
            df[col] = df[col].astype("int64")
        for col in df.columns:
            if col not in ("G", "M", "N", "K"):
                df[col] = df[col].map(
                    lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "nan"
                )
        print(
            df.to_markdown(
                index=False,
                tablefmt="github",
                disable_numparse=True,
                colalign=("right",) * len(df.columns),
            )
        )
    if not all_valid:
        raise RuntimeError("validation failed for one or more shapes")
    return results


def _parse_gmnk(value: str) -> list[int]:
    parts = [int(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Invalid gmnk '{value}'. Expected G,M,N,K.")
    if any(v < 1 for v in parts):
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected G,M,N,K >= 1."
        )
    return parts


_DEFAULT_GMNK = [
    [1, 2048, 2048, 7168],
    [1, 8192, 2048, 7168],
    [1, 32768, 2048, 7168],
    [1, 131072, 2048, 7168],
    [256, 2048, 2048, 7168],
    [256, 8192, 2048, 7168],
    [256, 32768, 2048, 7168],
    [256, 131072, 2048, 7168],
    [1, 2048, 7168, 2048],
    [1, 8192, 7168, 2048],
    [1, 32768, 7168, 2048],
    [1, 131072, 7168, 2048],
    [256, 2048, 7168, 2048],
    [256, 8192, 7168, 2048],
    [256, 32768, 7168, 2048],
    [256, 131072, 7168, 2048],
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gmnk",
        nargs="+",
        type=_parse_gmnk,
        default=_DEFAULT_GMNK,
        help="Problem sizes as G,M,N,K (space-separated, M = total rows across the group).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lhs-block", type=int, choices=(1, 128), default=1)
    parser.add_argument("--rhs-block", type=int, choices=(1, 128), default=128)
    parser.add_argument(
        "--grouping",
        choices=["balanced", "random"],
        default="balanced",
        help="How to split the total M across G groups.",
    )
    parser.add_argument("--rtol", type=float, default=7e-2)
    parser.add_argument("--atol", type=float, default=6e-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--backend",
        choices=["both", "reference", "cute"],
        default="both",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--emit-us-only",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--stat", choices=_STAT_CHOICES, default="median")
    parser.add_argument(
        "--layout",
        choices=["2d_3d", "2d_2d"],
        default="2d_3d",
        help="2d_3d: offs splits M, B is (G,K,N). 2d_2d: offs splits K, B is (K,N).",
    )
    args = parser.parse_args()

    benchmark_deepseek_scaled_grouped_mm(
        gmnk=args.gmnk,
        seed=args.seed,
        lhs_block=args.lhs_block,
        rhs_block=args.rhs_block,
        grouping=args.grouping,
        warmup=args.warmup,
        rep=args.rep,
        rtol=args.rtol,
        atol=args.atol,
        backend=args.backend,
        emit_us_only=args.emit_us_only,
        use_cuda_graphs=args.use_cuda_graphs,
        stat=args.stat,
        layout=args.layout,
    )
