# FIXME: move this to tritonbench project.

import argparse
import dataclasses
import gc
import json
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

from triton import runtime

import torch
from torch._inductor.utils import run_and_get_code


def is_blackwell():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[
        0
    ] == 10 and torch.cuda.get_device_capability()[1] in [0, 3]


def _major_label(is_k_major, other_major):
    return "k-major" if is_k_major else f"{other_major}-major"


def _normalize_major(value):
    return value.replace("-", "").replace("_", "")


def _parse_tensor_spec(value, allowed_majors, expected_str, example):
    value = value.lower()
    try:
        dim_str, layout_str = value.split(":")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_str} (e.g., {example})."
        ) from exc
    if dim_str not in {"2d", "3d"}:
        raise argparse.ArgumentTypeError(f"Expected {expected_str} (e.g., {example}).")
    major_norm = _normalize_major(layout_str)
    if major_norm not in allowed_majors:
        raise argparse.ArgumentTypeError(f"Expected {expected_str} (e.g., {example}).")
    return (int(dim_str[0]), major_norm == "kmajor")


def _parse_a_spec(value):
    return _parse_tensor_spec(
        value,
        {"kmajor", "mmajor"},
        "'<2d|3d>:<k-major|m-major>'",
        "2d:k-major",
    )


def _parse_b_spec(value):
    return _parse_tensor_spec(
        value,
        {"kmajor", "nmajor"},
        "'<2d|3d>:<k-major|n-major>'",
        "3d:k-major",
    )


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
    if any(value <= 1 for value in values):
        raise argparse.ArgumentTypeError(
            f"Invalid gmnk '{value}'. Expected G,M,N,K > 1."
        )
    return values


def _generate_offsets(total, groups, device, mode="random", align=1):
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
                    f"grouping='balanced' with M={total}, G={groups}: "
                    f"using base size {base} and placing tail "
                    f"{remainder} in the last group",
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
                    f"grouping='balanced' with M={total}, G={groups}, "
                    f"align={align}: using aligned base size "
                    f"{base_units * align} and placing "
                    f"{extra_units * align + remainder} values in the "
                    "last groups",
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


def _do_bench_cuda(fn, warmup=10, rep=100, cooldown_seconds=1.0):
    """Benchmark `fn` with a fixed number of iterations, an L2 cache
    clear before each measured call, and a cooldown before the first
    measured call to avoid thermal-throttling bias.

    triton.testing.do_bench's warmup/rep are milliseconds, not iteration
    counts: for slow (large-shape) calls this collapses to very few
    measured samples (e.g. ~1 warmup, ~9 reps at ~2ms/call with the
    warmup=2, rep=20 previously used here), which is too noisy for
    tracking single-digit-percent speedups. Fixed iteration counts give
    every shape the same statistical power regardless of how long it
    takes to run.
    """
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    fn()
    di.synchronize()
    time.sleep(cooldown_seconds)

    for _ in range(warmup):
        fn()
    di.synchronize()

    times_ms = []
    for _ in range(rep):
        runtime.driver.active.clear_cache(cache)
        start = di.Event(enable_timing=True)
        end = di.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        di.synchronize()
        times_ms.append(start.elapsed_time(end))

    times_ms.sort()
    mid = len(times_ms) // 2
    median_ms = (
        times_ms[mid] if len(times_ms) % 2 else (times_ms[mid - 1] + times_ms[mid]) / 2
    )
    return {
        "median_us": median_ms * 1e3,
        "mean_us": sum(times_ms) / len(times_ms) * 1e3,
        "min_us": times_ms[0] * 1e3,
        "max_us": times_ms[-1] * 1e3,
    }


def _maybe_wrap_cuda_graph(fn, label, use_cuda_graphs):
    """Capture `fn` into a CUDA graph and return a closure that just
    replays it, isolating GPU execution time from Python/dispatcher
    overhead (guard checks, view ops like .transpose(), etc.) that
    would otherwise be included in every measured call.
    """
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
            f"CUDA graph capture failed for backend '{label}', "
            f"falling back to eager: {exc}",
            stacklevel=2,
        )
        return fn


BACKEND_CHOICES = ["aten", "triton", "cutedsl", "gluon"]


def _autotune_winning_config(A, B, offs):
    """Autotune the Gluon backend and return the config that won.

    Timed without instrumentation, so the winner is the one that is
    actually fastest.
    """
    from torch._inductor.heuristics.template import gluon as gluon_heuristics

    seen = []
    orig_get_configs = gluon_heuristics.get_grouped_mm_configs

    def recording_get_configs(**kwargs):
        configs = orig_get_configs(**kwargs)
        seen.extend(configs)
        return configs

    gluon_heuristics.get_grouped_mm_configs = recording_get_configs
    try:
        torch._dynamo.reset()
        compiled = torch.compile(
            torch._grouped_mm,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "GLUON",
                "fx_graph_cache": False,
            },
            dynamic=False,
        )
        _, code = run_and_get_code(compiled, A, B.transpose(-2, -1), offs)
    finally:
        gluon_heuristics.get_grouped_mm_configs = orig_get_configs

    src = "\n".join(code)
    matches = [c for c in seen if _config_matches_source(c, src)]
    if len(matches) != 1:
        found = re.search(r"'config_args': \{[^}]*\}", src)
        raise RuntimeError(
            f"could not identify the winning Gluon config: {len(matches)} of "
            f"{len(seen)} candidates match the generated code; it says "
            f"{found.group(0) if found else 'nothing about config_args'}"
        )
    return matches[0]


def _config_matches_source(gluon_config, src):
    # Inductor emits config values either as an annotated constexpr,
    # "BLOCK_M : gl.constexpr = 128", or as a repr'd dict entry,
    # "'BLOCK_M': 128".
    for field in dataclasses.fields(gluon_config):
        value = getattr(gluon_config, field.name)
        pattern = rf"\b{field.name}\b['\"]?\s*(?::\s*[\w.]+\s*)?[:=]\s*{value}\b"
        if not re.search(pattern, src):
            return False
    return True


def _proton_profile_gluon(
    A,
    B,
    offs,
    proton_out,
    buffer_type,
    buffer_size,
    sample_warps,
    output_format,
    optimizations,
):
    """Profile one call of the Gluon kernel with Proton scopes.

    Proton instruments at compile time, so the compile has to happen
    inside the session. Autotuning there would put every candidate's
    trial launches into the profile, so the winner is picked first
    without instrumentation and then pinned as the only choice.
    """
    import triton.profiler as proton

    winner = _autotune_winning_config(A, B, offs)
    pinned = winner
    if buffer_type == "shared" and winner.NUM_LOAD_BUFFERS > 1:
        # Proton's shared buffer is whatever shared memory the kernel
        # leaves unused, and at the winning config that is nothing.
        # Give up one load buffer to make room: a shallower pipeline,
        # but the same pipeline.
        pinned = dataclasses.replace(
            winner, NUM_LOAD_BUFFERS=winner.NUM_LOAD_BUFFERS - 1
        )
    num_warps = pinned.NUM_STORE_WARPS + 4
    segments = len(sample_warps.split(",")) if sample_warps else num_warps
    if buffer_size == 0 and buffer_type == "global":
        # Proton splits buffer_size across the profiled warps and wants
        # each share to be a power of 2, and caps the total, which it
        # multiplies by the CTA count, at 4GB. Take the largest share
        # that fits: a dropped event is the oldest one, so the partition
        # scopes are the first thing lost.
        num_ctas = torch.cuda.get_device_properties(0).multi_processor_count
        share_limit = 4 * 1024**3 // (segments * num_ctas)
        buffer_size = segments * (1 << min(23, share_limit.bit_length() - 1))
    print(f"  Proton: pinning {pinned}")
    if buffer_size:
        print(f"  Proton: {buffer_size // segments // 8} events/warp")

    mode_kwargs = {
        "name": "default",
        "granularity": "warp",
        "buffer_type": buffer_type,
        "buffer_size": buffer_size,
        "optimizations": optimizations,
    }
    if sample_warps:
        mode_kwargs["sampling_strategy"] = "selective"
        mode_kwargs["sampling_options"] = sample_warps

    from torch._inductor.heuristics.template import gluon as gluon_heuristics

    orig_get_configs = gluon_heuristics.get_grouped_mm_configs
    gluon_heuristics.get_grouped_mm_configs = lambda **kwargs: [pinned]
    try:
        torch._dynamo.reset()
        compiled = torch.compile(
            torch._grouped_mm,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "GLUON",
                "gluon_enable_proton_profiling": True,
                "fx_graph_cache": False,
                # Proton instruments during the Triton compile and
                # installs itself in this process only, so a kernel
                # built by an async_compile worker is uninstrumented.
                "compile_threads": 1,
            },
            dynamic=False,
        )
        # time_shift, the only knob that removes Proton's own per-record
        # cost, is applied by the trace dump and not by the tree dump.
        session = proton.start(
            proton_out,
            data="trace" if output_format == "chrome_trace" else "tree",
            backend="instrumentation",
            mode=proton.mode.InstrumentationMode(**mode_kwargs),
        )
        compiled(A, B.transpose(-2, -1), offs)
        torch.cuda.synchronize()
        proton.finalize(session, output_format=output_format)
    finally:
        gluon_heuristics.get_grouped_mm_configs = orig_get_configs

    path = f"{proton_out}.{output_format}"
    with open(path) as f:
        if "compute_work" not in f.read():
            raise RuntimeError(
                f"{path} has no scopes: the kernel ran without "
                "instrumentation. Proton instruments in-process, so check "
                "that nothing moved the Triton compile off the main process "
                "or served it from a cache."
            )
    print(f"  Proton profile written to {path}")


# (num_experts, hidden_size, expert intermediate size, top_k), from each
# model's Hugging Face config.json. Each expert computes
# (silu(x @ W_gate) * (x @ W_up)) @ W_down; as in torchtitan's GroupedLinear
# (torchtitan/models/common/linear.py), W_gate and W_up are fused, giving one
# up GEMM with N = 2 * intermediate, K = hidden and one down GEMM with
# N = hidden, K = intermediate, over M = tokens * top_k routed rows.
_MOE_MODELS = {
    "Mixtral-8x7B": (8, 4096, 14336, 2),
    "Llama-4-Scout": (16, 5120, 8192, 1),
    "DeepSeek-V3": (256, 7168, 2048, 8),
    "gpt-oss-120b": (128, 2880, 2880, 4),
    "Qwen3-235B-A22B": (128, 4096, 1536, 8),
}
_MOE_TOKENS = (256, 16384)


def _moe_model_gmnk():
    gmnk = []
    for experts, hidden, intermediate, top_k in _MOE_MODELS.values():
        for tokens in _MOE_TOKENS:
            m = tokens * top_k
            gmnk.append([experts, m, 2 * intermediate, hidden])
            gmnk.append([experts, m, hidden, intermediate])
    return gmnk


def _default_gmnk(a_dim, b_dim):
    gmnk = _moe_model_gmnk()
    if a_dim == 2 and b_dim == 2:
        return [[g, n, k, m] for g, m, n, k in gmnk]
    if a_dim == 3 and b_dim == 2:
        return [[g, n, m, k] for g, m, n, k in gmnk]
    if a_dim == 3 and b_dim == 3:
        return [[g, m // g, n, k] for g, m, n, k in gmnk]
    return gmnk


def _first_call_seconds(fn, start):
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _save_progress(result_file, done, result):
    if result_file is not None:
        Path(result_file).write_text(json.dumps({"done": done, "result": result}))


def _print_table(columns):
    import pandas as pd

    df = pd.DataFrame(columns)
    floatfmt = tuple(
        ".0f" if pd.api.types.is_integer_dtype(dt) else ".2f" for dt in df.dtypes
    )
    df = df.astype(object).where(df.notna(), None)
    print(df.to_markdown(index=False, floatfmt=floatfmt, missingval=""))


def _print_results(results):
    if not results:
        return
    first = results[0]
    print(
        f"A: {first['A dim']}d {first['A layout']}, B: {first['B dim']}d {first['B layout']}"
        " | us = median time, x = speedup over ATen"
    )
    names = ("ATen", "Triton", "CuTeDSL", "Gluon")
    shape = {k: [r[k] for r in results] for k in ("G", "M", "N", "K")}
    nan = float("nan")

    perf = dict(shape)
    for b in names:
        if any(f"{b} (us)" in r for r in results):
            perf[f"{b} us"] = [r.get(f"{b} (us)", nan) for r in results]
            if b != "ATen":
                perf[f"{b} x"] = [r.get(f"{b} speedup", nan) for r in results]
    _print_table(perf)

    timing = dict(shape)
    for b in names:
        if any(f"{b} compile (s)" in r for r in results):
            timing[f"{b} compile s"] = [r.get(f"{b} compile (s)", nan) for r in results]
        if any(f"{b} eager (us)" in r for r in results):
            timing[f"{b} overhead us"] = [
                r.get(f"{b} eager (us)", nan) - r.get(f"{b} (us)", nan) for r in results
            ]
    if len(timing) > len(shape):
        print()
        _print_table(timing)


def benchmark_grouped_mm(
    gmnk=None,
    a_dim=2,
    a_k_major=True,
    b_dim=3,
    b_k_major=True,
    dtype=None,
    seed=0,
    rtol=1e-2,
    atol=1e-2,
    use_cuda_graphs=False,
    backends=None,
    warmup=10,
    rep=100,
    cooldown_seconds=1.0,
    grouping="random",
    proton_out=None,
    proton_buffer_size=0,
    proton_buffer_type="shared",
    proton_sample_warps="",
    proton_format="chrome_trace",
    proton_optimizations="clock32,time_shift",
    result_file=None,
    timing_details=False,
):
    torch.manual_seed(seed)
    if timing_details:
        from torch._inductor.async_compile import AsyncCompile

        AsyncCompile.warm_pool()
        AsyncCompile.wait_pool_ready()
    if backends is None:
        backends = BACKEND_CHOICES

    device = "cuda"
    if dtype is None:
        dtype = torch.bfloat16
    align = 16 // dtype.itemsize

    if gmnk is None:
        gmnk = _default_gmnk(a_dim, b_dim)

    results = []

    for G, M, N, K in gmnk:
        K_align = (K + align - 1) // align * align
        M_align = (M + align - 1) // align * align
        N_align = (N + align - 1) // align * align

        a_is_2d = a_dim == 2
        b_is_2d = b_dim == 2

        if a_is_2d:
            if a_k_major:
                A = torch.randn(M, K_align, device=device, dtype=dtype)[:, :K]
            else:
                A = torch.randn(K, M_align, device=device, dtype=dtype).t()[:M, :]
        else:
            if a_k_major:
                A = torch.randn(G, M, K_align, device=device, dtype=dtype)[:, :, :K]
            else:
                A = torch.randn(G, K, M_align, device=device, dtype=dtype).transpose(
                    -2, -1
                )[:, :M, :]

        if b_is_2d:
            if b_k_major:
                B = torch.randn(N, K_align, device=device, dtype=dtype)[:, :K]
            else:
                B = torch.randn(K, N_align, device=device, dtype=dtype).t()[:N, :]
        else:
            if b_k_major:
                B = torch.randn(G, N, K_align, device=device, dtype=dtype)[:, :, :K]
            else:
                B = torch.randn(G, K, N_align, device=device, dtype=dtype).transpose(
                    -2, -1
                )[:, :N, :]

        if a_is_2d and b_is_2d:
            offs_align = 1 if (not a_k_major and not b_k_major) else align
            offs = _generate_offsets(K, G, device, mode=grouping, align=offs_align)
        elif a_is_2d and not b_is_2d:
            offs_align = 1 if a_k_major else align
            offs = _generate_offsets(M, G, device, mode=grouping, align=offs_align)
        elif not a_is_2d and b_is_2d:
            offs = _generate_offsets(N, G, device, mode=grouping, align=align)
        else:
            offs = None

        print(f"G={G}, M={M}, N={N}, K={K}")

        flops = 2 * M * N * K
        result = {
            "G": G,
            "M": M,
            "N": N,
            "K": K,
            "A dim": a_dim,
            "B dim": b_dim,
            "A layout": _major_label(a_k_major, "m"),
            "B layout": _major_label(b_k_major, "n"),
        }

        C_ref = torch._grouped_mm(A, B.transpose(-2, -1), offs)
        done = []

        us_aten = None
        if "aten" in backends:
            fn_aten = lambda: torch._grouped_mm(  # noqa: E731
                A, B.transpose(-2, -1), offs
            )
            bench_aten = _do_bench_cuda(
                _maybe_wrap_cuda_graph(fn_aten, "aten", use_cuda_graphs),
                warmup=warmup,
                rep=rep,
                cooldown_seconds=cooldown_seconds,
            )
            us_aten = bench_aten["median_us"]
            tflops_aten = flops * 1e-12 / (us_aten * 1e-6)
            print(
                f"  ATen: {us_aten:.2f} us ({tflops_aten:.2f} TFLOPS; "
                f"min={bench_aten['min_us']:.2f}, max={bench_aten['max_us']:.2f})"
            )
            result["ATen (us)"] = us_aten
            if timing_details and use_cuda_graphs:
                result["ATen eager (us)"] = _do_bench_cuda(
                    fn_aten, warmup=warmup, rep=rep, cooldown_seconds=cooldown_seconds
                )["median_us"]
            gc.collect()
            torch.cuda.empty_cache()
            done.append("aten")
            _save_progress(result_file, done, result)

        if "triton" in backends:
            try:
                torch._dynamo.reset()
                compile_start = time.perf_counter()
                compiled_triton = torch.compile(
                    torch._grouped_mm,
                    options={
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "TRITON",
                    },
                    dynamic=False,
                )
                fn_triton = lambda: compiled_triton(  # noqa: E731
                    A, B.transpose(-2, -1), offs
                )
                if timing_details:
                    result["Triton compile (s)"] = _first_call_seconds(fn_triton, compile_start)
                bench_triton = _do_bench_cuda(
                    _maybe_wrap_cuda_graph(fn_triton, "triton", use_cuda_graphs),
                    warmup=warmup,
                    rep=rep,
                    cooldown_seconds=cooldown_seconds,
                )
                us_triton = bench_triton["median_us"]
                tflops_triton = flops * 1e-12 / (us_triton * 1e-6)
                print(
                    f"  Triton: {us_triton:.2f} us ({tflops_triton:.2f} TFLOPS; "
                    f"min={bench_triton['min_us']:.2f}, max={bench_triton['max_us']:.2f})"
                )
                result["Triton (us)"] = us_triton
                if timing_details and use_cuda_graphs:
                    result["Triton eager (us)"] = _do_bench_cuda(
                        fn_triton, warmup=warmup, rep=rep, cooldown_seconds=cooldown_seconds
                    )["median_us"]
                if us_aten is not None:
                    result["Triton speedup"] = us_aten / us_triton

                try:
                    C_triton = compiled_triton(A, B.transpose(-2, -1), offs)
                    torch.testing.assert_close(C_triton, C_ref, rtol=rtol, atol=atol)
                    print("  ✓ Triton correctness check passed")
                except AssertionError:
                    print("  ✗ Triton correctness check FAILED")
            except Exception as e:
                print(f"  Triton: Failed ({e})")
            gc.collect()
            torch.cuda.empty_cache()
            done.append("triton")
            _save_progress(result_file, done, result)

        if is_blackwell():
            if a_dim == 2 and b_dim == 3 and "cutedsl" in backends:
                try:
                    torch._dynamo.reset()
                    compile_start = time.perf_counter()
                    compiled_cutedsl = torch.compile(
                        torch._grouped_mm,
                        options={
                            "max_autotune": True,
                            "max_autotune_gemm_backends": "CUTEDSL",
                        },
                        dynamic=False,
                    )
                    fn_cutedsl = lambda: compiled_cutedsl(  # noqa: E731
                        A, B.transpose(-2, -1), offs
                    )
                    if timing_details:
                        result["CuTeDSL compile (s)"] = _first_call_seconds(fn_cutedsl, compile_start)
                    bench_cutedsl = _do_bench_cuda(
                        _maybe_wrap_cuda_graph(fn_cutedsl, "cutedsl", use_cuda_graphs),
                        warmup=warmup,
                        rep=rep,
                        cooldown_seconds=cooldown_seconds,
                    )
                    us_cutedsl = bench_cutedsl["median_us"]
                    tflops_cutedsl = flops * 1e-12 / (us_cutedsl * 1e-6)
                    print(
                        f"  CuTeDSL: {us_cutedsl:.2f} us ({tflops_cutedsl:.2f} TFLOPS; "
                        f"min={bench_cutedsl['min_us']:.2f}, "
                        f"max={bench_cutedsl['max_us']:.2f})"
                    )
                    result["CuTeDSL (us)"] = us_cutedsl
                    if timing_details and use_cuda_graphs:
                        result["CuTeDSL eager (us)"] = _do_bench_cuda(
                            fn_cutedsl, warmup=warmup, rep=rep, cooldown_seconds=cooldown_seconds
                        )["median_us"]
                    if us_aten is not None:
                        result["CuTeDSL speedup"] = us_aten / us_cutedsl

                    try:
                        C_cutedsl = compiled_cutedsl(A, B.transpose(-2, -1), offs)
                        torch.testing.assert_close(
                            C_cutedsl, C_ref, rtol=rtol, atol=atol
                        )
                        print("  ✓ CuTeDSL correctness check passed")
                    except AssertionError:
                        print("  ✗ CuTeDSL correctness check FAILED")
                except Exception as e:
                    print(f"  CuTeDSL: Failed ({e})")
                gc.collect()
                torch.cuda.empty_cache()
                done.append("cutedsl")
                _save_progress(result_file, done, result)

            if "gluon" in backends:
                try:
                    torch._dynamo.reset()
                    compile_start = time.perf_counter()
                    compiled_gluon = torch.compile(
                        torch._grouped_mm,
                        options={
                            "max_autotune": True,
                            "max_autotune_gemm_backends": "GLUON",
                        },
                        dynamic=False,
                    )
                    fn_gluon = lambda: compiled_gluon(  # noqa: E731
                        A, B.transpose(-2, -1), offs
                    )
                    if timing_details:
                        result["Gluon compile (s)"] = _first_call_seconds(fn_gluon, compile_start)
                    bench_gluon = _do_bench_cuda(
                        _maybe_wrap_cuda_graph(fn_gluon, "gluon", use_cuda_graphs),
                        warmup=warmup,
                        rep=rep,
                        cooldown_seconds=cooldown_seconds,
                    )
                    us_gluon = bench_gluon["median_us"]
                    tflops_gluon = flops * 1e-12 / (us_gluon * 1e-6)
                    print(
                        f"  Gluon: {us_gluon:.2f} us ({tflops_gluon:.2f} TFLOPS; "
                        f"min={bench_gluon['min_us']:.2f}, max={bench_gluon['max_us']:.2f})"
                    )
                    result["Gluon (us)"] = us_gluon
                    if timing_details and use_cuda_graphs:
                        result["Gluon eager (us)"] = _do_bench_cuda(
                            fn_gluon, warmup=warmup, rep=rep, cooldown_seconds=cooldown_seconds
                        )["median_us"]
                    if us_aten is not None:
                        result["Gluon speedup"] = us_aten / us_gluon

                    try:
                        C_gluon = compiled_gluon(A, B.transpose(-2, -1), offs)
                        torch.testing.assert_close(C_gluon, C_ref, rtol=rtol, atol=atol)
                        print("  ✓ Gluon correctness check passed")
                    except AssertionError:
                        print("  ✗ Gluon correctness check FAILED")

                    if proton_out is not None:
                        _proton_profile_gluon(
                            A,
                            B,
                            offs,
                            proton_out,
                            proton_buffer_type,
                            proton_buffer_size,
                            proton_sample_warps,
                            proton_format,
                            proton_optimizations,
                        )
                except Exception as e:
                    print(f"  Gluon: Failed ({e})")
                gc.collect()
                torch.cuda.empty_cache()
                done.append("gluon")
                _save_progress(result_file, done, result)

        results.append(result)
        print()

    if result_file is None:
        _print_results(results)
    return results


def _benchmark_isolated(gmnk, backends, child_args, layout):
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "result.json"
        for shape in gmnk:
            remaining = [b for b in BACKEND_CHOICES if b in backends]
            merged = dict(zip(("G", "M", "N", "K"), shape)) | layout
            while remaining:
                result_file.unlink(missing_ok=True)
                cmd = [sys.executable, __file__, *child_args, "--result-file", str(result_file)]
                cmd += ["--gmnk", ",".join(map(str, shape)), "--backends", *remaining]
                sys.stdout.flush()
                rc = subprocess.run(cmd).returncode
                done = []
                if result_file.exists():
                    progress = json.loads(result_file.read_text())
                    done = progress["done"]
                    merged.update(progress["result"])
                remaining = [b for b in remaining if b not in done]
                if rc == 0:
                    break
                if remaining:
                    print(f"  {remaining.pop(0)}: crashed the benchmark process (exit {rc}), skipped\n", flush=True)
            for b in ("Triton", "CuTeDSL", "Gluon"):
                if "ATen (us)" in merged and f"{b} (us)" in merged:
                    merged[f"{b} speedup"] = merged["ATen (us)"] / merged[f"{b} (us)"]
            results.append(merged)
    _print_results(results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark grouped MM with selectable row/col-major layouts."
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
        default=1e-2,
        help="Relative tolerance for correctness checks.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness checks.",
    )
    parser.add_argument(
        "--gmnk",
        nargs="+",
        type=_parse_gmnk,
        help="Problem sizes as G,M,N,K (space-separated).",
    )
    parser.add_argument(
        "--A",
        dest="a_spec",
        type=_parse_a_spec,
        default=_parse_a_spec("2d:k-major"),
        help="A spec: <2d|3d>:<k-major|m-major>.",
    )
    parser.add_argument(
        "--B",
        dest="b_spec",
        type=_parse_b_spec,
        default=_parse_b_spec("3d:k-major"),
        help="B spec: <2d|3d>:<k-major|n-major>.",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        default=False,
        help=(
            "Capture each backend's call in a CUDA graph and benchmark "
            "graph.replay(), isolating GPU execution time from Python/"
            "dispatcher overhead."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKEND_CHOICES,
        default=BACKEND_CHOICES,
        help="Which backends to benchmark (space-separated). Default: all.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations per shape/backend.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured iterations per shape/backend.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=1.0,
        help=(
            "Idle time before each shape/backend measurement, so every "
            "backend starts from the same thermal state."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=["random", "balanced"],
        default="random",
        help=(
            "How to split the ragged dimension across groups: 'random' "
            "(default, non-equal via a multinomial draw) or 'balanced' "
            "(equal-sized groups, remainder in the last group(s))."
        ),
    )
    parser.add_argument(
        "--proton-out",
        dest="proton_out",
        type=str,
        default=None,
        help=(
            "Path prefix for a Proton instrumentation profile of the Gluon "
            "kernel. Passing this also compiles the kernel with Proton scopes "
            "enabled, which adds significant runtime overhead."
        ),
    )
    parser.add_argument(
        "--proton-buffer-size",
        dest="proton_buffer_size",
        type=int,
        default=0,
        help=(
            "Proton per-CTA event buffer size in bytes (0 = 1 MB per warp). "
            "Proton splits it across warps and wants each share to be a "
            "power of 2. Total allocation is this times CTAs, capped at 4GB."
        ),
    )
    parser.add_argument(
        "--proton-buffer-type",
        dest="proton_buffer_type",
        choices=["global", "shared"],
        default="shared",
        help=(
            "Where Proton stages its event buffer. 'shared' is far cheaper "
            "per event, but it only gets the shared memory the kernel leaves "
            "free, so one load buffer is given up to make room."
        ),
    )
    parser.add_argument(
        "--proton-sample-warps",
        dest="proton_sample_warps",
        type=str,
        default="",
        help=(
            "Comma-separated warp indices to profile (e.g. '6,7'). Restricting "
            "to the warps of interest leaves a larger buffer for each of them."
        ),
    )
    parser.add_argument(
        "--proton-format",
        dest="proton_format",
        choices=["hatchet", "chrome_trace"],
        default="chrome_trace",
        help=(
            "Proton output format: 'chrome_trace' for a Perfetto timeline, "
            "the only one that gets the time_shift correction; 'hatchet' for "
            "proton-viewer aggregates, which are raw uncorrected cycles."
        ),
    )
    parser.add_argument(
        "--proton-optimizations",
        dest="proton_optimizations",
        default="clock32,time_shift",
        help=(
            "Comma-separated Proton Optimize flags. 'clock32' halves the "
            "record size and 'time_shift' subtracts Proton's own per-record "
            "cost, which only the trace dump applies. Pass '' to disable."
        ),
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Run each shape in its own subprocess. If a backend crashes the "
            "process (e.g. an illegal memory access poisoning the CUDA "
            "context), it is recorded as failed and the remaining backends "
            "are rerun in a fresh process."
        ),
    )
    parser.add_argument(
        "--timing-details",
        action="store_true",
        help=(
            "Also record each compiled backend's compile time (torch.compile "
            "through the end of the first call, including autotuning) and, "
            "with --use-cuda-graphs, each backend's eager time, whose "
            "difference from the graph time is the per-call host overhead."
        ),
    )
    parser.add_argument("--result-file", dest="result_file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    a_dim, a_k_major = args.a_spec
    b_dim, b_k_major = args.b_spec
    dtype = torch.bfloat16 if args.input_dtype == "bf16" else torch.float16
    gmnk = args.gmnk if args.gmnk is not None else None
    if args.isolate:
        if args.proton_out is not None:
            parser.error("--isolate does not support --proton-out")
        child_args = [
            "--input-dtype", args.input_dtype,
            "--seed", str(args.seed),
            "--rtol", str(args.rtol),
            "--atol", str(args.atol),
            "--A", f"{a_dim}d:{_major_label(a_k_major, 'm')}",
            "--B", f"{b_dim}d:{_major_label(b_k_major, 'n')}",
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--cooldown-seconds", str(args.cooldown_seconds),
            "--grouping", args.grouping,
        ]
        if args.use_cuda_graphs:
            child_args.append("--use-cuda-graphs")
        if args.timing_details:
            child_args.append("--timing-details")
        layout = {
            "A dim": a_dim,
            "B dim": b_dim,
            "A layout": _major_label(a_k_major, "m"),
            "B layout": _major_label(b_k_major, "n"),
        }
        _benchmark_isolated(gmnk or _default_gmnk(a_dim, b_dim), args.backends, child_args, layout)
        sys.exit(0)
    benchmark_grouped_mm(
        gmnk=gmnk,
        a_dim=a_dim,
        a_k_major=a_k_major,
        b_dim=b_dim,
        b_k_major=b_k_major,
        dtype=dtype,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        use_cuda_graphs=args.use_cuda_graphs,
        backends=args.backends,
        warmup=args.warmup,
        rep=args.iterations,
        cooldown_seconds=args.cooldown_seconds,
        grouping=args.grouping,
        proton_out=args.proton_out,
        proton_buffer_size=args.proton_buffer_size,
        proton_buffer_type=args.proton_buffer_type,
        proton_sample_warps=args.proton_sample_warps,
        proton_format=args.proton_format,
        proton_optimizations=args.proton_optimizations,
        result_file=args.result_file,
        timing_details=args.timing_details,
    )
