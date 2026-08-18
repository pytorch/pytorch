# FIXME: move this to tritonbench project.

import argparse
import gc
import time
import warnings

from triton import runtime

import torch


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


_BENCH_SETTLE_SECONDS = 0.1


def _do_bench_cuda(fn, warmup=10, rep=100, settle_seconds=_BENCH_SETTLE_SECONDS):
    """Benchmark `fn` with a fixed number of iterations, an L2 cache
    clear before each measured call, and a settle delay before each
    call to avoid thermal-throttling bias.

    triton.testing.do_bench's warmup/rep are milliseconds, not iteration
    counts: for slow (large-shape) calls this collapses to very few
    measured samples (e.g. ~1 warmup, ~9 reps at ~2ms/call with the
    warmup=2, rep=20 previously used here), which is too noisy for
    tracking single-digit-percent speedups. Fixed iteration counts give
    every shape the same statistical power regardless of how long it
    takes to run.

    Without a settle delay, back-to-back launches let the GPU heat up
    over the course of the rep loop, so later iterations can run
    measurably slower than earlier ones purely from clock throttling -
    a directional bias, not just noise, and one that can differ by
    backend depending on how much power each kernel draws. The delay
    (and the synchronize before it, so it's genuine idle time rather
    than the host stalling while queued work keeps running) gives the
    GPU a chance to cool between measured calls.
    """
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    fn()
    di.synchronize()

    for _ in range(warmup):
        time.sleep(settle_seconds)
        fn()
    di.synchronize()

    times_ms = []
    for _ in range(rep):
        runtime.driver.active.clear_cache(cache)
        time.sleep(settle_seconds)
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
    grouping="random",
):
    torch.manual_seed(seed)
    if backends is None:
        backends = BACKEND_CHOICES

    device = "cuda"
    if dtype is None:
        dtype = torch.bfloat16
    align = 16 // dtype.itemsize

    if gmnk is None:
        gmnk = [
            [2, 5, 16, 16],
            [3, 13, 16, 32],
            [8, 128, 16, 16],
            [7, 253, 24, 24],
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
        if a_dim == 2 and b_dim == 2:
            gmnk = [[g, k, n, m] for g, m, n, k in gmnk]
        elif a_dim == 3 and b_dim == 2:
            gmnk = [[g, n, m, k] for g, m, n, k in gmnk]
        elif a_dim == 3 and b_dim == 3:
            gmnk = [[g, m // g, n, k] for g, m, n, k in gmnk]

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

        us_aten = None
        if "aten" in backends:
            fn_aten = lambda: torch._grouped_mm(  # noqa: E731
                A, B.transpose(-2, -1), offs
            )
            bench_aten = _do_bench_cuda(
                _maybe_wrap_cuda_graph(fn_aten, "aten", use_cuda_graphs),
                warmup=warmup,
                rep=rep,
            )
            us_aten = bench_aten["median_us"]
            tflops_aten = flops * 1e-12 / (us_aten * 1e-6)
            print(
                f"  ATen: {us_aten:.2f} us ({tflops_aten:.2f} TFLOPS; "
                f"min={bench_aten['min_us']:.2f}, max={bench_aten['max_us']:.2f})"
            )
            result["ATen (us)"] = us_aten
            gc.collect()
            torch.cuda.empty_cache()

        if "triton" in backends:
            try:
                torch._dynamo.reset()
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
                bench_triton = _do_bench_cuda(
                    _maybe_wrap_cuda_graph(fn_triton, "triton", use_cuda_graphs),
                    warmup=warmup,
                    rep=rep,
                )
                us_triton = bench_triton["median_us"]
                tflops_triton = flops * 1e-12 / (us_triton * 1e-6)
                print(
                    f"  Triton: {us_triton:.2f} us ({tflops_triton:.2f} TFLOPS; "
                    f"min={bench_triton['min_us']:.2f}, max={bench_triton['max_us']:.2f})"
                )
                result["Triton (us)"] = us_triton
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

        if is_blackwell():
            if a_dim == 2 and b_dim == 3 and "cutedsl" in backends:
                try:
                    torch._dynamo.reset()
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
                    bench_cutedsl = _do_bench_cuda(
                        _maybe_wrap_cuda_graph(fn_cutedsl, "cutedsl", use_cuda_graphs),
                        warmup=warmup,
                        rep=rep,
                    )
                    us_cutedsl = bench_cutedsl["median_us"]
                    tflops_cutedsl = flops * 1e-12 / (us_cutedsl * 1e-6)
                    print(
                        f"  CuTeDSL: {us_cutedsl:.2f} us ({tflops_cutedsl:.2f} TFLOPS; "
                        f"min={bench_cutedsl['min_us']:.2f}, "
                        f"max={bench_cutedsl['max_us']:.2f})"
                    )
                    result["CuTeDSL (us)"] = us_cutedsl
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

            if "gluon" in backends:
                try:
                    torch._dynamo.reset()
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
                    bench_gluon = _do_bench_cuda(
                        _maybe_wrap_cuda_graph(fn_gluon, "gluon", use_cuda_graphs),
                        warmup=warmup,
                        rep=rep,
                    )
                    us_gluon = bench_gluon["median_us"]
                    tflops_gluon = flops * 1e-12 / (us_gluon * 1e-6)
                    print(
                        f"  Gluon: {us_gluon:.2f} us ({tflops_gluon:.2f} TFLOPS; "
                        f"min={bench_gluon['min_us']:.2f}, max={bench_gluon['max_us']:.2f})"
                    )
                    result["Gluon (us)"] = us_gluon
                    if us_aten is not None:
                        result["Gluon speedup"] = us_aten / us_gluon

                    try:
                        C_gluon = compiled_gluon(A, B.transpose(-2, -1), offs)
                        torch.testing.assert_close(C_gluon, C_ref, rtol=rtol, atol=atol)
                        print("  ✓ Gluon correctness check passed")
                    except AssertionError:
                        print("  ✗ Gluon correctness check FAILED")
                except Exception as e:
                    print(f"  Gluon: Failed ({e})")
                gc.collect()
                torch.cuda.empty_cache()

        results.append(result)
        print()

    import pandas as pd

    df = pd.DataFrame(results)
    floatfmt = tuple(
        ".0f"
        if pd.api.types.is_integer_dtype(dt)
        else ".2f"
        if pd.api.types.is_float_dtype(dt)
        else ""
        for dt in df.dtypes
    )
    print(df.to_markdown(index=False, floatfmt=floatfmt))
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
        "--grouping",
        choices=["random", "balanced"],
        default="random",
        help=(
            "How to split the ragged dimension across groups: 'random' "
            "(default, non-equal via a multinomial draw) or 'balanced' "
            "(equal-sized groups, remainder in the last group(s))."
        ),
    )
    args = parser.parse_args()
    a_dim, a_k_major = args.a_spec
    b_dim, b_k_major = args.b_spec
    dtype = torch.bfloat16 if args.input_dtype == "bf16" else torch.float16
    gmnk = args.gmnk if args.gmnk is not None else None
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
        grouping=args.grouping,
    )
