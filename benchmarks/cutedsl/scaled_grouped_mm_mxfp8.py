import argparse
import importlib
import statistics
import subprocess
import sys
import warnings
from collections.abc import Callable

import torch
from torch.nn.functional import scaled_grouped_mm, ScalingType, SwizzleType
from torch.testing._internal.common_quantized import (
    from_blocked_format,
    to_blocked,
    to_mxfp,
)


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


def _do_bench_cuda(fn, warmup=10, rep=100):
    if isinstance(fn, (list, tuple)):
        if len(fn) == 0:
            raise ValueError("fn list for benchmarking cannot be empty")
        fns = fn
        for i in range(warmup):
            fns[i % len(fns)]()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(rep):
            fns[i % len(fns)]()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1e3 / rep

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / rep


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty")
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _format_stat_triplet(values: list[float]) -> str:
    med = statistics.median(values)
    p10 = _percentile(values, 10.0)
    p90 = _percentile(values, 90.0)
    return f"{med:.2f} us (p10={p10:.2f}, p90={p90:.2f})"


def _nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _maybe_warn_perf_tuning_failure(
    context: str, proc: subprocess.CompletedProcess[str]
) -> None:
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    msg = stderr if stderr else stdout
    if proc.returncode != 0:
        warnings.warn(
            (
                f"GPU perf tuning failed during {context}: {msg}. "
                "Skipping further clock/persistence changes for this run."
            ),
            stacklevel=2,
        )


def _query_gpu_clock_info(device_index: int) -> str | None:
    query_candidates = [
        "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,pstate",
        "clocks.sm,clocks.mem,clocks.max.graphics,clocks.max.memory,pstate",
    ]
    for query in query_candidates:
        proc = _nvidia_smi(
            [
                "-i",
                str(device_index),
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ]
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0]
    return None


def _configure_gpu_perf_once(device_index: int) -> None:
    before = _query_gpu_clock_info(device_index)
    if before is not None:
        print(f"GPU[{device_index}] clocks before tuning: {before}")

    proc_pm = _nvidia_smi(["-i", str(device_index), "-pm", "1"])
    if proc_pm.returncode != 0:
        _maybe_warn_perf_tuning_failure("enabling persistence mode", proc_pm)
        return

    max_info = _query_gpu_clock_info(device_index)
    if max_info is None:
        warnings.warn(
            "Failed to query max GPU clocks; skipping GPU clock locking.",
            stacklevel=2,
        )
        return
    # Query returns: sm, mem, max_sm, max_mem, pstate
    parts = [x.strip() for x in max_info.split(",")]
    if len(parts) < 5:
        warnings.warn(
            f"Unexpected GPU clock query format: '{max_info}', skipping GPU clock locking.",
            stacklevel=2,
        )
        return
    max_sm = parts[2]
    max_mem = parts[3]

    proc_lgc = _nvidia_smi(["-i", str(device_index), "-lgc", f"{max_sm},{max_sm}"])
    if proc_lgc.returncode != 0:
        _maybe_warn_perf_tuning_failure("locking SM clocks", proc_lgc)
        return
    proc_lmc = _nvidia_smi(["-i", str(device_index), "-lmc", f"{max_mem},{max_mem}"])
    if proc_lmc.returncode != 0:
        _maybe_warn_perf_tuning_failure("locking memory clocks", proc_lmc)
        return

    after = _query_gpu_clock_info(device_index)
    if after is not None:
        print(f"GPU[{device_index}] clocks after tuning:  {after}")


def _maybe_wrap_cuda_graph(fn, label: str, use_cuda_graphs: bool):
    if not use_cuda_graphs:
        return fn

    keep_alive = [None]
    try:
        keep_alive[0] = fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph):
                keep_alive[0] = fn()
        torch.cuda.current_stream().wait_stream(capture_stream)

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


def _bench_backend_subprocess(
    g: int,
    m: int,
    n: int,
    k: int,
    input_dtype: str,
    seed: int,
    grouping: str,
    warmup: int,
    rep: int,
    backend: str,
    use_cuda_graphs: bool,
    mma_tile_mn: tuple[int, int] | None,
    cluster_shape_mn: tuple[int, int] | None,
    transpose_ab: bool | None,
    layout_mode: str,
) -> float:
    cmd = [
        sys.executable,
        __file__,
        "--gmnk",
        f"{g},{m},{n},{k}",
        "--input-dtype",
        input_dtype,
        "--seed",
        str(seed),
        "--grouping",
        grouping,
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
        "--backend",
        backend,
        "--emit-us-only",
        "--no-correctness",
    ]
    if use_cuda_graphs:
        cmd.append("--use-cuda-graphs")
    if mma_tile_mn is not None:
        cmd.extend(["--mma-tile-mn", f"{mma_tile_mn[0]},{mma_tile_mn[1]}"])
    if cluster_shape_mn is not None:
        cmd.extend(
            ["--cluster-shape-mn", f"{cluster_shape_mn[0]},{cluster_shape_mn[1]}"]
        )
    if transpose_ab is not None:
        cmd.extend(["--transpose-ab", "on" if transpose_ab else "off"])
    cmd.extend(["--layout-mode", layout_mode])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "subprocess benchmark failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("subprocess benchmark returned no output")
    return float(lines[-1])


def _generate_offsets(total, groups, device, mode="balanced", align=1):
    if total <= 0:
        return torch.zeros(groups, device=device, dtype=torch.int32)
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")
    if mode not in ("balanced", "random"):
        raise ValueError(f"mode must be 'balanced' or 'random', got {mode}")

    if mode == "balanced":
        base = total // groups
        remainder = total - base * groups
        if remainder != 0:
            warnings.warn(
                (
                    f"grouping='balanced' with M={total}, G={groups}: "
                    f"using base size {base} and placing tail {remainder} "
                    "in the last group"
                ),
                stacklevel=2,
            )
        counts = torch.full((groups,), base, device=device, dtype=torch.int64)
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


def _convert_to_mxfp8_with_hp_ref(t, block_size: int):
    t_scale, t_lp = to_mxfp(t, format="mxfp8")
    t_hp = from_blocked_format(t_lp, t_scale, blocksize=block_size)
    return t_hp, t_lp, t_scale


def _prepare_inputs(
    g: int,
    m: int,
    n: int,
    k: int,
    dtype,
    seed: int,
    grouping: str = "balanced",
    layout_mode: str = "2d/3d",
):
    torch.manual_seed(seed)
    device = "cuda"
    block_size = 32
    k_eff = ((k + block_size - 1) // block_size) * block_size
    align = 16 // dtype.itemsize
    k_align = (k_eff + align - 1) // align * align

    if layout_mode == "2d/3d":
        x = torch.randn((m, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.1
        w = (
            torch.randn((g, n, k_align), device=device, dtype=dtype)[:, :, :k_eff]
            * 0.01
        )
        if k_eff > k:
            x[:, k:] = 0
            w[:, :, k:] = 0
        offs = _generate_offsets(m, g, device, mode=grouping, align=16)

        wh_list = []
        wq_list = []
        w_blocked_scales = []
        for i in range(g):
            wh, wq, w_scale = _convert_to_mxfp8_with_hp_ref(w[i], block_size)
            w_scale_blocked = to_blocked(w_scale)
            wh_list.append(wh)
            wq_list.append(wq)
            w_blocked_scales.append(w_scale_blocked)
        wh = torch.stack(wh_list, dim=0).contiguous()
        wq = torch.stack(wq_list, dim=0).contiguous()
        w_blocked_scales = torch.stack(w_blocked_scales, dim=0).contiguous()

        xh_list = []
        xq_list = []
        x_scale_list = []
        x_scale_elems = 0
        for i in range(g):
            start = 0 if i == 0 else int(offs[i - 1].item())
            end = int(offs[i].item())
            if end == start:
                continue
            x_slice = x[start:end, :]
            xh, xq_slice, x_scale = _convert_to_mxfp8_with_hp_ref(x_slice, block_size)
            x_scale_blocked = to_blocked(x_scale)
            x_scale_list.append(x_scale_blocked)
            x_scale_elems += x_scale_blocked.numel()
            xh_list.append(xh)
            xq_list.append(xq_slice)
        xh = torch.cat(xh_list, dim=0).contiguous()
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
        xh = xh.view(-1, xh.shape[-1])
    else:
        x = torch.randn((m, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.1
        w = torch.randn((n, k_align), device=device, dtype=dtype)[:, :k_eff] * 0.01
        if k_eff > k:
            x[:, k:] = 0
            w[:, k:] = 0
        offs = _generate_offsets(k_eff, g, device, mode=grouping, align=block_size)
        m_rounded = ((m + 127) // 128) * 128
        n_rounded = ((n + 127) // 128) * 128

        xh_list = []
        xq_list = []
        x_scale_list = []
        wh_list = []
        wq_list = []
        w_scale_list = []
        for i in range(g):
            start = 0 if i == 0 else int(offs[i - 1].item())
            end = int(offs[i].item())
            if end == start:
                continue
            x_slice = x[:, start:end].contiguous()
            w_slice = w[:, start:end].contiguous()
            xh, xq, x_scale = _convert_to_mxfp8_with_hp_ref(x_slice, block_size)
            wh, wq, w_scale = _convert_to_mxfp8_with_hp_ref(w_slice, block_size)
            xh_list.append(xh)
            xq_list.append(xq)
            x_scale_list.append(to_blocked(x_scale))
            wh_list.append(wh)
            wq_list.append(wq)
            w_scale_list.append(to_blocked(w_scale))

        xh = torch.cat(xh_list, dim=1).contiguous()
        xq = torch.cat(xq_list, dim=1).contiguous()
        wh = torch.cat(wh_list, dim=1).contiguous()
        wq = torch.cat(wq_list, dim=1).contiguous()
        x_blocked_scales = torch.cat(x_scale_list, dim=0).contiguous()
        w_blocked_scales = torch.cat(w_scale_list, dim=0).contiguous()
        x_blocked_scales = x_blocked_scales.reshape(m_rounded, -1).contiguous()
        w_blocked_scales = w_blocked_scales.reshape(n_rounded, -1).contiguous()

    return xq, wq, x_blocked_scales, w_blocked_scales, offs, xh, wh, k_eff


def benchmark_scaled_grouped_mm(
    gmnk=None,
    dtype=None,
    seed=0,
    rtol=8e-2,
    atol=8e-2,
    grouping="balanced",
    use_subprocess=False,
    use_cuda_graphs=False,
    warmup=2,
    rep=20,
    paired_trials=1,
    backend="both",
    emit_us_only=False,
    do_correctness=True,
    mma_tile_mn: tuple[int, int] | None = None,
    cluster_shape_mn: tuple[int, int] | None = None,
    transpose_ab: bool | None = None,
    set_max_gpu_clocks=False,
    layout_mode: str = "2d/3d",
):
    if dtype is None:
        dtype = torch.bfloat16
    if backend not in ("both", "cpp", "cute"):
        raise ValueError(f"backend must be one of both/cpp/cute, got {backend}")
    if paired_trials < 1:
        raise ValueError(f"paired_trials must be >= 1, got {paired_trials}")
    if (mma_tile_mn is None) != (cluster_shape_mn is None):
        raise ValueError(
            "mma_tile_mn and cluster_shape_mn must be provided together or omitted together"
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
    scale_recipe = ScalingType.BlockWise1x32

    scale_recipe_int = [scale_recipe.value]
    swizzle_int = [swizzle.value]
    run_cpp = backend in ("both", "cpp")
    run_cute = backend in ("both", "cute")
    input_dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
    gpu_perf_configured = False

    results = []

    def _make_cute_call(
        fn: Callable[[], torch.Tensor],
    ) -> Callable[[], torch.Tensor]:
        if mma_tile_mn is None and cluster_shape_mn is None and transpose_ab is None:
            return fn

        sgmm_mod = importlib.import_module("torch._cutedsl.scaled_grouped_mm_mxfp8")

        def _call_with_override():
            old_select = sgmm_mod._select_kernel_config

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

            sgmm_mod._select_kernel_config = _select_override
            try:
                return fn()
            finally:
                sgmm_mod._select_kernel_config = old_select

        return _call_with_override

    for g, m, n, k in gmnk:
        if not is_blackwell():
            raise RuntimeError("Benchmark requires Blackwell (sm_100).")
        if set_max_gpu_clocks and not gpu_perf_configured:
            dev_idx = torch.cuda.current_device()
            _configure_gpu_perf_once(dev_idx)
            gpu_perf_configured = True

        (
            xq,
            wq,
            x_blocked_scales,
            w_blocked_scales,
            offs,
            xh,
            wh,
            k_eff,
        ) = _prepare_inputs(g, m, n, k, dtype, seed, grouping, layout_mode)

        if not emit_us_only:
            print(f"G={g}, M={m}, N={n}, K={k} (eff K={k_eff})")
        fn_cpp = lambda: torch._scaled_grouped_mm_v2(  # noqa: E731
            input=xq,
            mat2=wq.transpose(-2, -1),
            scale_a=[x_blocked_scales],
            recipe_a=scale_recipe_int,
            swizzle_a=swizzle_int,
            scale_b=[w_blocked_scales],
            recipe_b=scale_recipe_int,
            swizzle_b=swizzle_int,
            offs=offs,
            bias=None,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )

        fn_cute_base = lambda: scaled_grouped_mm(  # noqa: E731
            xq,
            wq.transpose(-2, -1),
            [x_blocked_scales],
            [scale_recipe],
            [w_blocked_scales],
            [scale_recipe],
            swizzle_a=[swizzle],
            swizzle_b=[swizzle],
            offs=offs,
            output_dtype=torch.bfloat16,
        )
        fn_cute = _make_cute_call(fn_cute_base)

        cpp_samples = []
        cute_samples = []
        speedup_samples = []

        if use_subprocess and backend == "both":
            for _ in range(paired_trials):
                trial_cpp = _bench_backend_subprocess(
                    g=g,
                    m=m,
                    n=n,
                    k=k,
                    input_dtype=input_dtype_name,
                    seed=seed,
                    grouping=grouping,
                    warmup=warmup,
                    rep=rep,
                    backend="cpp",
                    use_cuda_graphs=use_cuda_graphs,
                    mma_tile_mn=mma_tile_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    transpose_ab=transpose_ab,
                    layout_mode=layout_mode,
                )
                trial_cute = _bench_backend_subprocess(
                    g=g,
                    m=m,
                    n=n,
                    k=k,
                    input_dtype=input_dtype_name,
                    seed=seed,
                    grouping=grouping,
                    warmup=warmup,
                    rep=rep,
                    backend="cute",
                    use_cuda_graphs=use_cuda_graphs,
                    mma_tile_mn=mma_tile_mn,
                    cluster_shape_mn=cluster_shape_mn,
                    transpose_ab=transpose_ab,
                    layout_mode=layout_mode,
                )
                cpp_samples.append(trial_cpp)
                cute_samples.append(trial_cute)
                speedup_samples.append(trial_cpp / trial_cute)
        else:
            bench_fn_cpp = (
                _maybe_wrap_cuda_graph(fn_cpp, "cpp", use_cuda_graphs)
                if run_cpp
                else None
            )
            bench_fn_cute = (
                _maybe_wrap_cuda_graph(fn_cute, "cute", use_cuda_graphs)
                if run_cute
                else None
            )
            if run_cpp and run_cute:
                for _ in range(paired_trials):
                    trial_cpp = _do_bench_cuda(bench_fn_cpp, warmup=warmup, rep=rep)
                    trial_cute = _do_bench_cuda(bench_fn_cute, warmup=warmup, rep=rep)
                    cpp_samples.append(trial_cpp)
                    cute_samples.append(trial_cute)
                    speedup_samples.append(trial_cpp / trial_cute)
            else:
                if run_cpp:
                    cpp_samples = [
                        _do_bench_cuda(bench_fn_cpp, warmup=warmup, rep=rep)
                        for _ in range(paired_trials)
                    ]
                if run_cute:
                    cute_samples = [
                        _do_bench_cuda(bench_fn_cute, warmup=warmup, rep=rep)
                        for _ in range(paired_trials)
                    ]

        us_cpp = statistics.median(cpp_samples) if cpp_samples else None
        us_cute = statistics.median(cute_samples) if cute_samples else None
        speedup = (
            statistics.median(speedup_samples)
            if speedup_samples
            else (us_cpp / us_cute if us_cpp and us_cute else None)
        )

        if emit_us_only:
            print(f"{us_cpp if run_cpp else us_cute}")
            return [
                {"G": g, "M": m, "N": n, "K": k, "us": us_cpp if run_cpp else us_cute}
            ]

        if us_cpp is not None:
            print(f"  C++: {us_cpp:.2f} us")
        if us_cute is not None:
            print(f"  CuTeDSL: {us_cute:.2f} us")
        if paired_trials > 1:
            if cpp_samples:
                print(f"  C++ samples: {_format_stat_triplet(cpp_samples)}")
            if cute_samples:
                print(f"  CuTeDSL samples: {_format_stat_triplet(cute_samples)}")
            if speedup_samples:
                s_med = statistics.median(speedup_samples)
                s_p10 = _percentile(speedup_samples, 10.0)
                s_p90 = _percentile(speedup_samples, 90.0)
                print(
                    f"  speedup samples: {s_med:.3f} (p10={s_p10:.3f}, p90={s_p90:.3f})"
                )

        if do_correctness and run_cpp and run_cute:
            try:
                out_cpp = fn_cpp()
                out_cute = fn_cute()
                torch.testing.assert_close(out_cpp, out_cute, rtol=rtol, atol=atol)
                print("  ✓ correctness check passed")
            except AssertionError:
                print("  ✗ correctness check FAILED")

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
    for col in ("C++ (us)", "CuTeDSL (us)", "speedup"):
        df[col] = df[col].map(
            lambda x: f"{x:.3f}" if x is not None and pd.notna(x) else "nan"
        )
    print(
        df.to_markdown(
            index=False,
            disable_numparse=True,
            colalign=("right",) * len(df.columns),
        )
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark scaled grouped MM (MXFP8) C++ vs CuTeDSL."
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
        help="How to split the grouped dimension: M for 2d/3d, K for 2d/2d.",
    )
    parser.add_argument(
        "--layout-mode",
        choices=["2d/3d", "2d/2d"],
        default="2d/3d",
        help="Input layout mode for grouped MM benchmark.",
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        default=False,
        help="Run C++ and CuTeDSL timings in separate subprocesses per shape (recommended for precise measurement).",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        default=False,
        help="Capture backend call in a CUDA graph and benchmark replay().",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations for timing loops.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=20,
        help="Measured iterations for timing loops.",
    )
    parser.add_argument(
        "--paired-trials",
        type=int,
        default=1,
        help="Number of paired timing trials per shape (C++ then CuTeDSL). Reports median timings and median per-trial speedup.",
    )
    parser.add_argument(
        "--set-max-gpu-clocks",
        action="store_true",
        default=False,
        help="Attempt to enable persistence mode and lock SM/MEM clocks to max via nvidia-smi once at startup.",
    )
    parser.add_argument(
        "--backend",
        choices=["both", "cpp", "cute"],
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-us-only",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        default=False,
        help="Skip correctness check.",
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
        use_subprocess=args.use_subprocess,
        use_cuda_graphs=args.use_cuda_graphs,
        warmup=args.warmup,
        rep=args.rep,
        paired_trials=args.paired_trials,
        backend=args.backend,
        emit_us_only=args.emit_us_only,
        do_correctness=not args.no_correctness,
        mma_tile_mn=args.mma_tile_mn,
        cluster_shape_mn=args.cluster_shape_mn,
        transpose_ab=(None if args.transpose_ab is None else args.transpose_ab == "on"),
        set_max_gpu_clocks=args.set_max_gpu_clocks,
        layout_mode=args.layout_mode,
    )
