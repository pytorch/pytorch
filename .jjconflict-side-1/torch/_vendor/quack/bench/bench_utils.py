"""Shared helpers for triton perf_report-based benchmarks."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import pandas as pd
    from triton.testing import Benchmark


def run_and_print(mark, save_path=None):
    """Run a triton ``Mark`` (from ``perf_report``) and print/save results.

    Each runner is expected to return a ``dict[str, Any]`` mapping stat name to
    value, e.g. ``{"ms": 0.123, "GB/s": 1234}``. All providers in a benchmark
    must return the same set of keys. Values are written through unchanged --
    rounding/formatting is the caller's responsibility.

    Output columns are ``x_names + [f"{line_name} ({stat})" for ...]``.
    """
    benchmarks = mark.benchmarks if isinstance(mark.benchmarks, list) else [mark.benchmarks]
    for bench in benchmarks:
        df = _run_one(mark.fn, bench)
        print(bench.plot_name + ":")
        print(df.to_string())
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), index=False)


def _run_one(fn, bench: Benchmark) -> pd.DataFrame:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required to format benchmark results. "
            "Install it with `pip install pandas` or `pip install -e '.[bench]'`."
        ) from e

    x_names = list(bench.x_names)
    rows = []
    stat_keys = None  # locked in from the first runner result
    for x in bench.x_vals:
        if not isinstance(x, (list, tuple)):
            x = [x] * len(x_names)
        x_args = dict(zip(x_names, x))
        row = list(x)
        for line_val in bench.line_vals:
            stats = fn(**x_args, **{bench.line_arg: line_val}, **bench.args)
            if not isinstance(stats, dict):
                raise TypeError(f"runner must return dict[str, Any], got {type(stats).__name__}")
            if stat_keys is None:
                stat_keys = list(stats.keys())
            elif list(stats.keys()) != stat_keys:
                raise ValueError(f"runner returned keys {list(stats.keys())}, expected {stat_keys}")
            row.extend(stats[k] for k in stat_keys)
        rows.append(row)
    cols = list(x_names) + [
        f"{name} ({stat})" for name in bench.line_names for stat in (stat_keys or [])
    ]
    return pd.DataFrame(rows, columns=cols)


def _bench_cuda_graph_l2_rotate(
    fn,
    arg_sets,
    kwarg_sets,
    extra_kwargs,
    warmup_target_ms: float = 200.0,
    n_timed_calls: int = 200,
    quantiles=None,
):
    """L2-cold single-replay CUDA-graph benchmark.

    Warmup is time-based: probe a single kernel launch to estimate the
    per-call cost, then iterate round-robin over the pre-cloned
    ``(arg_sets[i], kwarg_sets[i])`` pairs as a plain Python loop for
    ``warmup_target_ms`` of wall-clock GPU work. Heavy pipelined configs
    (TMA + smem_stages=3) need enough warmup to drain the pipeline-fill
    phase or the timed window catches them artificially fast - a fixed
    count of warmup launches underwarms heavy configs while overpaying
    for cheap ones.

    The timed window is a single ``graph.replay()`` of a captured CUDA
    graph whose body records ``n_timed_calls`` round-robin invocations -
    no Python loop, no per-launch CPU overhead - so the measurement is
    just GPU work / total recorded calls.

    Round-robin over fresh tensor sets defeats the L2-resident caching
    that inflates short-kernel timing under ``triton.testing.do_bench``
    (which calls the kernel on the same single tensor each iteration).
    For cache-cold production workloads the round-robin number predicts
    real-world latency; the L2-hot number favours wider layouts / deeper
    smem stages that don't actually win once data has to come from HBM.

    ``fn`` is called as ``fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)``
    once per recorded launch. ``extra_kwargs`` holds the per-config kwargs
    (e.g. ``{"config": <RmsNormBwdConfig>}``) which don't need cloning;
    keys in ``extra_kwargs`` must not overlap with ``kwarg_sets[i]``.
    Returns ms/call, or a ``len(quantiles)``-list replicating that value
    when ``quantiles`` is provided (for API parity with
    ``triton.testing.do_bench``).
    """
    n_sets = len(arg_sets)
    # Round timed-call count to a multiple of n_sets for even L2 turnover.
    rounds_timed = max(1, n_timed_calls // n_sets)
    total_timed_calls = rounds_timed * n_sets

    # A few priming launches so the probe doesn't catch first-launch
    # driver / kernel-load overhead.
    for _ in range(3):
        fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)
    torch.cuda.synchronize()

    # Probe a single launch to estimate per-call ms; size the warmup loop
    # to hit ``warmup_target_ms`` of GPU work regardless of kernel cost.
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)
    probe_end.record()
    torch.cuda.synchronize()
    est_ms = max(probe_start.elapsed_time(probe_end), 1e-3)
    n_warmup_calls = max(50, int(warmup_target_ms / est_ms))

    # Warmup: plain Python loop over rotating sets, no graph capture.
    for i in range(n_warmup_calls):
        idx = i % n_sets
        fn(*arg_sets[idx], **kwarg_sets[idx], **extra_kwargs)
    torch.cuda.synchronize()

    # Capture timed graph: a single replay covers all timed kernel launches.
    timed_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(timed_graph):
        for _ in range(rounds_timed):
            for i in range(n_sets):
                fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    timed_graph.replay()
    end_evt.record()
    torch.cuda.synchronize()
    ms = start_evt.elapsed_time(end_evt) / total_timed_calls

    if quantiles:
        return [ms for _ in quantiles]
    return ms


def _clone_l2_rotate_inputs(args, kwargs, n_buffers: int):
    """Clone tensor args AND tensor kwargs ``n_buffers`` times.

    Returns ``(arg_sets, kwarg_sets)``: ``arg_sets[i]`` is a tuple matching
    ``args``' positional shape with tensors cloned to fresh memory;
    ``kwarg_sets[i]`` is a dict matching ``kwargs``' keys with tensor values
    cloned. Non-tensor values (ints, strings, None, dataclasses, etc.) are
    shared across all sets (no clone).

    Both args and kwargs are cloned so that every recorded launch in the
    L2-cold round-robin CUDA graph touches distinct GMEM addresses,
    including for write-target kwargs like ``dw_partial`` / ``dx``.
    """
    arg_sets = []
    kwarg_sets = []
    for _ in range(n_buffers):
        arg_sets.append(tuple(a.clone() if isinstance(a, Tensor) else a for a in args))
        kwarg_sets.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in kwargs.items()})
    return arg_sets, kwarg_sets


def _pick_l2_rotate_count(
    args, kwargs, target_ratio: int = 3, min_buffers: int = 4, max_buffers: int = 16
):
    """Pick ``n_bufs`` so cloned input bytes per round exceed
    ``target_ratio * L2_size`` (defeats L2 reuse), capped by HBM headroom and
    [min_buffers, max_buffers]. Counts tensor bytes across both ``args`` and
    ``kwargs`` so write-target kwargs (``dw_partial``, ``dx``, etc.) are
    included in the L2-turnover calculation.
    """
    if not torch.cuda.is_available():
        return min_buffers
    tensor_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, Tensor)) + sum(
        v.numel() * v.element_size() for v in kwargs.values() if isinstance(v, Tensor)
    )
    if tensor_bytes == 0:
        return min_buffers
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_size = props.L2_cache_size
    n_by_l2 = (target_ratio * l2_size + tensor_bytes - 1) // tensor_bytes
    free_bytes, _ = torch.cuda.mem_get_info()
    # Leave half of free memory headroom for the kernel's own scratch + the
    # user's other allocations.
    n_by_mem = max(1, int(free_bytes * 0.5) // tensor_bytes)
    return max(min_buffers, min(max_buffers, min(n_by_l2, n_by_mem)))
