"""CUPTI monitor *distortion* benchmark.

Measures how much each profiling approach perturbs a CUDA-graph workload's step
time, so the monitor's overhead can be compared against stock Kineto and against
a no-profiling baseline. Modes:

  baseline            : graph replay only (the reference step time)
  monitor[_hw]        : the CUPTI monitor running always-on (subscription +
                        per-buffer columnar decode + observer dispatch over the full
                        profiler field selection), outside any torch.profiler window
  node_timer[_hw]     : the monitor always-on over just the NodeTimerObserver timing
                        fields (single kind, vectorized decode -> the lower bound)
  stock_window[_hw]   : a stock Kineto torch.profiler window
  monitor_window[_hw] : a torch.profiler window using the cupti_monitor backend

``_hw`` variants enable HES (hardware kernel timestamps) before CUDA init.

Ported from ``cupti_monitor_bench.py``. API changes:
  * ``torch.profiler._cupti_monitor`` -> ``torch.profiler._cupti.monitor``
    (``enable_hes_early`` / ``is_hes_enabled`` are unchanged).
  * the old ``start_collection(output_dir)`` raw-dump-to-disk path is gone -- the
    new monitor is observer-based and always decodes. ``monitor`` now
    registers a no-op observer over the full profiler field selection, so the cost
    it reports is subscription + decode + dispatch (there is no raw-dump mode to
    measure).
  * the ``cupti_monitor`` backend now exports ASYNCHRONOUSLY. ``start_trace`` /
    ``stop_trace`` only stamp native-clock window boundaries -- no device sync inside
    the measured window (the distortion win), so ``active_step`` and
    ``context_exit_ms`` no longer carry a fence. ``export_chrome_trace`` just
    registers the output; the deferred flush + merge + write happens in
    ``prof.wait_for_exports()``. The window modes therefore report ``export_ms`` (the
    cheap deferred-setup call) and ``wait_exports_ms`` (where the real cost lands)
    separately; for stock Kineto ``wait_for_exports`` is a no-op and the write is in
    ``export_ms`` as before.

Run on a host with libcupti >= 13.3 visible to the monitor, e.g.
``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat python benchmarks/profiler_benchmark/cupti_monitor_bench.py --mode monitor_window``.
"""

import argparse
import json
import os
import shutil
import statistics
import tempfile
import threading
import time
from pathlib import Path

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity, schedule
from torch.profiler._cupti import monitor as cupti_monitor


def build_mixed_graph(
    groups: int, tensor_size: int, sleep_cycles: int, layers: int
):
    x = torch.randn(tensor_size, device="cuda")
    y = torch.randn(tensor_size, device="cuda")
    for _ in range(20):
        for _ in range(8):
            x.add_(y)
            x.relu_()
        torch.cuda._sleep(sleep_cycles)
        torch.cuda._sleep(sleep_cycles)
    torch.cuda.synchronize()

    # ``layers`` stacks the whole group-pattern, deepening the captured graph so the
    # step runs proportionally longer (more records per window) without changing the
    # per-record work -- lets the distortion be read as a function of step length.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(layers):
            for _ in range(groups):
                for _ in range(8):
                    x.add_(y)
                    x.relu_()
                torch.cuda._sleep(sleep_cycles)
                torch.cuda._sleep(sleep_cycles)
    torch.cuda.synchronize()
    return g


def build_multistream_mixed_graph(
    groups: int,
    tensor_size: int,
    sleep_cycles_main: int,
    sleep_cycles_side: int,
    layers: int,
):
    x_main = torch.randn(tensor_size, device="cuda")
    y_main = torch.randn(tensor_size, device="cuda")
    x_side = torch.randn(tensor_size, device="cuda")
    y_side = torch.randn(tensor_size, device="cuda")
    side_stream = torch.cuda.Stream()

    for _ in range(20):
        x_main.add_(y_main)
        x_main.relu_()
        capture_stream = torch.cuda.current_stream()
        with torch.cuda.stream(side_stream):
            side_stream.wait_stream(capture_stream)
            for _ in range(4):
                x_side.add_(y_side)
                x_side.relu_()
            torch.cuda._sleep(sleep_cycles_side)
        for _ in range(4):
            x_main.add_(y_main)
            x_main.relu_()
        torch.cuda._sleep(sleep_cycles_main)
        capture_stream.wait_stream(side_stream)
        x_main.mul_(1.0001)
    torch.cuda.synchronize()

    # ``layers`` stacks the whole group-pattern, deepening the captured graph so the
    # step runs proportionally longer (more records per window) without changing the
    # per-record work -- lets the distortion be read as a function of step length.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        capture_stream = torch.cuda.current_stream()
        for _ in range(layers):
            for _ in range(groups):
                x_main.add_(y_main)
                x_main.relu_()
                with torch.cuda.stream(side_stream):
                    side_stream.wait_stream(capture_stream)
                    for _ in range(4):
                        x_side.add_(y_side)
                        x_side.relu_()
                    torch.cuda._sleep(sleep_cycles_side)
                for _ in range(4):
                    x_main.add_(y_main)
                    x_main.relu_()
                torch.cuda._sleep(sleep_cycles_main)
                capture_stream.wait_stream(side_stream)
                x_main.mul_(1.0001)
    torch.cuda.synchronize()
    return g


def make_workload(args):
    if args.workload == "mixed":
        graph = build_mixed_graph(
            args.mixed_groups, args.tensor_size, args.sleep_cycles, args.layers
        )
    else:
        graph = build_multistream_mixed_graph(
            args.mixed_groups,
            args.tensor_size,
            args.sleep_cycles_main,
            args.sleep_cycles_side,
            args.layers,
        )

    def run_step():
        graph.replay()

    return run_step


def time_step(step_fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def time_step_block(step_fn, steps: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / steps


def summarize(samples):
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_baseline(step_fn, warmup_steps: int, samples: int, measure_steps: int = 1):
    for _ in range(warmup_steps):
        step_fn()
    torch.cuda.synchronize()
    if measure_steps == 1:
        values = [time_step(step_fn) for _ in range(samples)]
    else:
        values = [time_step_block(step_fn, measure_steps) for _ in range(samples)]
    return summarize(values)


def _profiler_fields():
    from torch.profiler._cupti.observers.profiler import PROFILER_FIELDS

    return PROFILER_FIELDS


def _node_timer_fields():
    # The NodeTimerObserver field selection: just the compact kernel timing fields.
    # Far fewer fields than PROFILER_FIELDS and a single kind -> the monitor's
    # vectorized stride decode, so its always-on cost should be the lower bound.
    from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

    from torch.profiler._cupti.records import Kernel

    return {
        ActivityKind.CONCURRENT_KERNEL: {
            Kernel.START,
            Kernel.END,
            Kernel.GRAPH_NODE_ID,
        }
    }


def run_monitor(step_fn, warmup_steps: int, samples: int, measure_steps: int, fields):
    # The monitor is observer-based: its always-on cost is the CUPTI subscription
    # + per-buffer columnar decode + per-observer dispatch, so register a no-op
    # observer over ``fields`` and run the workload under it. flush_period_s=0.0
    # means no background flush thread (buffers deliver as they fill), isolating the
    # monitor's collection overhead from any flushing.
    mon = cupti_monitor.CuptiMonitor(flush_period_s=0.0)
    obs = mon.register(fields, lambda _cols: None)
    try:
        for _ in range(warmup_steps):
            step_fn()
        torch.cuda.synchronize()
        values = [time_step_block(step_fn, measure_steps) for _ in range(samples)]
        return summarize(values)
    finally:
        mon.unregister(obs)  # drops the last observer -> stops + tears down the monitor


def make_experimental_config(mode: str):
    kwargs = {"trace_only": True}
    if mode.startswith("monitor_window"):
        kwargs["custom_profiler_config"] = json.dumps({"backend": "cupti_monitor"})
    return _ExperimentalConfig(**kwargs)


def run_window(
    step_fn,
    mode: str,
    samples: int,
):
    active_times = []
    warmup_times = []
    exit_times = []
    export_times = []
    wait_export_times = []

    for _ in range(samples):
        temp_root = Path(tempfile.mkdtemp(prefix=f"{mode}_"))
        trace_path = temp_root / "trace.json.gz"
        cfg = make_experimental_config(mode)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            experimental_config=cfg,
        )
        prof.__enter__()
        try:
            warmup_times.append(time_step(step_fn))
            prof.step()
            active_times.append(time_step(step_fn))
            prof.step()
        finally:
            # cupti_monitor: stop_trace only stamps the window boundary (no device
            # sync), so this no longer carries the fence the old sync-drain did.
            t0 = time.perf_counter()
            prof.__exit__(None, None, None)
            exit_times.append((time.perf_counter() - t0) * 1e3)

        # cupti_monitor: export_chrome_trace just registers the output path + captures
        # the Kineto CPU trace -- the merge + write is deferred. Stock Kineto writes
        # the gzip here as before.
        t1 = time.perf_counter()
        prof.export_chrome_trace(str(trace_path))
        export_times.append((time.perf_counter() - t1) * 1e3)

        # cupti_monitor: the deferred flush + merge + write lands here (off the
        # measured window, and callable off the training thread). No-op for stock.
        t2 = time.perf_counter()
        prof.wait_for_exports()
        wait_export_times.append((time.perf_counter() - t2) * 1e3)

        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "warmup_step": summarize(warmup_times),
        "active_step": summarize(active_times),
        "context_exit_ms": summarize(exit_times),
        "export_ms": summarize(export_times),
        "wait_exports_ms": summarize(wait_export_times),
    }


def run_window_async(step_fn, mode: str, samples: int, post_steps: int):
    """Full-async export (cupti_monitor backend only): hand the deferred finalize to a
    background thread and keep 'training' on the main thread, the way l4x's
    trace_handler does it. Measures what actually costs the training thread now --
    versus run_window's synchronous ``wait_exports_ms``:

      main_handoff_ms  : main-thread cost of export_chrome_trace (deferred) +
                         take_pending_cupti_export + spawning the worker -- the only
                         thing that still blocks the caller.
      post_window_step : step times WHILE the background finalize runs; should stay
                         ~= baseline (no hitch) since the finalize is off-thread and
                         does no flush (it rides the workload's natural delivery).
      bg_finalize_ms   : wall time of the off-thread join(force=False) (informational;
                         not on the training thread's critical path).
    """
    active_times = []
    handoff_times = []
    post_step_times = []
    bg_finalize_times = []

    for _ in range(samples):
        temp_root = Path(tempfile.mkdtemp(prefix=f"{mode}_"))
        trace_path = temp_root / "trace.json.gz"
        cfg = make_experimental_config(mode)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            experimental_config=cfg,
        )
        prof.__enter__()
        try:
            time_step(step_fn)  # warmup
            prof.step()
            active_times.append(time_step(step_fn))
            prof.step()
        finally:
            prof.__exit__(None, None, None)

        result: dict = {}
        t0 = time.perf_counter()
        prof.export_chrome_trace(str(trace_path))  # deferred
        obs = prof.take_pending_cupti_export()

        def _finalize(obs=obs, result=result):
            tb = time.perf_counter()
            if obs is not None:
                # force=False: no flush -> safe off-thread; relies on the workload's
                # continuing activity (the post_steps below) covering the window.
                obs.join(force=False)
            result["ms"] = (time.perf_counter() - tb) * 1e3

        th = threading.Thread(target=_finalize, daemon=True)
        th.start()
        handoff_times.append((time.perf_counter() - t0) * 1e3)

        # "Training" continues on the main thread, unblocked, while the worker
        # finalizes; its activity is what naturally covers the window boundary.
        for _ in range(post_steps):
            post_step_times.append(time_step(step_fn))

        th.join()
        bg_finalize_times.append(result.get("ms", float("nan")))
        shutil.rmtree(temp_root, ignore_errors=True)

    return {
        "active_step": summarize(active_times),
        "main_handoff_ms": summarize(handoff_times),
        "post_window_step": summarize(post_step_times),
        "bg_finalize_ms": summarize(bg_finalize_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "monitor",
            "monitor_hw",
            "node_timer",
            "node_timer_hw",
            "stock_window",
            "stock_window_hw",
            "monitor_window",
            "monitor_window_hw",
            "monitor_window_async",
            "monitor_window_async_hw",
        ],
        required=True,
    )
    parser.add_argument(
        "--workload",
        choices=["mixed", "multistream_mixed"],
        default="multistream_mixed",
    )
    parser.add_argument("--mixed-groups", type=int, default=256)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="stack the group-pattern N times -> deeper graph / longer step",
    )
    parser.add_argument("--tensor-size", type=int, default=2048)
    parser.add_argument("--sleep-cycles", type=int, default=180000)
    parser.add_argument("--sleep-cycles-main", type=int, default=180000)
    parser.add_argument("--sleep-cycles-side", type=int, default=180000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument(
        "--post-steps",
        type=int,
        default=40,
        help="async modes: steps to run on the main thread while the bg finalize runs",
    )
    args = parser.parse_args()

    if "async" in args.mode:
        # Short background flush period so the post-window workload's activity is
        # delivered promptly -> the off-thread join(force=False) covers the window
        # without a long wait. Must be set before the monitor singleton is created.
        os.environ.setdefault("TORCH_CUPTI_MONITOR_FLUSH_PERIOD_S", "0.05")

    if args.mode.endswith("_hw"):
        cupti_monitor.enable_hes_early()

    torch.cuda.init()
    step_fn = make_workload(args)

    result = {
        "mode": args.mode,
        "workload": args.workload,
        "hes_enabled": cupti_monitor.is_hes_enabled(),
    }
    if args.mode == "baseline":
        result["baseline"] = run_baseline(
            step_fn,
            args.warmup_steps,
            args.samples,
            args.measure_steps,
        )
    elif args.mode in {"monitor", "monitor_hw"}:
        result["monitor"] = run_monitor(
            step_fn,
            args.warmup_steps,
            args.samples,
            args.measure_steps,
            _profiler_fields(),
        )
    elif args.mode in {"node_timer", "node_timer_hw"}:
        result["node_timer"] = run_monitor(
            step_fn,
            args.warmup_steps,
            args.samples,
            args.measure_steps,
            _node_timer_fields(),
        )
    elif args.mode.startswith("monitor_window_async"):
        result["window"] = run_window_async(
            step_fn,
            args.mode,
            args.samples,
            args.post_steps,
        )
    else:
        result["window"] = run_window(
            step_fn,
            args.mode,
            args.samples,
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
