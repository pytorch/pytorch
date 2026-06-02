# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Step-time overhead benchmark for the CUPTI mux observers.

Adapted from the original CuptiMonitor benchmark onto the mux interface:
measures per-step time for a CUDA-graph workload under

  baseline         : no collection
  kineto           : the STANDARD torch profiler (kineto's CUPTI path, not our
                     mux) -- the apples-to-apples comparison
  timed            : a NodeTimerObserver over all kinds for the whole run
                     (multi-kind => per-record KIND-walk decode)
  timed_vectorized : a kernel-only NodeTimerObserver (single-kind =>
                     vectorized stride decode)
  profiler         : an always-on rich ProfilerObserver (full per-event
                     metadata); also reports records collected, distinct
                     device/stream lanes, trace event count + build time
  profiled         : torch.profiler.profile(cupti_mux=True) over a window

``--workload multistream`` captures a main + side stream graph (exercises the
profiler's per-stream lanes); the default ``mixed`` is single-stream.

Append ``_hw`` to a mode to arm HES (hardware kernel timestamps) before the
CUDA context. Each mode runs in its own subprocess, so a single invocation
with no ``--mode`` runs them all (HES must be armed before CUDA init, so
non-HES and ``_hw`` modes cannot share a process). Run with
``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat`` and a torch build that includes
torch.profiler.cupti (CUPTI >= 13.2).
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# IMPORTANT: load the cupti-python wheel's v2-capable libcupti BEFORE torch, so
# the mux resolves it rather than torch's bundled (non-v2) libcupti. Without
# this the mux can arm against the wrong libcupti instance and silently collect
# nothing (the test harness does the same). Harmless in the driver process.
try:
    from cupti import cupti as _cupti_preload  # noqa: E402

    _h = _cupti_preload.subscribe(lambda *a: None, 0)
    _cupti_preload.unsubscribe(_h)
except Exception:
    pass

import torch  # noqa: E402
from torch.profiler import (  # noqa: E402
    cupti,
    profile,
    ProfilerActivity,
    schedule,
)


# Each mode runs in its own subprocess (see main()): HES must be armed before
# the CUDA context exists, so non-HES and _hw modes can't share a process.
_MODES = [
    "baseline",
    "kineto",
    "timed",
    "timed_hw",
    "timed_vectorized",
    "timed_vectorized_hw",
    "profiler",
    "profiler_hw",
    "profiled",
    "profiled_hw",
]
# Marker the single-mode worker prints so the driver can parse its result out
# of stdout regardless of torch/triton import noise.
_RESULT_PREFIX = "__BENCH_RESULT__"


def build_mixed_graph(groups: int, tensor_size: int, sleep_cycles: int):
    x = torch.randn(tensor_size, device="cuda")
    y = torch.randn(tensor_size, device="cuda")

    def body(n):
        for _ in range(n):
            for _ in range(8):
                x.add_(y)
                x.relu_()
            if sleep_cycles:  # 0 => clean compute-only trace (no spin kernel)
                torch.cuda._sleep(sleep_cycles)

    body(20)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body(groups)
    torch.cuda.synchronize()
    return g


def build_multistream_mixed_graph(
    groups: int, tensor_size: int, sleep_main: int, sleep_side: int
):
    """A CUDA-graph workload that issues kernels on a main + side stream, so
    activity lands on two stream lanes (exercises the profiler's per-stream
    placement)."""
    x_main = torch.randn(tensor_size, device="cuda")
    y_main = torch.randn(tensor_size, device="cuda")
    x_side = torch.randn(tensor_size, device="cuda")
    y_side = torch.randn(tensor_size, device="cuda")
    side = torch.cuda.Stream()

    def body():
        cur = torch.cuda.current_stream()
        for _ in range(groups):
            x_main.add_(y_main)
            x_main.relu_()
            with torch.cuda.stream(side):
                side.wait_stream(cur)
                for _ in range(4):
                    x_side.add_(y_side)
                    x_side.relu_()
                if sleep_side:  # 0 => no spin kernel (clean trace)
                    torch.cuda._sleep(sleep_side)
            for _ in range(4):
                x_main.add_(y_main)
                x_main.relu_()
            if sleep_main:
                torch.cuda._sleep(sleep_main)
            cur.wait_stream(side)
            x_main.mul_(1.0001)

    for _ in range(5):  # warmup (fewer iters than groups; just to settle)
        body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    torch.cuda.synchronize()
    return g


def _make_step(args):
    if args.workload == "multistream":
        graph = build_multistream_mixed_graph(
            args.groups, args.tensor_size, args.sleep_cycles, args.sleep_cycles_side
        )
    else:
        graph = build_mixed_graph(args.groups, args.tensor_size, args.sleep_cycles)
    return lambda: graph.replay()


def time_step(step_fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def time_step_block(step_fn, steps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / steps


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_baseline(make_step, warmup, samples, measure_steps):
    step_fn = make_step()
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    return summarize([time_step_block(step_fn, measure_steps) for _ in range(samples)])


def _run_timer(make_step, warmup, samples, measure_steps, kinds):
    from torch.profiler.cupti.observers import NodeTimerObserver

    # Create the observer (enabling the activity kinds) BEFORE building/capturing
    # the graph: CUPTI instruments graph nodes at capture time, so kinds enabled
    # only afterwards yield no records on replay.
    obs = NodeTimerObserver(kinds)
    if not obs.available:
        return {"skipped": "CUPTI mux unavailable"}
    try:
        step_fn = make_step()
        for _ in range(warmup):
            step_fn()
        torch.cuda.synchronize()
        return summarize(
            [time_step_block(step_fn, measure_steps) for _ in range(samples)]
        )
    finally:
        obs.close()


def run_timed(make_step, warmup, samples, measure_steps):
    # All kinds enabled => multi-kind era => per-record KIND-walk decode.
    k = cupti.types.ActivityKind
    return _run_timer(
        make_step,
        warmup,
        samples,
        measure_steps,
        [k.CONCURRENT_KERNEL, k.MEMCPY, k.MEMSET],
    )


def run_timed_vectorized(make_step, warmup, samples, measure_steps):
    # Kernel only => homogeneous era => vectorized stride decode.
    kinds = [cupti.types.ActivityKind.CONCURRENT_KERNEL]
    return _run_timer(make_step, warmup, samples, measure_steps, kinds)


def run_profiler(make_step, warmup, samples, measure_steps, trace_out=None, trace_steps=1):
    """Always-on rich ProfilerObserver: per-step overhead, plus collection
    stats (records per kind, distinct device/stream lanes, trace event count +
    build time). Writes a chrome://tracing JSON to ``trace_out`` if given --
    load it in chrome://tracing or ui.perfetto.dev. The trace covers only
    ``trace_steps`` replays (not the whole run), so the same kernels don't
    repeat once per replay into a giant trace."""
    from torch.profiler.cupti import types as T
    from torch.profiler.cupti.observers import ProfilerObserver
    from torch.profiler.cupti.observers.profiler import _to_chrome_trace

    # Observer before graph capture (see _run_timer) so replays are recorded.
    obs = ProfilerObserver()
    if not obs.available:
        return {"skipped": "CUPTI mux unavailable"}
    try:
        step_fn = make_step()
        for _ in range(warmup):
            step_fn()
        torch.cuda.synchronize()
        step = summarize(
            [time_step_block(step_fn, measure_steps) for _ in range(samples)]
        )
        # Capture a BOUNDED trace window: discard everything accumulated during
        # the overhead phase, then record exactly `trace_steps` replays. The
        # always-on observer otherwise holds every step's records, so the trace
        # would be the whole run (each kernel repeated once per replay).
        obs.drain()
        for _ in range(trace_steps):
            step_fn()
        torch.cuda.synchronize()
        data = obs.drain()
    finally:
        obs.close()

    k = T.ActivityKind
    device_stream = {
        k.CONCURRENT_KERNEL: (T.KernelField.DEVICE_ID, T.KernelField.STREAM_ID),
        k.MEMCPY: (T.MemcpyField.DEVICE_ID, T.MemcpyField.STREAM_ID),
        k.MEMSET: (T.MemsetField.DEVICE_ID, T.MemsetField.STREAM_ID),
    }
    counts: dict[int, int] = {}
    lanes: set[tuple[int, int]] = set()
    for kind, cols in data.items():
        if not cols:
            continue
        counts[int(kind)] = int(len(next(iter(cols.values()))))
        dfid, sfid = device_stream.get(kind, (None, None))
        dev, strm = cols.get(dfid), cols.get(sfid)
        if dev is not None and strm is not None:
            lanes.update(zip(dev.tolist(), strm.tolist()))

    t0 = time.perf_counter()
    trace = _to_chrome_trace(data)
    build_ms = (time.perf_counter() - t0) * 1e3
    out = {
        "active_step": step,
        "records": counts,
        "lanes": sorted(f"GPU{int(d)}/stream{int(s)}" for d, s in lanes),
        "n_trace_events": len(trace["traceEvents"]),
        "build_trace_ms": build_ms,
    }
    if trace_out:
        if trace_out.endswith(".gz"):
            import gzip

            with gzip.open(trace_out, "wt") as f:
                json.dump(trace, f)
        else:
            with open(trace_out, "w") as f:
                json.dump(trace, f)
        out["trace_file"] = os.path.abspath(trace_out)
    return out


def run_kineto(make_step, warmup, samples, measure_steps):
    """Standard torch profiler (kineto's own CUPTI path), NOT our mux -- the
    apples-to-apples host-overhead comparison. The graph is captured inside the
    profile window so its kernels are instrumented."""
    prof = profile(activities=[ProfilerActivity.CUDA])
    prof.__enter__()
    try:
        step_fn = make_step()
        for _ in range(warmup):
            step_fn()
        torch.cuda.synchronize()
        return summarize(
            [time_step_block(step_fn, measure_steps) for _ in range(samples)]
        )
    finally:
        prof.__exit__(None, None, None)


def run_profiled(make_step, samples, measure_steps):
    warmup_times, active_times, exit_times, export_times = [], [], [], []
    step_fn = None
    for _ in range(samples):
        temp_root = Path(tempfile.mkdtemp(prefix="cupti_mux_"))
        trace_path = temp_root / "trace.json"
        prof = profile(
            activities=[ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=measure_steps, repeat=1),
            cupti_mux=True,
        )
        prof.__enter__()
        try:
            if step_fn is None:
                # Capture the graph while the profiler (mux) is active so its
                # nodes are instrumented; reused for later samples (the mux
                # singleton persists, so replays stay recorded).
                step_fn = make_step()
            warmup_times.append(time_step(step_fn))
            prof.step()
            # Time the active window as a BLOCK (apples-to-apples with the other
            # modes' time_step_block; single-replay timing is ~2-5% noisier and
            # would falsely inflate the apparent overhead).
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(measure_steps):
                step_fn()
                prof.step()
            torch.cuda.synchronize()
            active_times.append((time.perf_counter() - t0) * 1e3 / measure_steps)
        finally:
            t0 = time.perf_counter()
            prof.__exit__(None, None, None)
            exit_times.append((time.perf_counter() - t0) * 1e3)
        t1 = time.perf_counter()
        prof.export_chrome_trace(str(trace_path))
        export_times.append((time.perf_counter() - t1) * 1e3)
        shutil.rmtree(temp_root, ignore_errors=True)
    return {
        "warmup_step": summarize(warmup_times),
        "active_step": summarize(active_times),
        "context_exit_ms": summarize(exit_times),
        "export_ms": summarize(export_times),
    }


def _run_mode_subprocess(mode: str, args) -> dict:
    """Run one mode in a fresh process and return its parsed result. A new
    process is required so HES can be (or not be) armed before CUDA init and
    so libcupti load order is clean per mode. Env (e.g. LD_LIBRARY_PATH) is
    inherited."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        mode,
        "--groups",
        str(args.groups),
        "--tensor-size",
        str(args.tensor_size),
        "--sleep-cycles",
        str(args.sleep_cycles),
        "--warmup-steps",
        str(args.warmup_steps),
        "--samples",
        str(args.samples),
        "--measure-steps",
        str(args.measure_steps),
        "--workload",
        args.workload,
        "--sleep-cycles-side",
        str(args.sleep_cycles_side),
    ]
    cmd += ["--trace-steps", str(args.trace_steps)]
    if args.trace_out:
        # Per-mode file so modes don't clobber each other; only profiler modes
        # actually write one.
        base, ext = os.path.splitext(args.trace_out)
        cmd += ["--trace-out", f"{base}.{mode}{ext or '.json'}"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.mode_timeout
        )
    except subprocess.TimeoutExpired as e:
        tail = e.stderr if isinstance(e.stderr, str) else ""
        return {
            "mode": mode,
            "error": f"timed out after {args.mode_timeout}s (CUPTI likely "
            "stalled -- is cuda-compat on LD_LIBRARY_PATH?)",
            "stderr_tail": (tail or "")[-2000:],
        }
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX) :])
    return {
        "mode": mode,
        "error": f"no result (returncode={proc.returncode})",
        "stderr_tail": (proc.stderr or proc.stdout)[-2000:],
    }


def _run_worker(args) -> dict:
    """Run a single mode in this process (the subprocess entry point)."""
    if args.mode.endswith("_hw"):
        cupti.enable_hes_early()

    torch.cuda.init()
    # A graph BUILDER (not a built graph): each run_* creates its observer first,
    # then calls this, so the graph is captured with CUPTI instrumentation active.
    make_step = lambda: _make_step(args)  # noqa: E731

    result = {"mode": args.mode, "hes_enabled": cupti.hes_enabled()}
    if args.mode == "baseline":
        result["result"] = run_baseline(
            make_step, args.warmup_steps, args.samples, args.measure_steps
        )
    elif args.mode == "kineto":
        result["result"] = run_kineto(
            make_step, args.warmup_steps, args.samples, args.measure_steps
        )
    elif args.mode in ("timed", "timed_hw"):
        result["result"] = run_timed(
            make_step, args.warmup_steps, args.samples, args.measure_steps
        )
    elif args.mode in ("timed_vectorized", "timed_vectorized_hw"):
        result["result"] = run_timed_vectorized(
            make_step, args.warmup_steps, args.samples, args.measure_steps
        )
    elif args.mode in ("profiler", "profiler_hw"):
        result["result"] = run_profiler(
            make_step,
            args.warmup_steps,
            args.samples,
            args.measure_steps,
            args.trace_out,
            args.trace_steps,
        )
    else:
        result["result"] = run_profiled(make_step, args.samples, args.measure_steps)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default=None,
        help="single mode to run; omit to run every mode, each in its own "
        "subprocess",
    )
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--tensor-size", type=int, default=2048)
    parser.add_argument("--sleep-cycles", type=int, default=180000)
    parser.add_argument("--sleep-cycles-side", type=int, default=120000)
    parser.add_argument(
        "--workload", choices=["mixed", "multistream"], default="mixed"
    )
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=1,
        help="number of replays the profiler-mode trace covers (default 1); "
        "keep small -- each replay repeats every kernel in the graph",
    )
    parser.add_argument(
        "--trace-out",
        default=None,
        help="write the profiler mode's chrome://tracing JSON here (.gz =>"
        " gzipped; driver mode suffixes it per mode); load in chrome://tracing"
        " or perfetto. For a clean compute-only trace pass --sleep-cycles 0"
        " --sleep-cycles-side 0 (the spin kernel otherwise dominates).",
    )
    parser.add_argument(
        "--mode-timeout",
        type=int,
        default=600,
        help="per-mode subprocess timeout in seconds (driver mode only)",
    )
    args = parser.parse_args()

    # Driver: no mode given -> fan out one subprocess per mode and aggregate.
    # Each subprocess is silent until it finishes (output is captured), so log
    # progress to stderr -- otherwise a multi-minute run looks hung.
    if args.mode is None:
        if "cuda-compat" not in os.environ.get("LD_LIBRARY_PATH", ""):
            print(
                "[bench] warning: 'cuda-compat' not on LD_LIBRARY_PATH; CUPTI "
                "modes may hang or report unavailable. Prefix the command with "
                "LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat.",
                file=sys.stderr,
                flush=True,
            )
        results = []
        for i, m in enumerate(_MODES, 1):
            print(f"[bench] ({i}/{len(_MODES)}) {m} ...", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            res = _run_mode_subprocess(m, args)
            dt = time.perf_counter() - t0
            note = res["error"] if "error" in res else "ok"
            print(f"[bench]     {m} done in {dt:.1f}s ({note})", file=sys.stderr, flush=True)
            results.append(res)
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    # Worker: run the one mode and emit both a human-readable block and the
    # machine-parseable marker line the driver looks for.
    result = _run_worker(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(_RESULT_PREFIX + json.dumps(result))


if __name__ == "__main__":
    main()
