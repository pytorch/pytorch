
"""Profiler *distortion* benchmark for the CUPTI mux observers.

Where bench_observers measures host-side step overhead (wall clock), this
measures the *observer effect* on the GPU itself: the ground-truth GPU
execution time of the workload (via CUDA events) with and without collection,
so you can see how much CUPTI instrumentation perturbs the kernels it watches.

Modes (each in its own subprocess, for HES isolation):

  baseline    : no collection (the reference GPU time)
  timer       : always-on NodeTimerObserver over all kinds
  timer_hw    : + HES (hardware kernel timestamps)
  profiler    : always-on rich ProfilerObserver (full metadata)
  profiler_hw : + HES

The driver reports each mode's GPU ms/replay and its distortion vs baseline
(% slowdown of the GPU work). For the timer modes it also reports the summed
kernel duration CUPTI *attributes* per replay, so you can compare what the
profiler thinks the GPU did against the CUDA-event ground truth.

Run with ``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat``; ``--workload``,
``--groups`` etc. are shared with bench_observers (imported for the workloads).
"""

# Import bench_observers FIRST: its module top loads the v2 libcupti before
# torch (so the mux resolves it) and gives us the workload builders + helpers.
import bench_observers as _bo  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.profiler import cupti  # noqa: E402


_MODES = ["baseline", "kineto", "timer", "timer_hw", "profiler", "profiler_hw"]
_RESULT_PREFIX = _bo._RESULT_PREFIX


def _gpu_ms(step_fn, reps: int) -> float:
    """GPU execution time per replay (ms), measured with CUDA events -- this
    is on-device time, independent of host scheduling overhead."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        step_fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _make_observer(mode: str):
    """The observer to keep alive during the run (None for baseline). Created
    BEFORE the graph is captured (see bench_observers) so replays are
    instrumented."""
    if mode == "baseline":
        return None
    from torch.profiler.cupti import types as T
    from torch.profiler.cupti.observers import NodeTimerObserver, ProfilerObserver

    if mode.startswith("timer"):
        return NodeTimerObserver(
            [T.ActivityKind.CONCURRENT_KERNEL, T.ActivityKind.MEMCPY, T.ActivityKind.MEMSET]
        )
    return ProfilerObserver()


def _run_worker(args) -> dict:
    if args.mode == "kineto":
        # The STANDARD torch profiler (kineto's own CUPTI path), not our mux --
        # the apples-to-apples comparison. Capture the graph inside the profile
        # window so its kernels are instrumented.
        from torch.profiler import profile, ProfilerActivity

        torch.cuda.init()
        prof = profile(activities=[ProfilerActivity.CUDA])
        prof.__enter__()
        try:
            step_fn = _bo._make_step(args)
            for _ in range(args.warmup_steps):
                step_fn()
            torch.cuda.synchronize()
            gpu = _bo.summarize(
                [_gpu_ms(step_fn, args.measure_steps) for _ in range(args.samples)]
            )
        finally:
            prof.__exit__(None, None, None)
        return {
            "mode": args.mode,
            "hes_enabled": False,
            "gpu_ms": gpu,
            "reported_kernel_us_per_replay": None,
        }

    if args.mode.endswith("_hw"):
        cupti.enable_hes_early()
    torch.cuda.init()

    obs = _make_observer(args.mode)
    if obs is not None and not obs.available:
        return {"mode": args.mode, "skipped": "CUPTI mux unavailable"}
    try:
        step_fn = _bo._make_step(args)
        for _ in range(args.warmup_steps):
            step_fn()
        torch.cuda.synchronize()
        gpu = _bo.summarize([_gpu_ms(step_fn, args.measure_steps) for _ in range(args.samples)])
        reported_us = None
        if obs is not None and args.mode.startswith("timer"):
            # Summed kernel duration CUPTI attributes per replay (ground-truth
            # comparison): drain a clean window of measure_steps replays.
            obs.drain(flush=True)
            for _ in range(args.measure_steps):
                step_fn()
            torch.cuda.synchronize()
            _g, _s, _e = obs.drain(flush=True)  # (graph_node_id, start, end) cols
            total_ns = int((_e - _s).clip(min=0).sum())
            reported_us = total_ns / 1000.0 / args.measure_steps
    finally:
        if obs is not None:
            obs.close()

    return {
        "mode": args.mode,
        "hes_enabled": cupti.hes_enabled(),
        "gpu_ms": gpu,
        "reported_kernel_us_per_replay": reported_us,
    }


def _run_mode_subprocess(mode: str, args) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode",
        mode,
        "--workload",
        args.workload,
        "--groups",
        str(args.groups),
        "--tensor-size",
        str(args.tensor_size),
        "--sleep-cycles",
        str(args.sleep_cycles),
        "--sleep-cycles-side",
        str(args.sleep_cycles_side),
        "--warmup-steps",
        str(args.warmup_steps),
        "--samples",
        str(args.samples),
        "--measure-steps",
        str(args.measure_steps),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.mode_timeout
        )
    except subprocess.TimeoutExpired as e:
        tail = e.stderr if isinstance(e.stderr, str) else ""
        return {"mode": mode, "error": f"timed out after {args.mode_timeout}s",
                "stderr_tail": (tail or "")[-2000:]}
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX) :])
    return {
        "mode": mode,
        "error": f"no result (returncode={proc.returncode})",
        "stderr_tail": (proc.stderr or proc.stdout)[-2000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=_MODES, default=None)
    parser.add_argument("--workload", choices=["mixed", "multistream"], default="mixed")
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--tensor-size", type=int, default=2048)
    parser.add_argument("--sleep-cycles", type=int, default=180000)
    parser.add_argument("--sleep-cycles-side", type=int, default=120000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--mode-timeout", type=int, default=600)
    args = parser.parse_args()

    if args.mode is not None:
        result = _run_worker(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(_RESULT_PREFIX + json.dumps(result))
        return

    if "cuda-compat" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print(
            "[distortion] warning: 'cuda-compat' not on LD_LIBRARY_PATH; CUPTI "
            "modes may hang or report unavailable.",
            file=sys.stderr,
            flush=True,
        )
    results: dict[str, dict] = {}
    for i, m in enumerate(_MODES, 1):
        print(f"[distortion] ({i}/{len(_MODES)}) {m} ...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        results[m] = _run_mode_subprocess(m, args)
        dt = time.perf_counter() - t0
        note = results[m].get("error") or results[m].get("skipped") or "ok"
        print(f"[distortion]     {m} done in {dt:.1f}s ({note})", file=sys.stderr, flush=True)

    # Distortion = GPU-time slowdown vs the baseline reference.
    base = results.get("baseline", {}).get("gpu_ms", {}).get("median_ms")
    out = []
    for m in _MODES:
        r = results[m]
        gpu = r.get("gpu_ms", {}).get("median_ms")
        entry = {"mode": m, "gpu_ms": gpu, **{k: r[k] for k in ("error", "skipped") if k in r}}
        if base and gpu is not None:
            entry["gpu_distortion_pct"] = round((gpu / base - 1.0) * 100, 2)
        if r.get("reported_kernel_us_per_replay") is not None:
            entry["reported_kernel_us_per_replay"] = round(
                r["reported_kernel_us_per_replay"], 1
            )
        out.append(entry)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
