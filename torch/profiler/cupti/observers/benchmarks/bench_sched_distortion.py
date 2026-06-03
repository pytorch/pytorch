
"""Profiled-step distortion, matching the CUPTI-monitor prototype's test.

Where ``bench_distortion`` measures one steady block with the observer active
throughout: a ``torch.profiler`` schedule of ``wait=0, warmup=1, active=1`` (repeated for
statistics), reporting the **warmup-step** and **active-step** GPU-time
distortion separately, for the stock kineto profiler vs our mux backend
(``profile(cupti_mux=True)``), with and without HES. Same workload knobs as
bench_observers (default multistream_mixed, groups=256 -> 5376 kernels).

Run with ``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat``. Each config runs in its
own subprocess (HES is process-global; CUPTI state must be clean per config).
"""

import bench_observers as _bo  # noqa: E402  -- preloads v2 libcupti before torch

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402
from torch.profiler import cupti, ProfilerActivity, profile, schedule  # noqa: E402


_RESULT_PREFIX = _bo._RESULT_PREFIX
_CONFIGS = ["baseline", "kineto", "kineto_hw", "mux", "mux_hw"]


def _gpu_ms_single(step_fn) -> float:
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    step_fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def _run_worker(args) -> dict:
    cfg = args.config
    if cfg.endswith("_hw"):
        cupti.enable_hes_early()
    torch.cuda.init()
    step_fn = _bo._make_step(args)
    for _ in range(args.warmup_steps):  # untimed warmup (graph capture etc.)
        step_fn()
    torch.cuda.synchronize()

    # In-process baseline (no profiler) FIRST, so distortion is relative to the
    # same GPU clock state -- avoids cross-subprocess variance.
    base_ms = statistics.median([_gpu_ms_single(step_fn) for _ in range(4 * args.cycles)])
    if cfg == "baseline":
        return {"config": cfg, "base_ms": base_ms}

    # warmup=1/active=1 repeated `cycles` times -> alternating warmup/active.
    sched = schedule(wait=0, warmup=1, active=1, repeat=args.cycles)
    use_mux = cfg.startswith("mux")
    warmup_ms: list[float] = []
    active_ms: list[float] = []
    p = profile(
        activities=[ProfilerActivity.CUDA], schedule=sched, cupti_mux=use_mux
    )
    p.__enter__()
    try:
        for i in range(2 * args.cycles):
            t = _gpu_ms_single(step_fn)
            (warmup_ms if i % 2 == 0 else active_ms).append(t)
            p.step()
    finally:
        p.__exit__(None, None, None)
    w = statistics.median(warmup_ms)
    a = statistics.median(active_ms)
    return {
        "config": cfg,
        "hes_enabled": cupti.hes_enabled(),
        "base_ms": base_ms,
        "warmup_ms": w,
        "warmup_pct": round((w - base_ms) / base_ms * 100, 2),
        "active_ms": a,
        "active_pct": round((a - base_ms) / base_ms * 100, 2),
    }


def _run_config_subprocess(cfg: str, args) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--config", cfg,
        "--workload", args.workload,
        "--groups", str(args.groups),
        "--tensor-size", str(args.tensor_size),
        "--sleep-cycles", str(args.sleep_cycles),
        "--sleep-cycles-side", str(args.sleep_cycles_side),
        "--warmup-steps", str(args.warmup_steps),
        "--cycles", str(args.cycles),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.mode_timeout
        )
    except subprocess.TimeoutExpired as e:
        return {"config": cfg, "error": f"timeout {args.mode_timeout}s",
                "stderr_tail": (e.stderr or "")[-1500:] if isinstance(e.stderr, str) else ""}
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX):])
    return {"config": cfg, "error": f"no result (rc={proc.returncode})",
            "stderr_tail": (proc.stderr or proc.stdout)[-1500:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=_CONFIGS, default=None)
    parser.add_argument("--workload", choices=["mixed", "multistream"], default="multistream")
    parser.add_argument("--groups", type=int, default=256)
    parser.add_argument("--tensor-size", type=int, default=2048)
    parser.add_argument("--sleep-cycles", type=int, default=180000)
    parser.add_argument("--sleep-cycles-side", type=int, default=120000)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--mode-timeout", type=int, default=600)
    args = parser.parse_args()

    if args.config is not None:
        print(_RESULT_PREFIX + json.dumps(_run_worker(args)))
        return

    results = {}
    for cfg in _CONFIGS:
        print(f"[sched-distortion] {cfg} ...", file=sys.stderr, flush=True)
        results[cfg] = _run_config_subprocess(cfg, args)

    rows = []
    for cfg in _CONFIGS:
        r = results[cfg]
        if "error" in r:
            rows.append(r)
        elif cfg == "baseline":
            rows.append({"config": cfg, "base_ms": round(r["base_ms"], 3)})
        else:
            rows.append({
                "config": cfg,
                "base_ms": round(r["base_ms"], 3),
                "warmup_pct": r["warmup_pct"],
                "active_pct": r["active_pct"],
            })
    print(json.dumps({"results": rows}, indent=2))


if __name__ == "__main__":
    main()
