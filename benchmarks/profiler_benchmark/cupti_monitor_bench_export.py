"""CUPTI monitor export-latency benchmark.

Measures ``torch.profiler.export_chrome_trace`` latency, comparing stock Kineto
(``--mode trunk``) against the cupti_monitor backend (``--mode monitor``).

The cupti_monitor backend exports synchronously by default (parity with stock Kineto:
``export_chrome_trace`` builds + writes the merged trace before returning,
``wait_for_exports`` is a no-op); set ``cupti_monitor_async_export`` to hand the merge
off-thread. Reported per mode:

  export_ms        : the export_chrome_trace cost (the part on the calling/training
                     thread; the full merge when sync, just the handoff when async)
  wait_exports_ms  : the deferred merge + gzip (async only; off-thread in real use)
  total_ms         : export_ms + wait_exports_ms (the total export work)

Each ``--mode`` runs in its own process so the monitor gets a clean CUPTI subscriber.
``--kernels`` scales the eager workload's kernel count -> the trace's event count.

``--trace-file PATH`` runs the real ``--mode monitor`` ``export_chrome_trace`` over a
loaded chrome trace's *exact* events, measuring export at a real trace's scale/shape: a
stand-in Kineto profiler emits the trace's CPU-side events as the base, a stand-in
observer holds the GPU columnar window reconstructed from its GPU events, and the same
merge_trace_window_into_chrome_trace the live backend writes runs over them. No
CUDA/CUPTI needed. Monitor-only -- stock Kineto exports off a live in-memory profile and
cannot re-export a saved trace; for the kineto baseline run synthetic ``--mode trunk``.

Run on a host with libcupti >= 13.3 visible to the monitor (torch front-loads the
nvidia-cuda-cupti wheel; add ``LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat`` for the
graph-node-tools-id probe), e.g.
``python benchmarks/profiler_benchmark/cupti_monitor_bench_export.py --mode monitor``.
"""

import argparse
import gzip
import json
import os
import statistics
import tempfile
import time
import types
from pathlib import Path

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity


def run_export(
    mode: str,
    kernels: int,
    iters: int,
    trace_file: str | None = None,
    async_export: bool = False,
) -> dict:
    if trace_file is not None:
        return _run_export_trace_file(mode, trace_file, iters, async_export)
    use_monitor = mode == "monitor"
    backend: dict = {"backend": "cupti_monitor"} if use_monitor else {}
    if use_monitor and async_export:
        backend["cupti_monitor_async_export"] = True
    cfg = _ExperimentalConfig(
        custom_profiler_config=json.dumps(backend) if backend else ""
    )
    x = torch.randn(512, 512, device="cuda")
    for _ in range(50):  # warm up kernels / allocator
        x = torch.relu(x)
    torch.cuda.synchronize()

    export_ms: list[float] = []
    wait_ms: list[float] = []
    n_events = 0
    out_mb = 0.0
    with tempfile.TemporaryDirectory() as td:
        trace_path = str(Path(td) / "trace.json.gz")
        for _ in range(iters):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                experimental_config=cfg,
            ) as prof:
                for _ in range(kernels):
                    x = torch.relu(x)
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            prof.export_chrome_trace(trace_path)
            export_ms.append((time.perf_counter() - t0) * 1e3)
            t1 = time.perf_counter()
            prof.wait_for_exports()  # no-op for stock Kineto
            wait_ms.append((time.perf_counter() - t1) * 1e3)

            out_mb = os.path.getsize(trace_path) / 1e6
            with gzip.open(trace_path, "rt") as f:
                n_events = len(json.load(f)["traceEvents"])
            os.remove(trace_path)

    em = statistics.median(export_ms)
    wm = statistics.median(wait_ms)
    return {
        "mode": mode,
        "kernels": kernels,
        "iters": iters,
        "trace_events": n_events,
        "out_mb": round(out_mb, 1),
        "export_ms": round(em, 1),
        "wait_exports_ms": round(wm, 1),
        "total_ms": round(em + wm, 1),
        "us_per_event": round((em + wm) * 1e3 / max(n_events, 1), 2),
    }


def _columns_from_trace(events: list, base_ns: int) -> dict:
    """Reverse a merged chrome trace's GPU X events back into the monitor's columnar
    window ({kind: {col: np.ndarray}}) so the columnar merge can be timed at a real
    trace's scale/shape. Reconstructs the dominant kinds (kernel / memcpy / memset /
    runtime / driver / overhead); join-only inputs (external_correlation) are absent in
    a merged trace, so the annotation join is skipped -- the per-event chrome build,
    which dominates, is faithful."""
    import numpy as np
    from cupti.cupti import (  # pyrefly: ignore[missing-import]
        Driver_api_trace_cbid,
        Runtime_api_trace_cbid,
    )

    def _cbid_by_name(enum_cls) -> dict:
        out = {}
        for name, m in enum_cls.__members__.items():
            base = name.rsplit("_v", 1)[0] if "_v" in name else name
            out.setdefault(base, int(m.value))
            out.setdefault(name, int(m.value))
        return out

    rt_cbid = _cbid_by_name(Runtime_api_trace_cbid)
    dr_cbid = _cbid_by_name(Driver_api_trace_cbid)
    by_cat: dict[str, list] = {}
    for e in events:
        if e.get("ph") == "X":
            by_cat.setdefault(e.get("cat"), []).append(e)

    def i64(xs):
        return np.asarray(xs, dtype=np.int64)

    def times(rows):
        s = i64([round(r["ts"] * 1000) + base_ns for r in rows])
        en = i64([round((r["ts"] + r.get("dur", 0.0)) * 1000) + base_ns for r in rows])
        return s, en

    cols: dict[str, dict] = {}
    for kind in ("kernel", "gpu_memcpy", "gpu_memset"):
        rows = by_cat.get(kind, [])
        if not rows:
            continue
        a = [r.get("args") or {} for r in rows]
        s, en = times(rows)
        c = {
            "start_ns": s,
            "end_ns": en,
            "device_id": i64([r.get("pid", 0) for r in rows]),
            "context_id": i64([x.get("context", 0) for x in a]),
            "stream_id": i64([x.get("stream", 0) for x in a]),
            "correlation_id": i64([x.get("correlation", 0) for x in a]),
            "graph_id": i64([x.get("graph id", 0) for x in a]),
            "graph_node_id": i64([x.get("graph node id", 0) for x in a]),
            "annotation": np.array([None] * len(rows), dtype=object),
        }
        if kind == "kernel":
            grid = [x.get("grid") or [0, 0, 0] for x in a]
            blk = [x.get("block") or [0, 0, 0] for x in a]
            c.update(
                name=np.array([r.get("name", "") for r in rows], dtype=object),
                grid_x=i64([g[0] for g in grid]),
                grid_y=i64([g[1] for g in grid]),
                grid_z=i64([g[2] for g in grid]),
                block_x=i64([b[0] for b in blk]),
                block_y=i64([b[1] for b in blk]),
                block_z=i64([b[2] for b in blk]),
                registers_per_thread=i64([x.get("registers per thread", 0) for x in a]),
                static_shared_memory=i64([x.get("shared memory", 0) for x in a]),
                dynamic_shared_memory=i64([0] * len(rows)),
                priority=i64([x.get("priority", 0) for x in a]),
                queued=i64([x.get("queued", 0) for x in a]),
                channel=i64([x.get("channel", 0) for x in a]),
                channel_type=i64([x.get("channel_type", 0) for x in a]),
            )
        else:
            c.update(
                bytes=i64([x.get("bytes", 0) for x in a]),
                flags=i64([0] * len(rows)),
            )
            if kind == "gpu_memcpy":
                c.update(
                    copy_kind=i64([0] * len(rows)),
                    src_kind=i64([0] * len(rows)),
                    dst_kind=i64([0] * len(rows)),
                )
            else:
                c.update(value=i64([0] * len(rows)), memory_kind=i64([0] * len(rows)))
        cols[kind] = c
    for kind, table in (("cuda_runtime", rt_cbid), ("cuda_driver", dr_cbid)):
        rows = by_cat.get(kind, [])
        if not rows:
            continue
        a = [r.get("args") or {} for r in rows]
        s, en = times(rows)
        cols[kind] = {
            "cbid": i64([table.get(r.get("name", ""), 0) for r in rows]),
            "start_ns": s,
            "end_ns": en,
            "process_id": i64([r.get("pid", 0) for r in rows]),
            "thread_id": i64([r.get("tid", 0) for r in rows]),
            "correlation_id": i64([x.get("correlation", 0) for x in a]),
        }
    rows = by_cat.get("overhead", [])
    if rows:
        s, en = times(rows)
        cols["overhead"] = {
            "start_ns": s,
            "end_ns": en,
            "correlation_id": i64([0] * len(rows)),
            "name": np.array([r.get("name", "") for r in rows], dtype=object),
        }
    return cols


def _state_from_trace_file(path: str) -> tuple[dict, dict, int]:
    """Load a real exported chrome trace into the monitor's export state: the GPU columnar
    window (reconstructed from the trace's GPU events) and a CPU-only base trace. The
    columnar merge re-emits the GPU X events (+ ac2g flows) from the window, so the base
    keeps only the events it does NOT regenerate (cpu_op / user_annotation / python / the
    metadata records) -- splicing the GPU work back in reproduces the original trace."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    base_ns = int(data.get("baseTimeNanoseconds", 0))
    columns = _columns_from_trace(events, base_ns)
    window = {
        "columns": columns,
        "user_annotations": {},
        "thread_resource_map": {},
        "start_ns": base_ns,
    }
    gpu_cats = {
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
        "cuda_runtime",
        "cuda_driver",
        "overhead",
    }
    cpu_data = dict(data)
    cpu_data["traceEvents"] = [
        e
        for e in events
        if e.get("cat") not in gpu_cats and e.get("ph") not in ("f", "s", "t")
    ]
    return window, cpu_data, len(events)


class _LoadedCpuProfiler:
    """Stands in for the live Kineto profiler in export_chrome_trace: dumps the CPU-side
    events loaded from the trace file (the base the columnar merge splices GPU work into),
    just as the real backend dumps the live Kineto CPU trace to a temp file."""

    def __init__(self, cpu_data: dict) -> None:
        self._cpu_data = cpu_data

    def export_chrome_trace(self, path, metadata=None, use_python_export=False) -> None:
        with open(path, "w") as f:
            json.dump(self._cpu_data, f)


class _LoadedWindowObserver:
    """Stands in for the ProfilerObserver: holds the reconstructed GPU window and runs the
    real merge on join() (the monitor's actual export work) against the base set_export
    hands it -- exactly what ProfilerObserver._maybe_write does on a synchronous export."""

    def __init__(self, window: dict) -> None:
        self._window = window
        self._cpu: str | None = None
        self._out: str | None = None

    def set_export(self, window_id, cpu_path, out_path) -> None:
        self._cpu = os.fspath(cpu_path)
        out = os.fspath(out_path)
        self._out = out if out.endswith(".gz") else out + ".gz"

    def join(self, *, force=True, timeout_s=30.0) -> None:
        import contextlib

        from torch.profiler._cupti.monitor_trace import (
            merge_trace_window_into_chrome_trace,
        )

        try:
            merge_trace_window_into_chrome_trace(
                self._cpu, self._out, self._window, trace_name=self._out
            )
        finally:
            with contextlib.suppress(OSError, TypeError):
                os.remove(self._cpu)


def _run_export_trace_file(
    mode: str, path: str, iters: int, async_export: bool = False
) -> dict:
    """Run the real ``export_chrome_trace`` over a loaded trace's state: a stand-in
    profiler emits its CPU events and a stand-in observer holds its GPU window, so the
    monitor's actual export path (merge + serialize + gzip) is timed at a real trace's
    scale. Monitor-only: stock Kineto cannot re-export a saved trace.

    With ``async_export`` the merge is deferred just like the live backend: the timed
    ``export_chrome_trace`` is only the training-thread handoff (CPU-trace dump +
    set_export), and the merge + write is timed separately under ``wait_for_exports`` (it
    runs off-thread in real use)."""
    if mode != "monitor":
        raise SystemExit(
            "--trace-file supports --mode monitor only: stock Kineto exports off a live "
            "in-memory profile and cannot re-export a saved trace. For the kineto "
            "baseline, run synthetic --mode trunk."
        )
    window, cpu_data, n_in = _state_from_trace_file(path)
    export_ms: list[float] = []
    wait_ms: list[float] = []
    n_events = 0
    out_mb = 0.0
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "trace.json.gz")
        for _ in range(iters):
            prof = types.SimpleNamespace(
                profiler=_LoadedCpuProfiler(cpu_data),
                _use_cupti_monitor=True,
                _cupti_async_export=async_export,
                _trace_metadata={},
                _monitor_window_id=0,
                _cupti_profiler_observer=_LoadedWindowObserver(window),
            )
            t0 = time.perf_counter()
            profile.export_chrome_trace(prof, out)
            export_ms.append((time.perf_counter() - t0) * 1e3)
            # Sync export already merged + wrote inside export_chrome_trace; async deferred
            # the merge to wait_for_exports (the off-thread finalize in real use).
            t1 = time.perf_counter()
            profile.wait_for_exports(prof)
            wait_ms.append((time.perf_counter() - t1) * 1e3)
            out_mb = os.path.getsize(out) / 1e6
            with gzip.open(out, "rt") as f:
                n_events = len(json.load(f)["traceEvents"])
    em = statistics.median(export_ms)
    wm = statistics.median(wait_ms)
    return {
        "mode": "monitor",
        "async_export": async_export,
        "trace_file": path,
        "iters": iters,
        "input_events": n_in,
        "trace_events": n_events,
        "out_mb": round(out_mb, 1),
        "export_ms": round(em, 1),
        "wait_exports_ms": round(wm, 1),
        "total_ms": round(em + wm, 1),
        "us_per_event": round((em + wm) * 1e3 / max(n_events, 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trunk", "monitor"], default="monitor")
    parser.add_argument("--kernels", type=int, default=40000)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="run --mode monitor's real export_chrome_trace over a loaded chrome trace "
        "instead of a synthetic workload (CPU base + reconstructed GPU window). "
        "Monitor-only; No CUDA/CUPTI needed.",
    )
    parser.add_argument(
        "--async-export",
        action="store_true",
        help="cupti_monitor only: defer the merge off-thread (export_chrome_trace is just "
        "the handoff; the merge + write is timed under wait_exports_ms).",
    )
    args = parser.parse_args()

    if args.trace_file and args.mode != "monitor":
        raise SystemExit(
            "--trace-file supports --mode monitor only: stock Kineto exports off a live "
            "in-memory profile and cannot re-export a saved trace. For the kineto "
            "baseline, run synthetic --mode trunk."
        )
    if not args.trace_file:
        torch.cuda.init()
    print(
        json.dumps(
            run_export(
                args.mode,
                args.kernels,
                args.iters,
                trace_file=args.trace_file,
                async_export=args.async_export,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
