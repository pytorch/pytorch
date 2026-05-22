# mypy: allow-untyped-defs
"""
Validates Chrome traces emitted by ``torch.profiler`` against rules derived
from production issues.

Usage::

    from torch.profiler._trace_validator import validate_trace

    passed, violations = validate_trace("trace.pt.trace.json")
    for v in violations:
        print(v)
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections import defaultdict
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass
class Violation:
    """A single rule violation found in a trace."""

    rule_name: str
    message: str

    def __str__(self) -> str:
        return f"{self.rule_name}: {self.message}"


def _load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return [e for e in events if isinstance(e, dict)]


def _check_gpu_kernel_causality(events: list[dict]) -> list[Violation]:
    """For each (cudaLaunchKernel, GPU kernel) pair matched by External id,
    the GPU kernel must start at or after its cudaLaunchKernel."""
    cpu_launches: dict[int, dict] = {}
    gpu_kernels: dict[int, dict] = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        ext_id = args.get("External id")
        if ext_id is None:
            continue
        ext_id = int(ext_id)
        cat, name, ts = ev.get("cat", ""), ev.get("name", ""), float(ev.get("ts", 0))
        corr = args.get("correlation")

        if cat == "cuda_runtime" and name == "cudaLaunchKernel":
            if ext_id not in cpu_launches or ts < cpu_launches[ext_id]["ts"]:
                cpu_launches[ext_id] = {"ts": ts, "name": name, "corr": corr}
        elif cat == "kernel":
            if ext_id not in gpu_kernels or ts < gpu_kernels[ext_id]["ts"]:
                gpu_kernels[ext_id] = {"ts": ts, "name": name, "corr": corr}

    violations = []
    for ext_id, gpu in gpu_kernels.items():
        launch = cpu_launches.get(ext_id)
        if launch is None:
            continue
        if gpu["ts"] < launch["ts"]:
            skew = launch["ts"] - gpu["ts"]
            violations.append(
                Violation(
                    rule_name="_check_gpu_kernel_causality",
                    message=(
                        f"GPU kernel '{gpu['name']}' (External id={ext_id}, "
                        f"correlation={gpu['corr']}) starts {skew:.1f}us before "
                        f"its cudaLaunchKernel (External id={ext_id}, "
                        f"correlation={launch['corr']}), "
                        f"gpu_ts={gpu['ts']:.1f}, cpu_ts={launch['ts']:.1f}"
                    ),
                )
            )
    return violations


def _check_stream_wait_corr_id_populated(events: list[dict]) -> list[Violation]:
    """Stream Wait Events and Event Synchronize must have
    wait_on_cuda_event_record_corr_id >= 0."""
    TARGET_KINDS = {"Stream Wait Event", "Event Sync"}

    violations = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        sync_kind = args.get("cuda_sync_kind")
        if sync_kind not in TARGET_KINDS:
            continue
        raw_corr = args.get("wait_on_cuda_event_record_corr_id")
        if raw_corr is None or int(raw_corr) < 0:
            ts = float(ev.get("ts", 0))
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_populated",
                    message=(
                        f"'{sync_kind}' event at ts={ts:.1f}us on "
                        f"device={args.get('device')} stream={args.get('stream')} "
                        f"has invalid wait_on_cuda_event_record_corr_id={raw_corr!r}"
                    ),
                )
            )
    return violations


def _check_stream_sync_overlap(events: list[dict]) -> list[Violation]:
    """For each Stream Synchronize on (device, stream), no kernel on that
    stream should still be running when the sync starts."""
    stream_syncs = []
    for ev in events:
        if (
            ev.get("ph") == "X"
            and ev.get("cat") == "cuda_sync"
            and ev.get("args", {}).get("cuda_sync_kind") == "Stream Sync"
        ):
            args = ev.get("args", {})
            stream_syncs.append(
                {
                    "ts": float(ev.get("ts", 0)),
                    "dur": float(ev.get("dur", 0)),
                    "stream": args.get("stream"),
                    "device": args.get("device"),
                }
            )
    if not stream_syncs:
        return []

    kernels_by_stream: dict[tuple, list[dict]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") == "kernel":
            args = ev.get("args", {})
            key = (args.get("device"), args.get("stream"))
            ts = float(ev.get("ts", 0))
            kernels_by_stream[key].append(
                {
                    "ts": ts,
                    "end": ts + float(ev.get("dur", 0)),
                    "name": ev.get("name", ""),
                }
            )

    violations = []
    for sync in stream_syncs:
        key = (sync["device"], sync["stream"])
        for k in kernels_by_stream.get(key, []):
            if k["ts"] < sync["ts"] < k["end"]:
                overlap = k["end"] - sync["ts"]
                violations.append(
                    Violation(
                        rule_name="_check_stream_sync_overlap",
                        message=(
                            f"StreamSynchronize on device={sync['device']} "
                            f"stream={sync['stream']} at ts={sync['ts']:.1f}us "
                            f"but kernel '{k['name']}' (ends {k['end']:.1f}us) "
                            f"is still running ({overlap:.1f}us overlap)"
                        ),
                    )
                )
    return violations


_CUDA_EVENT_RECORD_NAMES = {
    "cudaEventRecord",
    "cudaEventRecord_ptsz",
    "cudaEventRecordWithFlags",
    "cudaEventRecordWithFlags_ptsz",
}


def _check_stream_wait_corr_id_in_past(events: list[dict]) -> list[Violation]:
    """wait_on_cuda_event_record_corr_id must point to a cudaEventRecord
    with cudaEventRecord.ts <= stream_wait.ts."""
    event_record_ts: dict[int, float] = {}
    for ev in events:
        if (
            ev.get("ph") == "X"
            and ev.get("cat") in ("cuda_runtime", "cuda_driver")
            and ev.get("name") in _CUDA_EVENT_RECORD_NAMES
        ):
            args = ev.get("args", {})
            ts = float(ev.get("ts", 0))
            for field in ("External id", "correlation"):
                cid = args.get(field)
                if cid is not None:
                    cid = int(cid)
                    if cid not in event_record_ts or ts < event_record_ts[cid]:
                        event_record_ts[cid] = ts

    violations = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        args = ev.get("args", {})
        if args.get("cuda_sync_kind") != "Stream Wait Event":
            continue
        ref = args.get("wait_on_cuda_event_record_corr_id")
        if ref is None or int(ref) < 0:
            continue
        ref = int(ref)
        sw_ts = float(ev.get("ts", 0))
        record_ts = event_record_ts.get(ref)

        if record_ts is None:
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_in_past",
                    message=(
                        f"Stream Wait Event at ts={sw_ts:.1f}us references "
                        f"corr_id={ref} but no matching cudaEventRecord in trace"
                    ),
                )
            )
        elif record_ts > sw_ts:
            lag = record_ts - sw_ts
            violations.append(
                Violation(
                    rule_name="_check_stream_wait_corr_id_in_past",
                    message=(
                        f"Stream Wait Event at ts={sw_ts:.1f}us references "
                        f"cudaEventRecord (corr_id={ref}) {lag:.1f}us in the future "
                        f"(event_record_ts={record_ts:.1f})"
                    ),
                )
            )
    return violations


_NCCL_REQUIRED_FIELDS = {
    "Collective name",
    "dtype",
    "In msg nelems",
    "Out msg nelems",
    "Group size",
}


def _check_nccl_metadata(events: list[dict]) -> list[Violation]:
    """record_param_comms events must carry: Collective name, dtype,
    In msg nelems, Out msg nelems, Group size."""
    violations = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("name") != "record_param_comms":
            continue
        args = ev.get("args", {})
        missing = _NCCL_REQUIRED_FIELDS - set(args.keys())
        if missing:
            violations.append(
                Violation(
                    rule_name="_check_nccl_metadata",
                    message=(
                        f"'record_param_comms' at ts={float(ev.get('ts', 0)):.1f}us "
                        f"missing metadata: {sorted(missing)}"
                    ),
                )
            )
    return violations


def _check_backward_seq_id_uniqueness(events: list[dict]) -> list[Violation]:
    """Per Sequence number, at most one distinct backward op name."""
    seq_to_ops: dict[int, list[str]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if "autograd::engine::evaluate_function:" not in name:
            continue
        args = ev.get("args", {})
        seq = args.get("Sequence number") or args.get("seq_num")
        if seq is None:
            continue
        seq = int(seq)
        op = name.split(":", 1)[-1].strip() if ":" in name else name
        if op not in seq_to_ops[seq]:
            seq_to_ops[seq].append(op)

    violations = []
    for seq, ops in seq_to_ops.items():
        if len(ops) > 1:
            violations.append(
                Violation(
                    rule_name="_check_backward_seq_id_uniqueness",
                    message=(
                        f"Sequence number {seq} shared by {len(ops)} backward "
                        f"ops: {ops}"
                    ),
                )
            )
    return violations


_RULES: list[Callable[[list[dict]], list[Violation]]] = [
    _check_gpu_kernel_causality,
    _check_stream_wait_corr_id_populated,
    _check_stream_sync_overlap,
    _check_stream_wait_corr_id_in_past,
    _check_nccl_metadata,
    _check_backward_seq_id_uniqueness,
]


def validate_trace(trace_path: str) -> tuple[bool, list[Violation]]:
    """
    Run all validation rules against a Chrome trace JSON file.

    Args:
        trace_path: Path to ``.pt.trace.json`` or ``.pt.trace.json.gz``.

    Returns:
        A ``(passed, violations)`` tuple.  ``passed`` is ``True`` when no
        violations were found.
    """
    events = _load_events(trace_path)
    all_violations: list[Violation] = []
    for rule in _RULES:
        all_violations.extend(rule(events))
    return len(all_violations) == 0, all_violations
