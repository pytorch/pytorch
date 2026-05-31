# mypy: allow-untyped-defs
from __future__ import annotations

import gzip
import json
import math
import os
import re
import struct
import time as _time
from pathlib import Path
from typing import IO

import torch

from . import _cupti_monitor as _mon


_TRIMONTH_SECONDS = 7889238

_WORKLOAD_TYPES = {
    _mon._RECORD_KIND_KERNEL: "kernel",
    _mon._RECORD_KIND_MEMCPY: "gpu_memcpy",
    _mon._RECORD_KIND_MEMSET: "gpu_memset",
}

_MEMCPY_KIND_NAMES = {
    1: "HtoD",
    2: "DtoH",
    3: "HtoA",
    4: "AtoH",
    5: "AtoA",
    6: "AtoD",
    7: "DtoA",
    8: "DtoD",
    10: "PtoP",
}

_MEMORY_KIND_NAMES = {
    0: "unknown",
    1: "pageable",
    2: "pinned",
    3: "device",
    4: "array",
    5: "managed",
    6: "device_static",
    7: "managed_static",
}

_FLOW_CATEGORY = "ac2g"
_OVERHEAD_PID = -1
_CBID_NAME_RE = re.compile(r"^\s*[A-Z0-9_]+_(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<value>\d+),")
_RUNTIME_CBID_HEADER = Path("/usr/local/cuda/include/cupti_runtime_cbid.h")
_DRIVER_CBID_HEADER = Path("/usr/local/cuda/include/cupti_driver_cbid.h")
_RUNTIME_CBID_NAMES: dict[int, str] | None = None
_DRIVER_CBID_NAMES: dict[int, str] | None = None
_RUNTIME_BLOCKLIST = {
    "cudaGetDevice",
    "cudaSetDevice",
    "cudaGetLastError",
    "cudaEventCreate",
    "cudaEventCreateWithFlags",
    "cudaEventDestroy",
}
_RUNTIME_FLOW_NAMES = {
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaLaunchKernelExC",
    "cudaGraphLaunch",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaStreamWaitEvent",
}
_DRIVER_REGISTERED = {
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cuMemCreate",
    "cuMemMap",
    "cuMemUnmap",
    "cuMemRelease",
    "cuMemExportToShareableHandle",
    "cuMemImportFromShareableHandle",
}

_EXCLUDED_OVERHEAD_NAMES = {
    "Activity Buffer Request",
}
_DRIVER_FLOW_NAMES = {
    "cuLaunchKernel",
    "cuLaunchKernelEx",
}


def _trimester_base_ns() -> int:
    return (int(_time.time()) // _TRIMONTH_SECONDS) * _TRIMONTH_SECONDS * 1_000_000_000


def _ns_to_us(time_ns: int) -> str:
    return f"{time_ns // 1000}.{time_ns % 1000:03d}"


def _json_escape(s: str) -> str:
    return json.dumps(s)


def _write_metadata_event(
    f: IO[str], name: str, ts: str, pid, tid, arg_key: str, arg_value: str
) -> None:
    f.write(
        f'{{"ph":"M","name":"{name}","ts":{ts},'
        f'"pid":{pid},"tid":{tid},"args":{{"{arg_key}":{arg_value}}}}},\n'
    )


def _metadata_event(
    name: str,
    ts_us: float,
    pid,
    tid,
    arg_key: str,
    arg_value,
) -> dict[str, object]:
    return {
        "ph": "M",
        "name": name,
        "ts": ts_us,
        "pid": pid,
        "tid": tid,
        "args": {arg_key: arg_value},
    }


def _annotation_to_args(args: dict[str, object], annotation: object) -> None:
    if annotation is None:
        return
    try:
        decoded = json.loads(annotation) if isinstance(annotation, str) else annotation
    except json.JSONDecodeError:
        args["annotation"] = annotation
        return
    if isinstance(decoded, list):
        args["annotation"] = json.dumps(decoded)
    elif isinstance(decoded, dict):
        for key, value in decoded.items():
            args[str(key)] = value
    else:
        args["annotation"] = decoded


def _load_cbid_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not path.exists():
        return names
    for line in path.read_text(errors="ignore").splitlines():
        match = _CBID_NAME_RE.match(line)
        if match is None:
            continue
        raw_name = match.group("name")
        normalized = re.sub(r"_v\d+$", "", raw_name)
        names[int(match.group("value"))] = normalized
    return names


def _runtime_cbid_name(cbid: int) -> str:
    global _RUNTIME_CBID_NAMES
    if _RUNTIME_CBID_NAMES is None:
        _RUNTIME_CBID_NAMES = _load_cbid_names(_RUNTIME_CBID_HEADER)
    return _RUNTIME_CBID_NAMES.get(cbid, f"cbid_{cbid}")


def _driver_cbid_name(cbid: int) -> str:
    global _DRIVER_CBID_NAMES
    if _DRIVER_CBID_NAMES is None:
        _DRIVER_CBID_NAMES = _load_cbid_names(_DRIVER_CBID_HEADER)
    return _DRIVER_CBID_NAMES.get(cbid, f"cbid_{cbid}")


def _runtime_is_registered(name: str) -> bool:
    return name not in _RUNTIME_BLOCKLIST


def _runtime_requires_flow(name: str) -> bool:
    return name in _RUNTIME_FLOW_NAMES or name.startswith("cudaMemcpy") or name.startswith("cudaMemset")


def _driver_is_registered(name: str) -> bool:
    return name in _DRIVER_REGISTERED


def _driver_requires_flow(name: str) -> bool:
    return name in _DRIVER_FLOW_NAMES


def _load_annotations(path: Path) -> dict[int, str]:
    annotations: dict[int, str] = {}
    if not path.exists():
        return annotations

    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        annotation_id, size = _mon._ANNOTATION_HEADER.unpack_from(data, offset)
        offset += _mon._ANNOTATION_HEADER.size
        payload = data[offset : offset + size]
        offset += size
        try:
            annotations[annotation_id] = payload.decode()
        except UnicodeDecodeError:
            annotations[annotation_id] = repr(payload)
    return annotations


def _iter_record_chunks(path: Path):
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        magic, version, count, payload_size, min_ts, max_ts = _mon._CHUNK_HEADER.unpack_from(
            data, offset
        )
        if magic != _mon._CHUNK_MAGIC:
            raise RuntimeError(f"Unexpected chunk magic in {path}")
        offset += _mon._CHUNK_HEADER.size
        payload = data[offset : offset + payload_size]
        offset += payload_size
        yield version, count, min_ts, max_ts, payload


def load_monitor_records(output_dir: str | os.PathLike[str]) -> list[dict]:
    output_dir = Path(output_dir)
    record_path = output_dir / _mon._RECORD_FILE
    annotation_path = output_dir / _mon._ANNOTATION_FILE
    annotations = _load_annotations(annotation_path)

    events = []
    for version, count, _min_ts, _max_ts, payload in _iter_record_chunks(record_path):
        if version != _mon._CHUNK_VERSION:
            raise RuntimeError(
                f"Unsupported record version {version}; expected {_mon._CHUNK_VERSION}"
            )
        stride = _mon._RECORD_STRUCT.size
        for i in range(count):
            (
                record_kind,
                device_id,
                context_id,
                stream_id,
                correlation_id,
                annotation_id,
                graph_node_id,
                start_ns,
                end_ns,
                value0,
                value1,
                value2,
                value3,
                graph_id,
            ) = _mon._RECORD_STRUCT.unpack_from(payload, i * stride)
            events.append(
                {
                    "record_kind": record_kind,
                    "type": _WORKLOAD_TYPES.get(record_kind, "unknown"),
                    "device_id": device_id,
                    "context_id": context_id,
                    "stream_id": stream_id,
                    "correlation_id": correlation_id,
                    "annotation_id": annotation_id,
                    "annotation_json": annotations.get(annotation_id),
                    "graph_node_id": graph_node_id,
                    "graph_id": graph_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "value0": value0,
                    "value1": value1,
                    "value2": value2,
                    "value3": value3,
                }
            )
    return events


def _trace_window_entries(
    trace_window: dict[str, object],
    *,
    base_ns: int,
    cpu_main_thread_by_pid: dict[int, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events = trace_window["events"]
    cpu_main_thread_by_pid = cpu_main_thread_by_pid or {}
    max_non_overhead_end_ns = 0
    for event in events:
        if event.get("kind") == "overhead":
            continue
        max_non_overhead_end_ns = max(
            max_non_overhead_end_ns,
            int(event.get("end_ns", 0)),
        )
    metadata_events: list[dict[str, object]] = []
    trace_events: list[dict[str, object]] = []
    seen_devices: dict[int, int] = {}
    seen_streams: set[tuple[int, int]] = set()
    seen_cpu_processes: dict[int, int] = {}
    seen_cpu_threads: set[tuple[int, int]] = set()
    need_overhead_metadata = False

    for event in events:
        kind = event["kind"]
        if kind in {"kernel", "gpu_memcpy", "gpu_memset"}:
            device_id = int(event["device_id"])
            stream_id = int(event["stream_id"])
            seen_devices.setdefault(device_id, int(event["start_ns"]))
            seen_streams.add((device_id, stream_id))
        elif kind in {"cuda_runtime", "cuda_driver"}:
            if kind == "cuda_runtime":
                cbid_name = _runtime_cbid_name(int(event["cbid"]))
                if not _runtime_is_registered(cbid_name):
                    continue
            else:
                cbid_name = _driver_cbid_name(int(event["cbid"]))
                if not _driver_is_registered(cbid_name):
                    continue
            process_id = int(event["process_id"])
            thread_id = cpu_main_thread_by_pid.get(
                process_id,
                int(event["thread_id"]),
            )
            seen_cpu_processes.setdefault(process_id, int(event["start_ns"]))
            seen_cpu_threads.add((process_id, thread_id))
        elif kind == "overhead":
            if str(event.get("name")) in _EXCLUDED_OVERHEAD_NAMES:
                continue
            need_overhead_metadata = True

    for did, first_ts in sorted(seen_devices.items()):
        ts_us = max((first_ts - base_ns) / 1000.0, 0.0)
        metadata_events.extend(
            [
                _metadata_event("process_name", ts_us, did, 0, "name", "python"),
                _metadata_event(
                    "process_labels", ts_us, did, 0, "labels", f"GPU {did}"
                ),
                _metadata_event(
                    "process_sort_index", ts_us, did, 0, "sort_index", 5000000 + did
                ),
            ]
        )

    for pid, first_ts in sorted(seen_cpu_processes.items()):
        ts_us = max((first_ts - base_ns) / 1000.0, 0.0)
        metadata_events.extend(
            [
                _metadata_event("process_name", ts_us, pid, 0, "name", "python"),
                _metadata_event("process_labels", ts_us, pid, 0, "labels", "CPU"),
                _metadata_event("process_sort_index", ts_us, pid, 0, "sort_index", pid),
            ]
        )

    for pid, tid in sorted(seen_cpu_threads):
        metadata_events.extend(
            [
                _metadata_event("thread_name", 0.0, pid, tid, "name", f"thread {tid}"),
                _metadata_event("thread_sort_index", 0.0, pid, tid, "sort_index", tid),
            ]
        )

    for did, rid in sorted(seen_streams):
        ts_us = 0.0
        metadata_events.extend(
            [
                _metadata_event("thread_name", ts_us, did, rid, "name", f"stream {rid} "),
                _metadata_event("thread_sort_index", ts_us, did, rid, "sort_index", rid),
            ]
        )

    if need_overhead_metadata:
        metadata_events.extend(
            [
                _metadata_event("process_name", 0.0, _OVERHEAD_PID, 0, "name", "python"),
                _metadata_event(
                    "process_labels", 0.0, _OVERHEAD_PID, 0, "labels", "Overhead"
                ),
                _metadata_event(
                    "process_sort_index",
                    0.0,
                    _OVERHEAD_PID,
                    0,
                    "sort_index",
                    0x1000000,
                ),
                _metadata_event("thread_name", 0.0, _OVERHEAD_PID, 0, "name", "thread 0"),
                _metadata_event(
                    "thread_sort_index", 0.0, _OVERHEAD_PID, 0, "sort_index", 0
                ),
            ]
        )

    for event in events:
        kind = event["kind"]
        if kind not in {"kernel", "gpu_memcpy", "gpu_memset", "overhead"}:
            if kind in {"cuda_runtime", "cuda_driver"}:
                pid = int(event["process_id"])
                tid = cpu_main_thread_by_pid.get(pid, int(event["thread_id"]))
                cat = str(kind)
                name = (
                    _runtime_cbid_name(int(event["cbid"]))
                    if kind == "cuda_runtime"
                    else _driver_cbid_name(int(event["cbid"]))
                )
                if kind == "cuda_runtime":
                    if not _runtime_is_registered(name):
                        continue
                    requires_flow = _runtime_requires_flow(name)
                else:
                    if not _driver_is_registered(name):
                        continue
                    requires_flow = _driver_requires_flow(name)
                args = {
                    "cbid": int(event["cbid"]),
                    "correlation": int(event["correlation_id"]),
                }
                ts_us = max((int(event["start_ns"]) - base_ns) / 1000.0, 0.0)
                dur_us = max(
                    (int(event["end_ns"]) - int(event["start_ns"])) / 1000.0, 0.0
                )
                trace_events.append(
                    {
                        "ph": "X",
                        "cat": cat,
                        "name": name,
                        "pid": pid,
                        "tid": tid,
                        "ts": ts_us,
                        "dur": dur_us,
                        "args": args,
                    }
                )
                correlation_id = int(event.get("correlation_id", 0))
                if correlation_id and requires_flow:
                    trace_events.append(
                        {
                            "ph": "s",
                            "id": correlation_id,
                            "pid": pid,
                            "tid": tid,
                            "ts": ts_us,
                            "cat": _FLOW_CATEGORY,
                            "name": _FLOW_CATEGORY,
                        }
                    )
            continue

        if kind == "overhead":
            if str(event.get("name")) in _EXCLUDED_OVERHEAD_NAMES:
                continue
            pid = _OVERHEAD_PID
            tid = 0
            cat = "overhead"
            name = str(event["name"])
            args: dict[str, object] = {}
        else:
            pid = int(event["device_id"])
            tid = int(event["stream_id"])
            cat = str(kind)
            name = str(event["name"])
            args = {
                "device": int(event["device_id"]),
                "context": int(event["context_id"]),
                "stream": int(event["stream_id"]),
                "correlation": int(event["correlation_id"]),
            }
            if int(event.get("graph_id", 0)):
                args["graph id"] = int(event["graph_id"])
            if int(event.get("graph_node_id", 0)):
                args["graph node id"] = int(event["graph_node_id"])
            _annotation_to_args(args, event.get("annotation"))

            if kind == "gpu_memcpy":
                args["bytes"] = int(event["bytes"])
                args["copy kind"] = _MEMCPY_KIND_NAMES.get(
                    int(event["copy_kind"]), int(event["copy_kind"])
                )
                args["src kind"] = _MEMORY_KIND_NAMES.get(
                    int(event["src_kind"]), int(event["src_kind"])
                )
                args["dst kind"] = _MEMORY_KIND_NAMES.get(
                    int(event["dst_kind"]), int(event["dst_kind"])
                )
                args["flags"] = int(event["flags"])
            elif kind == "gpu_memset":
                args["bytes"] = int(event["bytes"])
                args["value"] = int(event["value"])
                args["memory kind"] = int(event["memory_kind"])
                args["flags"] = int(event["flags"])

        ts_us = max((int(event["start_ns"]) - base_ns) / 1000.0, 0.0)
        dur_us = max((int(event["end_ns"]) - int(event["start_ns"])) / 1000.0, 0.0)
        trace_events.append(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": ts_us,
                "dur": dur_us,
                "args": args,
            }
        )

        correlation_id = int(event.get("correlation_id", 0))
        if kind in {"kernel", "gpu_memcpy", "gpu_memset"} and correlation_id:
            trace_events.append(
                {
                    "ph": "f",
                    "id": correlation_id,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts_us,
                    "cat": _FLOW_CATEGORY,
                    "name": _FLOW_CATEGORY,
                    "bp": "e",
                }
            )

    return metadata_events, trace_events


def merge_trace_window_into_chrome_trace(
    cpu_trace_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    trace_window: dict[str, object],
    *,
    trace_name: str | None = None,
) -> None:
    cpu_trace_path = str(cpu_trace_path)
    output_path = str(output_path)
    input_opener = gzip.open if cpu_trace_path.endswith(".gz") else open
    with input_opener(cpu_trace_path, "rt") as f:
        data = json.load(f)

    base_ns = int(data.get("baseTimeNanoseconds", _trimester_base_ns()))
    original_events = list(data.get("traceEvents", []))
    cpu_main_thread_by_pid: dict[int, int] = {}
    for event in original_events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X":
            continue
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        if isinstance(pid, int) and isinstance(tid, int):
            cpu_main_thread_by_pid.setdefault(pid, tid)

    metadata_events, trace_events = _trace_window_entries(
        trace_window,
        base_ns=base_ns,
        cpu_main_thread_by_pid=cpu_main_thread_by_pid,
    )
    events = [
        event
        for event in original_events
        if not (
            (event.get("cat") == "Trace" and event.get("name") == "PyTorch Profiler (0)")
            or event.get("name") in {
                "Iteration Start: PyTorch Profiler",
                "Record Window End",
            }
        )
    ]

    metadata_insert = 0
    while metadata_insert < len(events) and events[metadata_insert].get("ph") == "M":
        metadata_insert += 1
    events[metadata_insert:metadata_insert] = metadata_events

    events.extend(trace_events)

    min_ts = math.inf
    max_end_ts = 0.0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") == "Trace":
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        min_ts = min(min_ts, ts)
        max_end_ts = max(max_end_ts, ts + max(dur, 0.0))

    if not math.isfinite(min_ts):
        raise RuntimeError("Merged trace did not contain any duration events")

    events.extend(
        [
            {
                "ph": "X",
                "cat": "Trace",
                "name": "PyTorch Profiler (0)",
                "pid": "Spans",
                "tid": "PyTorch Profiler",
                "ts": min_ts,
                "dur": max(max_end_ts - min_ts, 0.0),
                "args": {"Op count": 0},
            },
            {
                "ph": "i",
                "s": "g",
                "name": "Iteration Start: PyTorch Profiler",
                "pid": "Traces",
                "tid": "Trace PyTorch Profiler",
                "ts": min_ts,
            },
            {
                "ph": "i",
                "s": "g",
                "name": "Record Window End",
                "pid": "",
                "tid": "",
                "ts": max_end_ts + 0.001,
            },
        ]
    )

    data["traceEvents"] = events
    data["traceName"] = trace_name or output_path

    output_opener = gzip.open if output_path.endswith(".gz") else open
    with output_opener(output_path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def export_trace_window_chrome_trace(
    trace_window: dict[str, object],
    path: str | os.PathLike[str],
    *,
    trace_name: str | None = None,
) -> None:
    base_ns = _trimester_base_ns()
    metadata_events, trace_events = _trace_window_entries(trace_window, base_ns=base_ns)
    if not trace_events:
        raise RuntimeError("Trace window did not contain any exportable events")

    min_ts = min(float(event["ts"]) for event in trace_events if event["ph"] == "X")
    max_end_ts = max(
        float(event["ts"]) + float(event.get("dur", 0.0))
        for event in trace_events
        if event["ph"] == "X"
    )

    device_props = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        device_props.append(
            {
                "id": i,
                "name": p.name,
                "totalGlobalMem": p.total_memory,
                "computeMajor": p.major,
                "computeMinor": p.minor,
                "maxThreadsPerBlock": p.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
                "maxThreadsPerMultiprocessor": p.multi_processor_count,
                "regsPerMultiprocessor": p.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
                "warpSize": p.warp_size,
            }
        )

    trace_events = metadata_events + trace_events + [
        {
            "ph": "X",
            "cat": "Trace",
            "name": "PyTorch Profiler (0)",
            "pid": "Spans",
            "tid": "PyTorch Profiler",
            "ts": min_ts,
            "dur": max(max_end_ts - min_ts, 0.0),
            "args": {"Op count": 0},
        },
        _metadata_event(
            "process_sort_index",
            min_ts,
            "Spans",
            0,
            "sort_index",
            0x20000000,
        ),
        {
            "ph": "i",
            "s": "g",
            "name": "Iteration Start: PyTorch Profiler",
            "pid": "Traces",
            "tid": "Trace PyTorch Profiler",
            "ts": min_ts,
        },
        {
            "ph": "i",
            "s": "g",
            "name": "Record Window End",
            "pid": "",
            "tid": "",
            "ts": max_end_ts + 0.001,
        },
    ]

    data = {
        "schemaVersion": 1,
        "deviceProperties": device_props,
        "displayTimeUnit": "ms",
        "baseTimeNanoseconds": base_ns,
        "traceEvents": trace_events,
        "traceName": trace_name or str(path),
    }

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        json.dump(data, f, separators=(",", ":"))


def export_monitor_trace(
    output_dir: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    trace_name: str | None = None,
) -> None:
    events = load_monitor_records(output_dir)
    if not events:
        raise RuntimeError(f"No monitor records found in {output_dir}")

    base_ns = _trimester_base_ns()
    min_ts = min(e["start_ns"] for e in events)
    max_end_ts = max(e["end_ns"] for e in events)

    def _rel(ns: int) -> str:
        return _ns_to_us(max(ns - base_ns, 0))

    device_props = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        device_props.append(
            {
                "id": i,
                "name": p.name,
                "totalGlobalMem": p.total_memory,
                "computeMajor": p.major,
                "computeMinor": p.minor,
                "maxThreadsPerBlock": p.max_threads_per_block,  # pyrefly: ignore[missing-attribute]
                "maxThreadsPerMultiprocessor": p.multi_processor_count,
                "regsPerMultiprocessor": p.regs_per_multiprocessor,  # pyrefly: ignore[missing-attribute]
                "warpSize": p.warp_size,
            }
        )

    seen_devices = {}
    seen_resources = {}
    for event in events:
        ts = event["start_ns"]
        seen_devices.setdefault(event["device_id"], ts)
        seen_resources.setdefault((event["device_id"], event["stream_id"]), ts)

    path = str(path)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt") as f:
        f.write("{\n")
        f.write('"schemaVersion": 1,\n')
        f.write(f'"deviceProperties": {json.dumps(device_props)},\n')
        f.write('"displayTimeUnit": "ms",\n')
        f.write(f'"baseTimeNanoseconds": {base_ns},\n')
        f.write('"traceEvents": [\n')

        for did, ts in sorted(seen_devices.items()):
            ts_str = _rel(ts)
            _write_metadata_event(f, "process_name", ts_str, did, 0, "name", '"python"')
            _write_metadata_event(
                f,
                "process_labels",
                ts_str,
                did,
                0,
                "labels",
                f'"GPU {did}"',
            )
            _write_metadata_event(
                f,
                "process_sort_index",
                ts_str,
                did,
                0,
                "sort_index",
                str(5000000 + did),
            )

        for (did, rid), ts in sorted(seen_resources.items()):
            ts_str = _rel(ts)
            _write_metadata_event(
                f,
                "thread_name",
                ts_str,
                did,
                rid,
                "name",
                f'"stream {rid} "',
            )
            _write_metadata_event(
                f,
                "thread_sort_index",
                ts_str,
                did,
                rid,
                "sort_index",
                str(rid),
            )

        for event in events:
            cat = event["type"]
            ts_str = _rel(event["start_ns"])
            dur_str = _ns_to_us(max(event["end_ns"] - event["start_ns"], 0))
            pid = event["device_id"]
            tid = event["stream_id"]
            name = cat
            args = {
                "device": event["device_id"],
                "context": event["context_id"],
                "stream": event["stream_id"],
                "correlation": event["correlation_id"],
            }

            if event["graph_id"]:
                args["graph id"] = event["graph_id"]
            if event["graph_node_id"]:
                args["graph node id"] = event["graph_node_id"]
            if event["annotation_json"] is not None:
                try:
                    decoded = json.loads(event["annotation_json"])
                    if isinstance(decoded, list):
                        args["annotation"] = json.dumps(decoded)
                    elif isinstance(decoded, dict):
                        for key, value in decoded.items():
                            args[str(key)] = value
                    else:
                        args["annotation"] = decoded
                except json.JSONDecodeError:
                    args["annotation"] = event["annotation_json"]

            if event["record_kind"] == _mon._RECORD_KIND_MEMCPY:
                aux = event["value2"]
                copy_kind = aux & 0xFF
                src_kind = (aux >> 8) & 0xFF
                dst_kind = (aux >> 16) & 0xFF
                flags = (aux >> 24) & 0xFF
                args["bytes"] = event["value0"]
                args["runtime correlation"] = event["value1"]
                args["copy kind"] = _MEMCPY_KIND_NAMES.get(copy_kind, copy_kind)
                args["src kind"] = _MEMORY_KIND_NAMES.get(src_kind, src_kind)
                args["dst kind"] = _MEMORY_KIND_NAMES.get(dst_kind, dst_kind)
                args["flags"] = flags
                name = args["copy kind"]
            elif event["record_kind"] == _mon._RECORD_KIND_MEMSET:
                args["bytes"] = event["value0"]
                args["value"] = event["value1"]
                args["memory kind"] = event["value2"] & 0xFFFF
                args["flags"] = event["value2"] >> 16
                name = "Memset"

            f.write(
                f'{{"ph":"X","cat":{_json_escape(cat)},'
                f'"name":{_json_escape(str(name))},'
                f'"pid":{pid},"tid":{tid},'
                f'"ts":{ts_str},"dur":{dur_str},'
                f'"args":{json.dumps(args, separators=(",", ":"))}}},\n'
            )

        its = _rel(min_ts)
        trace_dur = _ns_to_us(max(max_end_ts - min_ts, 0))
        f.write(
            f'{{"ph":"X","cat":"Trace","name":"PyTorch Profiler (0)",'
            f'"pid":"Spans","tid":"PyTorch Profiler",'
            f'"ts":{its},"dur":{trace_dur},"args":{{"Op count": 0}}}},\n'
        )
        _write_metadata_event(
            f,
            "process_sort_index",
            its,
            '"Spans"',
            0,
            "sort_index",
            str(0x20000000),
        )
        f.write(
            f'{{"ph":"i","s":"g","name":"Iteration Start: PyTorch Profiler",'
            f'"pid":"Traces","tid":"Trace PyTorch Profiler","ts":{its}}},\n'
        )
        end_ts = _rel(max_end_ts + 1000)
        f.write(
            f'{{"ph":"i","s":"g","name":"Record Window End",'
            f'"pid":"","tid":"","ts":{end_ts}}}\n'
        )
        trace_name = trace_name or path
        f.write(f'],\n"traceName": {_json_escape(trace_name)}\n}}\n')
