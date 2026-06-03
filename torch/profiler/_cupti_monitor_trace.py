# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import gzip
import json
import math
import time as _time
from typing import cast, IO, TYPE_CHECKING


if TYPE_CHECKING:
    import os

import torch

from . import _cupti_monitor as _mon


_TRIMONTH_SECONDS = 7889238


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
_USER_EXTERNAL_CORRELATION_KIND = int(_mon._cc.ExternalCorrelationKind.CUSTOM1)
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


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(int | float | str, value))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(int | float | str, value))
    except (TypeError, ValueError):
        return default


def _sanitize_tid(tid: int) -> int:
    if tid == -(1 << 63):
        return 0
    return abs(tid)


def _export_tid(tid):
    if isinstance(tid, int):
        return _sanitize_tid(tid)
    return tid


def _write_metadata_event(
    f: IO[str], name: str, ts: str, pid, tid, arg_key: str, arg_value: str
) -> None:
    tid = _export_tid(tid)
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
        "tid": _export_tid(tid),
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


def _load_cbid_names(enum_cls) -> dict[int, str]:
    names: dict[int, str] = {}
    for name, member in enum_cls.__members__.items():
        normalized = name
        if "_v" in normalized:
            prefix, maybe_version = normalized.rsplit("_v", 1)
            if maybe_version.isdigit():
                normalized = prefix
        names[int(member.value)] = normalized
    return names


def _runtime_cbid_name(cbid: int) -> str:
    global _RUNTIME_CBID_NAMES
    if _RUNTIME_CBID_NAMES is None:
        _RUNTIME_CBID_NAMES = _load_cbid_names(_mon._cc.Runtime_api_trace_cbid)
    return _RUNTIME_CBID_NAMES.get(cbid, f"cbid_{cbid}")


def _driver_cbid_name(cbid: int) -> str:
    global _DRIVER_CBID_NAMES
    if _DRIVER_CBID_NAMES is None:
        _DRIVER_CBID_NAMES = _load_cbid_names(_mon._cc.Driver_api_trace_cbid)
    return _DRIVER_CBID_NAMES.get(cbid, f"cbid_{cbid}")


def _runtime_is_registered(name: str) -> bool:
    return name not in _RUNTIME_BLOCKLIST


def _runtime_requires_flow(name: str) -> bool:
    return name in _RUNTIME_FLOW_NAMES or name.startswith(("cudaMemcpy", "cudaMemset"))


def _driver_is_registered(name: str) -> bool:
    return name in _DRIVER_REGISTERED


def _driver_requires_flow(name: str) -> bool:
    return name in _DRIVER_FLOW_NAMES


def _trace_window_entries(
    trace_window: dict[str, object],
    *,
    base_ns: int,
    cpu_thread_by_external_id: dict[int, tuple[int, int]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events = cast(list[dict[str, object]], trace_window["events"])
    cpu_thread_by_external_id = cpu_thread_by_external_id or {}
    thread_resource_map = cast(
        dict[int, dict[int, int]], trace_window.get("thread_resource_map", {})
    )
    cpu_thread_by_correlation_id: dict[int, tuple[int, int]] = {}
    for event in events:
        if event.get("kind") != "external_correlation":
            continue
        correlation_id = _as_int(event.get("correlation_id", 0))
        external_id = _as_int(event.get("external_id", 0))
        if correlation_id == 0:
            continue
        linked = cpu_thread_by_external_id.get(external_id)
        if linked is not None:
            cpu_thread_by_correlation_id[correlation_id] = linked

    def _runtime_thread_id(
        process_id: int, correlation_id: int, raw_thread_id: int
    ) -> int:
        linked = cpu_thread_by_correlation_id.get(correlation_id)
        if linked is not None and linked[0] == process_id:
            return linked[1]
        process_map = thread_resource_map.get(process_id, {})
        normalized_raw_thread_id = ctypes.c_int32(raw_thread_id & 0xFFFFFFFF).value
        return int(process_map.get(normalized_raw_thread_id, normalized_raw_thread_id))

    max_non_overhead_end_ns = 0
    for event in events:
        if event.get("kind") == "overhead":
            continue
        max_non_overhead_end_ns = max(
            max_non_overhead_end_ns,
            _as_int(event.get("end_ns", 0)),
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
            device_id = _as_int(event["device_id"])
            stream_id = _as_int(event["stream_id"])
            seen_devices.setdefault(device_id, _as_int(event["start_ns"]))
            seen_streams.add((device_id, stream_id))
        elif kind in {"cuda_runtime", "cuda_driver"}:
            if kind == "cuda_runtime":
                cbid_name = _runtime_cbid_name(_as_int(event["cbid"]))
                if not _runtime_is_registered(cbid_name):
                    continue
            else:
                cbid_name = _driver_cbid_name(_as_int(event["cbid"]))
                if not _driver_is_registered(cbid_name):
                    continue
            process_id = _as_int(event["process_id"])
            thread_id = _runtime_thread_id(
                process_id,
                _as_int(event["correlation_id"]),
                _as_int(event["thread_id"]),
            )
            seen_cpu_processes.setdefault(process_id, _as_int(event["start_ns"]))
            seen_cpu_threads.add((process_id, thread_id))
        elif kind == "overhead":
            if (
                str(event.get("name")) == "Activity Buffer Request"
                and max_non_overhead_end_ns
                and _as_int(event.get("start_ns", 0)) > max_non_overhead_end_ns
            ):
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
                _metadata_event(
                    "thread_name", ts_us, did, rid, "name", f"stream {rid} "
                ),
                _metadata_event(
                    "thread_sort_index", ts_us, did, rid, "sort_index", rid
                ),
            ]
        )

    if need_overhead_metadata:
        metadata_events.extend(
            [
                _metadata_event(
                    "process_name", 0.0, _OVERHEAD_PID, 0, "name", "python"
                ),
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
                _metadata_event(
                    "thread_name", 0.0, _OVERHEAD_PID, 0, "name", "thread 0"
                ),
                _metadata_event(
                    "thread_sort_index", 0.0, _OVERHEAD_PID, 0, "sort_index", 0
                ),
            ]
        )

    for event in events:
        kind = event["kind"]
        if kind not in {"kernel", "gpu_memcpy", "gpu_memset", "overhead"}:
            if kind in {"cuda_runtime", "cuda_driver"}:
                pid = _as_int(event["process_id"])
                tid = _runtime_thread_id(
                    pid,
                    _as_int(event["correlation_id"]),
                    _as_int(event["thread_id"]),
                )
                cat = str(kind)
                name = (
                    _runtime_cbid_name(_as_int(event["cbid"]))
                    if kind == "cuda_runtime"
                    else _driver_cbid_name(_as_int(event["cbid"]))
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
                    "cbid": _as_int(event["cbid"]),
                    "correlation": _as_int(event["correlation_id"]),
                }
                ts_us = max((_as_int(event["start_ns"]) - base_ns) / 1000.0, 0.0)
                dur_us = max(
                    (_as_int(event["end_ns"]) - _as_int(event["start_ns"])) / 1000.0,
                    0.0,
                )
                trace_events.append(
                    {
                        "ph": "X",
                        "cat": cat,
                        "name": name,
                        "pid": pid,
                        "tid": _export_tid(tid),
                        "ts": ts_us,
                        "dur": dur_us,
                        "args": args,
                    }
                )
                correlation_id = _as_int(event.get("correlation_id", 0))
                if correlation_id and requires_flow:
                    trace_events.append(
                        {
                            "ph": "s",
                            "id": correlation_id,
                            "pid": pid,
                            "tid": _export_tid(tid),
                            "ts": ts_us,
                            "cat": _FLOW_CATEGORY,
                            "name": _FLOW_CATEGORY,
                        }
                    )
            continue

        if kind == "overhead":
            if (
                str(event.get("name")) == "Activity Buffer Request"
                and max_non_overhead_end_ns
                and _as_int(event.get("start_ns", 0)) > max_non_overhead_end_ns
            ):
                continue
            pid = _OVERHEAD_PID
            tid = 0
            cat = "overhead"
            name = str(event["name"])
            args: dict[str, object] = {}
        else:
            pid = _as_int(event["device_id"])
            tid = _as_int(event["stream_id"])
            cat = str(kind)
            name = str(event["name"])
            args = {
                "device": _as_int(event["device_id"]),
                "context": _as_int(event["context_id"]),
                "stream": _as_int(event["stream_id"]),
                "correlation": _as_int(event["correlation_id"]),
            }
            if _as_int(event.get("graph_id", 0)):
                args["graph id"] = _as_int(event["graph_id"])
            if _as_int(event.get("graph_node_id", 0)):
                args["graph node id"] = _as_int(event["graph_node_id"])
            _annotation_to_args(args, event.get("annotation"))

            if kind == "gpu_memcpy":
                args["bytes"] = _as_int(event["bytes"])
                args["copy kind"] = _MEMCPY_KIND_NAMES.get(
                    _as_int(event["copy_kind"]), _as_int(event["copy_kind"])
                )
                args["src kind"] = _MEMORY_KIND_NAMES.get(
                    _as_int(event["src_kind"]), _as_int(event["src_kind"])
                )
                args["dst kind"] = _MEMORY_KIND_NAMES.get(
                    _as_int(event["dst_kind"]), _as_int(event["dst_kind"])
                )
                args["flags"] = _as_int(event["flags"])
            elif kind == "gpu_memset":
                args["bytes"] = _as_int(event["bytes"])
                args["value"] = _as_int(event["value"])
                args["memory kind"] = _as_int(event["memory_kind"])
                args["flags"] = _as_int(event["flags"])

        ts_us = max((_as_int(event["start_ns"]) - base_ns) / 1000.0, 0.0)
        dur_us = max(
            (_as_int(event["end_ns"]) - _as_int(event["start_ns"])) / 1000.0, 0.0
        )
        trace_events.append(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": pid,
                "tid": _export_tid(tid),
                "ts": ts_us,
                "dur": dur_us,
                "args": args,
            }
        )

        correlation_id = _as_int(event.get("correlation_id", 0))
        if kind in {"kernel", "gpu_memcpy", "gpu_memset"} and correlation_id:
            trace_events.append(
                {
                    "ph": "f",
                    "id": correlation_id,
                    "pid": pid,
                    "tid": _export_tid(tid),
                    "ts": ts_us,
                    "cat": _FLOW_CATEGORY,
                    "name": _FLOW_CATEGORY,
                    "bp": "e",
                }
            )

    trace_events.extend(
        _gpu_user_annotation_events(
            trace_window,
            base_ns=base_ns,
        )
    )

    return metadata_events, trace_events


def _gpu_user_annotation_events(
    trace_window: dict[str, object],
    *,
    base_ns: int,
) -> list[dict[str, object]]:
    user_annotations = trace_window.get("user_annotations", {})
    if not isinstance(user_annotations, dict) or not user_annotations:
        return []
    trace_window_events = cast(list[dict[str, object]], trace_window["events"])

    correlation_to_user_external: dict[int, int] = {}
    for event in trace_window_events:
        if event.get("kind") != "external_correlation":
            continue
        if _as_int(event.get("external_kind", 0)) != _USER_EXTERNAL_CORRELATION_KIND:
            continue
        external_id = _as_int(event.get("external_id", 0))
        correlation_id = _as_int(event.get("correlation_id", 0))
        if external_id in user_annotations and correlation_id != 0:
            correlation_to_user_external[correlation_id] = external_id

    if not correlation_to_user_external:
        return []

    span_map: dict[tuple[int, int, int], dict[str, int]] = {}
    for event in trace_window_events:
        if event.get("kind") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        correlation_id = _as_int(event.get("correlation_id", 0))
        external_id = correlation_to_user_external.get(correlation_id)
        if external_id is None:
            continue
        device_id = _as_int(event["device_id"])
        stream_id = _as_int(event["stream_id"])
        key = (external_id, device_id, stream_id)
        start_ns = _as_int(event["start_ns"])
        end_ns = _as_int(event["end_ns"])
        span = span_map.get(key)
        if span is None:
            span_map[key] = {"start_ns": start_ns, "end_ns": end_ns}
        else:
            span["start_ns"] = min(span["start_ns"], start_ns)
            span["end_ns"] = max(span["end_ns"], end_ns)

    gpu_user_events: list[dict[str, object]] = []
    for (external_id, device_id, stream_id), span in sorted(span_map.items()):
        name = user_annotations.get(external_id)
        if not isinstance(name, str):
            continue
        start_us = max((span["start_ns"] - base_ns) / 1000.0 - 0.001, 0.0)
        dur_us = max((span["end_ns"] - span["start_ns"]) / 1000.0 + 0.002, 0.0)
        gpu_user_events.append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": device_id,
                "tid": _export_tid(stream_id),
                "ts": start_us,
                "dur": dur_us,
                "args": {"External id": external_id},
            }
        )

    return gpu_user_events


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
    cpu_thread_by_external_id: dict[int, tuple[int, int]] = {}
    for event in original_events:
        if not isinstance(event, dict):
            continue
        if event.get("ph") != "X":
            continue
        if event.get("cat") not in {"cpu_op", "user_annotation"}:
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        args = event.get("args")
        if not (
            isinstance(pid, int) and isinstance(tid, int) and isinstance(args, dict)
        ):
            continue
        external_id = args.get("External id")
        if external_id is None:
            continue
        try:
            cpu_thread_by_external_id[int(external_id)] = (pid, tid)
        except (TypeError, ValueError):
            continue

    metadata_events, trace_events = _trace_window_entries(
        trace_window,
        base_ns=base_ns,
        cpu_thread_by_external_id=cpu_thread_by_external_id,
    )
    events = [
        event
        for event in original_events
        if not (
            (
                event.get("cat") == "Trace"
                and event.get("name") == "PyTorch Profiler (0)"
            )
            or event.get("name")
            in {
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
        ts = _as_float(event.get("ts", 0.0))
        dur = _as_float(event.get("dur", 0.0))
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

    min_ts = min(_as_float(event["ts"]) for event in trace_events if event["ph"] == "X")
    max_end_ts = max(
        _as_float(event["ts"]) + _as_float(event.get("dur", 0.0))
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

    trace_events = (
        metadata_events
        + trace_events
        + [
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
    )

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
