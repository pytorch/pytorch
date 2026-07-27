# mypy: allow-untyped-defs
from __future__ import annotations

import gzip
import json
import math
import time as _time
from typing import Any, cast, TYPE_CHECKING

import numpy as np


# orjson serializes ~3-8x faster than stdlib json on large traces and emits bytes; not a
# torch dep (absent in CI), so use it when present and fall back to json.
try:
    import orjson as _orjson  # pyrefly: ignore[missing-import]
except ImportError:
    _orjson = None  # type: ignore[assignment]


if TYPE_CHECKING:
    import os

from cupti.cupti import (  # pyrefly: ignore[missing-import]
    Driver_api_trace_cbid,
    Runtime_api_trace_cbid,
)


# The value Kineto uses to round the trace base time down to a ~3-month boundary (seconds);
# reused to derive the default baseTimeNanoseconds so timestamps match Kineto's range.
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

# CUpti_ActivitySynchronizationType -> kineto cuda_sync name.
_SYNC_TYPE_NAMES = {
    0: "Unknown",
    1: "Event Sync",
    2: "Stream Wait Event",
    3: "Stream Sync",
    4: "Context Sync",
}
# CUPTI sentinel for "not applicable" stream/context on a synchronization record.
_SYNC_INVALID = 0xFFFFFFFF


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


def _default_base_ns() -> int:
    # Fallback trace base time (ns) when the trace has no baseTimeNanoseconds:
    # round "now" down to a _TRIMONTH_SECONDS boundary, matching Kineto.
    return (int(_time.time()) // _TRIMONTH_SECONDS) * _TRIMONTH_SECONDS * 1_000_000_000


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
        names[member.value] = normalized
    return names


def _runtime_cbid_name(cbid: int) -> str:
    global _RUNTIME_CBID_NAMES
    if _RUNTIME_CBID_NAMES is None:
        _RUNTIME_CBID_NAMES = _load_cbid_names(Runtime_api_trace_cbid)
    return _RUNTIME_CBID_NAMES.get(cbid, f"cbid_{cbid}")


def _driver_cbid_name(cbid: int) -> str:
    global _DRIVER_CBID_NAMES
    if _DRIVER_CBID_NAMES is None:
        _DRIVER_CBID_NAMES = _load_cbid_names(Driver_api_trace_cbid)
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
    # Columnar: each kind is a dict of named numpy columns. Events are built by bulk-
    # converting the columns to lists once (tolist) and zipping, not boxing per record.
    columns = cast("dict[str, dict[str, Any]]", trace_window.get("columns", {}))
    cpu_thread_by_external_id = cpu_thread_by_external_id or {}
    thread_resource_map = cast(
        "dict[int, dict[int, int]]", trace_window.get("thread_resource_map", {})
    )

    def _col(kind_str: str):
        c = columns.get(kind_str)
        return c if c and len(next(iter(c.values()))) else None

    # context -> device (cuda_sync records carry no device id) and cuda_event_sync_id ->
    # cudaEventRecord correlation, for the wait_on join on Event Sync / Stream Wait Event.
    context_to_device: dict[int, int] = {}
    for ks in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_event"):
        c = _col(ks)
        if c is None:
            continue
        for ctx, dev in zip(c["context_id"].tolist(), c["device_id"].tolist()):
            context_to_device.setdefault(ctx, dev)
    event_sync_to_corr: dict[int, int] = {}
    ce = _col("cuda_event")
    if ce is not None:
        event_sync_to_corr = {
            sid: corr
            for sid, corr in zip(
                ce["cuda_event_sync_id"].tolist(), ce["correlation_id"].tolist()
            )
            if sid
        }

    cpu_thread_by_correlation_id: dict[int, tuple[int, int]] = {}
    ext = _col("external_correlation")
    if ext is not None:
        for corr, external_id in zip(
            ext["correlation_id"].tolist(), ext["external_id"].tolist()
        ):
            if corr == 0:
                continue
            linked = cpu_thread_by_external_id.get(external_id)
            if linked is not None:
                cpu_thread_by_correlation_id[corr] = linked

    def _runtime_thread_id(
        process_id: int, correlation_id: int, normalized_thread_id: int
    ) -> int:
        # normalized_thread_id: the raw CUPTI threadId reduced to signed 32-bit, vectorized
        # at the call site (ctypes.c_int32 per record is otherwise the hot cost).
        linked = cpu_thread_by_correlation_id.get(correlation_id)
        if linked is not None and linked[0] == process_id:
            return linked[1]
        process_map = thread_resource_map.get(process_id, {})
        return int(process_map.get(normalized_thread_id, normalized_thread_id))

    # Drop the trailing "Activity Buffer Request" overhead that lands after the last
    # real activity: the cutoff is the max non-overhead end (converted ns).
    max_non_overhead_end_ns = 0
    for ks, c in columns.items():
        if ks in ("overhead", "external_correlation", "cuda_event"):
            continue
        if not c or "end_ns" not in c or not len(c["end_ns"]):
            continue
        max_non_overhead_end_ns = max(max_non_overhead_end_ns, int(c["end_ns"].max()))

    trace_events: list[dict[str, object]] = []
    seen_devices: dict[int, int] = {}
    seen_streams: set[tuple[int, int]] = set()
    lane_names: dict[tuple[int, int], str] = {}
    seen_cpu_processes: dict[int, int] = {}
    seen_cpu_threads: set[tuple[int, int]] = set()
    need_overhead_metadata = False

    # --- GPU ops (kernel / memcpy / memset): one X event + a terminating ac2g flow ---
    # Each kind builds X events from one dict literal per row over the bulk-converted
    # columns; graph-id/node and annotation keys (absent for eager kernels) are patched on
    # only when the column carries them.
    for ks in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = _col(ks)
        if c is None:
            continue
        starts = c["start_ns"]
        ts_l = np.maximum((starts - base_ns) / 1000.0, 0.0).tolist()
        dur_l = np.maximum((c["end_ns"] - starts) / 1000.0, 0.0).tolist()
        start_l = starts.tolist()
        dev_l = c["device_id"].tolist()
        ctx_l = c["context_id"].tolist()
        str_l = c["stream_id"].tolist()
        corr_l = c["correlation_id"].tolist()
        tid_l = [_export_tid(s) for s in str_l]
        n = len(start_l)
        if ks == "kernel":
            name_l = c["name"].tolist()
            gx, gy, gz = (
                c["grid_x"].tolist(),
                c["grid_y"].tolist(),
                c["grid_z"].tolist(),
            )
            bx, by, bz = (
                c["block_x"].tolist(),
                c["block_y"].tolist(),
                c["block_z"].tolist(),
            )
            reg_l = c["registers_per_thread"].tolist()
            shmem_l = (c["static_shared_memory"] + c["dynamic_shared_memory"]).tolist()
            prio_l = c["priority"].tolist()
            queued_l = c["queued"].tolist()
            chan_l = c["channel"].tolist()
            chant_l = c["channel_type"].tolist()
            events = [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name_l[i],
                    "pid": dev_l[i],
                    "tid": tid_l[i],
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {
                        "device": dev_l[i],
                        "context": ctx_l[i],
                        "stream": str_l[i],
                        "correlation": corr_l[i],
                        "grid": [gx[i], gy[i], gz[i]],
                        "block": [bx[i], by[i], bz[i]],
                        "registers per thread": reg_l[i],
                        "shared memory": shmem_l[i],
                        "priority": prio_l[i],
                        "queued": queued_l[i],
                        "channel": chan_l[i],
                        "channel_type": chant_l[i],
                    },
                }
                for i in range(n)
            ]
        elif ks == "gpu_memcpy":
            bytes_l = c["bytes"].tolist()
            ck_l, sk_l, dk_l = (
                c["copy_kind"].tolist(),
                c["src_kind"].tolist(),
                c["dst_kind"].tolist(),
            )
            fl_l = c["flags"].tolist()
            events = [
                {
                    "ph": "X",
                    "cat": "gpu_memcpy",
                    "name": "Memcpy",
                    "pid": dev_l[i],
                    "tid": tid_l[i],
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {
                        "device": dev_l[i],
                        "context": ctx_l[i],
                        "stream": str_l[i],
                        "correlation": corr_l[i],
                        "bytes": bytes_l[i],
                        "copy kind": _MEMCPY_KIND_NAMES.get(ck_l[i], ck_l[i]),
                        "src kind": _MEMORY_KIND_NAMES.get(sk_l[i], sk_l[i]),
                        "dst kind": _MEMORY_KIND_NAMES.get(dk_l[i], dk_l[i]),
                        "flags": fl_l[i],
                    },
                }
                for i in range(n)
            ]
        else:
            bytes_l = c["bytes"].tolist()
            val_l = c["value"].tolist()
            mk_l = c["memory_kind"].tolist()
            fl_l = c["flags"].tolist()
            events = [
                {
                    "ph": "X",
                    "cat": "gpu_memset",
                    "name": "Memset",
                    "pid": dev_l[i],
                    "tid": tid_l[i],
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {
                        "device": dev_l[i],
                        "context": ctx_l[i],
                        "stream": str_l[i],
                        "correlation": corr_l[i],
                        "bytes": bytes_l[i],
                        "value": val_l[i],
                        "memory kind": mk_l[i],
                        "flags": fl_l[i],
                    },
                }
                for i in range(n)
            ]
        # Graph ids, annotations, and the comms metadata blob are absent for eager kernels;
        # patch them on only when the column has any. The metadata blob (collective
        # descriptor JSON) is spread into args so its fields show up in the chrome trace.
        gid = c["graph_id"]
        gnid = c["graph_node_id"]
        ann_l = c["annotation"].tolist()
        meta_col = c.get("metadata")
        meta_l = meta_col.tolist() if meta_col is not None else None
        # Pluggable lane assignment: a graph lane resolver (if installed) supplies per-op
        # (logical_lane, lane_name) columns; a graphed op whose logical lane differs from
        # its CUDA stream is moved onto that lane below. CUPTI reports graph-replay ops on
        # whatever streams the graph executor placed them on -- often hundreds of distinct
        # streams -- which scatters one logical replay across a wall of stream lanes and makes
        # for a very confusing profile (the ops don't overlap or disappear, there are just far
        # too many lanes). The resolver collapses them onto a few meaningful logical lanes.
        # Baking the move into this export pass (tid + args["stream"] moved, the op's CUDA
        # stream kept as original_stream) means consumers need no read/reassign/rewrite round
        # trip. Absent a resolver, ops render on their CUDA stream lane.
        lane_col = c.get("logical_lane")
        lane_name_col = c.get("lane_name")
        lane_l = lane_col.tolist() if lane_col is not None else None
        lane_name_l = lane_name_col.tolist() if lane_name_col is not None else None
        display_tid_l = tid_l
        if (
            gid.any()
            or gnid.any()
            or any(ann_l)  # any non-empty annotation (None / empty skip)
            or meta_l is not None
            or lane_l is not None
        ):
            gid_l = gid.tolist()
            gnid_l = gnid.tolist()
            display_tid_l = list(tid_l)
            for i, ev in enumerate(events):
                a = ev["args"]
                if gid_l[i]:
                    a["graph id"] = gid_l[i]
                if gnid_l[i]:
                    a["graph node id"] = gnid_l[i]
                _annotation_to_args(a, ann_l[i])
                if meta_l is not None and meta_l[i] is not None:
                    _annotation_to_args(a, meta_l[i])
                if gnid_l[i] and lane_l is not None and lane_l[i] != str_l[i]:
                    lane_id = lane_l[i]
                    lane = _export_tid(lane_id)
                    ev["tid"] = lane
                    a["stream"] = lane_id
                    a["original_stream"] = str_l[i]
                    display_tid_l[i] = lane
                    seen_streams.add((dev_l[i], lane_id))
                    if lane_name_l is not None and lane_name_l[i] is not None:
                        lane_names[(dev_l[i], lane_id)] = lane_name_l[i]
        trace_events.extend(events)
        trace_events.extend(
            {
                "ph": "f",
                "id": corr_l[i],
                "pid": dev_l[i],
                "tid": display_tid_l[i],
                "ts": ts_l[i],
                "cat": _FLOW_CATEGORY,
                "name": _FLOW_CATEGORY,
                "bp": "e",
            }
            for i in range(n)
            if corr_l[i]
        )
        seen_streams.update(zip(dev_l, str_l))
        for dev, s in zip(dev_l, start_l):
            seen_devices.setdefault(dev, s)

    # --- runtime / driver API: registered names only, remapped onto their CPU thread ---
    for ks in ("cuda_runtime", "cuda_driver"):
        c = _col(ks)
        if c is None:
            continue
        is_runtime = ks == "cuda_runtime"
        starts = c["start_ns"]
        ts_l = np.maximum((starts - base_ns) / 1000.0, 0.0).tolist()
        dur_l = np.maximum((c["end_ns"] - starts) / 1000.0, 0.0).tolist()
        start_l = starts.tolist()
        cbid_l = c["cbid"].tolist()
        pid_l = c["process_id"].tolist()
        # Reduce the raw CUPTI threadId to a signed 32-bit value for the whole column at
        # once (ctypes.c_int32 per record is the hot cost on API-heavy windows).
        normtid_l = (
            c["thread_id"].astype(np.uint32).astype(np.int32).astype(np.int64).tolist()
        )
        corr_l = c["correlation_id"].tolist()
        for i in range(len(cbid_l)):
            name = (
                _runtime_cbid_name(cbid_l[i])
                if is_runtime
                else _driver_cbid_name(cbid_l[i])
            )
            if is_runtime:
                if not _runtime_is_registered(name):
                    continue
                requires_flow = _runtime_requires_flow(name)
            else:
                if not _driver_is_registered(name):
                    continue
                requires_flow = _driver_requires_flow(name)
            pid = pid_l[i]
            tid = _runtime_thread_id(pid, corr_l[i], normtid_l[i])
            seen_cpu_processes.setdefault(pid, start_l[i])
            seen_cpu_threads.add((pid, tid))
            export_tid = _export_tid(tid)
            trace_events.append(
                {
                    "ph": "X",
                    "cat": ks,
                    "name": name,
                    "pid": pid,
                    "tid": export_tid,
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {"cbid": cbid_l[i], "correlation": corr_l[i]},
                }
            )
            if corr_l[i] and requires_flow:
                trace_events.append(
                    {
                        "ph": "s",
                        "id": corr_l[i],
                        "pid": pid,
                        "tid": export_tid,
                        "ts": ts_l[i],
                        "cat": _FLOW_CATEGORY,
                        "name": _FLOW_CATEGORY,
                    }
                )

    # --- overhead (own lane), dropping the trailing buffer-request artifact ---
    c = _col("overhead")
    if c is not None:
        starts = c["start_ns"]
        ts_l = np.maximum((starts - base_ns) / 1000.0, 0.0).tolist()
        dur_l = np.maximum((c["end_ns"] - starts) / 1000.0, 0.0).tolist()
        start_l = starts.tolist()
        name_l = c["name"].tolist()
        for i in range(len(name_l)):
            name = name_l[i]
            if (
                name == "Activity Buffer Request"
                and max_non_overhead_end_ns > 0
                and start_l[i] > max_non_overhead_end_ns
            ):
                continue
            need_overhead_metadata = True
            trace_events.append(
                {
                    "ph": "X",
                    "cat": "overhead",
                    "name": name,
                    "pid": _OVERHEAD_PID,
                    "tid": 0,
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {},
                }
            )

    # --- cuda_sync: device via context, stream via the sync record, wait_on join ---
    c = _col("cuda_sync")
    if c is not None:
        starts = c["start_ns"]
        ts_l = np.maximum((starts - base_ns) / 1000.0, 0.0).tolist()
        dur_l = np.maximum((c["end_ns"] - starts) / 1000.0, 0.0).tolist()
        start_l = starts.tolist()
        st_l = c["sync_type"].tolist()
        ctx_l = c["context_id"].tolist()
        rawstream_l = c["stream_id"].tolist()
        corr_l = c["correlation_id"].tolist()
        evid_l = c["cuda_event_id"].tolist()
        evsync_l = c["cuda_event_sync_id"].tolist()
        for i in range(len(st_l)):
            device = context_to_device.get(ctx_l[i], 0)
            s = rawstream_l[i]
            stream = s if s != _SYNC_INVALID else -1
            sync_type = st_l[i]
            kind_name = _SYNC_TYPE_NAMES.get(sync_type, f"sync_{sync_type}")
            seen_devices.setdefault(device, start_l[i])
            seen_streams.add((device, stream))
            args = {
                "cuda_sync_kind": kind_name,
                "stream": stream,
                "correlation": corr_l[i],
                "device": device,
                "context": ctx_l[i],
            }
            if sync_type in (1, 2):  # Event Sync, Stream Wait Event
                args["wait_on_stream"] = -1
                args["wait_on_cuda_event_id"] = evid_l[i]
                args["wait_on_cuda_event_record_corr_id"] = event_sync_to_corr.get(
                    evsync_l[i], -1
                )
            trace_events.append(
                {
                    "ph": "X",
                    "cat": "cuda_sync",
                    "name": kind_name,
                    "pid": device,
                    "tid": _export_tid(stream),
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": args,
                }
            )

    metadata_events: list[dict[str, object]] = []
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
        lane_name = lane_names.get((did, rid), f"stream {rid} ")
        metadata_events.extend(
            [
                _metadata_event("thread_name", ts_us, did, rid, "name", lane_name),
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

    trace_events.extend(_gpu_user_annotation_events(trace_window, base_ns=base_ns))

    return metadata_events, trace_events


def _gpu_user_annotation_events(
    trace_window: dict[str, object],
    *,
    base_ns: int,
) -> list[dict[str, object]]:
    user_annotations = trace_window.get("user_annotations", {})
    if not isinstance(user_annotations, dict) or not user_annotations:
        return []
    columns = cast("dict[str, dict[str, Any]]", trace_window.get("columns", {}))
    ext = columns.get("external_correlation")
    if not ext or not len(ext["correlation_id"]):
        return []

    # `user_external_id` is the innermost ENCLOSING named-region id (resolved at decode via
    # the monitor's active-id chain), falling back to the raw external_id.
    correlation_to_user_external = {
        corr: uext
        for corr, uext in zip(
            ext["correlation_id"].tolist(), ext["user_external_id"].tolist()
        )
        if corr != 0 and uext in user_annotations
    }
    if not correlation_to_user_external:
        return []

    span_map: dict[tuple[int, int, int], dict[str, int]] = {}
    for ks in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(ks)
        if not c or not len(c["correlation_id"]):
            continue
        corr_l = c["correlation_id"].tolist()
        dev_l = c["device_id"].tolist()
        # Follow graphed ops onto their reassigned logical lane so the spanning annotation
        # lands on the same lane as its kernels (else it stays on the capture stream).
        stream_arr = c["stream_id"]
        lane_col = c.get("logical_lane")
        if lane_col is not None:
            reassign = (c["graph_node_id"] != 0) & (lane_col != stream_arr)
            stream_arr = np.where(reassign, lane_col, stream_arr)
        str_l = stream_arr.tolist()
        start_l = c["start_ns"].tolist()
        end_l = c["end_ns"].tolist()
        for i in range(len(corr_l)):
            external_id = correlation_to_user_external.get(corr_l[i])
            if external_id is None:
                continue
            key = (external_id, dev_l[i], str_l[i])
            start_ns = start_l[i]
            end_ns = end_l[i]
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


# PM counter descriptor ids: local (0-based, by metric order); _merge_counters maps them onto the
# global monotonic id sequence, alongside the env and cycle counters.

# Friendly display names for common PM metrics, keyed by the metric base (before the rollup); a
# "% of peak" rollup gets a "(%)" suffix. Unknown metrics fall back to their raw metric name.
_PM_METRIC_LABELS = {
    "sm__cycles_active": "SM Active",
    "gpu__dram_throughput": "DRAM BW",
    "dram__throughput": "HBM BW",
    "dram__read_throughput": "HBM Read",
    "dram__write_throughput": "HBM Write",
    "nvlrx__bytes": "NVLink RX",
    "nvltx__bytes": "NVLink TX",
    "pcie__throughput": "PCIe",
}


def _pm_label(metric: str) -> str:
    base, _, rollup = metric.partition(".")
    friendly = _PM_METRIC_LABELS.get(base)
    if friendly is None:
        return metric
    return f"{friendly} (%)" if "pct_of_peak" in rollup else friendly


def _build_pm_counters(pm: dict | None, active_devices: set):
    """Build the GpuCounterEvent payload from PM-sampling columns (start_ns/device_id plus one
    value column per metric, keyed by the CUPTI metric name): same tuple shape as
    :func:`_build_gpu_counters`. The metric columns are self-describing, so each is assigned a
    local descriptor id here (mapped to a global id in :func:`_merge_counters`) and labeled with a
    friendly display name (:func:`_pm_label`, or the raw metric name if unmapped). Restricted to
    devices that ran GPU work."""
    if not pm or not len(pm.get("start_ns", ())):
        return None
    ts = np.ascontiguousarray(pm["start_ns"], dtype=np.int64)
    dev = np.asarray(pm["device_id"], dtype=np.int64)
    base = (
        np.isin(dev, list(active_devices))
        if active_devices
        else np.ones(len(dev), dtype=bool)
    )
    if not base.any():
        return None
    metric_cols = [k for k in pm if k not in ("start_ns", "device_id")]
    specs, gpu_l, ts_l, cid_l, val_l = [], [], [], [], []
    for name in metric_cols:
        col = pm.get(name)
        if col is None:
            continue
        cid = len(specs)  # local id; _merge_counters assigns the global id
        specs.append((cid, _pm_label(name)))
        gpu_l.append(dev[base])
        ts_l.append(ts[base])
        cid_l.append(np.full(int(base.sum()), cid, dtype=np.int32))
        val_l.append(np.asarray(col, dtype=np.float64)[base])
    if not specs:
        return None
    return (
        specs,
        np.concatenate(gpu_l).astype(np.int32),
        np.concatenate(ts_l).astype(np.int64),
        np.concatenate(cid_l).astype(np.int32),
        np.concatenate(val_l).astype(np.float64),
    )


def _merge_counters(*parts):
    """Concatenate GpuCounterEvent payloads (the tuples from the per-source builders) into a single
    payload for the encoder. Each source uses its own local descriptor ids; here they are assigned
    onto a single monotonic sequence (0, 1, 2, ...) so ids are globally unique by construction, and
    each source's cid column, COMPUTE-group ids (6th element), and int-valued ids (7th) are remapped
    through it."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    specs: list = []
    gpu_l, ts_l, cid_l, val_l = [], [], [], []
    compute_group: list = []
    int_value_ids: list = []
    next_id = 0
    for p in parts:
        s, g, t, c, v = p[:5]
        remap = {old: next_id + k for k, (old, _name) in enumerate(s)}
        next_id += len(s)
        specs.extend((remap[old], name) for old, name in s)
        gpu_l.append(g)
        ts_l.append(t)
        lut = np.zeros(max(remap) + 1, dtype=np.int32)
        for old, new in remap.items():
            lut[old] = new
        cid_l.append(lut[np.asarray(c)])
        val_l.append(v)
        if len(p) > 5 and p[5]:
            compute_group.extend(remap[i] for i in p[5])
        if len(p) > 6 and p[6]:
            int_value_ids.extend(remap[i] for i in p[6])
    return (
        specs,
        np.concatenate(gpu_l),
        np.concatenate(ts_l),
        np.concatenate(cid_l),
        np.concatenate(val_l),
        compute_group,
        int_value_ids,
    )


def _active_devices(columns: dict) -> set:
    """Devices that ran GPU work this window (so idle GPUs get no counters)."""
    active: set = set()
    for ks in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(ks)
        if c is not None and len(c.get("device_id", ())):
            active.update(np.unique(c["device_id"]).tolist())
    return active


def _gpu_counter_process(device_id: int) -> str:
    """String "pid" for a device's GPU counter row. A string pid makes Perfetto label the process
    row with the string verbatim ("GPU N Counters", no numeric suffix -- the same way kineto's
    "Spans"/"Traces" rows work) and keeps it a distinct process: an integer pid would show its
    number, and a pid <= 0 collapses into the device/unknown process."""
    return f"GPU {device_id} Counters"


def _build_chrome_counters(counters, base_ns: int) -> list[dict]:
    """Emit the merged GPU counter payload as chrome-trace "C" (counter) events -- one per
    sample -- in a per-device "GPU N Counters" process row, separate from the device's kernel/
    stream work (the JSON counterpart of the .pftrace "GPU / Counters / <gpu>" tracks)."""
    if counters is None:
        return []
    specs, gpu, ts, cid, val = counters[:5]
    int_ids = set(counters[6]) if len(counters) > 6 else set()
    name_by_id = dict(specs)
    ts_us = (ts - base_ns) / 1000.0
    out: list[dict] = []
    counter_procs: dict[int, str] = {}
    for i in range(len(cid)):
        name = name_by_id.get(int(cid[i]))
        if name is None:
            continue
        did = int(gpu[i])
        proc = _gpu_counter_process(did)
        counter_procs[did] = proc
        v = int(val[i]) if int(cid[i]) in int_ids else float(val[i])
        out.append(
            {
                "ph": "C",
                "name": name,
                "pid": proc,
                "ts": float(ts_us[i]),
                # single unnamed series (empty key): keying it by the metric name instead
                # makes Perfetto render the track label doubled ("name name").
                "args": {"": v},
            }
        )
    # Low sort index (below the CPU pid and the 5000000 + did device rows) so Perfetto floats
    # the counter rows to the top of the trace, above the CPU and GPU work.
    meta: list[dict] = [
        _metadata_event("process_sort_index", 0.0, proc, 0, "sort_index", 100 + did)
        for did, proc in sorted(counter_procs.items())
    ]
    return meta + out


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
    with input_opener(cpu_trace_path, "rb") as f:
        raw = f.read()
    data = _orjson.loads(raw) if _orjson is not None else json.loads(raw)

    base_ns = int(data.get("baseTimeNanoseconds", _default_base_ns()))
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

    # GPU counters (PM utilization) as chrome "C" events on the device pid -- the JSON
    # counterpart of the .pftrace GPU counter tracks.
    columns = cast("dict[str, dict[str, Any]]", trace_window.get("columns", {}))
    active_devices = _active_devices(columns)
    events.extend(
        _build_chrome_counters(
            _merge_counters(
                _build_pm_counters(columns.get("pm_sampling"), active_devices),
            ),
            base_ns,
        )
    )

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

    # Encode once and write the whole buffer (json.dump streaming through gzip's text wrapper
    # is ~3-5x slower on large traces). compresslevel=1 favors throughput over file size.
    if _orjson is not None:
        payload = _orjson.dumps(data)
    else:
        payload = json.dumps(data, separators=(",", ":")).encode()
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as f:
            f.write(payload)
    else:
        with open(output_path, "wb") as f:
            f.write(payload)
