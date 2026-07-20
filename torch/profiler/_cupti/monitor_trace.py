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


_RUNTIME_DROPPED_CBIDS: frozenset[int] | None = None
_DRIVER_KEPT_CBIDS: frozenset[int] | None = None


def runtime_dropped_cbids() -> frozenset[int]:
    """cbids of RUNTIME APIs the trace drops (names in the blocklist), so the decoder can
    filter the noise (e.g. cudaGetDevice/GetLastError) out of the window before it is
    built/merged. CUPTI's own per-cbid activity filter is NOT_COMPATIBLE under
    user-defined records, so it cannot be done in CUPTI; this is the post-decode
    equivalent, drop-set-identical to the merge's name blocklist."""
    global _RUNTIME_DROPPED_CBIDS
    if _RUNTIME_DROPPED_CBIDS is None:
        names = _load_cbid_names(Runtime_api_trace_cbid)
        _RUNTIME_DROPPED_CBIDS = frozenset(
            cb for cb, n in names.items() if n in _RUNTIME_BLOCKLIST
        )
    return _RUNTIME_DROPPED_CBIDS


def driver_kept_cbids() -> frozenset[int]:
    """cbids of DRIVER APIs the trace keeps (names in the registered allowlist); the
    decoder drops every other driver record (the driver kind is an allowlist, unlike the
    runtime blocklist). Keep-set-identical to the merge's allowlist."""
    global _DRIVER_KEPT_CBIDS
    if _DRIVER_KEPT_CBIDS is None:
        names = _load_cbid_names(Driver_api_trace_cbid)
        _DRIVER_KEPT_CBIDS = frozenset(
            cb for cb, n in names.items() if n in _DRIVER_REGISTERED
        )
    return _DRIVER_KEPT_CBIDS


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
        if (
            gid.any()
            or gnid.any()
            or any(a is not None for a in ann_l)
            or meta_l is not None
        ):
            gid_l = gid.tolist()
            gnid_l = gnid.tolist()
            for i, ev in enumerate(events):
                a = ev["args"]
                if gid_l[i]:
                    a["graph id"] = gid_l[i]
                if gnid_l[i]:
                    a["graph node id"] = gnid_l[i]
                _annotation_to_args(a, ann_l[i])
                if meta_l is not None and meta_l[i] is not None:
                    _annotation_to_args(a, meta_l[i])
        trace_events.extend(events)
        trace_events.extend(
            {
                "ph": "f",
                "id": corr_l[i],
                "pid": dev_l[i],
                "tid": tid_l[i],
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
        str_l = c["stream_id"].tolist()
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
