# mypy: allow-untyped-defs
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time as _time
from typing import Any, cast

import numpy as np

import torch


# orjson serializes ~3-8x faster than stdlib json on large traces and emits bytes; not a
# torch dep (absent in CI), so use it when present and fall back to json.
try:
    import orjson as _orjson  # pyrefly: ignore[missing-import]
except ImportError:
    _orjson = None  # type: ignore[assignment]


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


_GRAPH_DEP_FLOW_ID_BASE = (
    1 << 40
)  # keep node->node arrow flow ids clear of correlation ids


def _graph_dependency_flow_events(
    node_rows: list[tuple[int, int, int, int, int, int]],
    graph_deps: dict[int, list[int]],
    *,
    base_ns: int,
) -> list[dict[str, object]]:
    """Chrome flow (s/f) events drawing CUDA-graph node->node dependency arrows in the JSON
    export. ``node_rows`` is (correlation_id, graph_node_id, device, tid, start_ns, end_ns) per
    graphed op as displayed (tid is the resolver lane). Dependencies (graph_deps: node_id ->
    predecessor node_ids) are resolved within the same replay (correlation_id) so arrows never
    cross replays; each edge is one flow from the predecessor's end to the successor's start.
    Empty without deps.

    KNOWN LIMITATION (CUPTI bug, reported to NVIDIA): a CUDA-graph host node keeps the launching
    cuGraphLaunch's correlationId only on a replay enqueued after the previous replay finished;
    a replay queued behind an in-flight one reports correlationId 0 instead (kernel/memcpy/memset
    nodes of the same graph always keep theirs). Since a profiler cannot synchronize between a
    workload's replays, the id is unrecoverable, and those rows collapse into a correlationId-0
    bucket holding no other node of their replay -- so a host node's arrows draw only on the
    replays whose id survived.

    NOTE: under cross-stream clock skew the recorded predecessor end can land after the successor
    start, and chrome/Perfetto JSON flows cannot render backwards in time, so such an arrow may
    misrender (or bind to the wrong slice when spans overlap on one lane). The .pftrace export
    encodes the same edges as native render-stage waits (GpuRenderStageEvent.event_wait_ids) and
    renders them correctly regardless -- prefer it when dependency arrows matter."""
    if not graph_deps:
        return []
    by_corr: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for corr, gnid, dev, tid, start_ns, end_ns in node_rows:
        by_corr.setdefault(corr, {})[gnid] = (dev, tid, start_ns, end_ns)
    events: list[dict[str, object]] = []
    fid = _GRAPH_DEP_FLOW_ID_BASE
    for node_map in by_corr.values():
        for gnid, (dev, tid, start_ns, end_ns) in node_map.items():
            for pred in graph_deps.get(gnid, ()):
                p = node_map.get(pred)
                if p is None:
                    continue
                pdev, ptid, _pstart_ns, pend_ns = p
                # Arrow from the predecessor's end to the successor's start (the dependency
                # handoff). See the NOTE in the docstring: this can render backwards under clock
                # skew in the JSON export; the .pftrace export does not have that limitation.
                events.append(
                    {
                        "ph": "s",
                        "id": fid,
                        "pid": pdev,
                        "tid": ptid,
                        "ts": max((pend_ns - base_ns) / 1000.0, 0.0),
                        "cat": _FLOW_CATEGORY,
                        "name": _FLOW_CATEGORY,
                    }
                )
                events.append(
                    {
                        "ph": "f",
                        "id": fid,
                        "pid": dev,
                        "tid": tid,
                        "ts": max((start_ns - base_ns) / 1000.0, 0.0),
                        "cat": _FLOW_CATEGORY,
                        "name": _FLOW_CATEGORY,
                        "bp": "e",
                    }
                )
                fid += 1
    return events


# Eager kernel/transfer launches (NOT cudaGraphLaunch). These 1:1 correlate to an eager GPU
# op, so they drive the host-launch -> render-stage gpu_correlation link (event_id == corr).
# Syncs are excluded -- they correlate to no kernel, so a link on them would point nowhere
# pointing nowhere.
_RUNTIME_LAUNCH_NAMES = {
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cudaLaunchCooperativeKernelMultiDevice",
    "cudaLaunchKernelExC",
}


def _runtime_is_eager_launch(name: str) -> bool:
    return name in _RUNTIME_LAUNCH_NAMES or name.startswith(
        ("cudaMemcpy", "cudaMemset")
    )


def _runtime_is_launch(name: str) -> bool:
    # The set for the gpu_correlation channel link: eager launches plus cudaGraphLaunch
    # (links to the replayed kernels; gpu_correlation is a per-selection details link, so the
    # fan-out does not spiderweb).
    return _runtime_is_eager_launch(name) or name == "cudaGraphLaunch"


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
        return process_map.get(normalized_thread_id, normalized_thread_id)

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
    # CUDA-graph node->node dependency arrows (drawn as flows below). Collect each graphed op as
    # displayed so the arrows land on the resolver lanes, then emit after all GPU kinds.
    graph_deps = cast("dict[int, list[int]]", trace_window.get("graph_deps") or {})
    # event-record node graph_node_id -> cudaEvent_t handle, to tag EventRecord spans with the
    # event they record (the CUDA_EVENT record carries no handle); empty for eager/other runs.
    event_record_events = cast(
        "dict[int, int]", trace_window.get("graph_event_record_events") or {}
    )
    dep_node_rows: list[tuple[int, int, int, int, int, int]] = []

    # --- GPU ops (kernel / memcpy / memset): one X event + a terminating ac2g flow ---
    # Each kind builds X events from one dict literal per row over the bulk-converted
    # columns; graph-id/node and annotation keys (absent for eager kernels) are patched on
    # only when the column carries them.
    for ks in (
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
        "graph_event_node",
        "graph_host_node",
    ):
        c = _col(ks)
        if c is None:
            continue
        starts = c["start_ns"]
        ts_l = np.maximum((starts - base_ns) / 1000.0, 0.0).tolist()
        dur_l = np.maximum((c["end_ns"] - starts) / 1000.0, 0.0).tolist()
        start_l = starts.tolist()
        end_l = c["end_ns"].tolist()
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
            chan_l = c["channel"].tolist()
            chant_l = c["channel_type"].tolist()
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
                        "channel": chan_l[i],
                        "channel_type": chant_l[i],
                    },
                }
                for i in range(n)
            ]
        elif ks == "gpu_memset":
            bytes_l = c["bytes"].tolist()
            val_l = c["value"].tolist()
            mk_l = c["memory_kind"].tolist()
            fl_l = c["flags"].tolist()
            chan_l = c["channel"].tolist()
            chant_l = c["channel_type"].tolist()
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
                        "channel": chan_l[i],
                        "channel_type": chant_l[i],
                    },
                }
                for i in range(n)
            ]
        elif ks == "graph_event_node":
            # A CUDA-graph event-record node (e.g. the nodes NCCL inserts under
            # NCCL_GRAPH_MIXING_SUPPORT), rendered as a point span on its stream lane so it is a
            # visible dependency-arrow endpoint. graph id / node / annotation are patched on
            # below; its device timestamp gives ts and dur is ~0.
            events = [
                {
                    "ph": "X",
                    "cat": "graph_event_node",
                    "name": "EventRecord",
                    "pid": dev_l[i],
                    "tid": tid_l[i],
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {
                        "device": dev_l[i],
                        "context": ctx_l[i],
                        "stream": str_l[i],
                        "correlation": corr_l[i],
                    },
                }
                for i in range(n)
            ]
        else:
            # graph_host_node: a CPU callback run as a CUDA-graph node. Rendered on the
            # waiting stream's lane like the other graphed ops; graph id / node / annotation
            # and the recorded callback name/address are patched on below (host nodes always
            # carry a graph_node_id).
            pid_l = c["process_id"].tolist()
            htid_l = c["thread_id"].tolist()
            events = [
                {
                    "ph": "X",
                    "cat": "graph_host_node",
                    "name": "HostNode",
                    "pid": dev_l[i],
                    "tid": tid_l[i],
                    "ts": ts_l[i],
                    "dur": dur_l[i],
                    "args": {
                        "device": dev_l[i],
                        "context": ctx_l[i],
                        "stream": str_l[i],
                        "correlation": corr_l[i],
                        "host process": pid_l[i],
                        "host thread": htid_l[i],
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
        # Host-node callback name/address (resolved at graph instantiate; the CUPTI record
        # carries no fn field). Present only on the graph_host_node column group.
        fn_col = c.get("host_fn")
        fn_l = fn_col.tolist() if fn_col is not None else None
        addr_col = c.get("host_fn_addr")
        addr_l = addr_col.tolist() if addr_col is not None else None
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
        # Event-record node -> cudaEvent handle it records (topology, not in the CUPTI record),
        # used to tag EventRecord spans; the dependency arrows then show what waits on the event.
        ev_events = event_record_events if ks == "graph_event_node" else None
        display_tid_l = tid_l
        if (
            gid.any()
            or gnid.any()
            or any(ann_l)  # any non-empty annotation (None / empty skip)
            or meta_l is not None
            or lane_l is not None
            or fn_l is not None
        ):
            gid_l = gid.tolist()
            gnid_l = gnid.tolist()
            display_tid_l = list(tid_l)
            for i, ev in enumerate(events):
                a = ev["args"]
                if gid_l[i]:
                    a["graph id"] = gid_l[i]
                if gnid_l[i]:
                    # Upper 32 bits are the graph id (emitted separately above); keep only the
                    # lower 32-bit node id here rather than the full 9-digit packed value.
                    a["graph node id"] = gnid_l[i] & 0xFFFFFFFF
                _annotation_to_args(a, ann_l[i])
                if meta_l is not None and meta_l[i] is not None:
                    _annotation_to_args(a, meta_l[i])
                if ev_events is not None and gnid_l[i] in ev_events:
                    a["cuda event"] = hex(ev_events[gnid_l[i]])
                if fn_l is not None and fn_l[i]:
                    a["host fn"] = fn_l[i]
                    ev["name"] = f"HostNode: {fn_l[i]}"
                if addr_l is not None and addr_l[i]:
                    a["host fn addr"] = hex(addr_l[i])
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
            for i in range(n):
                if gnid_l[i]:
                    dep_node_rows.append(
                        (
                            corr_l[i],
                            gnid_l[i],
                            dev_l[i],
                            display_tid_l[i],
                            start_l[i],
                            end_l[i],
                        )
                    )
        trace_events.extend(events)
        # CPU-launch -> GPU-op flow (the terminating end of the host launch's correlation).
        # When graph node->node arrows are drawn (graph_deps present), a graphed op that has a
        # predecessor in the graph gets its incoming edge from that arrow, so drop this launch
        # flow for it: only root graph nodes (and eager ops) keep it, so a cudaGraphLaunch links
        # to the graph's roots instead of fanning out to every replayed kernel. A graph_deps key
        # is a node with predecessors; 0 (eager) and root node ids are absent -> kept.
        drop_launch_flow = (
            [g in graph_deps for g in gnid.tolist()] if graph_deps else None
        )
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
            if corr_l[i] and not (drop_launch_flow and drop_launch_flow[i])
        )
        seen_streams.update(zip(dev_l, str_l))
        for dev, s in zip(dev_l, start_l):
            seen_devices.setdefault(dev, s)

    trace_events.extend(
        _graph_dependency_flow_events(dep_node_rows, graph_deps, base_ns=base_ns)
    )

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
    # Cuspy's active-id chain), falling back to the raw external_id.
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
    # (device, stream) lanes that still render real GPU work after lane reassignment. A graphed
    # op the resolver moved to a logical lane no longer displays on its capture stream (mirrors
    # the display rule in _trace_window_entries); a span stranded on such a stream is dropped
    # below.
    streams_with_display: set[tuple[int, int]] = set()
    for ks in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(ks)
        if not c or not len(c["correlation_id"]):
            continue
        corr_l = c["correlation_id"].tolist()
        dev_l = c["device_id"].tolist()
        # Keep the annotation on the kernel's real capture stream; do not follow graphed
        # ops onto the synthetic logical lanes assigned by the lane resolver.
        str_l = c["stream_id"].tolist()
        start_l = c["start_ns"].tolist()
        end_l = c["end_ns"].tolist()
        lane_col = c.get("logical_lane")
        gnid_col = c.get("graph_node_id")
        lane_l = lane_col.tolist() if lane_col is not None else None
        gnid_l = gnid_col.tolist() if gnid_col is not None else None
        for i in range(len(corr_l)):
            reassigned = (
                lane_l is not None
                and gnid_l is not None
                and gnid_l[i]
                and lane_l[i] != str_l[i]
            )
            if not reassigned:
                streams_with_display.add((dev_l[i], str_l[i]))
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
        # Orphaned span: the capture stream's ops all moved to logical lanes, so it would sit
        # alone on an empty lane divorced from its kernels -- drop it instead.
        if (device_id, stream_id) not in streams_with_display:
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


# --- Perfetto-native (.pftrace) encoding -------------------------------------
# The wire encoding is done natively (protozero via the perfetto SDK) in
# torch/csrc/profiler/cuspy/cuspy_pftrace.cpp; here we only shape the window into the
# flat arrays + track list it consumes.


def _nest_track_slices(groups: list, track_uuids: set) -> None:
    """Make each track's slices nest, so Perfetto can pair them.

    Perfetto's TrackEvent model pairs SLICE_BEGIN/SLICE_END per track with a stack, so a
    track cannot represent partially-overlapping (non-nested) slices. CUPTI reports serial
    GPU-stream slices whose start/end can jitter into ~ns overlaps; left as-is the viewer's
    stack pairing mis-pairs them into 0-duration / inflated slices. For each track in
    ``track_uuids`` (GPU stream tracks, whose slices are serial), walk its slices in start
    order and truncate any still-open slice that a later overlapping slice extends past --
    serial slices become adjacent, genuine nesting is preserved. Only ``end`` is mutated
    (in place), so per-slice annotation/flow alignment is untouched."""
    if not track_uuids:
        return
    want = np.fromiter(
        (np.uint64(u) for u in track_uuids), dtype=np.uint64, count=len(track_uuids)
    )
    ts_p, end_p, uu_p, gi_p, li_p = [], [], [], [], []
    for gi, g in enumerate(groups):
        idx = np.nonzero(np.isin(g["uuid"], want))[0]
        if not len(idx):
            continue
        ts_p.append(g["ts"][idx])
        end_p.append(g["end"][idx])
        uu_p.append(g["uuid"][idx])
        gi_p.append(np.full(len(idx), gi, dtype=np.int64))
        li_p.append(idx.astype(np.int64))
    if not ts_p:
        return
    ts, end, uu = np.concatenate(ts_p), np.concatenate(end_p), np.concatenate(uu_p)
    gi, li = np.concatenate(gi_p), np.concatenate(li_p)
    # Per track, parents before nested children: sort by (track, start asc, end desc).
    order = np.lexsort((-end, ts, uu))
    tsl, endl, uul = ts[order].tolist(), end[order].tolist(), uu[order].tolist()
    stack: list = []
    prev = None
    for k in range(len(tsl)):
        if uul[k] != prev:
            stack.clear()
            prev = uul[k]
        s_ts, s_end = tsl[k], endl[k]
        while stack:
            p = stack[-1]
            if endl[p] <= s_ts:
                stack.pop()
            elif endl[p] < s_end:
                # partial overlap: truncate the open parent to this slice's start so the
                # two serialize instead of mis-pairing.
                endl[p] = s_ts
                stack.pop()
            else:
                break  # endl[p] >= s_end: this slice nests within the parent
        stack.append(k)
    new_end = np.empty(len(endl), dtype=np.int64)
    new_end[order] = np.asarray(endl, dtype=np.int64)  # undo the sort permutation
    for gv in np.unique(gi):
        m = gi == gv
        groups[int(gv)]["end"][li[m]] = new_end[m]


# GPU render-stage definitions: (column key, stage iid, stage name, RenderStageCategory).
# COMPUTE == 2 (matches the reference); the rest are OTHER (0). Stage iids are small and
# share the gpu_specifications iid space with the stream lanes below.
_RENDER_STAGES = (
    ("kernel", 1, "Kernel", 2),
    ("gpu_memcpy", 2, "Memcpy", 0),
    ("gpu_memset", 3, "Memset", 0),
    # GPU-side user annotations (collective/phase ranges) on the stream lane; not a kernel
    # (no launch), so they render as a range nesting over the kernels they span.
    ("gpu_annotation", 4, "Annotation", 0),
    # CUDA-graph event-record / host-callback nodes on their stream lane. Rendered as render
    # stages (not kernels: no launch) so they carry graph id/node/annotation and become
    # dependency-arrow endpoints, matching the JSON export. Point-ish spans from their device
    # timestamps. Both are empty unless the CUDA_EVENT bridge / host-node collection is enabled.
    ("graph_event_node", 5, "EventRecord", 0),
    ("graph_host_node", 6, "HostNode", 0),
)
_STREAM_IID_BASE = 100  # stream-lane iids start past the stage iids

# Chrome-trace categories NOT rendered as CPU track_event slices: "Trace" metadata, and the
# GPU-side kinds we emit from the columnar window instead (kernels/transfers as GPU Render
# Stages, runtime/driver remapped onto their CPU thread). gpu_user_annotation in particular
# is a GPU-stream-side mirror of a CPU user_annotation -- rendering it here would put duplicate
# stream-named tracks (forward_backward, all_reduce, ...) under the CPU process.
_NON_CPU_CATS = frozenset(
    {
        "Trace",
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
        "gpu_user_annotation",
        "cuda_runtime",
        "cuda_driver",
        "cuda_sync",
        "overhead",
        "graph_event_node",
        "graph_host_node",
    }
)

# ComputeKernelLaunch.args, interned as InternedComputeArgName (field 1001 on InternedData).
# Stable iids + the names the viewer's GPU Compute panel joins on (ExtraComputeArg.name_iid ->
# InternedComputeArgName.name). value getter maps a kind's column dict to the per-event column
# (None -> 0 / skipped, e.g. on memcpy/memset).
_COMPUTE_ARGS = (
    (1, "registers_per_thread", lambda c: c.get("registers_per_thread")),
    (2, "shared_mem_dynamic", lambda c: c.get("dynamic_shared_memory")),
    (
        3,
        "thread_count",
        lambda c: (
            c["block_x"] * c["block_y"] * c["block_z"] if "block_x" in c else None
        ),
    ),
)


# Per-event GpuRenderStageEvent.extra_data: the scalar args that used to live on the
# track_event GPU slices, now that the render stage is the single GPU representation.
# (key, getter, skip_zero); skip_zero drops 0 for fields where 0 means absent (so kernel-
# only / memcpy-only fields don't show up on the other kinds). grid/block/registers/shared/
# thread_count ride the structured ComputeKernelLaunch instead; collective metadata is
# carried separately as JSON extra_data.
# NB: device/stream/context are kept under " id"-suffixed keys, not the bare names. The
# GPU-by-process viewer plugin reads extract_arg(arg_set, 'device'/'stream') and requires
# them to be NUM, but GpuRenderStageEvent.extra_data.value is always a string -- the bare
# names would collide with that numeric expectation and crash the plugin. (device == the
# render stage's gpu_id and stream == the lane, so this is detail-panel only.)
def _lower_node_id(c: dict):
    # graphNodeId packs the graph id in its upper 32 bits (emitted separately as "graph id");
    # keep only the lower 32-bit node id, matching the chrome export.
    gnid = c.get("graph_node_id")
    return None if gnid is None else gnid & 0xFFFFFFFF


_RENDER_EXTRA = (
    ("device id", lambda c: c.get("device_id"), False),
    ("context id", lambda c: c.get("context_id"), False),
    ("stream id", lambda c: c.get("stream_id"), False),
    ("correlation", lambda c: c.get("correlation_id"), True),
    ("priority", lambda c: c.get("priority"), True),
    ("queued", lambda c: c.get("queued"), True),
    ("channel", lambda c: c.get("channel"), True),
    ("channel_type", lambda c: c.get("channel_type"), True),
    ("graph id", lambda c: c.get("graph_id"), True),
    ("graph node id", _lower_node_id, True),
    ("bytes", lambda c: c.get("bytes"), True),
    ("value", lambda c: c.get("value"), True),
    ("memory kind", lambda c: c.get("memory_kind"), True),
)
# Grid + workgroup dimension columns (kernels only; 0 elsewhere -> no launch emitted).
_LAUNCH_DIMS = ("grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z")


def _process_name() -> str:
    try:
        with open("/proc/self/comm") as f:
            return f.read().strip() or "python"
    except OSError:
        return os.path.basename(sys.executable) or "python"


def _gpu_panel_const_extra(gpu: np.ndarray) -> list[tuple[str, str]]:
    """Trace-wide string args the GPU Compute panel reads off each kernel slice
    (extract_arg 'arch'/'process_id'/'process_name'): the traced device's arch
    (compute capability) plus the owning process. Constant across the trace, so
    emitted once per event rather than per-kernel-profiled."""
    out: list[tuple[str, str]] = []
    try:
        dev = int(gpu[0]) if len(gpu) else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(dev)
        out.append(("arch", f"sm_{major}{minor}"))
    except Exception:
        pass
    out.append(("process_id", str(os.getpid())))
    out.append(("process_name", _process_name()))
    return out


def _merge_annotation_metadata(
    annotation: Any, metadata: Any, extra: dict[str, Any] | None = None
) -> str | None:
    """Merge a row's graph annotation(s) and collective-metadata blob into one JSON object whose
    top-level fields the native encoder spreads onto the render stage as extra_data -- the
    pftrace analog of the chrome path's per-arg spread (``_annotation_to_args``). ``annotation``
    is the resolved graph annotation (a list of dicts, a dict, or a str, from the graph
    annotation registry); ``metadata`` is the collective-descriptor JSON blob (str, from the
    eager external-metadata join). Returns None when neither contributes anything.

    Without this the render stage carried only the collective ``metadata`` column, so
    graph-node annotations (``mark_kernels`` regions, e.g. a collective's process-group name
    recorded at capture) never reached the .pftrace GPU rows even though the chrome path spread
    them -- a chrome/pftrace parity gap."""
    merged: dict[str, Any] = {}
    if annotation is not None:
        items = annotation if isinstance(annotation, list) else [annotation]
        for it in items:
            if isinstance(it, dict):
                for k, v in it.items():
                    merged[str(k)] = v
            elif isinstance(it, str):
                merged.setdefault("annotation", it)
    if metadata is not None:
        try:
            decoded = (
                json.loads(metadata) if isinstance(metadata, (str, bytes)) else metadata
            )
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, dict):
            for k, v in decoded.items():
                merged[str(k)] = v
    if extra:
        merged.update(extra)
    return json.dumps(merged) if merged else None


def _lists_to_csr(lists: list, dtype) -> tuple:
    """Flatten a list-of-lists into a CSR: an int32 offsets array (length n + 1) and a flat
    values array of ``dtype``; slice i is values[offsets[i]:offsets[i + 1]]."""
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int32, count=len(lists))
    offsets = np.zeros(len(lists) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.fromiter(
        (v for x in lists for v in x), dtype=dtype, count=int(offsets[-1])
    )
    return offsets, flat


def _render_stage_cols(render_columns: dict) -> list:
    """Authoritative surviving-stage selection: ``(ks, iid, name, cat, c)`` for each stage in
    ``_RENDER_STAGES`` order whose column is present and nonempty. Row ordering across the render
    stages is defined once here, so ``_assign_render_event_ids`` and ``_build_render_stages``
    emit rows in the same order (event_id[i] aligns with render-stage row i)."""
    out = []
    for ks, iid, name, cat in _RENDER_STAGES:
        c = render_columns.get(ks)
        if not c or not len(c.get("start_ns", ())):
            continue
        out.append((ks, iid, name, cat, c))
    return out


def _assign_render_event_ids(render_columns: dict, graph_deps: dict) -> tuple | None:
    """Assign a unique event_id per GPU render-stage row and derive the host-launch link and the
    graph node->node dependency arrows. Iterates :func:`_render_stage_cols` (which defines the
    render-stage row order shared with :func:`_build_render_stages`), so ``event_id[i]`` aligns
    with render-stage row i. Returns ``None`` when there are no render-stage rows, else a tuple
    of:

    - ``event_id``: uint64 arange(n) + 1 (0 stays reserved for "none").
    - ``corr_to_event_ids``: correlation_id -> [event_id, ...] grouping, for the CPU launch
      slice's gpu_correlation (an eager launch -> 1 id, a graph replay -> its ROOT node ids).
      When graph deps are known the launch links only to the graph's roots (nodes with no
      recorded predecessor); the rest are reached through the node->node dependency arrows.
    - ``(wait_offsets, wait_ids)``: the event_wait_ids CSR. Each graphed row (graph_node_id != 0)
      resolves ``graph_deps[graph_node_id]`` predecessor tools_ids to the event_ids of the nodes
      in the SAME replay (grouped by correlation_id), skipping predecessors absent from it.
    """
    corr_p, gnid_p = [], []
    for _ks, _iid, _name, _cat, c in _render_stage_cols(render_columns):
        n = len(c["start_ns"])
        z = np.zeros(n, dtype=np.int64)
        corr_p.append(np.ascontiguousarray(c.get("correlation_id", z), dtype=np.int64))
        gnid_p.append(np.ascontiguousarray(c.get("graph_node_id", z), dtype=np.int64))
    if not corr_p:
        return None
    corr = np.concatenate(corr_p)
    gnid = np.concatenate(gnid_p)
    # A graphed row reporting correlationId 0 (a host node on a replay queued behind an in-flight
    # one) strands in a bucket of its own: same CUPTI bug, same lost arrows, as described in
    # :func:`_graph_dependency_flow_events`.
    n = len(corr)
    event_id = np.arange(n, dtype=np.uint64) + np.uint64(1)
    # correlation_id -> event_ids, grouped vectorized (event_id[i] == i + 1). A launch's
    # gpu_correlation reads this: eager -> one id, a graph replay (one shared correlation) ->
    # its root node ids. When graph deps are known, drop non-root graphed nodes (those present
    # as a graph_deps key, i.e. with a recorded predecessor) so the launch links only to the
    # roots; the rest are reached through the node->node dependency arrows. Eager kernels
    # (graph_node_id 0) are never a graph_deps key, so they always link.
    if graph_deps:
        dep_keys = np.fromiter(graph_deps.keys(), dtype=np.int64, count=len(graph_deps))
        link = ~np.isin(gnid, dep_keys)
        corr_l, eid_l = corr[link], event_id[link]
    else:
        corr_l, eid_l = corr, event_id
    order = np.argsort(corr_l, kind="stable")
    uniq_corr, starts = np.unique(corr_l[order], return_index=True)
    groups = np.split(eid_l[order], starts[1:])
    corr_to_event_ids = dict(zip(uniq_corr.tolist(), groups))
    # event_wait_ids: only graphed replays carry node dependencies, so build the per-row CSR
    # only when deps exist; otherwise every render stage waits on nothing.
    if graph_deps:
        corr_l, gnid_l = corr.tolist(), gnid.tolist()
        # Per-replay graph_node_id -> event_id (a correlation is one replay), to resolve each
        # graphed row's predecessors to node event_ids within its own replay.
        node_map_by_corr: dict[int, dict[int, int]] = {}
        for i, g in enumerate(gnid_l):
            if g:
                node_map_by_corr.setdefault(corr_l[i], {})[g] = i + 1
        wait_lists: list[list[int]] = []
        for i, g in enumerate(gnid_l):
            node_map = node_map_by_corr.get(corr_l[i]) if g else None
            wait_lists.append(
                [node_map[p] for p in graph_deps.get(g, ()) if p in node_map]
                if node_map
                else []
            )
        wait_offsets, wait_ids = _lists_to_csr(wait_lists, np.uint64)
    else:
        wait_offsets = np.zeros(n + 1, dtype=np.int32)
        wait_ids = np.empty(0, dtype=np.uint64)
    return event_id, corr_to_event_ids, wait_offsets, wait_ids


def _build_render_stages(
    columns: dict,
    gfx_pid: int,
    iid_of: dict,
    name_table: list,
    event_id: Any,
    event_wait: tuple,
):
    """Build the GpuRenderStageEvent payload (gpu_specs, gfx_contexts, stage_cols, extra,
    launch, tables) for the native GPU Render Stages stream lanes, or None if there are
    no GPU ops. Each GPU op becomes one event on a (gpu_id, stream) lane tagged by stage
    (kind). Compute kernels additionally carry kernel_iid -> InternedComputeKernel (which names
    the slice) and a structured ComputeKernelLaunch (grid/workgroup Dim3 + args named via
    name_iid -> InternedComputeArgName) -> the viewer's GPU Compute "Launch Statistics" panel;
    memcpy/memset carry their byte count as generic extra_data. Perfetto calls the lane a
    hardware queue (hw_queue_iid on the wire); for CUDA it is the stream, so all streams are
    preserved -- as are the synthetic logical lanes, which share that id space. The scalar args
    (device/stream/correlation/channel/...) ride along as extra_data. Each event has its own
    duration -- no stack pairing -- so durations are exact (no nesting)."""
    ts_p, dur_p, gpu_p, stage_p, kname_p = [], [], [], [], []
    dim_p: dict[str, list] = {k: [] for k in _LAUNCH_DIMS}
    arg_p: dict[int, list] = {iid: [] for iid, _n, _g in _COMPUTE_ARGS}
    extra_p: dict[str, list] = {k: [] for k, _g, _s in _RENDER_EXTRA}
    meta_p: list = []  # per-row spread blob (graph annotation + collective descriptor)
    stream_p: list = []
    lane_names: dict[tuple[int, int], str] = {}  # (device, lane) -> resolver name
    for _ks, stage_iid, _name, _cat, c in _render_stage_cols(columns):
        n = len(c["start_ns"])
        z = np.zeros(n, dtype=np.int64)
        ts_p.append(np.ascontiguousarray(c["start_ns"], dtype=np.int64))
        dur_p.append(np.maximum(c["end_ns"] - c["start_ns"], 0).astype(np.int64))
        gpu_p.append(np.ascontiguousarray(c["device_id"], dtype=np.int64))
        # One lane per stream; preserves all streams (channel is
        # 0/unpopulated in many captures, which would otherwise collapse every stream
        # into a single lane). The CUPTI channel is kept as extra_data instead.
        # Reassign graphed ops onto their logical lane (mirrors the chrome path): CUPTI piles
        # graph-replay ops on the one replay stream, so on that stream's lane they overlap;
        # the resolver-assigned lane (+ its name) spreads them out. Absent a resolver the
        # logical_lane column defaults to the CUDA stream, so this is a no-op.
        stream_col = np.ascontiguousarray(c["stream_id"], dtype=np.int64)
        lane_col = c.get("logical_lane")
        if lane_col is not None:
            reassign = lane_col != stream_col
            gnid_col = c.get("graph_node_id")
            if gnid_col is not None:
                reassign &= gnid_col != 0
            lname_col = c.get("lane_name")
            if lname_col is not None:
                for dv, lv, nm, rs in zip(
                    c["device_id"].tolist(),
                    lane_col.tolist(),
                    lname_col.tolist(),
                    reassign.tolist(),
                ):
                    if rs and nm is not None:
                        lane_names[(int(dv), int(lv))] = str(nm)
            stream_p.append(np.where(reassign, lane_col, stream_col).astype(np.int64))
        else:
            stream_p.append(stream_col)
        stage_p.append(np.full(n, stage_iid, dtype=np.uint64))
        # Per-event kernel name (only kernels have one) -> InternedComputeKernel + kernel_iid.
        nm = c.get("name")
        kname_p.append(nm if nm is not None else np.full(n, "", dtype=object))
        for k in _LAUNCH_DIMS:
            dim_p[k].append(np.ascontiguousarray(c.get(k, z), dtype=np.int64))
        for iid, _n, getter in _COMPUTE_ARGS:
            v = getter(c)
            arg_p[iid].append(z if v is None else np.ascontiguousarray(v, np.int64))
        for key, getter, _skip in _RENDER_EXTRA:
            v = getter(c)
            extra_p[key].append(z if v is None else np.ascontiguousarray(v, np.int64))
        # Per-row spread blob: the graph annotation (mark_kernels regions -- e.g. a collective's
        # process-group name) merged with the collective-descriptor metadata, spread onto the
        # render stage as extra_data below. Fast path when a kind has neither.
        mc = c.get("metadata")
        ac = c.get("annotation")
        # Host-node callback name/address (graph_host_node only; string/pointer columns, so they
        # ride the spread blob rather than the numeric extra_data). This is the pftrace analog of
        # the chrome path's "host fn"/"host fn addr" args -- host nodes otherwise all read "HostNode".
        fc = c.get("host_fn")
        addr_c = c.get("host_fn_addr")
        has_ann = ac is not None and bool(ac.astype(bool).any())
        if mc is None and not has_ann and fc is None:
            meta_p.append([None] * n)
        else:
            ml = mc.tolist() if mc is not None else [None] * n
            al = ac.tolist() if has_ann else [None] * n
            fl = fc.tolist() if fc is not None else [None] * n
            addr_l = addr_c.tolist() if addr_c is not None else [None] * n
            blobs = []
            for a, m, f, ad in zip(al, ml, fl, addr_l):
                host_extra: dict[str, str] = {}
                if f:
                    host_extra["host fn"] = f
                if ad:
                    host_extra["host fn addr"] = hex(ad)
                blobs.append(_merge_annotation_metadata(a, m, host_extra or None))
            meta_p.append(blobs)
    if not ts_p:
        return None
    ts, dur, gpu = np.concatenate(ts_p), np.concatenate(dur_p), np.concatenate(gpu_p)
    stream, stage = np.concatenate(stream_p), np.concatenate(stage_p)
    # Serialize overlapping kernel/memcpy/memset render stages per lane. A CUDA stream is
    # serial, but CUPTI graph-replay timestamps overlap ~13%, which renders as messy multi-depth
    # kernels instead of a clean depth-1 row beneath the spanning Annotation (which is the depth-0
    # parent and must keep its full extent -- so it is excluded). Clamp each op's end to the next
    # op's start on the same (gpu, stream) lane.
    ann_iid = next(i for ks, i, _n, _c in _RENDER_STAGES if ks == "gpu_annotation")
    end = ts + dur
    sub = np.nonzero(stage != ann_iid)[0]
    # A lane is (gpu, stream), never a bare stream: the same stream id on two devices is two
    # lanes. Shared by the clamp below and the stream-lane interning further down.
    lane = (gpu.astype(np.int64) << np.int64(32)) | (stream & 0xFFFFFFFF)
    if len(sub):
        order = sub[np.lexsort((ts[sub], lane[sub]))]
        nxt_ts = np.empty(len(order), dtype=np.int64)
        nxt_ts[:-1] = ts[order][1:]
        nxt_ts[-1] = np.iinfo(np.int64).max
        same = np.zeros(len(order), dtype=bool)
        same[:-1] = lane[order][1:] == lane[order][:-1]
        cap = np.where(same, nxt_ts, np.iinfo(np.int64).max)
        end[order] = np.maximum(ts[order], np.minimum(end[order], cap))
        dur = end - ts
    # One lane per (gpu, stream); lanes are split per gpu_id by Perfetto from
    # (gpu_id, stream_iid). Interning on the same (gpu, stream) key keeps each device's names
    # its own -- the iid table is global, so interning on the bare stream would let one device's
    # resolver name ("DP") relabel another device's identically-numbered CUDA stream. The
    # spanning annotation and its kernels share the lane and the viewer depth-nests them
    # (annotation at depth 0, kernels at depth 1) once the kernels are clamped to not overlap
    # each other. Perfetto sorts lanes lexicographically by name, so prefix every lane
    # (resolver-named or "stream N") with a bracketed zero-padded lane id -> the lane order
    # matches the chrome path's numeric lane-id order (otherwise "stream 7" sorts after
    # "stream 26674", and resolver-named lanes like "DP" interleave alphabetically). The "["
    # is a constant leading char, so the padded digits still decide the order.
    uniq, inv = np.unique(lane, return_inverse=True)
    uniq_dev, uniq_lane = (uniq >> 32).tolist(), (uniq & 0xFFFFFFFF).tolist()
    width = len(str(max(uniq_lane))) if len(uniq) else 1
    specs = [(iid, name, cat) for _ks, iid, name, cat in _RENDER_STAGES]
    specs += [
        (
            _STREAM_IID_BASE + j,
            f"[{k:0{width}d}] {lane_names.get((d, k), f'stream {k}')}",
            0,
        )
        for j, (d, k) in enumerate(zip(uniq_dev, uniq_lane))
    ]
    stream_iid = (inv + _STREAM_IID_BASE).astype(np.uint64)
    # event_id is a unique per-row id (assigned by _assign_render_event_ids and passed in, in
    # this exact row order). The CPU launch's gpu_correlation lists the event_ids of the
    # render stages it launched (a graph replay -> all its nodes); event_wait_ids (below) draw
    # the graph node->node dependency arrows between those render stages.
    event_id = np.ascontiguousarray(event_id, dtype=np.uint64)
    # Graphics context 1 -> gfx_pid (the owning process), matching the reference traces
    # (context_id 1 / upid 1). This also makes the GpuByProcess plugin label its per-process
    # container "<proc> / GPU" rather than "Process 0"; disable that plugin in Perfetto if the
    # native "GPU / Hardware Queues" view alone is wanted.
    context = np.ones(len(ts), dtype=np.uint64)
    # Per kernel: kernel_iid -> InternedComputeKernel (launch panel) and name_iid ->
    # EventName (the timeline slice label). Both 0 for memcpy/memset (they fall back to
    # the stage name). name_iid reuses the global interning so kernel names are shared
    # with the track_event slices.
    kernel_iid = np.zeros(len(ts), dtype=np.uint64)
    name_iid = np.zeros(len(ts), dtype=np.uint64)
    name_to_iid: dict[str, int] = {}
    compute_kernels: list = []
    for i, name in enumerate(np.concatenate(kname_p).tolist()):
        if not name:
            continue
        kid = name_to_iid.get(name)
        if kid is None:
            kid = name_to_iid[name] = len(name_to_iid) + 1
            compute_kernels.append((kid, name))
        kernel_iid[i] = kid
        j = iid_of.get(name)
        if j is None:
            j = iid_of[name] = len(name_table)
            name_table.append(name)
        name_iid[i] = j + 1
    stage_cols = (
        ts,
        dur,
        event_id,
        gpu,
        stream_iid,
        stage,
        context,
        name_iid,
        event_wait,
    )
    extra = [(k, np.concatenate(extra_p[k]), s) for k, _g, s in _RENDER_EXTRA]
    dims = tuple(np.concatenate(dim_p[k]) for k in _LAUNCH_DIMS)
    launch_args = [
        (iid, np.concatenate(arg_p[iid]), True) for iid, _n, _g in _COMPUTE_ARGS
    ]
    launch = (*dims, kernel_iid, launch_args)
    arg_names = [(iid, name) for iid, name, _g in _COMPUTE_ARGS]
    # CSR of the per-row spread blobs (offsets int32 length n+1 + concatenated bytes;
    # empty slice = nothing to spread), mirroring the CPU json_anno column. The native encoder
    # spreads each blob's top-level fields onto the render stage as extra_data.
    meta_l = [b for sub in meta_p for b in sub]
    meta_enc = [
        b"" if b is None else (b if isinstance(b, bytes) else b.encode())
        for b in meta_l
    ]
    meta_offsets = np.zeros(len(meta_enc) + 1, dtype=np.int32)
    np.cumsum(
        np.fromiter((len(b) for b in meta_enc), dtype=np.int32, count=len(meta_enc)),
        out=meta_offsets[1:],
    )
    return (
        specs,
        [
            (1, gfx_pid)
        ],  # graphics context 1 -> gfx_pid (the lanes' owning process, upid 1)
        stage_cols,
        extra,
        launch,
        (compute_kernels, arg_names),
        _gpu_panel_const_extra(gpu),
        (meta_offsets, b"".join(meta_enc)),
    )


# GPU counter specs over the environment union's first 8 bytes (data, u64): (name,
# environment_kind, value-from-data). POWER/SPEED pack two u32s (low | high<<32); TEMPERATURE/
# COOLING are a single u32. powerLimit (the constant high half of POWER) is omitted. Descriptor
# ids are assigned locally (by position) and mapped to global ids in _merge_counters; the viewer
# groups the counters per gpu_id under "GPU / Counters".
_ENV_COUNTERS = (
    ("Power (W)", 3, lambda d: (d & 0xFFFFFFFF).astype(np.float64) / 1000.0),
    ("Temperature (C)", 2, lambda d: d.astype(np.float64)),
    ("SM Clock (MHz)", 1, lambda d: (d & 0xFFFFFFFF).astype(np.float64)),
    ("Memory Clock (MHz)", 1, lambda d: (d >> np.uint64(32)).astype(np.float64)),
    ("Fan Speed (%)", 4, lambda d: d.astype(np.float64)),
)


def _build_gpu_counters(env: dict | None, active_devices: set):
    """Build the GpuCounterEvent payload from the sampled environment column:
    (specs, gpu_id[], ts[], counter_id[], value[]) or None. specs = [(counter_id, name), ...];
    the viewer renders these under "GPU / Counters / <gpu>" (sibling of Hardware Queues), keyed
    by gpu_id. Restricted to devices that ran GPU work so idle GPUs show no counters."""
    if not env or not len(env.get("start_ns", ())):
        return None
    ek = np.asarray(env["environment_kind"])
    data = np.asarray(env["data"], dtype=np.uint64)
    ts = np.ascontiguousarray(env["start_ns"], dtype=np.int64)
    dev = np.asarray(env["device_id"], dtype=np.int64)
    base = (
        np.isin(dev, list(active_devices))
        if active_devices
        else np.ones(len(dev), dtype=bool)
    )
    specs, gpu_l, ts_l, cid_l, val_l = [], [], [], [], []
    for name, kind_val, value_of in _ENV_COUNTERS:
        m = base & (ek == kind_val)
        if not m.any():
            continue
        cid = len(specs)  # local id; _merge_counters assigns the global id
        specs.append((cid, name))
        gpu_l.append(dev[m])
        ts_l.append(ts[m])
        cid_l.append(np.full(int(m.sum()), cid, dtype=np.int32))
        val_l.append(value_of(data[m]))
    if not specs:
        return None
    return (
        specs,
        np.concatenate(gpu_l).astype(np.int32),
        np.concatenate(ts_l).astype(np.int64),
        np.concatenate(cid_l).astype(np.int32),
        np.concatenate(val_l).astype(np.float64),
    )


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


# Per-kernel derived "Cycles" counter for the GPU Compute panel. It is *elapsed* cycles =
# duration * clock -- pure scalar math from the kernel's duration + the device SM clock, no perf
# counters (so it is a friendly display name, not the raw gpc__cycles_elapsed metric). SM
# Frequency is intentionally not emitted: it is the same value as the always-on "SM Clock (MHz)"
# environment counter. Local descriptor id 0 -- _merge_counters assigns the global id.
_CYCLE_COUNTER = (0, "Cycles (Current Kernel)")


def _device_clocks_hz(env: dict | None) -> dict[int, float]:
    """Median SM clock (Hz) per device from the sampled ENVIRONMENT SPEED records
    (environment_kind==1; smClock is the low u32 of the union, in MHz)."""
    if not env or not len(env.get("start_ns", ())):
        return {}
    ek = np.asarray(env["environment_kind"])
    m = ek == 1
    if not m.any():
        return {}
    sm_mhz = (
        np.asarray(env["data"], dtype=np.uint64)[m] & np.uint64(0xFFFFFFFF)
    ).astype(np.float64)
    dev = np.asarray(env["device_id"], dtype=np.int64)[m]
    return {int(d): float(np.median(sm_mhz[dev == d])) * 1e6 for d in np.unique(dev)}


def _build_cycle_counters(kernel: dict | None, env: dict | None, active_devices: set):
    """Per-kernel Cycles (gpc__cycles_elapsed = duration * clock) for the GPU Compute panel,
    derived as scalar math from each kernel's duration (activity record) and the device SM clock
    (env counter). One sample per kernel at its start, so every kernel gets an exact value (no
    sampling sparseness); placed in the COMPUTE group so the panel's kernel<->counter time-window
    join finds it, and emitted as an int (a cycle count). Returns the 5-tuple plus a 6th
    compute_group and 7th int_value_ids."""
    if not kernel or not len(kernel.get("start_ns", ())):
        return None
    clocks = _device_clocks_hz(env)
    if not clocks:
        return None
    cid = _CYCLE_COUNTER[0]
    ts = np.ascontiguousarray(kernel["start_ns"], dtype=np.int64)
    end = np.asarray(kernel["end_ns"], dtype=np.int64)
    dev = np.asarray(kernel["device_id"], dtype=np.int64)
    clk = np.array([clocks.get(int(d), 0.0) for d in dev], dtype=np.float64)
    base = (clk > 0) & (
        np.isin(dev, list(active_devices))
        if active_devices
        else np.ones(len(dev), dtype=bool)
    )
    if not base.any():
        return None
    devb = dev[base].astype(np.int32)
    cycles = (np.maximum(end - ts, 0).astype(np.float64)[base] / 1e9) * clk[base]
    return (
        [_CYCLE_COUNTER],
        devb,
        ts[base].astype(np.int64),
        np.full(len(devb), cid, np.int32),
        cycles,
        [cid],  # compute_group
        [cid],  # int_value_ids
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


def _gpu_annotation_render_column(
    trace_window: dict, base_ns: int
) -> dict[str, Any] | None:
    """Synthetic ``gpu_annotation`` render-stage column for the pftrace GPU stream lanes,
    built from the columnar window via the same synthesizer as the chrome gpu_user_annotation
    events -- so it lands on the kernels' capture stream, never a reassigned logical lane.
    None when there are no GPU annotations. Kineto emits no gpu_user_annotation in
    cuspy mode, so cpu_data cannot be the source (the chrome path synthesizes them too)."""
    gua = _gpu_user_annotation_events(trace_window, base_ns=base_ns)
    if not gua:
        return None
    ts_us = np.asarray([e["ts"] for e in gua], dtype=np.float64)
    dur_us = np.asarray([e["dur"] for e in gua], dtype=np.float64)
    a_ts = base_ns + np.round(ts_us * 1000.0).astype(np.int64)
    a_dur = np.maximum(np.round(dur_us * 1000.0).astype(np.int64), 0)
    return {
        "start_ns": a_ts,
        "end_ns": a_ts + a_dur,
        "device_id": np.asarray([cast(int, e["pid"]) for e in gua], dtype=np.int64),
        "stream_id": np.asarray([cast(int, e["tid"]) for e in gua], dtype=np.int64),
        "name": np.asarray([str(e["name"]) for e in gua], dtype=object),
    }


def _window_to_pftrace(
    cpu_data: dict,
    trace_window: dict,
    base_ns: int,
    output_path: str,
    compression_level: int = 1,
) -> None:
    """Encode Cuspy's columnar window straight to a Perfetto-native trace (.pftrace),
    concatenated with the Kineto CPU events -- NO chrome-dict materialization. Full parity with
    the chrome path's per-event args, ac2g flows, and collective metadata, emitted as
    TrackEvent debug_annotations + flow_ids by the native encoder. The GPU kinds are assembled
    vectorized from their numpy columns into per-kind groups; runtime/driver use a per-record
    CPU-thread join (as the chrome path does)."""
    columns = cast("dict[str, dict[str, Any]]", trace_window.get("columns", {}))
    thread_resource_map = cast(
        "dict[int, dict[int, int]]", trace_window.get("thread_resource_map", {})
    )
    uuids: dict = {}
    pid_ints: dict = {}
    proc_name: dict = {}
    thr_name: dict = {}
    tracks: list = []
    groups: list = []  # raw group dicts; name_iid filled in after global name interning
    gpu_track_uuids: set = set()  # stream tracks to nest post-assembly (serial slices)

    def _col(ks):
        c = columns.get(ks)
        return c if c and len(next(iter(c.values()))) else None

    def pid_int(pid):
        # int32 descriptor id (cosmetic); map any id to a stable NON-negative value.
        if isinstance(pid, int):
            return pid & 0x7FFFFFFF
        return pid_ints.setdefault(pid, len(pid_ints) + 1)

    def add_track(uuid, parent, is_proc, pid, tid, name):
        tracks.append((uuid, parent, is_proc, pid_int(pid), pid_int(tid), str(name)))

    def track_for(pid, tid, proc_label="", thr_label=""):
        if ("p", pid) not in uuids:
            uuids[("p", pid)] = len(uuids) + 1
            add_track(
                uuids[("p", pid)],
                0,
                True,
                pid,
                0,
                str(proc_name.get(pid) or proc_label or pid),
            )
        key = ("t", pid, tid)
        if key not in uuids:
            uuids[key] = len(uuids) + 1
            add_track(
                uuids[key],
                uuids[("p", pid)],
                False,
                pid,
                tid,
                str(thr_name.get(key) or thr_label or tid),
            )
        return uuids[key]

    def gpu_uuids(pid_arr, tid_arr, proc_fmt, thr_fmt, is_gpu=True):
        # (pid, tid) -> uuid for whole columns: unique pairs (packed into one int64 key) each
        # register a track; map back via the inverse index (no per-event Python). is_gpu marks
        # GPU stream tracks (serial slices) for the post-assembly overlap nesting.
        key = (pid_arr.astype(np.int64) << np.int64(32)) | (
            tid_arr.astype(np.int64) & 0xFFFFFFFF
        )
        uniq, inv = np.unique(key, return_inverse=True)
        ids = np.empty(len(uniq), dtype=np.uint64)
        for j, k in enumerate(uniq.tolist()):
            pid = int(k >> 32)
            tid = int(k & 0xFFFFFFFF)
            if tid >= 0x80000000:  # sign-extend the packed int32 tid
                tid -= 0x100000000
            ids[j] = track_for(pid, tid, proc_fmt.format(pid), thr_fmt.format(tid))
            if is_gpu:
                gpu_track_uuids.add(int(ids[j]))
        return ids[inv]

    def int_anno(key, col, skip_zero=False, present=None):
        # present: optional per-slice uint8 mask (emit the int only where set).
        return (
            key,
            np.ascontiguousarray(col, dtype=np.int64),
            skip_zero,
            None if present is None else np.ascontiguousarray(present, dtype=np.uint8),
        )

    def json_anno(blobs):
        # blobs: per-slice str/bytes/None -> a CSR (offsets, buffer) varlen column; empty None.
        enc = [
            b"" if b is None else (b if isinstance(b, bytes) else b.encode())
            for b in blobs
        ]
        lengths = np.fromiter((len(b) for b in enc), dtype=np.int32, count=len(enc))
        offsets = np.zeros(len(enc) + 1, dtype=np.int32)
        np.cumsum(lengths, out=offsets[1:])
        return (offsets, b"".join(enc))

    def add_group(
        ts,
        end,
        uu,
        names,
        ints=(),
        strs=(),
        arrs=(),
        jsons=(),
        flow=None,
        gpu_corr=None,
        cats=None,
    ):
        # cats: per-slice category strings (or None), interned into category_iids so
        # the viewer distinguishes cpu_op / cuda_runtime / cuda_driver / user_annotation.
        groups.append(
            {
                "ts": np.ascontiguousarray(ts, dtype=np.int64),
                "end": np.ascontiguousarray(end, dtype=np.int64),
                "uuid": np.ascontiguousarray(uu, dtype=np.uint64),
                "names": names,
                "ints": list(ints),
                "strs": list(strs),
                "arrs": list(arrs),
                "jsons": list(jsons),
                "flow": None
                if flow is None
                else np.ascontiguousarray(flow, dtype=np.int64),
                # gpu_corr: per-slice list of render-stage event_ids this slice launched
                # (empty for non-launch slices), stored as a CSR (offsets, ids) for the native
                # GpuCorrelation.render_stage_submission_event_ids. None when no gpu_corr given.
                "gpu_corr": None
                if gpu_corr is None
                else _lists_to_csr(gpu_corr, np.int64),
                "cats": cats,
            }
        )

    # --- joins shared with the chrome path ---
    context_to_device: dict = {}
    for ks in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_event"):
        c = _col(ks)
        if c is None or "context_id" not in c:
            continue
        for ctx, dev in zip(c["context_id"].tolist(), c["device_id"].tolist()):
            context_to_device.setdefault(ctx, dev)
    event_sync_to_corr: dict = {}
    ce = _col("cuda_event")
    if ce is not None:
        event_sync_to_corr = {
            sid: corr
            for sid, corr in zip(
                ce["cuda_event_sync_id"].tolist(), ce["correlation_id"].tolist()
            )
            if sid
        }

    # --- CPU side: Kineto chrome events (M names + X slices with their args) ---
    for e in cpu_data.get("traceEvents", []):
        if isinstance(e, dict) and e.get("ph") == "M":
            a = e.get("args") or {}
            if e.get("name") == "process_name":
                proc_name[e.get("pid")] = a.get("name", "")
            elif e.get("name") == "thread_name":
                thr_name[("t", e.get("pid"), e.get("tid"))] = a.get("name", "")
    cpu_thread_by_external_id: dict = {}
    cpu_ts_raw: list = []
    cpu_dur_raw: list = []
    cpu_uuid: list = []
    cpu_names: list = []
    cpu_cats: list = []
    cpu_args: list = []
    has_args = False
    for e in cpu_data.get("traceEvents", []):
        if (
            not isinstance(e, dict)
            or e.get("ph") != "X"
            or e.get("cat") in _NON_CPU_CATS
        ):
            continue
        a = e.get("args") if isinstance(e.get("args"), dict) else {}
        if e.get("cat") in ("cpu_op", "user_annotation") and "External id" in a:
            try:
                cpu_thread_by_external_id[int(a["External id"])] = (
                    e.get("pid"),
                    e.get("tid"),
                )
            except (TypeError, ValueError):
                pass
        cpu_ts_raw.append(e.get("ts", 0.0))
        cpu_dur_raw.append(e.get("dur", 0.0))
        cpu_uuid.append(track_for(e.get("pid"), e.get("tid")))
        cpu_names.append(str(e.get("name", "")))
        cpu_cats.append(str(e.get("cat", "")))
        # orjson (when present) emits bytes ~5x faster than stdlib json; json_anno takes
        # either. The native side parses the blob, so this avoids a slow per-event dumps.
        cpu_args.append(
            (_orjson.dumps(a) if _orjson is not None else json.dumps(a)) if a else None
        )
        has_args = has_args or bool(a)
    if cpu_ts_raw:
        ts_arr = base_ns + np.round(
            np.asarray(cpu_ts_raw, dtype=np.float64) * 1000.0
        ).astype(np.int64)
        end_arr = ts_arr + np.round(
            np.asarray(cpu_dur_raw, dtype=np.float64) * 1000.0
        ).astype(np.int64)
        jsons = [json_anno(cpu_args)] if has_args else ()
        add_group(ts_arr, end_arr, cpu_uuid, cpu_names, jsons=jsons, cats=cpu_cats)

    # corr -> CPU thread, for the runtime/driver thread remap (via the external-corr column).
    cpu_thread_by_correlation_id: dict = {}
    ext = _col("external_correlation")
    if ext is not None:
        for corr, eid in zip(
            ext["correlation_id"].tolist(), ext["external_id"].tolist()
        ):
            if corr and eid in cpu_thread_by_external_id:
                cpu_thread_by_correlation_id[corr] = cpu_thread_by_external_id[eid]

    # GPU ops (kernel / memcpy / memset) are emitted only as native GPU Render Stages
    # (one lane per stream, with launch stats + scalar args + collective metadata) by
    # _build_render_stages -- not as duplicate track_event slices here.

    # Pre-pass: assign a unique event_id per GPU render-stage row (row order matches
    # _build_render_stages) and derive the host-launch link (corr_to_event_ids) + the graph
    # node->node dependency arrows (the event_wait CSR). Done before the runtime/driver group so
    # its gpu_correlation can list the per-replay node event_ids; _build_render_stages is still
    # called later with these.
    graph_deps = {int(k): v for k, v in (trace_window.get("graph_deps") or {}).items()}
    # GPU-side user annotations (gpu_user_annotation): a synthetic "Annotation" render stage on
    # the kernels' capture-stream lane -- never a reassigned logical lane (matching the chrome
    # path). Added to render_columns before event-id assignment so the annotation rows share the
    # render-stage row order. Synthesized from the columnar window (kineto emits no
    # gpu_user_annotation in Cuspy mode; the chrome path builds them the same way).
    render_columns = columns
    ann_col = _gpu_annotation_render_column(trace_window, base_ns)
    if ann_col is not None:
        render_columns = {**columns, "gpu_annotation": ann_col}
    render_event = _assign_render_event_ids(render_columns, graph_deps)
    if render_event is not None:
        render_event_id, corr_to_event_ids, wait_offsets, wait_ids = render_event
    else:
        render_event_id, corr_to_event_ids = None, {}
        wait_offsets, wait_ids = None, None

    # --- runtime / driver API: registered names only, remapped onto their CPU thread ---
    for ks in ("cuda_runtime", "cuda_driver"):
        c = _col(ks)
        if c is None:
            continue
        is_rt = ks == "cuda_runtime"
        namer = _runtime_cbid_name if is_rt else _driver_cbid_name
        is_reg = _runtime_is_registered if is_rt else _driver_is_registered
        # gpu_correlation (host launch -> its GPU render-stage kernel) on launches, by the
        # kernel whose event_id == this correlation. Eager launches pair 1:1; cudaGraphLaunch
        # shares one correlation across all of a graph's replayed kernels, so it links to the
        # whole graph at once -- fine here because gpu_correlation only draws on selection (no
        # spiderweb, unlike always-on flow_ids). Syncs are excluded (they match no kernel).
        launch_pred = _runtime_is_launch if is_rt else _driver_requires_flow
        uniq_cb, inv = np.unique(c["cbid"], return_inverse=True)
        names_u = [namer(int(x)) for x in uniq_cb.tolist()]
        reg_u = np.array([is_reg(nm) for nm in names_u], dtype=bool)
        corr_u = np.array([launch_pred(nm) for nm in names_u], dtype=bool)
        keep = np.nonzero(reg_u[inv])[0]
        if not len(keep):
            continue
        pid = c["process_id"][keep].tolist()
        corr = c["correlation_id"][keep]
        normtid = (
            c["thread_id"][keep].astype(np.uint32).astype(np.int32).astype(np.int64)
        )
        tid = np.fromiter(
            (
                cpu_thread_by_correlation_id[cc][1]
                if cc in cpu_thread_by_correlation_id
                and cpu_thread_by_correlation_id[cc][0] == p
                else thread_resource_map.get(p, {}).get(nt, nt)
                for p, cc, nt in zip(pid, corr.tolist(), normtid.tolist())
            ),
            dtype=np.int64,
            count=len(keep),
        )
        uu = gpu_uuids(
            c["process_id"][keep], tid, "process {}", "thread {}", is_gpu=False
        )
        names = [names_u[i] for i in inv[keep].tolist()]
        # Per launch slice, the render-stage event_ids it links to (a graph launch -> its
        # replay's ROOT node event_ids, an eager launch -> the one kernel's). Non-launch slices
        # get an empty list. Links the CPU launch to its GPU render stages via gpu_correlation.
        launch_mask = corr_u[inv[keep]].tolist()
        gpu_corr = [
            corr_to_event_ids.get(int(cc), []) if lm else []
            for cc, lm in zip(corr.tolist(), launch_mask)
        ]
        ints = [int_anno("cbid", c["cbid"][keep]), int_anno("correlation", corr)]
        add_group(
            c["start_ns"][keep],
            c["end_ns"][keep],
            uu,
            names,
            ints,
            gpu_corr=gpu_corr,
            cats=[ks] * len(keep),
        )

    # --- overhead (own lane), dropping the trailing buffer-request artifact ---
    c = _col("overhead")
    if c is not None:
        max_non_overhead_end = 0
        for ks, col in columns.items():
            if ks in ("overhead", "external_correlation", "cuda_event"):
                continue
            if col and "end_ns" in col and len(col["end_ns"]):
                max_non_overhead_end = max(
                    max_non_overhead_end, int(col["end_ns"].max())
                )
        name_l = c["name"].tolist()
        starts = c["start_ns"]
        keep = np.array(
            [
                not (
                    nm == "Activity Buffer Request"
                    and max_non_overhead_end > 0
                    and int(starts[i]) > max_non_overhead_end
                )
                for i, nm in enumerate(name_l)
            ],
            dtype=bool,
        )
        idx = np.nonzero(keep)[0]
        if len(idx):
            u = track_for(_OVERHEAD_PID, 0, "Overhead", "Overhead")
            uu = np.full(len(idx), u, dtype=np.uint64)
            add_group(
                starts[idx], c["end_ns"][idx], uu, [name_l[i] for i in idx.tolist()]
            )

    # --- cuda_sync: device via context, stream via the sync record, wait_on join ---
    c = _col("cuda_sync")
    if c is not None:
        st = c["sync_type"]
        ctx = c["context_id"]
        device = np.fromiter(
            (context_to_device.get(int(x), 0) for x in ctx.tolist()),
            dtype=np.int64,
            count=len(ctx),
        )
        stream = np.where(c["stream_id"] == _SYNC_INVALID, -1, c["stream_id"]).astype(
            np.int64
        )
        corr = c["correlation_id"]
        # Split on wait-bearing sync types (Event Sync / Stream Wait Event) so only those rows
        # carry the wait_on_* annotations, matching the chrome path.
        wait = np.isin(st, (1, 2))
        for sub in (np.nonzero(~wait)[0], np.nonzero(wait)[0]):
            if not len(sub):
                continue
            uu = gpu_uuids(device[sub], stream[sub], "GPU {}", "stream {}")
            names = [
                _SYNC_TYPE_NAMES.get(int(s), f"sync_{int(s)}") for s in st[sub].tolist()
            ]
            ints = [
                int_anno("stream", stream[sub]),
                int_anno("correlation", corr[sub]),
                int_anno("device", device[sub]),
                int_anno("context", ctx[sub]),
            ]
            # cuda_sync_kind is the slice name (string, with the sync_<n> fallback) -- not
            # the raw-int fallback the copy/memory enums use.
            uniqn, name_inv = np.unique(
                np.asarray(names, dtype=object), return_inverse=True
            )
            strs = [
                (
                    "cuda_sync_kind",
                    name_inv.astype(np.int64),
                    [str(n) for n in uniqn.tolist()],
                )
            ]
            if np.isin(st[sub], (1, 2)).all():
                rec_corr = np.fromiter(
                    (
                        event_sync_to_corr.get(int(s), -1)
                        for s in c["cuda_event_sync_id"][sub].tolist()
                    ),
                    dtype=np.int64,
                    count=len(sub),
                )
                ints += [
                    int_anno("wait_on_stream", np.full(len(sub), -1, dtype=np.int64)),
                    int_anno("wait_on_cuda_event_id", c["cuda_event_id"][sub]),
                    int_anno("wait_on_cuda_event_record_corr_id", rec_corr),
                ]
            add_group(c["start_ns"][sub], c["end_ns"][sub], uu, names, ints, strs)

    _nest_track_slices(groups, gpu_track_uuids)

    # Global name interning: one EventName table across all groups; each slice carries a
    # name_iid into it. A first-seen dict intern (O(n) hashes) beats np.unique here, whose
    # argsort over the object-array of names is the dominant assembly cost.
    iid_of: dict = {}
    name_table: list = []
    # Categories are interned in their own iid space (EventCategory), referenced
    # per-slice by cat_iid; a "" category maps to 0 (no category emitted).
    cat_iid_of: dict = {}
    cat_table: list = []
    group_tuples = []
    for g in groups:
        names = g["names"]
        names = names.tolist() if isinstance(names, np.ndarray) else names
        ids = []
        for nm in names:
            j = iid_of.get(nm)
            if j is None:
                j = iid_of[nm] = len(name_table)
                name_table.append(nm)
            ids.append(j + 1)
        cats = g["cats"]
        if cats is None:
            cat_ids = None
        else:
            cids = []
            for ct in cats:
                if not ct:
                    cids.append(0)
                    continue
                j = cat_iid_of.get(ct)
                if j is None:
                    j = cat_iid_of[ct] = len(cat_table)
                    cat_table.append(ct)
                cids.append(j + 1)
            cat_ids = np.asarray(cids, dtype=np.uint64)
        group_tuples.append(
            (
                g["ts"],
                g["end"],
                g["uuid"],
                np.asarray(ids, dtype=np.uint64),
                g["ints"],
                g["strs"],
                g["arrs"],
                g["jsons"],
                g["flow"],
                g["gpu_corr"],
                cat_ids,
            )
        )
    # render_columns and the per-row event_id / event_wait CSR were built in the pre-pass above.
    # Graphics context attaches the render stages to the main (first-registered => upid 1)
    # process, matching the reference traces.
    gfx_pid = next((k[1] for k in uuids if isinstance(k, tuple) and k[0] == "p"), 0)
    render = (
        _build_render_stages(
            render_columns,
            gfx_pid,
            iid_of,
            name_table,
            render_event_id,
            (wait_offsets, wait_ids),
        )
        if render_event_id is not None
        else None
    )
    active_devices = _active_devices(columns)
    # GPU counters (power/temp/clocks) -> GpuCounterEvents: the viewer renders them under
    # "GPU / Counters / <gpu>", a sibling of the render-stage stream lanes, keyed by gpu_id.
    counters = _merge_counters(
        _build_gpu_counters(columns.get("environment"), active_devices),
        _build_pm_counters(columns.get("pm_sampling"), active_devices),
        _build_cycle_counters(
            columns.get("kernel"), columns.get("environment"), active_devices
        ),
    )
    # encode_pftrace returns gzip-compressed bytes (compressed in C++), so write as-is.
    out = torch._C._profiler._cuspy.encode_pftrace(
        base_ns,
        tracks,
        name_table,
        cat_table,
        group_tuples,
        render,
        counters,
        compression_level,
    )
    with open(output_path, "wb") as f:
        f.write(out)


def merge_trace_window_into_chrome_trace(
    cpu_trace_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    trace_window: dict[str, object],
    *,
    trace_name: str | None = None,
    pftrace_compression_level: int = 1,
) -> None:
    cpu_trace_path = str(cpu_trace_path)
    output_path = str(output_path)
    input_opener = gzip.open if cpu_trace_path.endswith(".gz") else open
    with input_opener(cpu_trace_path, "rb") as f:
        raw = f.read()
    data = _orjson.loads(raw) if _orjson is not None else json.loads(raw)

    base_ns = int(data.get("baseTimeNanoseconds", _default_base_ns()))
    # Perfetto-native output: encode straight from the columnar window (skip building the
    # chrome event dicts entirely -- that build dominates the JSON path).
    if ".pftrace" in output_path:
        _window_to_pftrace(
            data, trace_window, base_ns, output_path, pftrace_compression_level
        )
        return
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

    # GPU counters (env power/temp/clocks, PM utilization, per-kernel cycles) as chrome "C" events
    # on the device pid -- the JSON counterpart of the .pftrace GPU counter tracks.
    columns = cast("dict[str, dict[str, Any]]", trace_window.get("columns", {}))
    active_devices = _active_devices(columns)
    events.extend(
        _build_chrome_counters(
            _merge_counters(
                _build_gpu_counters(columns.get("environment"), active_devices),
                _build_pm_counters(columns.get("pm_sampling"), active_devices),
                _build_cycle_counters(
                    columns.get("kernel"), columns.get("environment"), active_devices
                ),
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

    # (.pftrace is handled earlier, straight from the columnar window.)
    if _orjson is not None:
        payload = _orjson.dumps(data)
    else:
        payload = json.dumps(data, separators=(",", ":")).encode()
    # Encode once and write the whole buffer, gzipped when the path ends .gz (compresslevel=1
    # favors throughput over size).
    if output_path.endswith(".gz"):
        with gzip.open(output_path, "wb", compresslevel=1) as f:
            f.write(payload)
    else:
        with open(output_path, "wb") as f:
            f.write(payload)
