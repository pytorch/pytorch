
"""Profiler feature on the CUPTI mux: observer, session, and backend.

  ``ProfilerObserver`` -- records full GPU activity (kernel/memcpy/memset)
    by registering a field selection on the shared mux; ``drain()``
    force-flushes the mux and returns the demuxed columns.
  ``ProfilerSession``  -- start/collect/stop lifecycle around one observer.
  ``CuptiProfiler``    -- context manager that bounds a window and exports a
    chrome trace (via the module-local ``_to_chrome_trace``).

The mux owns the CUPTI subscriber, the v2 user-defined config, and the
parse; everything here just consumes and serializes, costing only what the
field selection adds to the union. Timestamps come through in CUPTI's clock
domain; map them to unix-epoch ns with ``convert_time``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import threading
from typing import Any, Callable, Optional

from torch.profiler.cupti.observers.base import MuxObserver
from torch.profiler.cupti.types import (
    ActivityKind,
    ApiField,
    ExternalCorrelationField,
    KernelField,
    MemcpyField,
    MemsetField,
    OverheadField,
    SyncField,
)


# Full per-event metadata for a rich chrome trace (mirrors the standalone
# CuptiMonitor): span (START/END), kernel NAME (works eager + graph), placement
# (DEVICE/CONTEXT/STREAM), linkage (CORRELATION_ID), graph context (GRAPH_NODE_
# ID/GRAPH_ID), and per-kind payload (memcpy BYTES + copy/src/dst kinds + flags;
# memset BYTES + VALUE + MEMORY_KIND + flags). This is the windowed profiler --
# the always-on NodeTimerObserver stays minimal; these wider records only cost
# bandwidth during a capture.
_DEFAULT_WANTS: dict[int, "set[int]"] = {
    ActivityKind.CONCURRENT_KERNEL: {
        KernelField.START,
        KernelField.END,
        KernelField.NAME,
        KernelField.DEVICE_ID,
        KernelField.CONTEXT_ID,
        KernelField.STREAM_ID,
        KernelField.CORRELATION_ID,
        KernelField.GRAPH_NODE_ID,
        KernelField.GRAPH_ID,
    },
    ActivityKind.MEMCPY: {
        MemcpyField.START,
        MemcpyField.END,
        MemcpyField.BYTES,
        MemcpyField.COPY_KIND,
        MemcpyField.SRC_KIND,
        MemcpyField.DST_KIND,
        MemcpyField.FLAGS,
        MemcpyField.DEVICE_ID,
        MemcpyField.CONTEXT_ID,
        MemcpyField.STREAM_ID,
        MemcpyField.CORRELATION_ID,
        MemcpyField.GRAPH_NODE_ID,
        MemcpyField.GRAPH_ID,
    },
    ActivityKind.MEMSET: {
        MemsetField.START,
        MemsetField.END,
        MemsetField.BYTES,
        MemsetField.VALUE,
        MemsetField.MEMORY_KIND,
        MemsetField.FLAGS,
        MemsetField.DEVICE_ID,
        MemsetField.CONTEXT_ID,
        MemsetField.STREAM_ID,
        MemsetField.CORRELATION_ID,
        MemsetField.GRAPH_NODE_ID,
        MemsetField.GRAPH_ID,
    },
    # CPU-side activities for a merged trace (CPU timeline + launch->kernel
    # flow arrows). RUNTIME/DRIVER are CUpti_ActivityAPI (cbid identifies the
    # call); they join GPU records by CORRELATION_ID. These raise the profiled-
    # window cost vs a GPU-only trace, but only during a capture window.
    ActivityKind.RUNTIME: {
        ApiField.CBID,
        ApiField.START,
        ApiField.END,
        ApiField.PROCESS_ID,
        ApiField.THREAD_ID,
        ApiField.CORRELATION_ID,
    },
    ActivityKind.DRIVER: {
        ApiField.CBID,
        ApiField.START,
        ApiField.END,
        ApiField.PROCESS_ID,
        ApiField.THREAD_ID,
        ApiField.CORRELATION_ID,
    },
    ActivityKind.OVERHEAD: {
        OverheadField.OVERHEAD_KIND,
        OverheadField.PROCESS_ID,
        OverheadField.THREAD_ID,
        OverheadField.START,
        OverheadField.END,
        OverheadField.CORRELATION_ID,
    },
    ActivityKind.SYNCHRONIZATION: {
        SyncField.TYPE,
        SyncField.START,
        SyncField.END,
        SyncField.CORRELATION_ID,
        SyncField.CONTEXT_ID,
        SyncField.STREAM_ID,
    },
    # External correlation maps a CUDA correlation_id to a user-pushed
    # external_id (see ProfilerObserver.annotate) -> gpu_user_annotation.
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelationField.EXTERNAL_KIND,
        ExternalCorrelationField.EXTERNAL_ID,
        ExternalCorrelationField.CORRELATION_ID,
    },
}

# A GPU-only field selection (no CPU-side kinds) for callers that want the
# cheap, low-distortion trace (the always-on regime); pass as ``wants``.
GPU_ONLY_WANTS: dict[int, "set[int]"] = {
    k: set(v)
    for k, v in _DEFAULT_WANTS.items()
    if k
    in (
        ActivityKind.CONCURRENT_KERNEL,
        ActivityKind.MEMCPY,
        ActivityKind.MEMSET,
    )
}

# CUpti_ActivityMemcpyKind / CUpti_ActivityMemoryKind -> readable labels.
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

_OVERHEAD_KIND_NAMES = {
    0: "unknown",
    1: "driver_compiler",
    2: "cupti_buffer_flush",
    3: "cupti_instrumentation",
    4: "cupti_resource",
    5: "runtime_trigger",
    6: "command_buffer",
}

# CUpti_ActivitySynchronizationType.
_SYNC_TYPE_NAMES = {
    0: "unknown",
    1: "event_synchronize",
    2: "stream_wait_event",
    3: "stream_synchronize",
    4: "context_synchronize",
}

# Runtime cbids whose CPU events are pure noise; dropped from the trace
# (mirrors the standalone CuptiMonitor's blocklist).
_RUNTIME_BLOCKLIST = frozenset(
    {
        "cudaGetDevice",
        "cudaSetDevice",
        "cudaGetLastError",
        "cudaEventCreate",
        "cudaEventCreateWithFlags",
        "cudaEventDestroy",
        "cudaPeekAtLastError",
    }
)
# Runtime/driver cbids that launch device work -> get a flow arrow (ac2g) to
# the GPU activity sharing their correlation_id.
_RUNTIME_FLOW_NAMES = frozenset(
    {
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaLaunchCooperativeKernel",
        "cudaLaunchCooperativeKernelMultiDevice",
        "cudaGraphLaunch",
    }
)
_DRIVER_FLOW_NAMES = frozenset({"cuLaunchKernel", "cuLaunchKernelEx"})

# cupti cbid-enum -> {id: name}, loaded lazily/best-effort from the cupti
# python binding. Without cupti we fall back to "cbid_<n>".
_RUNTIME_CBIDS: "dict[int, str] | None" = None
_DRIVER_CBIDS: "dict[int, str] | None" = None


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


def _cbid_name(kind: int, cbid: int) -> str:
    """Map a runtime/driver callback id to its API name (e.g. cudaLaunchKernel).
    Best-effort: needs the cupti binding's cbid enums."""
    global _RUNTIME_CBIDS, _DRIVER_CBIDS
    is_runtime = kind == ActivityKind.RUNTIME
    table = _RUNTIME_CBIDS if is_runtime else _DRIVER_CBIDS
    if table is None:
        try:
            from cupti import cupti as _cc

            enum_cls = (
                _cc.Runtime_api_trace_cbid if is_runtime else _cc.Driver_api_trace_cbid
            )
            table = _load_cbid_names(enum_cls)
        except Exception:
            table = {}
        if is_runtime:
            _RUNTIME_CBIDS = table
        else:
            _DRIVER_CBIDS = table
    return table.get(cbid, f"cbid_{cbid}")

# kind -> chrome-trace spec: span fields, placement (device/stream -> pid/tid
# lanes), the default track label, an optional per-record NAME field, and
# {arg_name: field id} extras surfaced in the event ``args``.
_TRACE = {
    ActivityKind.CONCURRENT_KERNEL: {
        "start": KernelField.START,
        "end": KernelField.END,
        "device": KernelField.DEVICE_ID,
        "stream": KernelField.STREAM_ID,
        "track": "kernel",
        "name_field": KernelField.NAME,
        "args": {
            "context_id": KernelField.CONTEXT_ID,
            "correlation_id": KernelField.CORRELATION_ID,
            "graph_node_id": KernelField.GRAPH_NODE_ID,
            "graph_id": KernelField.GRAPH_ID,
        },
    },
    ActivityKind.MEMCPY: {
        "start": MemcpyField.START,
        "end": MemcpyField.END,
        "device": MemcpyField.DEVICE_ID,
        "stream": MemcpyField.STREAM_ID,
        "track": "memcpy",
        "name_field": None,
        "args": {
            "bytes": MemcpyField.BYTES,
            "copy_kind": MemcpyField.COPY_KIND,
            "src_kind": MemcpyField.SRC_KIND,
            "dst_kind": MemcpyField.DST_KIND,
            "flags": MemcpyField.FLAGS,
            "context_id": MemcpyField.CONTEXT_ID,
            "correlation_id": MemcpyField.CORRELATION_ID,
            "graph_node_id": MemcpyField.GRAPH_NODE_ID,
            "graph_id": MemcpyField.GRAPH_ID,
        },
    },
    ActivityKind.MEMSET: {
        "start": MemsetField.START,
        "end": MemsetField.END,
        "device": MemsetField.DEVICE_ID,
        "stream": MemsetField.STREAM_ID,
        "track": "memset",
        "name_field": None,
        "args": {
            "bytes": MemsetField.BYTES,
            "value": MemsetField.VALUE,
            "memory_kind": MemsetField.MEMORY_KIND,
            "flags": MemsetField.FLAGS,
            "context_id": MemsetField.CONTEXT_ID,
            "correlation_id": MemsetField.CORRELATION_ID,
            "graph_node_id": MemsetField.GRAPH_NODE_ID,
            "graph_id": MemsetField.GRAPH_ID,
        },
    },
}


def _demangle(name: str, _cache: dict[str, str] = {}) -> str:
    """Best-effort C++ demangle of a kernel symbol (``_Z...``) via libstdc++'s
    ``__cxa_demangle``; returns the input unchanged if demangling isn't
    available or fails. Cached, since kernel names repeat heavily."""
    if not name or not name.startswith("_Z"):
        return name
    cached = _cache.get(name)
    if cached is not None:
        return cached
    out = name
    try:
        import ctypes

        lib = ctypes.CDLL("libstdc++.so.6")
        fn = lib.__cxa_demangle
        fn.restype = ctypes.c_void_p
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        status = ctypes.c_int()
        res = fn(name.encode(), None, None, ctypes.byref(status))
        if status.value == 0 and res:
            out = ctypes.string_at(res).decode("utf-8", "replace")
            # The malloc'd buffer is intentionally NOT freed: results are cached
            # per unique name (bounded), and a mistyped free() across ctypes
            # truncates the 64-bit pointer and corrupts the heap.
    except Exception:
        pass
    _cache[name] = out
    return out


def _annotation_for(intervals: "list[tuple[int, int, str]]", ts: int) -> "str | None":
    """Innermost (latest-started) annotation interval [start, end] containing
    ``ts``. ``intervals`` is sorted by start. Linear scan -- annotation counts
    per window are small."""
    best: "str | None" = None
    best_start = -1
    for start, end, name in intervals:
        if start > ts:
            break
        if ts <= end and start > best_start:
            best, best_start = name, start
    return best


def _gpu_events(
    data: dict[int, dict[int, Any]],
    name_resolver: Optional[Callable[[int], "str | None"]],
    launch_ts_by_corr: dict[int, int],
    intervals: "list[tuple[int, int, str]]",
    gpu_by_corr: dict[int, "tuple[str, str, float]"],
    corr_to_name: dict[int, str],
) -> "list[dict[str, Any]]":
    """GPU activity events (kernel/memcpy/memset). Also records each event's
    (pid,tid,ts) by correlation_id for flow-arrow ends, and attaches the
    enclosing user annotation -- preferring the external-correlation mapping
    (correlation_id -> external_id -> name), falling back to bracketing the
    CPU launch time within an annotation span."""
    events: list[dict[str, Any]] = []
    for kind, fields in data.items():
        spec = _TRACE.get(kind)
        if spec is None:
            continue
        start = fields.get(spec["start"])
        end = fields.get(spec["end"])
        if start is None or end is None:
            continue
        device = fields.get(spec["device"])
        stream = fields.get(spec["stream"])
        names = fields.get(spec["name_field"]) if spec["name_field"] else None
        gnode = fields.get(KernelField.GRAPH_NODE_ID)
        corr = fields.get(spec["args"].get("correlation_id"))
        argspec = spec["args"]
        track = spec["track"]
        for i in range(len(start)):
            s = int(start[i])
            e = int(end[i])
            args: dict[str, Any] = {}
            for arg_name, fid in argspec.items():
                col = fields.get(fid)
                if col is not None:
                    args[arg_name] = int(col[i])
            if "copy_kind" in args:
                args["copy_kind"] = _MEMCPY_KIND_NAMES.get(
                    args["copy_kind"], args["copy_kind"]
                )
            if "memory_kind" in args:
                args["memory_kind"] = _MEMORY_KIND_NAMES.get(
                    args["memory_kind"], args["memory_kind"]
                )
            if kind == ActivityKind.CONCURRENT_KERNEL:
                name = None
                if names is not None and names[i]:
                    name = _demangle(str(names[i]))
                if name is None and name_resolver is not None and gnode is not None:
                    name = name_resolver(int(gnode[i]))
                name = name or "kernel"
            elif kind == ActivityKind.MEMCPY:
                ck = args.get("copy_kind")
                name = f"Memcpy {ck}" if isinstance(ck, str) else "Memcpy"
            else:
                name = "Memset"
            pid = f"GPU {int(device[i])}" if device is not None else "GPU"
            tid = f"stream {int(stream[i])}" if stream is not None else track
            cid = int(corr[i]) if corr is not None else None
            if cid is not None:
                gpu_by_corr.setdefault(cid, (pid, tid, s / 1000.0))
                ann = corr_to_name.get(cid)
                if ann is None and intervals:
                    launch = launch_ts_by_corr.get(cid)
                    ann = _annotation_for(intervals, launch) if launch else None
                if ann is not None:
                    args["annotation"] = ann
            events.append(
                {
                    "name": name,
                    "ph": "X",
                    "ts": s / 1000.0,
                    "dur": (e - s) / 1000.0,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
    return events


def _cpu_events(
    data: dict[int, dict[int, Any]],
    launch_ts_by_corr: dict[int, int],
    flow_src_by_corr: dict[int, "tuple[str, str, float]"],
) -> "list[dict[str, Any]]":
    """Runtime/driver API events on CPU process/thread lanes (named by cbid),
    plus SYNCHRONIZATION and OVERHEAD lanes. Records launch timestamps (by
    correlation_id) for annotation bracketing and flow-arrow starts."""
    events: list[dict[str, Any]] = []
    for kind in (ActivityKind.RUNTIME, ActivityKind.DRIVER):
        fields = data.get(kind)
        if not fields:
            continue
        cbid = fields.get(ApiField.CBID)
        start = fields.get(ApiField.START)
        end = fields.get(ApiField.END)
        pid_c = fields.get(ApiField.PROCESS_ID)
        tid_c = fields.get(ApiField.THREAD_ID)
        corr = fields.get(ApiField.CORRELATION_ID)
        if start is None or end is None or cbid is None:
            continue
        flow_names = (
            _RUNTIME_FLOW_NAMES if kind == ActivityKind.RUNTIME else _DRIVER_FLOW_NAMES
        )
        for i in range(len(start)):
            name = _cbid_name(kind, int(cbid[i]))
            if name in _RUNTIME_BLOCKLIST:
                continue
            s = int(start[i])
            e = int(end[i])
            pid = f"CPU {int(pid_c[i])}" if pid_c is not None else "CPU"
            tid = f"thread {int(tid_c[i])}" if tid_c is not None else "cpu"
            cid = int(corr[i]) if corr is not None else None
            if cid is not None:
                launch_ts_by_corr.setdefault(cid, s)
                if name in flow_names or name.startswith(
                    ("cudaMemcpy", "cudaMemset")
                ):
                    flow_src_by_corr.setdefault(cid, (pid, tid, s / 1000.0))
            events.append(
                {
                    "name": name,
                    "ph": "X",
                    "ts": s / 1000.0,
                    "dur": (e - s) / 1000.0,
                    "pid": pid,
                    "tid": tid,
                    "args": {"correlation_id": cid} if cid is not None else {},
                }
            )
    # Synchronization (CPU-initiated) on a dedicated lane.
    sync = data.get(ActivityKind.SYNCHRONIZATION)
    if sync:
        start = sync.get(SyncField.START)
        end = sync.get(SyncField.END)
        styp = sync.get(SyncField.TYPE)
        if start is not None and end is not None:
            for i in range(len(start)):
                t = int(styp[i]) if styp is not None else 0
                events.append(
                    {
                        "name": _SYNC_TYPE_NAMES.get(t, f"sync_{t}"),
                        "ph": "X",
                        "ts": int(start[i]) / 1000.0,
                        "dur": (int(end[i]) - int(start[i])) / 1000.0,
                        "pid": "CPU sync",
                        "tid": "sync",
                        "args": {},
                    }
                )
    # Overhead on its own lane.
    ovh = data.get(ActivityKind.OVERHEAD)
    if ovh:
        start = ovh.get(OverheadField.START)
        end = ovh.get(OverheadField.END)
        okind = ovh.get(OverheadField.OVERHEAD_KIND)
        if start is not None and end is not None:
            for i in range(len(start)):
                k = int(okind[i]) if okind is not None else 0
                events.append(
                    {
                        "name": _OVERHEAD_KIND_NAMES.get(k, f"overhead_{k}"),
                        "ph": "X",
                        "ts": int(start[i]) / 1000.0,
                        "dur": (int(end[i]) - int(start[i])) / 1000.0,
                        "pid": "Overhead",
                        "tid": "overhead",
                        "args": {},
                    }
                )
    return events


def _flow_arrows(
    flow_src_by_corr: dict[int, "tuple[str, str, float]"],
    gpu_by_corr: dict[int, "tuple[str, str, float]"],
) -> "list[dict[str, Any]]":
    """ac2g flow arrows: a start at the CPU launch and an end at the GPU
    activity sharing the correlation_id."""
    arrows: list[dict[str, Any]] = []
    for cid, (spid, stid, sts) in flow_src_by_corr.items():
        dst = gpu_by_corr.get(cid)
        if dst is None:
            continue
        dpid, dtid, dts = dst
        arrows.append(
            {
                "ph": "s",
                "id": cid,
                "cat": "ac2g",
                "name": "ac2g",
                "ts": sts,
                "pid": spid,
                "tid": stid,
            }
        )
        arrows.append(
            {
                "ph": "f",
                "bp": "e",
                "id": cid,
                "cat": "ac2g",
                "name": "ac2g",
                "ts": dts,
                "pid": dpid,
                "tid": dtid,
            }
        )
    return arrows


def _annotation_events(
    intervals: "list[tuple[int, int, str]]",
) -> "list[dict[str, Any]]":
    """User annotation scopes as their own CPU lane (gpu_user_annotation
    substitute: kernels also carry args['annotation'] via bracketing)."""
    out: list[dict[str, Any]] = []
    for start, end, name in intervals:
        out.append(
            {
                "name": name,
                "ph": "X",
                "ts": start / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": "Annotations",
                "tid": "user",
                "args": {},
            }
        )
    return out


def _to_chrome_trace(
    data: dict[int, dict[int, Any]],
    name_resolver: Optional[Callable[[int], "str | None"]] = None,
    annotations: "list[tuple[str, int, int]] | None" = None,
    ext_names: "dict[int, str] | None" = None,
) -> dict[str, Any]:
    """Build a chrome://tracing (Perfetto) dict from mux column data. Timestamps
    are CUPTI's clock domain (consistent within the trace).

    GPU activity goes on per-device/per-stream lanes (pid=GPU<device>,
    tid=stream<id>); kernels are labeled by their demangled NAME (falling back
    to ``name_resolver(graph_node_id)``). CPU-side runtime/driver calls go on
    per-process/thread lanes named by cbid, with ``ac2g`` flow arrows linking
    each launch to the GPU activity that shares its correlation_id. User
    ``annotations`` (name, start_ns, end_ns) become their own lane, and each
    GPU event is tagged with the annotation enclosing its CPU launch time."""
    # Order matters: CPU pass first (builds launch timestamps + flow sources),
    # then GPU pass (uses them for annotation bracketing + flow ends).
    intervals = sorted((s, e, n) for (n, s, e) in (annotations or []))
    # Map CUDA correlation_id -> annotation name via EXTERNAL_CORRELATION
    # records (correlation_id -> external_id) and the external_id -> name table.
    corr_to_name: dict[int, str] = {}
    extc = data.get(ActivityKind.EXTERNAL_CORRELATION)
    if extc is not None and ext_names:
        corr_col = extc.get(ExternalCorrelationField.CORRELATION_ID)
        extid_col = extc.get(ExternalCorrelationField.EXTERNAL_ID)
        if corr_col is not None and extid_col is not None:
            for i in range(len(corr_col)):
                nm = ext_names.get(int(extid_col[i]))
                if nm is not None:
                    corr_to_name[int(corr_col[i])] = nm
    launch_ts_by_corr: dict[int, int] = {}
    flow_src_by_corr: dict[int, "tuple[str, str, float]"] = {}
    gpu_by_corr: dict[int, "tuple[str, str, float]"] = {}
    cpu = _cpu_events(data, launch_ts_by_corr, flow_src_by_corr)
    gpu = _gpu_events(
        data, name_resolver, launch_ts_by_corr, intervals, gpu_by_corr, corr_to_name
    )
    flows = _flow_arrows(flow_src_by_corr, gpu_by_corr)
    ann = _annotation_events(intervals)
    return {
        "traceEvents": gpu + cpu + flows + ann,
        "displayTimeUnit": "ns",
    }


_EXTCORR_KIND = 4  # CUpti_ExternalCorrelationKind.CUSTOM1


class ProfilerObserver(MuxObserver):
    def __init__(self, wants: "dict[int, set[int] | str] | None" = None) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[int, list[dict[int, Any]]] = {}
        self._ext_names: dict[int, str] = {}
        self._annotations: list[tuple[str, int, int]] = []
        self._ext_ids = itertools.count(1)
        # Register last (base __init__) so _chunks is ready before the mux
        # poll thread can deliver records.
        super().__init__(wants or _DEFAULT_WANTS)

    def _on_records(self, kind: int, fields: dict[int, Any]) -> None:
        # Mux poll thread: just stash the column dict; concatenation happens
        # in drain(), off the hot path.
        with self._lock:
            self._chunks.setdefault(kind, []).append(fields)

    @contextlib.contextmanager
    def annotate(self, name: str):
        """Tag CUDA activities launched in this scope with ``name``. Pushes an
        external-correlation id (mapped to ``name``) so kernels are attributed
        to the region in the trace (gpu_user_annotation via correlation_id ->
        external_id), and records the scope's span for the annotation lane.
        Eager only -- external ids do not survive CUDA-graph capture/replay
        (graph kernels are attributed by graph_node_id instead)."""
        ext_id = next(self._ext_ids)
        with self._lock:
            self._ext_names[ext_id] = name
        start = self.now_ns()
        pushed = self._mux.push_external_id(ext_id, _EXTCORR_KIND)
        try:
            yield
        finally:
            if pushed:
                self._mux.pop_external_id(_EXTCORR_KIND)
            end = self.now_ns()
            with self._lock:
                self._annotations.append((name, start, end))

    def annotations(self) -> "list[tuple[str, int, int]]":
        """User annotation spans (name, start_ns, end_ns) recorded so far."""
        with self._lock:
            return list(self._annotations)

    def ext_names(self) -> "dict[int, str]":
        """external_id -> annotation name, for joining EXTERNAL_CORRELATION
        records to kernels."""
        with self._lock:
            return dict(self._ext_names)

    def drain(self, flush: bool = True) -> dict[int, dict[int, Any]]:
        """Return ``{kind: {field_id: ndarray}}`` recorded since the last
        call (concatenated per field) and reset. Timestamps are in CUPTI's
        clock domain; use ``convert_time`` for unix-epoch ns.

        Defaults to ``flush=True``: the profiler is a bounded window, so a
        drain wants a complete snapshot (force a CUPTI flush first). Pass
        ``flush=False`` to read only what the poll thread has already
        delivered, without the synchronous flush cost."""
        import numpy as np

        if flush:
            # Flush CUPTI so records still buffered land in _chunks first. Must
            # happen before taking _lock -- the flush delivers via _on_records.
            self._mux.force_drain()
        with self._lock:
            chunks = self._chunks
            self._chunks = {}
        out: dict[int, dict[int, Any]] = {}
        for kind, lst in chunks.items():
            if not lst:
                continue
            fids = lst[0].keys()
            out[kind] = {fid: np.concatenate([c[fid] for c in lst]) for fid in fids}
        return out


class ProfilerSession:
    """start/collect/stop lifecycle around a single ProfilerObserver."""

    def __init__(self, wants: "dict[int, set[int] | str] | None" = None) -> None:
        self._wants = wants
        self._obs: "ProfilerObserver | None" = None
        self._annotations: list[tuple[str, int, int]] = []
        self._ext_names: dict[int, str] = {}

    @property
    def active(self) -> bool:
        return self._obs is not None

    def annotate(self, name: str):
        """Tag CUDA activities in this scope with ``name`` (no-op if inactive)."""
        if self._obs is None:
            return contextlib.nullcontext()
        return self._obs.annotate(name)

    def annotations(self) -> "list[tuple[str, int, int]]":
        return self._annotations

    def ext_names(self) -> "dict[int, str]":
        return self._ext_names

    def start(self) -> bool:
        """Begin recording. Returns False if the mux/CUPTI isn't available
        (e.g. CUPTI < 13.2), in which case the session stays inactive."""
        if self._obs is not None:
            return True
        obs = ProfilerObserver(self._wants)
        if not obs.available:
            return False
        self._obs = obs
        return True

    def collect(self) -> dict[int, dict[int, Any]]:
        """Return everything recorded so far without stopping (drain()
        force-flushes first, so it's a complete snapshot). Empty if inactive."""
        return self._obs.drain() if self._obs is not None else {}

    def stop(self) -> dict[int, dict[int, Any]]:
        """Drain the window (force-flushed), unregister, and return columns."""
        if self._obs is None:
            return {}
        data = self._obs.drain()
        self._annotations = self._obs.annotations()
        self._ext_names = self._obs.ext_names()
        self._obs.close()
        self._obs = None
        return data

    def convert_time(self, value: int) -> int:
        if self._obs is None:
            return value
        return self._obs.convert_time(value)


class CuptiProfiler:
    """Context manager that records GPU activity over a window and exports a
    chrome trace -- the mux-based replacement for the standalone CuptiMonitor
    backend.

    Usage::

        with CuptiProfiler() as prof:
            ...  # run GPU work
        prof.export_chrome_trace("trace.json")

    Pass ``name_resolver(graph_node_id) -> str`` to label kernels (e.g. an
    l4x SubgraphName lookup); without it kernels are labeled generically.
    """

    def __init__(
        self,
        wants: "dict[int, set[int] | str] | None" = None,
        name_resolver: Optional[Callable[[int], "str | None"]] = None,
    ) -> None:
        self._session = ProfilerSession(wants)
        self._name_resolver = name_resolver
        self._data: dict[int, dict[int, Any]] = {}
        self._annotations: list[tuple[str, int, int]] = []
        self._ext_names: dict[int, str] = {}

    def start(self) -> bool:
        """Begin recording; False if CUPTI/mux is unavailable."""
        return self._session.start()

    def stop(self) -> None:
        self._data = self._session.stop()
        self._annotations = self._session.annotations()
        self._ext_names = self._session.ext_names()

    def annotate(self, name: str):
        """Tag CUDA activities launched in this scope with ``name`` (becomes a
        gpu_user_annotation in the trace). No-op if the profiler isn't active."""
        return self._session.annotate(name)

    def __enter__(self) -> "CuptiProfiler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False

    def chrome_trace(self) -> dict[str, Any]:
        return _to_chrome_trace(
            self._data, self._name_resolver, self._annotations, self._ext_names
        )

    def export_chrome_trace(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.chrome_trace(), f)
