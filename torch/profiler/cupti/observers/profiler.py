# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

import json
import threading
from typing import Any, Callable, Optional

from torch.profiler.cupti.observers.base import MuxObserver
from torch.profiler.cupti.types import (
    ActivityKind,
    KernelField,
    MemcpyField,
    MemsetField,
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


def _to_chrome_trace(
    data: dict[int, dict[int, Any]],
    name_resolver: Optional[Callable[[int], "str | None"]] = None,
) -> dict[str, Any]:
    """Build a chrome://tracing (Perfetto) dict from mux column data. GPU
    activity only; timestamps are CUPTI's clock domain (consistent within the
    trace). Events are placed on per-device/per-stream lanes (pid=GPU<device>,
    tid=stream<id>) when those fields are present, else a single GPU/kind lane.
    Kernels are labeled by their own NAME field (demangled), falling back to
    ``name_resolver(graph_node_id)`` then ``"kernel"``; memcpy is ``Memcpy
    <copy-kind>`` and memset ``Memset``. Per-event metadata (correlation, graph
    ids, bytes, copy/memory kinds, flags) is surfaced in ``args``."""
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
    return {"traceEvents": events, "displayTimeUnit": "ns"}


class ProfilerObserver(MuxObserver):
    def __init__(self, wants: "dict[int, set[int] | str] | None" = None) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[int, list[dict[int, Any]]] = {}
        # Register last (base __init__) so _chunks is ready before the mux
        # poll thread can deliver records.
        super().__init__(wants or _DEFAULT_WANTS)

    def _on_records(self, kind: int, fields: dict[int, Any]) -> None:
        # Mux poll thread: just stash the column dict; concatenation happens
        # in drain(), off the hot path.
        with self._lock:
            self._chunks.setdefault(kind, []).append(fields)

    def drain(self) -> dict[int, dict[int, Any]]:
        """Return ``{kind: {field_id: ndarray}}`` recorded since the last
        call (concatenated per field) and reset. Timestamps are in CUPTI's
        clock domain; use ``convert_time`` for unix-epoch ns."""
        import numpy as np

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

    @property
    def active(self) -> bool:
        return self._obs is not None

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

    def start(self) -> bool:
        """Begin recording; False if CUPTI/mux is unavailable."""
        return self._session.start()

    def stop(self) -> None:
        self._data = self._session.stop()

    def __enter__(self) -> "CuptiProfiler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False

    def chrome_trace(self) -> dict[str, Any]:
        return _to_chrome_trace(self._data, self._name_resolver)

    def export_chrome_trace(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.chrome_trace(), f)
