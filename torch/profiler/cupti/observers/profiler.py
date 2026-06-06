# mypy: allow-untyped-defs
"""Chrome-trace observer for the CUPTI activity multiplexer.

``ProfilerObserver`` is the multiplexer-backed form of the old monitor trace
path: it wants the trace activity kinds, accumulates the decoded GPU/API records
the monitor delivers, tracks the user-annotation and thread metadata the chrome
trace needs, and on ``drain()`` hands them back as a trace-window dict -- the
exact shape ``monitor_trace.merge_trace_window_into_chrome_trace`` consumes to
splice CUPTI activity into a stock Kineto chrome trace. The trace assembly is
entirely ``monitor_trace``'s existing logic; this class is the collection,
annotation join, and windowing around it.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import torch
from torch.profiler.cupti.cupti_python import ActivityKind, OVERHEAD_KIND_NAMES
from torch.profiler.cupti.fields import (
    ApiField,
    ExternalCorrelationField,
    KernelField,
    MemcpyField,
    MemsetField,
    OverheadField,
)
from torch.profiler.cupti.monitor_trace import merge_trace_window_into_chrome_trace
from torch.profiler.cupti.observers.base import CuptiMonitorObserver


def _current_thread_resource_tuple() -> tuple[int, int, int]:
    # (pid, opaque 32-bit thread id, system thread id) for trace lane naming.
    opaque_tid = ctypes.c_int32(threading.get_ident() & 0xFFFFFFFF).value
    return (os.getpid(), opaque_tid, threading.get_native_id())


if TYPE_CHECKING:
    from collections.abc import Iterator


# (graph_node_id, activity_kind, correlation_id) -> annotation, or None.
AnnotationResolver = Callable[[int, int, int], "Any | None"]

_DEMANGLE_CACHE: dict[str, str] = {}


def _demangle_symbol(name: str) -> str:
    cached = _DEMANGLE_CACHE.get(name)
    if cached is None:
        cached = _DEMANGLE_CACHE[name] = torch._C._demangle(name)
    return cached


def default_graph_annotation_resolver(
    graph_node_id: int, activity_kind: int, correlation_id: int
) -> Any | None:
    """Default resolver: map a CUDA-graph node id to its registered annotation."""
    del activity_kind, correlation_id
    if graph_node_id == 0:
        return None
    try:
        from torch.cuda._graph_annotations import get_kernel_annotations

        annotations = get_kernel_annotations()
    except Exception:
        return None
    return annotations.get(graph_node_id)


# The CUPTI fields ProfilerObserver requests: GPU work plus the CPU-side
# runtime/driver/overhead records and external correlation for the annotation join.
# (Omits fields the chrome trace never consumes, e.g. memcpy runtime_correlation_id
# / overhead object_kind -- which also have no v2 user-defined-record id.)
PROFILER_FIELDS: dict[ActivityKind, set[int]] = {
    ActivityKind.CONCURRENT_KERNEL: {
        KernelField.START,
        KernelField.END,
        KernelField.DEVICE_ID,
        KernelField.CONTEXT_ID,
        KernelField.STREAM_ID,
        KernelField.CORRELATION_ID,
        KernelField.GRAPH_NODE_ID,
        KernelField.GRAPH_ID,
        KernelField.NAME,
    },
    ActivityKind.MEMCPY: {
        MemcpyField.START,
        MemcpyField.END,
        MemcpyField.DEVICE_ID,
        MemcpyField.CONTEXT_ID,
        MemcpyField.STREAM_ID,
        MemcpyField.CORRELATION_ID,
        MemcpyField.GRAPH_NODE_ID,
        MemcpyField.GRAPH_ID,
        MemcpyField.BYTES,
        MemcpyField.COPY_KIND,
        MemcpyField.SRC_KIND,
        MemcpyField.DST_KIND,
        MemcpyField.FLAGS,
    },
    ActivityKind.MEMSET: {
        MemsetField.START,
        MemsetField.END,
        MemsetField.DEVICE_ID,
        MemsetField.CONTEXT_ID,
        MemsetField.STREAM_ID,
        MemsetField.CORRELATION_ID,
        MemsetField.GRAPH_NODE_ID,
        MemsetField.GRAPH_ID,
        MemsetField.BYTES,
        MemsetField.VALUE,
        MemsetField.MEMORY_KIND,
        MemsetField.FLAGS,
    },
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
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelationField.EXTERNAL_KIND,
        ExternalCorrelationField.EXTERNAL_ID,
        ExternalCorrelationField.CORRELATION_ID,
    },
    ActivityKind.OVERHEAD: {
        OverheadField.OVERHEAD_KIND,
        OverheadField.START,
        OverheadField.END,
        OverheadField.CORRELATION_ID,
    },
}


class ProfilerObserver(CuptiMonitorObserver):
    """Accumulates decoded activity records into a trace window for chrome-trace
    export. Construct it to start collecting, optionally bracket regions with
    ``annotate(name)``, then ``build_chrome_trace()`` (or ``drain()``) to
    emit/snapshot it. The observer keeps collecting after a drain, so it can be
    drained repeatedly (one window per drain).

    It requests the chrome-trace field selection (``PROFILER_FIELDS``) -- GPU work
    plus the CPU-side runtime/driver/overhead records and external correlation for
    the annotation join -- and the monitor hands it those fields as columns.
    """

    def __init__(self, annotation_resolver: AnnotationResolver | None = None) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        # external_id -> user-annotation name (this observer's metadata for the
        # monitor's global external-correlation pushes).
        self._ext_names: dict[int, str] = {}
        # pid -> {opaque_tid: system_tid}, for naming GPU/CPU lanes in the trace.
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        self._window_start_ns = 0
        self._annotation_resolver = (
            annotation_resolver or default_graph_annotation_resolver
        )
        super().__init__({k: set(v) for k, v in PROFILER_FIELDS.items()})
        if self.available:
            self._window_start_ns = self.now_ns()

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: the monitor has already demuxed the buffer to the columns
        # we selected; turn each kind's columns into chrome-trace event dicts and
        # accumulate them.
        new_events: list[dict[str, Any]] = []
        for kind, cols in columns.items():
            new_events.extend(
                events_from_columns(
                    int(kind),
                    cols,
                    convert_time=self.convert_time,
                    annotation_resolver=self._annotation_resolver,
                )
            )
        if new_events:
            with self._lock:
                self._events.extend(new_events)

    def push_annotation(self, name: str) -> int | None:
        """Push a global external-correlation id (mapped here to ``name``) so
        kernels recorded until the matching pop are attributed to the region in
        the trace via correlation_id -> external_id -> name. The push is the
        monitor's (global) concern; the id->name mapping is this observer's.
        Eager only -- external ids do not survive CUDA-graph capture/replay."""
        if not self.available:
            return None
        self._record_calling_thread()
        ext_id = self._monitor.push_external_correlation_id()
        if ext_id is not None:
            with self._lock:
                self._ext_names[ext_id] = name
        return ext_id

    def pop_annotation(self) -> int | None:
        if not self.available:
            return None
        return self._monitor.pop_external_correlation_id()

    @contextlib.contextmanager
    def annotate(self, name: str) -> Iterator[int | None]:
        """Context-manager form of push_annotation/pop_annotation."""
        ext_id = self.push_annotation(name)
        try:
            yield ext_id
        finally:
            self.pop_annotation()

    def _record_calling_thread(self) -> None:
        pid, opaque_tid, sys_tid = _current_thread_resource_tuple()
        with self._lock:
            self._thread_resource_map.setdefault(pid, {})[opaque_tid] = sys_tid

    def drain(self) -> dict[str, Any]:
        """Return the collected records + annotation/thread metadata as a
        trace-window dict and reset, so the next drain covers only what arrives
        after this call. Synchronously flushes CUPTI and waits for the worker to
        process all outstanding records first, so the window is complete."""
        if self._monitor is not None:
            self._monitor.flush(forced=True, sync=True)
        now = self.now_ns()
        with self._lock:
            events = self._events
            self._events = []
            user_annotations = self._ext_names
            self._ext_names = {}
            thread_resource_map = {
                pid: dict(mapping) for pid, mapping in self._thread_resource_map.items()
            }
            start_ns = self._window_start_ns
            self._window_start_ns = now
        return {
            "events": events,
            "user_annotations": user_annotations,
            "thread_resource_map": thread_resource_map,
            "start_ns": start_ns,
        }

    def build_chrome_trace(
        self,
        cpu_trace_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        trace_name: str | None = None,
    ) -> None:
        """Splice the drained CUPTI activity into the stock Kineto chrome trace
        at ``cpu_trace_path``, writing the merged trace to ``output_path``."""
        merge_trace_window_into_chrome_trace(
            cpu_trace_path,
            output_path,
            self.drain(),
            trace_name=trace_name,
        )


# --- columns -> chrome-trace event dicts -------------------------------------
# Turn the per-kind columns the monitor delivers into the event dicts
# monitor_trace splices into the chrome trace. Owns the per-kind event shape, the
# symbol demangling, the clock conversion, and the graph-annotation resolution --
# all ProfilerObserver/presentation concerns (the monitor only produces columns).


def events_from_columns(
    kind: int,
    cols: dict[int, Any],
    *,
    convert_time: Callable[[int], int],
    annotation_resolver: AnnotationResolver,
) -> list[dict[str, Any]]:
    """Turn one kind's columns (``{field_id: column}``) into chrome-trace event
    dicts. Returns [] for kinds this builder doesn't render."""
    builder = _BUILDERS.get(kind)
    if builder is None:
        return []
    return builder(cols, convert_time, annotation_resolver)


def _col_len(cols: dict[int, Any]) -> int:
    return len(next(iter(cols.values()))) if cols else 0


def _kernel_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[KernelField.GRAPH_NODE_ID][i])
        correlation_id = int(cols[KernelField.CORRELATION_ID][i])
        events.append(
            {
                "kind": "kernel",
                "device_id": int(cols[KernelField.DEVICE_ID][i]),
                "context_id": int(cols[KernelField.CONTEXT_ID][i]),
                "stream_id": int(cols[KernelField.STREAM_ID][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[KernelField.GRAPH_ID][i]),
                "start_ns": convert_time(int(cols[KernelField.START][i])),
                "end_ns": convert_time(int(cols[KernelField.END][i])),
                "annotation": resolver(
                    graph_node_id, ActivityKind.CONCURRENT_KERNEL, correlation_id
                ),
                "name": _demangle_symbol(cols[KernelField.NAME][i]),
            }
        )
    return events


def _memcpy_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[MemcpyField.GRAPH_NODE_ID][i])
        correlation_id = int(cols[MemcpyField.CORRELATION_ID][i])
        events.append(
            {
                "kind": "gpu_memcpy",
                "device_id": int(cols[MemcpyField.DEVICE_ID][i]),
                "context_id": int(cols[MemcpyField.CONTEXT_ID][i]),
                "stream_id": int(cols[MemcpyField.STREAM_ID][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[MemcpyField.GRAPH_ID][i]),
                "start_ns": convert_time(int(cols[MemcpyField.START][i])),
                "end_ns": convert_time(int(cols[MemcpyField.END][i])),
                "bytes": int(cols[MemcpyField.BYTES][i]),
                "copy_kind": int(cols[MemcpyField.COPY_KIND][i]),
                "src_kind": int(cols[MemcpyField.SRC_KIND][i]),
                "dst_kind": int(cols[MemcpyField.DST_KIND][i]),
                "flags": int(cols[MemcpyField.FLAGS][i]),
                "annotation": resolver(
                    graph_node_id, ActivityKind.MEMCPY, correlation_id
                ),
                "name": "Memcpy",
            }
        )
    return events


def _memset_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[MemsetField.GRAPH_NODE_ID][i])
        correlation_id = int(cols[MemsetField.CORRELATION_ID][i])
        events.append(
            {
                "kind": "gpu_memset",
                "device_id": int(cols[MemsetField.DEVICE_ID][i]),
                "context_id": int(cols[MemsetField.CONTEXT_ID][i]),
                "stream_id": int(cols[MemsetField.STREAM_ID][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[MemsetField.GRAPH_ID][i]),
                "start_ns": convert_time(int(cols[MemsetField.START][i])),
                "end_ns": convert_time(int(cols[MemsetField.END][i])),
                "bytes": int(cols[MemsetField.BYTES][i]),
                "value": int(cols[MemsetField.VALUE][i]),
                "memory_kind": int(cols[MemsetField.MEMORY_KIND][i]),
                "flags": int(cols[MemsetField.FLAGS][i]),
                "annotation": resolver(
                    graph_node_id, ActivityKind.MEMSET, correlation_id
                ),
                "name": "Memset",
            }
        )
    return events


def _api_events(kind_name):
    def build(cols, convert_time, resolver):
        del resolver
        events = []
        for i in range(_col_len(cols)):
            cbid = int(cols[ApiField.CBID][i])
            events.append(
                {
                    "kind": kind_name,
                    "cbid": cbid,
                    "start_ns": convert_time(int(cols[ApiField.START][i])),
                    "end_ns": convert_time(int(cols[ApiField.END][i])),
                    "process_id": int(cols[ApiField.PROCESS_ID][i]),
                    "thread_id": int(cols[ApiField.THREAD_ID][i]),
                    "correlation_id": int(cols[ApiField.CORRELATION_ID][i]),
                    "name": f"cbid_{cbid}",
                }
            )
        return events

    return build


def _external_correlation_events(cols, convert_time, resolver):
    del convert_time, resolver
    events = []
    for i in range(_col_len(cols)):
        events.append(
            {
                "kind": "external_correlation",
                "external_kind": int(cols[ExternalCorrelationField.EXTERNAL_KIND][i]),
                "external_id": int(cols[ExternalCorrelationField.EXTERNAL_ID][i]),
                "correlation_id": int(cols[ExternalCorrelationField.CORRELATION_ID][i]),
                "name": "external_correlation",
            }
        )
    return events


def _overhead_events(cols, convert_time, resolver):
    del resolver
    events = []
    for i in range(_col_len(cols)):
        overhead_kind = int(cols[OverheadField.OVERHEAD_KIND][i])
        events.append(
            {
                "kind": "overhead",
                "object_id": 0,
                "start_ns": convert_time(int(cols[OverheadField.START][i])),
                "end_ns": convert_time(int(cols[OverheadField.END][i])),
                "correlation_id": int(cols[OverheadField.CORRELATION_ID][i]),
                "name": OVERHEAD_KIND_NAMES.get(
                    overhead_kind, f"overhead_{overhead_kind}"
                ),
            }
        )
    return events


_BUILDERS: dict[int, Callable[..., list[dict[str, Any]]]] = {
    ActivityKind.CONCURRENT_KERNEL: _kernel_events,
    ActivityKind.MEMCPY: _memcpy_events,
    ActivityKind.MEMSET: _memset_events,
    ActivityKind.RUNTIME: _api_events("cuda_runtime"),
    ActivityKind.DRIVER: _api_events("cuda_driver"),
    ActivityKind.EXTERNAL_CORRELATION: _external_correlation_events,
    ActivityKind.OVERHEAD: _overhead_events,
}


# The active ProfilerObserver (if any) that record_function user annotations route
# to. The CUPTI external-correlation push is global (the monitor owns the stack);
# this just names which observer records the per-push id->name metadata. Set by the
# torch.profiler backend around a profiling session. This is a ProfilerObserver
# concern, not the monitor engine's, so it lives here.
_active_observer: ProfilerObserver | None = None


def set_active_profiler_observer(observer: ProfilerObserver | None) -> None:
    """Set (or clear, with None) the observer that push_user_annotation routes to."""
    global _active_observer
    _active_observer = observer


def push_user_annotation(name: str) -> int | None:
    """Push a record_function user annotation onto the active ProfilerObserver (if
    any). No-op returning None when no observer is active."""
    observer = _active_observer
    return observer.push_annotation(name) if observer is not None else None


def pop_user_annotation() -> int | None:
    """Pop the most recent user annotation off the active ProfilerObserver."""
    observer = _active_observer
    return observer.pop_annotation() if observer is not None else None
