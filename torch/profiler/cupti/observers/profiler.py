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

import ctypes
import os
import threading
from typing import Any, TYPE_CHECKING

import torch
from torch.profiler.cupti.cupti_python import ActivityKind, OVERHEAD_KIND_NAMES
from torch.profiler.cupti.monitor_trace import merge_trace_window_into_chrome_trace
from torch.profiler.cupti.observers.base import (
    CuptiMonitorObserver,
    ObserverAnnotationSettings,
)
from torch.profiler.cupti.records import (
    Api,
    ExternalCorrelation,
    Field,
    Kernel,
    Memcpy,
    Memset,
    Overhead,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.profiler.cupti.observers.base import AnnotationResolver


def _current_thread_resource_tuple() -> tuple[int, int, int]:
    # (pid, opaque 32-bit thread id, system thread id) for trace lane naming.
    opaque_tid = ctypes.c_int32(threading.get_ident() & 0xFFFFFFFF).value
    return (os.getpid(), opaque_tid, threading.get_native_id())


_DEMANGLE_CACHE: dict[str, str] = {}


def _demangle_symbol(name: str) -> str:
    cached = _DEMANGLE_CACHE.get(name)
    if cached is None:
        cached = _DEMANGLE_CACHE[name] = torch._C._demangle(name)
    return cached


# The CUPTI fields ProfilerObserver requests: GPU work plus the CPU-side
# runtime/driver/overhead records and external correlation for the annotation join.
# (Omits fields the chrome trace never consumes, e.g. memcpy runtime_correlation_id
# / overhead object_kind -- which also have no v2 user-defined-record id.)
PROFILER_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.CONCURRENT_KERNEL: {
        Kernel.START,
        Kernel.END,
        Kernel.DEVICE_ID,
        Kernel.CONTEXT_ID,
        Kernel.STREAM_ID,
        Kernel.CORRELATION_ID,
        Kernel.GRAPH_NODE_ID,
        Kernel.GRAPH_ID,
        Kernel.NAME,
    },
    ActivityKind.MEMCPY: {
        Memcpy.START,
        Memcpy.END,
        Memcpy.DEVICE_ID,
        Memcpy.CONTEXT_ID,
        Memcpy.STREAM_ID,
        Memcpy.CORRELATION_ID,
        Memcpy.GRAPH_NODE_ID,
        Memcpy.GRAPH_ID,
        Memcpy.BYTES,
        Memcpy.COPY_KIND,
        Memcpy.SRC_KIND,
        Memcpy.DST_KIND,
        Memcpy.FLAGS,
    },
    ActivityKind.MEMSET: {
        Memset.START,
        Memset.END,
        Memset.DEVICE_ID,
        Memset.CONTEXT_ID,
        Memset.STREAM_ID,
        Memset.CORRELATION_ID,
        Memset.GRAPH_NODE_ID,
        Memset.GRAPH_ID,
        Memset.BYTES,
        Memset.VALUE,
        Memset.MEMORY_KIND,
        Memset.FLAGS,
    },
    ActivityKind.RUNTIME: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.DRIVER: {
        Api.CBID,
        Api.START,
        Api.END,
        Api.PROCESS_ID,
        Api.THREAD_ID,
        Api.CORRELATION_ID,
    },
    ActivityKind.EXTERNAL_CORRELATION: {
        ExternalCorrelation.EXTERNAL_KIND,
        ExternalCorrelation.EXTERNAL_ID,
        ExternalCorrelation.CORRELATION_ID,
    },
    ActivityKind.OVERHEAD: {
        Overhead.OVERHEAD_KIND,
        Overhead.START,
        Overhead.END,
        Overhead.CORRELATION_ID,
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
        # pid -> {opaque_tid: system_tid}, for naming GPU/CPU lanes in the trace.
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        self._window_start_ns = 0
        # Graph-node naming via the base resolver (self._resolver; custom or default).
        # PROFILER_FIELDS already selects RUNTIME + EXTERNAL_CORRELATION + correlation
        # ids and the eager join happens in monitor_trace, so this doesn't opt into the
        # base's eager augmentation.
        super().__init__(
            {k: set(v) for k, v in PROFILER_FIELDS.items()},
            annotations=ObserverAnnotationSettings(
                graph=True, custom_graph_annotation_resolver=annotation_resolver
            ),
        )
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
                    annotation_resolver=self._resolver,
                )
            )
        if new_events:
            with self._lock:
                self._events.extend(new_events)

    def push_annotation(self, name: str) -> int | None:
        # Record the calling thread (for trace-lane naming) on top of the base
        # external-correlation push (which owns the id -> name mapping).
        self._record_calling_thread()
        return super().push_annotation(name)

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
        user_annotations = self.annotation_names(reset=True)
        with self._lock:
            events = self._events
            self._events = []
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
    annotation_resolver: AnnotationResolver | None,
) -> list[dict[str, Any]]:
    """Turn one kind's columns (``{field_id: column}``) into chrome-trace event
    dicts. Returns [] for kinds this builder doesn't render. A None resolver means
    no graph-node naming (every annotation resolves to None)."""
    builder = _BUILDERS.get(kind)
    if builder is None:
        return []
    resolver = annotation_resolver or (lambda *_: None)
    return builder(cols, convert_time, resolver)


def _col_len(cols: dict[int, Any]) -> int:
    return len(next(iter(cols.values()))) if cols else 0


def _kernel_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[Kernel.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Kernel.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "kernel",
                "device_id": int(cols[Kernel.DEVICE_ID.id][i]),
                "context_id": int(cols[Kernel.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Kernel.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Kernel.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Kernel.START.id][i])),
                "end_ns": convert_time(int(cols[Kernel.END.id][i])),
                "annotation": resolver(
                    graph_node_id, ActivityKind.CONCURRENT_KERNEL, correlation_id
                ),
                "name": _demangle_symbol(cols[Kernel.NAME.id][i]),
            }
        )
    return events


def _memcpy_events(cols, convert_time, resolver):
    events = []
    for i in range(_col_len(cols)):
        graph_node_id = int(cols[Memcpy.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Memcpy.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "gpu_memcpy",
                "device_id": int(cols[Memcpy.DEVICE_ID.id][i]),
                "context_id": int(cols[Memcpy.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Memcpy.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Memcpy.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Memcpy.START.id][i])),
                "end_ns": convert_time(int(cols[Memcpy.END.id][i])),
                "bytes": int(cols[Memcpy.BYTES.id][i]),
                "copy_kind": int(cols[Memcpy.COPY_KIND.id][i]),
                "src_kind": int(cols[Memcpy.SRC_KIND.id][i]),
                "dst_kind": int(cols[Memcpy.DST_KIND.id][i]),
                "flags": int(cols[Memcpy.FLAGS.id][i]),
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
        graph_node_id = int(cols[Memset.GRAPH_NODE_ID.id][i])
        correlation_id = int(cols[Memset.CORRELATION_ID.id][i])
        events.append(
            {
                "kind": "gpu_memset",
                "device_id": int(cols[Memset.DEVICE_ID.id][i]),
                "context_id": int(cols[Memset.CONTEXT_ID.id][i]),
                "stream_id": int(cols[Memset.STREAM_ID.id][i]),
                "correlation_id": correlation_id,
                "graph_node_id": graph_node_id,
                "graph_id": int(cols[Memset.GRAPH_ID.id][i]),
                "start_ns": convert_time(int(cols[Memset.START.id][i])),
                "end_ns": convert_time(int(cols[Memset.END.id][i])),
                "bytes": int(cols[Memset.BYTES.id][i]),
                "value": int(cols[Memset.VALUE.id][i]),
                "memory_kind": int(cols[Memset.MEMORY_KIND.id][i]),
                "flags": int(cols[Memset.FLAGS.id][i]),
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
            cbid = int(cols[Api.CBID.id][i])
            events.append(
                {
                    "kind": kind_name,
                    "cbid": cbid,
                    "start_ns": convert_time(int(cols[Api.START.id][i])),
                    "end_ns": convert_time(int(cols[Api.END.id][i])),
                    "process_id": int(cols[Api.PROCESS_ID.id][i]),
                    "thread_id": int(cols[Api.THREAD_ID.id][i]),
                    "correlation_id": int(cols[Api.CORRELATION_ID.id][i]),
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
                "external_kind": int(cols[ExternalCorrelation.EXTERNAL_KIND.id][i]),
                "external_id": int(cols[ExternalCorrelation.EXTERNAL_ID.id][i]),
                "correlation_id": int(cols[ExternalCorrelation.CORRELATION_ID.id][i]),
                "name": "external_correlation",
            }
        )
    return events


def _overhead_events(cols, convert_time, resolver):
    del resolver
    events = []
    for i in range(_col_len(cols)):
        overhead_kind = int(cols[Overhead.OVERHEAD_KIND.id][i])
        events.append(
            {
                "kind": "overhead",
                "object_id": 0,
                "start_ns": convert_time(int(cols[Overhead.START.id][i])),
                "end_ns": convert_time(int(cols[Overhead.END.id][i])),
                "correlation_id": int(cols[Overhead.CORRELATION_ID.id][i]),
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
