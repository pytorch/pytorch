# mypy: allow-untyped-defs
"""Chrome-trace observer for the CUPTI activity multiplexer: accumulates the monitor's
decoded records + trace metadata into the window dict monitor_trace splices into a Kineto
trace. Collection, annotation join, and window management live here."""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from typing import Any, TYPE_CHECKING

import numpy as np
from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

import torch
from torch.profiler._cupti.cupti_python import OVERHEAD_KIND_NAMES
from torch.profiler._cupti.monitor_trace import merge_trace_window_into_chrome_trace
from torch.profiler._cupti.observers.base import (
    CuptiMonitorObserver,
    default_graph_annotation_resolver,
    ObserverAnnotationSettings,
)
from torch.profiler._cupti.observers.observation_window import WindowFinalizerMixin
from torch.profiler._cupti.records import (
    Api,
    CudaEvent,
    ExternalCorrelation,
    Field,
    Kernel,
    Memcpy,
    Memcpy2,
    Memset,
    Overhead,
    Sync,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


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


# GPU work plus the CPU-side runtime/driver/overhead records and external correlation for
# the annotation join. Omits fields the chrome trace never consumes (which also have no v2
# user-defined-record id).
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
        # Launch config (kineto reports these; the only source for eager kernels).
        Kernel.GRID_X,
        Kernel.GRID_Y,
        Kernel.GRID_Z,
        Kernel.BLOCK_X,
        Kernel.BLOCK_Y,
        Kernel.BLOCK_Z,
        Kernel.REGISTERS_PER_THREAD,
        Kernel.STATIC_SHARED_MEMORY,
        Kernel.DYNAMIC_SHARED_MEMORY,
        Kernel.LAUNCH_PRIORITY,
        Kernel.QUEUED,
        Kernel.CHANNEL_ID,
        Kernel.CHANNEL_TYPE,
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
    # Peer-to-peer / cross-device copies (e.g. tensor.to(other_gpu), pipeline sends). CUPTI
    # records these under MEMCPY2, NOT MEMCPY, so without this they never appear as GPU spans
    # even though they drive NVLink. Folded into the same "gpu_memcpy" frame (see
    # _memcpy2_columns) so they render as Memcpy spans on the issuing device's lane.
    ActivityKind.MEMCPY2: {
        Memcpy2.START,
        Memcpy2.END,
        Memcpy2.DEVICE_ID,
        Memcpy2.CONTEXT_ID,
        Memcpy2.STREAM_ID,
        Memcpy2.CORRELATION_ID,
        Memcpy2.GRAPH_NODE_ID,
        Memcpy2.GRAPH_ID,
        Memcpy2.BYTES,
        Memcpy2.COPY_KIND,
        Memcpy2.SRC_KIND,
        Memcpy2.DST_KIND,
        Memcpy2.FLAGS,
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


# CUDA sync + event fields, selected only under enable_cuda_sync_events (matching kineto).
# SYNCHRONIZATION carries the sync spans; CUDA_EVENT records are the wait_on join inputs
# (which cudaEventRecord a wait refers to) resolved in monitor_trace.
SYNC_FIELDS: dict[ActivityKind, set[Field]] = {
    ActivityKind.SYNCHRONIZATION: {
        Sync.TYPE,
        Sync.START,
        Sync.END,
        Sync.CORRELATION_ID,
        Sync.CONTEXT_ID,
        Sync.STREAM_ID,
        Sync.CUDA_EVENT_ID,
        Sync.CUDA_EVENT_SYNC_ID,
    },
    ActivityKind.CUDA_EVENT: {
        CudaEvent.CORRELATION_ID,
        CudaEvent.CONTEXT_ID,
        CudaEvent.STREAM_ID,
        CudaEvent.EVENT_ID,
        CudaEvent.DEVICE_ID,
        CudaEvent.CUDA_EVENT_SYNC_ID,
    },
}


class ProfilerObserver(WindowFinalizerMixin, CuptiMonitorObserver):
    """Accumulates decoded records and exports them as chrome-trace windows. A window opens
    at trace start (:meth:`open_window`), closes at stop (:meth:`close_window`), and its
    merge + write is deferred until the records are naturally delivered -- no device sync on
    the measured timeline (see :class:`WindowFinalizerMixin`)."""

    def __init__(
        self,
        metadata_resolver: Callable[[int], str | None] | None = None,
        enable_cuda_sync: bool = False,
        defer_export: bool = True,
        enable_pm_sampling: bool = False,
        pm_metrics: Iterable[str] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        # Decoded activity kept COLUMNAR (frames of named numpy columns, not per-record
        # dicts), so window bucketing is a mask and the build is vectorized. Timed frames
        # (with a start_ns column) bucket into windows by start time.
        self._timed_frames: list[tuple[str, dict[str, Any]]] = []
        # Untimestamped join inputs (EXTERNAL_CORRELATION, CUDA_EVENT): each finalized window
        # consumes whatever is buffered (a harmless superset) and clears them.
        self._ext_frames: list[tuple[str, dict[str, Any]]] = []
        # {external_id: blob} from the metadata store, joined via correlation_id ->
        # external_id -> blob at finalize. Single-consumer (the store is process-global).
        self._ext_metadata: dict[int, str] = {}
        # graph_node_id -> blob for graph-captured collectives (no external-correlation link
        # on replay); resolved like graph-node names. NOT reclaimed per graph, reset per run.
        self._metadata_resolver = metadata_resolver
        # pid -> {opaque_tid: system_tid}, for naming GPU/CPU lanes.
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        self._open_start: int | None = None  # open window start (None when none open)
        # window_id -> pending state (start, annotations/thread map, paths, built dict);
        # written + dropped once built AND paths are present.
        self._windows: dict[int, dict[str, Any]] = {}
        selection = {k: set(v) for k, v in PROFILER_FIELDS.items()}
        if enable_cuda_sync:
            selection.update({k: set(v) for k, v in SYNC_FIELDS.items()})
        # Graph naming on via the default registry resolver (the profiler always wants graph
        # captures named -- it's free when there are none and a no-op for eager-only runs).
        # Eager naming stays off (the default): the full profiler already selects
        # EXTERNAL_CORRELATION + RUNTIME and does the correlation -> external_id -> annotation
        # join in the trace build itself, so the base eager machinery would be redundant.
        super().__init__(
            selection,
            annotations=ObserverAnnotationSettings(
                graph_annotation_resolver=default_graph_annotation_resolver,
            ),
        )
        if self.available:
            # defer_export=False (synchronous export) finalizes on the calling thread in
            # join(), so the background poll thread is never needed -- don't spawn one.
            self._init_observation_window(
                poll_interval_ms=20,
                thread_name="cupti-profiler-export",
                auto_start_poller=defer_export,
            )
        # Opt-in PM sampling (true SM-active % + DRAM-throughput %) is a feature of the CUPTI
        # monitor: it registers us as a consumer (with our metrics) of the current device's shared
        # session, delivering decoded frames to on_pm_samples (they render as GPU counter tracks).
        # Off by default; also a no-op when no metrics are configured (pm_metrics).
        self._pm_metrics = list(pm_metrics or [])
        self._pm_enabled = (
            enable_pm_sampling
            and bool(self._pm_metrics)
            and self.available
            and torch.cuda.is_available()
        )
        # PM samples are timestamped, so they bucket into windows exactly like the activity
        # records (via _timed_frames); no separate buffer. Per-device max start_ns delivered:
        # a monotonic guard so each sample is enqueued at most once.
        self._pm_last_ns: dict[int, int] = {}
        if self._pm_enabled and self._monitor is not None:
            self._monitor.request_pm_sampling(self.on_pm_samples, self._pm_metrics)

    def _boundary_clock_ns(self) -> int:
        # Stamp the boundary in the converted clock the events' start_ns use (convert_time
        # is monotonic, so the comparison stays order-equivalent).
        return self.convert_time(self.now_record_ns())

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        # Worker thread: build a named-column frame per kind (convert/demangle/resolve while
        # the active-id chain is live) and accumulate -- but only while a window is open or
        # pending, else drop the columns no window will consume.
        if not columns:
            return
        with self._lock:
            active = self._open_start is not None or bool(self._windows)
        if not active:
            return
        convert = self.convert_time_array
        timed: list[tuple[str, dict[str, Any]]] = []
        ext: list[tuple[str, dict[str, Any]]] = []
        for kind, cols in columns.items():
            spec = _COLUMN_BUILDERS.get(int(kind))
            if spec is None:
                continue
            kind_str, builder, is_timed = spec
            frame = builder(cols, convert, self._annotation_resolver)
            if frame is None or _named_len(frame) == 0:
                continue
            # Pluggable graphed-op lane assignment: attach (logical_lane, lane_name) columns
            # for work kinds so the trace builder can reassign graphed ops onto a dedicated
            # lane. No-op when no resolver is installed.
            if (
                self._lane_resolver is not None
                and "graph_node_id" in frame
                and "stream_id" in frame
            ):
                frame["logical_lane"], frame["lane_name"] = _resolve_lane_columns(
                    self._lane_resolver, frame
                )
            (timed if is_timed else ext).append((kind_str, frame))
        if not timed and not ext:
            return
        for kind_str, frame in ext:
            if kind_str == "external_correlation":
                self._resolve_user_external_ids(frame)
        meta = (
            self._monitor.take_external_metadata() if self._monitor is not None else {}
        )
        with self._lock:
            self._timed_frames.extend(timed)
            self._ext_frames.extend(ext)
            if meta:
                self._ext_metadata.update(meta)

    def _resolve_user_external_ids(self, frame: dict[str, Any]) -> None:
        """Override each EXTERNAL_CORRELATION row's ``user_external_id`` with the innermost
        ENCLOSING named-region id (via the monitor's live active-id chain), so a kernel
        nested below a named region gets that region's name. Defaults to the raw id."""
        if self._monitor is None:
            return
        names = set(self.annotation_names())
        if not names:
            return
        chain = self._monitor.external_id_chain
        resolved = frame["user_external_id"].tolist()
        cache: dict[int, int] = {}
        for i, eid in enumerate(frame["external_id"].tolist()):
            mapped = cache.get(eid)
            if mapped is None:
                mapped = cache[eid] = next(
                    (c for c in reversed(chain(eid)) if c in names), eid
                )
            resolved[i] = mapped
        frame["user_external_id"] = np.asarray(resolved, dtype=np.int64)

    def push_annotation(self, name: str) -> int | None:
        # Record the calling thread (for lane naming) on top of the base push.
        self._record_calling_thread()
        return super().push_annotation(name)

    def _record_calling_thread(self) -> None:
        pid, opaque_tid, sys_tid = _current_thread_resource_tuple()
        with self._lock:
            self._thread_resource_map.setdefault(pid, {})[opaque_tid] = sys_tid

    # --- async window API (the cupti_monitor profiler backend drives these) ----

    def on_pm_samples(self, frame: dict[str, Any]) -> None:
        # Monitor flush-thread hook: enqueue the frame as a timed frame so each finalized window
        # slices its [start, boundary) samples (a frame spanning a boundary is split like any other
        # timed frame -- see _finalize_window). Keep only samples newer than the last per device:
        # decode drains (each delivered once, in increasing start_ns), so this is a cheap monotonic
        # guard against any duplicate or out-of-order delivery.
        ts = frame.get("start_ns")
        if ts is None or not len(ts):
            return
        dev = frame["device_id"]
        with self._lock:
            keep = np.zeros(len(ts), dtype=bool)
            for d in np.unique(dev):
                keep |= (dev == d) & (ts > self._pm_last_ns.get(int(d), -1))
            if not keep.any():
                return
            kept = _slice_frame(frame, keep)
            kts, kdev = kept["start_ns"], kept["device_id"]
            for d in np.unique(kdev):
                self._pm_last_ns[int(d)] = int(kts[kdev == d].max())
            self._timed_frames.append(("pm_sampling", kept))

    def open_window(self) -> None:
        """Start a trace window; records before this are excluded (no prepare-phase leak)."""
        # Capture the starting thread so its RUNTIME/DRIVER records map to the OS tid
        # (matching its cpu_ops) -- else CUPTI's raw threadId lands them on a phantom lane.
        self._record_calling_thread()
        with self._lock:
            self._open_start = self._boundary_clock_ns()

    def close_window(self) -> int | None:
        """End the open window and queue it for deferred export; snapshots its annotations +
        thread map now. Pair with :meth:`set_export` for the paths. Returns the window id."""
        if not self.available:
            return None
        with self._lock:
            start = self._open_start if self._open_start is not None else 0
            self._open_start = None
            annotations = self.annotation_names(reset=True)
            thread_map = {
                pid: dict(mapping) for pid, mapping in self._thread_resource_map.items()
            }
        window_id = self.mark_boundary()
        with self._lock:
            self._windows[window_id] = {
                "start": start,
                "annotations": annotations,
                "thread_map": thread_map,
                "cpu": None,
                "out": None,
                "built": None,
            }
        return window_id

    def set_export(
        self,
        window_id: int,
        cpu_trace_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
    ) -> None:
        """Supply the captured Kineto CPU trace + output path for a closed window. The merged
        file is written once the window's records are covered (now, or by the poller)."""
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["cpu"] = os.fspath(cpu_trace_path)
            # Monitor traces are always gzipped; the writer keys gzip off the .gz suffix.
            out = os.fspath(output_path)
            w["out"] = out if out.endswith(".gz") else out + ".gz"
        self._maybe_write(window_id)

    def join(self, *, force: bool = True, timeout_s: float = 30.0) -> None:
        """Finalize + write every pending window, then unregister. Idempotent. ``force``
        (default) sync-flushes the tail, for use on the training thread. ``force=False`` (an
        off-thread finalize) must NOT flush, so it waits up to ``timeout_s`` for the poller to
        cover the windows, force-draining only if it stalls."""
        # Release PM sampling first: its final tail decode must land in _timed_frames BEFORE the
        # windows finalize, so those samples can be sliced into the closing window.
        if self._pm_enabled and self._monitor is not None:
            self._monitor.release_pm_sampling(self.on_pm_samples)
        if getattr(self, "_boundaries", None) is not None:
            sync = force
            if not force:
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    with self._win_lock:
                        if not self._boundaries:
                            break
                    time.sleep(self._poll_interval_s)
                with self._win_lock:
                    sync = bool(
                        self._boundaries
                    )  # stalled -> fall back to a forced drain
            self._stop_observation_window(sync=sync)
        # Write on the foreground (here and in set_export), never the poll thread, so it
        # stays inside the caller's temp-dir lifetime.
        for window_id in list(self._windows):
            self._maybe_write(window_id)
        super().close()

    # --- WindowFinalizerMixin hooks -------------------------------------------

    def _collect_delivered(self, *, sync: bool) -> None:
        # Events build in _on_activities as buffers arrive; only flush the tail at teardown.
        if sync and self._monitor is not None:
            self._monitor.flush(sync=True)

    def _window_watermark_ns(self) -> int:
        with self._lock:
            return max(
                (
                    int(frame["start_ns"].max())
                    for _, frame in self._timed_frames
                    if len(frame["start_ns"])
                ),
                default=-1,
            )

    def _finalize_window(self, window_id: int, boundary_ns: int) -> None:
        # Poll thread: records are all in hand, so BUILD the trace-window dict (rows in
        # [start, boundary)) and drop them. Writing is foreground-only (set_export / join).
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            start = w["start"]
            in_window: list[tuple[str, dict[str, Any]]] = []
            keep: list[tuple[str, dict[str, Any]]] = []
            for kind_str, frame in self._timed_frames:
                s = frame["start_ns"]
                in_mask = (s >= start) & (s < boundary_ns)
                if in_mask.any():
                    in_window.append((kind_str, _slice_frame(frame, in_mask)))
                # Rows at/after the boundary belong to a later window; rows before
                # ``start`` are prepare-phase noise and dropped (as the dict path did).
                keep_mask = s >= boundary_ns
                if keep_mask.any():
                    keep.append((kind_str, _slice_frame(frame, keep_mask)))
            self._timed_frames = keep
            # Untimestamped join frames ride along; consume the buffered ones now.
            ext, self._ext_frames = self._ext_frames, []
            meta, self._ext_metadata = self._ext_metadata, {}
        columns = _concat_frames_by_kind(in_window + ext)
        # Attach the per-collective blob as a "metadata" column on the GPU-op kinds (these
        # columns are this thread's now, no lock). No-op without comms metadata.
        _attach_metadata(columns, meta, self._metadata_resolver)
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return
            w["built"] = {
                "columns": columns,
                "user_annotations": w["annotations"],
                "thread_resource_map": w["thread_map"],
                "start_ns": start,
            }

    def _maybe_write(self, window_id: int) -> None:
        # Write once the window is both built and has its paths; foreground-only. The
        # del-under-lock makes the writer single.
        with self._lock:
            w = self._windows.get(window_id)
            if w is None or w["built"] is None or w["out"] is None:
                return
            built, cpu, out = w["built"], w["cpu"], w["out"]
            del self._windows[window_id]
        try:
            merge_trace_window_into_chrome_trace(cpu, out, built, trace_name=out)
        finally:
            with contextlib.suppress(OSError):  # the CPU trace is a throwaway snapshot
                os.remove(cpu)


# --- raw columns -> named-column frames --------------------------------------
# Turn the monitor's per-kind columns ({field_id: array}) into named numpy columns the merge
# consumes directly. Owns column shape, demangling, clock conversion, and graph-annotation
# resolution -- on the worker thread while the active-id chain is live.


def _named_len(frame: dict[str, Any]) -> int:
    for col in frame.values():
        return len(col)
    return 0


def _slice_frame(frame: dict[str, Any], mask: Any) -> dict[str, Any]:
    return {name: col[mask] for name, col in frame.items()}


def _concat_frames_by_kind(
    frames: list[tuple[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Group same-kind frames and concatenate their columns (same-kind frames share a column
    set); a single frame is passed through without a copy."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kind_str, frame in frames:
        grouped.setdefault(kind_str, []).append(frame)
    out: dict[str, dict[str, Any]] = {}
    for kind_str, group in grouped.items():
        if len(group) == 1:
            out[kind_str] = group[0]
        else:
            out[kind_str] = {
                name: np.concatenate([g[name] for g in group]) for name in group[0]
            }
    return out


def _attach_metadata(
    columns: dict[str, dict[str, Any]],
    external_metadata: dict[int, str],
    metadata_resolver: Any,
) -> None:
    """Attach the per-collective blob as a ``metadata`` object column on the GPU-op kinds by
    two routes: eager ``correlation_id -> external_id -> blob`` and graph ``graph_node_id``
    via ``metadata_resolver``. Mutates ``columns`` in place; no-op when neither applies."""
    if not external_metadata and metadata_resolver is None:
        return
    # Scope to ids we have a blob for: an op may carry an EXTERNAL_CORRELATION record per
    # active kind, so this picks the collective's and avoids a region id shadowing it.
    corr_to_ext: dict[int, int] = {}
    ext = columns.get("external_correlation")
    if external_metadata and ext is not None:
        corr_to_ext = {
            corr: external_id
            for corr, external_id in zip(
                ext["correlation_id"].tolist(), ext["external_id"].tolist()
            )
            if external_id in external_metadata
        }
    for kind_str in ("kernel", "gpu_memcpy", "gpu_memset"):
        c = columns.get(kind_str)
        if not c or not len(c["correlation_id"]):
            continue
        corr_l = c["correlation_id"].tolist()
        gnid_l = c["graph_node_id"].tolist()
        meta = np.empty(len(corr_l), dtype=object)
        meta[:] = None
        for i in range(len(corr_l)):
            if external_metadata:
                external_id = corr_to_ext.get(corr_l[i])
                if external_id is not None:
                    blob = external_metadata.get(external_id)
                    if blob is not None:
                        meta[i] = blob
                        continue  # eager hit; don't also consult the graph resolver
            if metadata_resolver is not None:
                node = gnid_l[i] or 0
                if node:
                    blob = metadata_resolver(node)
                    if blob is not None:
                        meta[i] = blob
        c["metadata"] = meta


def _demangle_column(names: Any) -> Any:
    out = np.empty(len(names), dtype=object)
    for i, raw in enumerate(names.tolist()):
        out[i] = _demangle_symbol(raw)
    return out


def _resolve_annotation_column(resolver, gnid: Any) -> Any:
    """Per-row graph annotation as an object column. None resolver -> all-None column, no
    calls. The resolver is memoized per graph_node_id by the observer (see
    CuptiMonitorObserver._annotation_resolver), so distinct nodes resolve once for its
    lifetime."""
    n = len(gnid)
    out = np.empty(n, dtype=object)
    if resolver is None:
        out[:] = None
        return out
    for i, g in enumerate(gnid.tolist()):
        out[i] = resolver(g)
    return out


def _resolve_lane_columns(lane_resolver, frame: dict[str, Any]) -> Any:
    """Per-row (logical_lane, lane_name) for graphed ops. Graphed rows (graph_node_id != 0)
    get the resolver's (lane, name), or keep their CUDA stream when it returns None; eager rows
    keep their CUDA stream and no name (the monitor names those "stream N"). The resolver is
    keyed and memoized on graph_node_id by the observer (see CuptiMonitorObserver._lane_resolver),
    so distinct nodes resolve once for its lifetime."""
    gnid = frame["graph_node_id"]
    n = len(gnid)
    logical = np.array(
        frame["stream_id"], dtype=np.int64
    )  # default: the op's CUDA stream
    names = np.full(n, None, dtype=object)
    for i, g in enumerate(gnid.tolist()):
        if not g:
            continue
        res = lane_resolver(g)
        if res is not None:
            logical[i], names[i] = res
    return logical, names


def _kernel_columns(cols, convert, resolver):
    gnid = cols[Kernel.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Kernel.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Kernel.START.id]),
        "end_ns": convert(cols[Kernel.END.id]),
        "device_id": cols[Kernel.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Kernel.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Kernel.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Kernel.GRAPH_ID.id].astype(np.int64),
        "name": _demangle_column(cols[Kernel.NAME.id]),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "grid_x": cols[Kernel.GRID_X.id].astype(np.int64),
        "grid_y": cols[Kernel.GRID_Y.id].astype(np.int64),
        "grid_z": cols[Kernel.GRID_Z.id].astype(np.int64),
        "block_x": cols[Kernel.BLOCK_X.id].astype(np.int64),
        "block_y": cols[Kernel.BLOCK_Y.id].astype(np.int64),
        "block_z": cols[Kernel.BLOCK_Z.id].astype(np.int64),
        "registers_per_thread": cols[Kernel.REGISTERS_PER_THREAD.id].astype(np.int64),
        "static_shared_memory": cols[Kernel.STATIC_SHARED_MEMORY.id].astype(np.int64),
        "dynamic_shared_memory": cols[Kernel.DYNAMIC_SHARED_MEMORY.id].astype(np.int64),
        "priority": cols[Kernel.LAUNCH_PRIORITY.id].astype(np.int64),
        "queued": convert(cols[Kernel.QUEUED.id]),
        "channel": cols[Kernel.CHANNEL_ID.id].astype(np.int64),
        "channel_type": cols[Kernel.CHANNEL_TYPE.id].astype(np.int64),
    }


def _memcpy_columns(cols, convert, resolver):
    gnid = cols[Memcpy.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memcpy.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memcpy.START.id]),
        "end_ns": convert(cols[Memcpy.END.id]),
        "device_id": cols[Memcpy.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memcpy.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memcpy.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memcpy.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memcpy.BYTES.id].astype(np.int64),
        "copy_kind": cols[Memcpy.COPY_KIND.id].astype(np.int64),
        "src_kind": cols[Memcpy.SRC_KIND.id].astype(np.int64),
        "dst_kind": cols[Memcpy.DST_KIND.id].astype(np.int64),
        "flags": cols[Memcpy.FLAGS.id].astype(np.int64),
    }


def _memcpy2_columns(cols, convert, resolver):
    # Peer-to-peer (MEMCPY2): same output columns as _memcpy_columns so the frames concatenate
    # under one "gpu_memcpy" kind; reads the MEMCPY2 field ids (src/dst device fields shift
    # correlation/graph ids). src/dst device aren't surfaced (the span on the issuing device's
    # lane is what's wanted), but they're available on Memcpy2 if needed later.
    gnid = cols[Memcpy2.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memcpy2.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memcpy2.START.id]),
        "end_ns": convert(cols[Memcpy2.END.id]),
        "device_id": cols[Memcpy2.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memcpy2.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memcpy2.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memcpy2.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memcpy2.BYTES.id].astype(np.int64),
        "copy_kind": cols[Memcpy2.COPY_KIND.id].astype(np.int64),
        "src_kind": cols[Memcpy2.SRC_KIND.id].astype(np.int64),
        "dst_kind": cols[Memcpy2.DST_KIND.id].astype(np.int64),
        "flags": cols[Memcpy2.FLAGS.id].astype(np.int64),
    }


def _memset_columns(cols, convert, resolver):
    gnid = cols[Memset.GRAPH_NODE_ID.id].astype(np.int64)
    corr = cols[Memset.CORRELATION_ID.id].astype(np.int64)
    return {
        "start_ns": convert(cols[Memset.START.id]),
        "end_ns": convert(cols[Memset.END.id]),
        "device_id": cols[Memset.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[Memset.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Memset.STREAM_ID.id].astype(np.int64),
        "correlation_id": corr,
        "graph_node_id": gnid,
        "graph_id": cols[Memset.GRAPH_ID.id].astype(np.int64),
        "annotation": _resolve_annotation_column(resolver, gnid),
        "bytes": cols[Memset.BYTES.id].astype(np.int64),
        "value": cols[Memset.VALUE.id].astype(np.int64),
        "memory_kind": cols[Memset.MEMORY_KIND.id].astype(np.int64),
        "flags": cols[Memset.FLAGS.id].astype(np.int64),
    }


def _api_columns(cols, convert, resolver):
    del resolver
    return {
        "cbid": cols[Api.CBID.id].astype(np.int64),
        "start_ns": convert(cols[Api.START.id]),
        "end_ns": convert(cols[Api.END.id]),
        "process_id": cols[Api.PROCESS_ID.id].astype(np.int64),
        "thread_id": cols[Api.THREAD_ID.id].astype(np.int64),
        "correlation_id": cols[Api.CORRELATION_ID.id].astype(np.int64),
    }


def _external_correlation_columns(cols, convert, resolver):
    del convert, resolver
    external_id = cols[ExternalCorrelation.EXTERNAL_ID.id].astype(np.int64)
    return {
        "external_kind": cols[ExternalCorrelation.EXTERNAL_KIND.id].astype(np.int64),
        "external_id": external_id,
        "correlation_id": cols[ExternalCorrelation.CORRELATION_ID.id].astype(np.int64),
        # Default to the raw innermost id; _resolve_user_external_ids overrides the
        # rows that resolve to an enclosing named region (chain is live at dispatch).
        "user_external_id": external_id.copy(),
    }


def _overhead_columns(cols, convert, resolver):
    del resolver
    kinds = cols[Overhead.OVERHEAD_KIND.id].astype(np.int64)
    names = np.empty(len(kinds), dtype=object)
    for i, k in enumerate(kinds.tolist()):
        names[i] = OVERHEAD_KIND_NAMES.get(k, f"overhead_{k}")
    return {
        "start_ns": convert(cols[Overhead.START.id]),
        "end_ns": convert(cols[Overhead.END.id]),
        "correlation_id": cols[Overhead.CORRELATION_ID.id].astype(np.int64),
        "name": names,
    }


def _sync_columns(cols, convert, resolver):
    del resolver
    return {
        "sync_type": cols[Sync.TYPE.id].astype(np.int64),
        "start_ns": convert(cols[Sync.START.id]),
        "end_ns": convert(cols[Sync.END.id]),
        "context_id": cols[Sync.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[Sync.STREAM_ID.id].astype(np.int64),
        "correlation_id": cols[Sync.CORRELATION_ID.id].astype(np.int64),
        "cuda_event_id": cols[Sync.CUDA_EVENT_ID.id].astype(np.int64),
        "cuda_event_sync_id": cols[Sync.CUDA_EVENT_SYNC_ID.id].astype(np.int64),
    }


def _cuda_event_columns(cols, convert, resolver):
    del convert, resolver
    return {
        "cuda_event_sync_id": cols[CudaEvent.CUDA_EVENT_SYNC_ID.id].astype(np.int64),
        "correlation_id": cols[CudaEvent.CORRELATION_ID.id].astype(np.int64),
        "device_id": cols[CudaEvent.DEVICE_ID.id].astype(np.int64),
        "context_id": cols[CudaEvent.CONTEXT_ID.id].astype(np.int64),
        "stream_id": cols[CudaEvent.STREAM_ID.id].astype(np.int64),
        "event_id": cols[CudaEvent.EVENT_ID.id].astype(np.int64),
    }


# kind -> (chrome-trace tag, column builder, is_timed). Timed kinds bucket into windows;
# untimed kinds (external_correlation, cuda_event) are join inputs that ride along.
_COLUMN_BUILDERS: dict[int, tuple[str, Any, bool]] = {
    int(ActivityKind.CONCURRENT_KERNEL): ("kernel", _kernel_columns, True),
    int(ActivityKind.MEMCPY): ("gpu_memcpy", _memcpy_columns, True),
    int(ActivityKind.MEMCPY2): ("gpu_memcpy", _memcpy2_columns, True),
    int(ActivityKind.MEMSET): ("gpu_memset", _memset_columns, True),
    int(ActivityKind.RUNTIME): ("cuda_runtime", _api_columns, True),
    int(ActivityKind.DRIVER): ("cuda_driver", _api_columns, True),
    int(ActivityKind.EXTERNAL_CORRELATION): (
        "external_correlation",
        _external_correlation_columns,
        False,
    ),
    int(ActivityKind.OVERHEAD): ("overhead", _overhead_columns, True),
    int(ActivityKind.SYNCHRONIZATION): ("cuda_sync", _sync_columns, True),
    int(ActivityKind.CUDA_EVENT): ("cuda_event", _cuda_event_columns, False),
}
