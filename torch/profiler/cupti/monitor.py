# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np

import torch

from . import cupti_python
from .cupti_python import (
    ActivityAPI,
    ActivityCudaEvent2,
    ActivityExternalCorrelation,
    ActivityKernel11,
    ActivityKind,
    ActivityMemcpy6,
    ActivityMemset4,
    ActivityOverhead3,
    ActivitySynchronization2,
)
from .fields import (
    FIELD_ENUMS,
    FIELD_REGISTRY,
    MemcpyField,
    MemsetField,
    plan_padding,
    record_layout,
    STRING_FIELDS,
    SyncField,
)


# A registration request: either a plain iterable of activity kinds (meaning "all
# fields"), or a field map {kind: iterable of field ids | "all"} selecting specific
# fields per kind. The monitor demuxes the selected fields to columns.
ActivitiesSpec = Mapping[ActivityKind, "Iterable[int] | str"] | Iterable[ActivityKind]


_PY_PROFILER = torch._C._profiler
# The native CUPTI buffer-pool / layout-capture module (C++ side of the monitor).
_cupti_monitor_native = _PY_PROFILER._cupti_monitor

logger = logging.getLogger(__name__)

# Buffers are a recycling pool bounded by peak concurrent demand, so the count
# only keeps climbing if the worker can't drain completed buffers as fast as
# CUPTI fills them. Warn once past this many outstanding buffers (1GB at the
# default 4MB size) as a sign of that backpressure.
_OUTSTANDING_WARN_THRESHOLD = 256

_DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
_DEFAULT_FLUSH_PERIOD_S = 1.0

# flush(sync=True) fences at a SYNC point: it enables SYNCHRONIZATION, captures
# CUPTI's clock, device-syncs (which produces a SYNCHRONIZATION record at a
# timestamp past that point), waits until the worker decodes a sync record that
# recent, then disables SYNCHRONIZATION again. CUPTI delivers buffers in fill
# order, so seeing the sync record means everything before it is delivered too. A
# device sync -- unlike a tracer kernel -- adds no kernel, no cudaLaunchKernel, and
# no dispatcher op to the trace; and enabling SYNCHRONIZATION only for the fence
# means the session doesn't record every sync between flushes. KIND + END are the
# fields the fence decodes.
_FENCE_KIND = ActivityKind.SYNCHRONIZATION
_FENCE_END_FIELD = int(SyncField.END)
_FENCE_FIELDS = frozenset({0, _FENCE_END_FIELD})


def _has_active_cuda_context() -> bool:
    try:
        from cuda.bindings import (  # pyrefly: ignore[missing-import]
            driver as cuda_driver,
        )
    except ImportError:
        return False
    rc, ctx = cuda_driver.cuCtxGetCurrent()
    if rc == cuda_driver.CUresult.CUDA_SUCCESS:
        return ctx is not None
    if rc == cuda_driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return False
    raise RuntimeError(f"cuCtxGetCurrent failed with rc={rc}")


def _cuda_version_string() -> str:
    return torch.version.cuda or ""


# --- columnar (v2 / user-defined-record) buffer demux ------------------------
# The monitor (not the observer) demuxes each completed buffer into per-kind
# columns -- ``{kind: {field_id: column}}`` -- against the record layout it
# computed from the field spec (``record_layout``). Numeric fields become
# little-endian unsigned numpy columns; ``const char*`` string fields (e.g.
# kernel ``NAME``) are dereferenced to object arrays here, while the buffer (and
# the strings it points to) is still alive. The v1 whole-record counterpart is
# ``decode_columns_v1`` below.

# A record layout as produced by ``record_layout``:
# list of (kind, record_size, [(field_id, offset, size), ...]).
RecordLayouts = list[tuple[int, int, list[tuple[int, int, int]]]]


def _deref_cstr(ptr: int) -> str:
    if not ptr:
        return ""
    value = ctypes.cast(ptr, ctypes.c_char_p).value
    return value.decode(errors="replace") if value is not None else ""


def decode_columns_v2(
    buffer_ptr: int,
    valid_size: int,
    record_layouts: RecordLayouts,
    wanted: dict[int, set[int]],
) -> dict[int, dict[int, Any]]:
    """Demux a completed buffer into ``{kind: {field_id: column}}``.

    record_layouts: the layout (from the spec) for the buffer's epoch.
    wanted: ``{kind: {field_id}}`` -- only these kinds/fields are gathered.
    """
    # kind -> (record_size, {field_id: (offset, size)}); size is authoritative.
    layouts: dict[int, tuple[int, dict[int, tuple[int, int]]]] = {}
    for kind, rsz, fields in record_layouts:
        if kind not in wanted or rsz <= 0:
            continue
        layouts[kind] = (rsz, {fid: (off, sz) for fid, off, sz in fields})
    if not layouts or valid_size == 0:
        return {}

    raw = np.ctypeslib.as_array((ctypes.c_uint8 * valid_size).from_address(buffer_ptr))

    # Record start offsets per kind. Records begin with *_FIELD_KIND (id 0, a
    # 4-byte kind) at offset 0 and are sized by their kind's record_size. Three
    # cases, fastest first:
    #   * one kind        -> homogeneous buffer; starts are a fixed stride.
    #   * uniform size    -> kinds vary but all share a record_size; stride to
    #                        the starts, read the KIND column, group by mask.
    #   * variable/multi  -> sequential walk reading each record's KIND (CUPTI
    #                        records aren't self-synchronizing).
    rszs = {rsz for rsz, _ in layouts.values()}
    positions: dict[int, Any] = {}
    if len(layouts) == 1:
        ((kind, (rsz, _)),) = layouts.items()
        n = valid_size // rsz
        if n:
            positions[kind] = np.arange(n, dtype=np.int64) * rsz
    elif len(rszs) == 1:
        rsz = next(iter(rszs))
        n = valid_size // rsz
        if n:
            starts = np.arange(n, dtype=np.int64) * rsz
            kinds_col = raw[starts[:, None] + np.arange(4)].copy().view("<u4").ravel()
            for kind in layouts:
                sel = starts[kinds_col == kind]
                if sel.size:
                    positions[kind] = sel
    else:
        pos_lists: dict[int, list[int]] = {k: [] for k in layouts}
        pos = 0
        while pos + 4 <= valid_size:
            kind = int(raw[pos : pos + 4].view("<u4")[0])
            ent = layouts.get(kind)
            if ent is None:
                break  # unknown kind: can't size it, stop
            pos_lists[kind].append(pos)
            pos += ent[0]
        positions = {k: np.array(v, dtype=np.int64) for k, v in pos_lists.items() if v}

    out: dict[int, dict[int, Any]] = {}
    for kind, pos_arr in positions.items():
        fields = layouts[kind][1]
        str_fields = STRING_FIELDS.get(kind, frozenset())
        cols: dict[int, Any] = {}
        for fid in wanted[kind]:
            ent = fields.get(fid)
            if ent is None:
                continue
            off, size = ent
            if fid in str_fields and size == 8:
                # const char* field: deref each pointer to a str now.
                ptrs = (
                    raw[pos_arr[:, None] + np.arange(off, off + 8)]
                    .copy()
                    .view("<u8")
                    .ravel()
                )
                cols[fid] = np.array([_deref_cstr(int(p)) for p in ptrs], dtype=object)
                continue
            if size not in (1, 2, 4, 8):
                continue
            idx = pos_arr[:, None] + np.arange(off, off + size)
            cols[fid] = raw[idx].copy().view(f"<u{size}").ravel()
        if cols:
            out[kind] = cols
    return out


# --- v1 (whole-record) buffer demux ------------------------------------------
# v1 has no field-layout API, and the cupti-python record classes are native
# getset-descriptor objects (not ctypes structs), so their byte offsets aren't
# introspectable. So v1 reads each record's fields via cupti-python's ``from_ptr``
# -- the authoritative, version-safe path. Record *positions* are still strided
# like v2 when the buffer is homogeneous (records are whole structs packed at
# sizeof(struct)), so the per-record cuptiActivityGetNextRecord cursor walk is only
# needed for mixed struct sizes. Either way it yields the *identical* column
# contract the v2 path produces, so observers never see the difference. Columns are
# RAW field values (``NAME`` mangled, timestamps CUPTI-clock); semantic
# interpretation is the observer's job, exactly as in v2.

# kind -> the cupti-python whole-record class. RUNTIME and DRIVER share
# CUpti_ActivityAPI.
_V1_RECORD_CLASSES: dict[int, type] = {
    ActivityKind.CONCURRENT_KERNEL: ActivityKernel11,
    ActivityKind.MEMCPY: ActivityMemcpy6,
    ActivityKind.MEMSET: ActivityMemset4,
    ActivityKind.EXTERNAL_CORRELATION: ActivityExternalCorrelation,
    ActivityKind.OVERHEAD: ActivityOverhead3,
    ActivityKind.CUDA_EVENT: ActivityCudaEvent2,
    ActivityKind.SYNCHRONIZATION: ActivitySynchronization2,
    ActivityKind.RUNTIME: ActivityAPI,
    ActivityKind.DRIVER: ActivityAPI,
}

# A record's attribute for a given field id is just the field-enum member name
# lowercased (KernelField.GRAPH_NODE_ID -> "graph_node_id"); these are the only
# exceptions, where the cupti-python binding renames the member (`flags` is a
# Python builtin-ish clash, exposed as `flags_`). Keyed by (kind, field id)
# because the bare id is ambiguous across kinds.
_V1_ATTR_OVERRIDES: dict[tuple[int, int], str] = {
    (ActivityKind.MEMCPY, MemcpyField.FLAGS): "flags_",
    (ActivityKind.MEMSET, MemsetField.FLAGS): "flags_",
}


def _v1_attr_name(kind: int, field_id: int) -> str:
    """The cupti-python record attribute that holds CUPTI field ``field_id``."""
    override = _V1_ATTR_OVERRIDES.get((kind, field_id))
    if override is not None:
        return override
    return FIELD_ENUMS[kind](field_id).name.lower()


def _v1_struct_size(kind: int) -> int:
    """Byte size of the v1 record for ``kind`` -- its stride in the buffer."""
    return cupti_python.record_struct_sizes()[_V1_RECORD_CLASSES[kind].__name__]


def _v1_record_positions(
    iter_records: Callable[[int, int], Iterator[tuple[int, int]]],
    buffer_ptr: int,
    valid_size: int,
    kinds: list[int],
) -> dict[int, list[int]]:
    """``kind -> [record address]`` for the wanted ``kinds`` in the buffer.

    v1 records are whole structs packed back-to-back at sizeof(struct), so when the
    buffer holds a single kind (or all wanted kinds share a struct size) we address
    records directly by striding -- no per-record cuptiActivityGetNextRecord cursor
    call (cf. the v2 demux). Mixed struct sizes fall back to walking the cursor."""
    sizes = {k: _v1_struct_size(k) for k in kinds}
    if len(kinds) == 1 and valid_size % sizes[kinds[0]] == 0:
        kind = kinds[0]
        size = sizes[kind]
        return {kind: [buffer_ptr + i * size for i in range(valid_size // size)]}

    distinct = set(sizes.values())
    if len(distinct) == 1 and valid_size % next(iter(distinct)) == 0:
        size = next(iter(distinct))
        n = valid_size // size
        raw = np.ctypeslib.as_array(
            (ctypes.c_uint8 * valid_size).from_address(buffer_ptr)
        )
        starts = np.arange(n, dtype=np.int64) * size
        kinds_col = raw[starts[:, None] + np.arange(4)].copy().view("<u4").ravel()
        wanted_kinds = set(kinds)
        # If a record's kind isn't one we enabled, the homogeneous-stride
        # assumption is wrong (e.g. a foreign subscriber's records of another
        # size) -- fall back to the cursor walk rather than misread.
        if {int(k) for k in np.unique(kinds_col)} <= wanted_kinds:
            positions: dict[int, list[int]] = {}
            for kind in kinds:
                sel = starts[kinds_col == kind]
                if sel.size:
                    positions[kind] = [buffer_ptr + int(s) for s in sel]
            return positions

    walked: dict[int, list[int]] = {}
    wanted_kinds = set(kinds)
    for kind, addr in iter_records(buffer_ptr, valid_size):
        if kind in wanted_kinds:
            walked.setdefault(kind, []).append(addr)
    return walked


def decode_columns_v1(
    iter_records: Callable[[int, int], Iterator[tuple[int, int]]],
    buffer_ptr: int,
    valid_size: int,
    wanted: dict[int, set[int]],
) -> dict[int, dict[int, Any]]:
    """Demux a completed v1 buffer into ``{kind: {field_id: column}}`` for the
    requested fields, reading each record via cupti-python.

    iter_records: the monitor's record walker yielding ``(kind, record_addr)``.
    wanted: ``{kind: {field_id}}`` -- only these kinds/fields are gathered.
    """
    kinds = [k for k in wanted if k in _V1_RECORD_CLASSES]
    if not kinds or valid_size == 0:
        return {}
    positions = _v1_record_positions(iter_records, buffer_ptr, valid_size, kinds)

    # kind -> {field_id: list of raw values}, appended per record.
    accum: dict[int, dict[int, list]] = {}
    for kind, addrs in positions.items():
        record_class = _V1_RECORD_CLASSES[kind]
        want = wanted[kind]
        # Resolve each requested field to its record attribute once, from the first
        # record (dropping any field the record doesn't expose).
        attrs: dict[int, str] | None = None
        cols: dict[int, list] = {}
        for addr in addrs:
            record = record_class.from_ptr(addr, readonly=True)
            if attrs is None:
                attrs = {}
                for fid in want:
                    name = _v1_attr_name(kind, fid)
                    if hasattr(record, name):
                        attrs[fid] = name
                if not attrs:
                    break
                cols = accum.setdefault(kind, {fid: [] for fid in attrs})
            for fid, name in attrs.items():
                cols[fid].append(getattr(record, name))

    out: dict[int, dict[int, Any]] = {}
    for kind, cols in accum.items():
        str_fields = STRING_FIELDS.get(kind, frozenset())
        out[kind] = {
            fid: np.array(values, dtype=object)
            if fid in str_fields
            else np.array(values, dtype=np.int64)
            for fid, values in cols.items()
        }
    return out


class _Observer:
    """A registered consumer of the monitor's records: the activity kinds it
    requested, its per-kind field selection (``{kind: frozenset(field_ids)}``), and
    its ``callback(columns)`` -- invoked on the worker thread once per completed
    buffer with the demuxed columns sliced to its selection (see
    ``CuptiMonitor.register``)."""

    def __init__(
        self,
        activities: Iterable[ActivityKind],
        fields: Mapping[ActivityKind, frozenset[int]],
        callback: Callable[..., None],
    ) -> None:
        self.activities: frozenset[ActivityKind] = frozenset(activities)
        # activity -> the set of field ids wanted for it (the columns to demux).
        self.fields: dict[ActivityKind, frozenset[int]] = dict(fields)
        self.callback = callback


class CuptiMonitor:
    def __init__(
        self,
        *,
        version: int = 1,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        flush_period_s: float = _DEFAULT_FLUSH_PERIOD_S,
    ) -> None:
        # The monitor is the engine and the multiplexer: it owns the single CUPTI
        # subscription + buffer pool + decode worker, demuxes each completed buffer
        # into columns, and hands every observer the columns it selected. It reaches
        # CUPTI only through the self._cupti.activity_* wrappers -- no ctypes here.
        #
        # version selects the CUPTI activity API: 1 = classic whole-record
        # activities (decoded per-record via cupti-python); 2 = user-defined records
        # (a subscriber + per-field selection, decoded columnar against a record
        # layout the monitor computes from the field-size spec -- no captured layout
        # needed). v1 is kept for backward compatibility and as a fallback if the v2
        # API misbehaves on a given libcupti/driver. A monitor is one version for
        # its lifetime; either way observers get columns.
        self.version = version
        self.buffer_size = buffer_size
        self.flush_period_s = flush_period_s
        self._cupti = cupti_python.pylibcupti()
        # The CUPTI subscriber handle (v2 only).
        self._subscriber: int | None = None
        # Layout state -- a function of registration, recomputed only when the
        # selection changes (never per buffer): the fields enabled per kind, the
        # fields to extract (the observers' union), and -- for v2 -- the record
        # layouts the demux decodes against, computed from the field-size spec
        # (padded for a uniform record size when uniform padding is on). v1 enables
        # whole kinds and reads whole records, so it ignores _record_layouts.
        self._enabled: dict[int, frozenset[int]] = {}
        self._wanted: dict[int, set[int]] = {}
        self._record_layouts: list[tuple[int, int, list[tuple[int, int, int]]]] = []
        self._pad_to_uniform = False

        self._lock = threading.Lock()
        self._started = False
        self._callbacks_registered = False
        self._worker_stop = threading.Event()
        self._flush_stop = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._flush_thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._observers: list[_Observer] = []
        self._next_external_id = 1
        self._time_converter = None
        self._timestamp_callback = None
        self._session_start_unix_ns = 0
        self._session_start_approx_ns = 0
        self._session_start_calibrated_unix_ns = 0

        self._buffers_completed = 0
        # Max END timestamp (CUPTI clock) the worker has decoded; flush(sync=True)
        # waits on this condition until it reaches a fence point. Monotonic,
        # worker-written, fence-read.
        self._max_decoded_ns = 0
        self._decoded_cv = threading.Condition()
        # Number of flush(sync=True) calls currently waiting for a sync point. The
        # worker only advances the decoded clock while this is non-zero, so normal
        # buffers skip the scan when no one is fencing.
        self._fence_waiters = 0
        # Snapshot of the native pool size taken before stop() frees it, so
        # stats() stays meaningful after the monitor has been stopped.
        self._final_allocated_buffers = 0
        self._valid_bytes = 0
        self._outstanding_warned = False
        self._dropped_records = 0

    def register_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        native = _cupti_monitor_native
        if self.version >= 2:
            if not self._cupti.has_v2():
                raise RuntimeError(
                    "CuptiMonitor(version=2) requires a libcupti with the v2 "
                    "user-defined-record API (>= 13.2); loaded "
                    f"{cupti_python.find_cupti_library()} reports "
                    f"{self._cupti.get_version()}"
                )
            request_addr = native.buffer_request_callback_address(2)
            complete_addr = native.buffer_complete_callback_address(2)
            # The v2 activity API is subscription-scoped: subscribe, turn on
            # user-defined records, and register the v2 buffer callbacks.
            self._subscriber = self._cupti.subscribe()
            self._cupti.arm_user_defined_records(
                self._subscriber, request_addr, complete_addr
            )
        else:
            request_addr = native.buffer_request_callback_address(1)
            complete_addr = native.buffer_complete_callback_address(1)
            self._cupti.activity_register_callbacks(request_addr, complete_addr)
        self._callbacks_registered = True

    def register_timestamp_callback(self) -> None:
        callback_addr = _cupti_monitor_native.approximate_time_callback_address()
        self._cupti.activity_register_timestamp_callback(callback_addr)
        self._timestamp_callback = callback_addr

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CUPTI monitor is already started")
        _cupti_monitor_native.reset_buffers()
        _cupti_monitor_native.configure_buffers(self.buffer_size)
        self.register_callbacks()
        self._time_converter = _PY_PROFILER._ApproximateClockToUnixTimeConverter()
        # The approximate-clock timestamp callback is incompatible with the v2
        # user-defined-record subscriber (cuptiActivityRegisterTimestampCallback
        # -> CUPTI_ERROR_NOT_COMPATIBLE), so v2 record timestamps stay in CUPTI's
        # native clock (durations are unaffected; absolute-time alignment is a v2
        # follow-up).
        if self.version < 2:
            self.register_timestamp_callback()
        self._session_start_unix_ns = time.time_ns()
        self._session_start_approx_ns = _PY_PROFILER._get_approximate_time()
        self._session_start_calibrated_unix_ns = self._convert_time(
            self._session_start_approx_ns
        )
        self._worker_stop.clear()
        self._flush_stop.clear()
        self._worker_error = None
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="torch-cupti-monitor-worker",
            daemon=True,
        )
        self._worker_thread.start()
        if self.flush_period_s > 0:
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name="torch-cupti-monitor-flush",
                daemon=True,
            )
            self._flush_thread.start()
        # Kinds/fields are enabled by the layout manager as observers register.
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Drain everything in flight (incl. CUPTI's async deliveries) before we
        # disable kinds and tear the worker down, so the final window is complete.
        self.flush(forced=True, sync=True)
        # Disable everything we enabled, then tear down the subscription (v2).
        self._disable(self._enabled.keys())
        self._enabled = {}
        self._wanted = {}
        self._record_layouts = []
        if self.version >= 2:
            if self._subscriber is not None:
                self._cupti.unsubscribe(self._subscriber)
            self._subscriber = None
            # Force a fresh subscribe on a subsequent start().
            self._callbacks_registered = False
        self._started = False
        self._flush_stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            if self._flush_thread.is_alive():
                logger.warning("CUPTI monitor flush thread did not stop within 5s")
            self._flush_thread = None
        self._worker_stop.set()
        # Unblock the decode thread waiting in get_completed().
        _cupti_monitor_native.shutdown_buffers()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                logger.warning("CUPTI monitor worker thread did not stop within 5s")
            self._worker_thread = None
        if self._worker_error is not None:
            raise RuntimeError("CUPTI monitor worker failed") from self._worker_error
        self._final_allocated_buffers = _cupti_monitor_native.allocated_buffers()
        _cupti_monitor_native.reset_buffers()
        self._time_converter = None
        self._timestamp_callback = None

    def flush(
        self, *, forced: bool = False, sync: bool = False, timeout_s: float = 5.0
    ) -> None:
        """Flush CUPTI's activity buffers to the processing worker.

        Plain (``sync=False``) just issues cuptiActivityFlushAll -- used by the
        background flush loop. With ``sync=True`` it blocks until the worker has
        processed every record up to the call, so the caller (drain, reconfigure,
        stop) sees a complete window.

        CUPTI invokes our buffer-complete callback on its own thread a beat *after*
        cuptiActivityFlushAll returns, so a single flush + idle-wait can race ahead
        of that async delivery and miss a just-flushed buffer. To fence
        deterministically we enable SYNCHRONIZATION just for this call, device-sync
        (which produces a SYNCHRONIZATION record past a captured CUPTI timestamp),
        then flush/drain until the worker decodes a sync record that recent. CUPTI
        delivers buffers in fill order, so seeing it means everything before is
        delivered too -- no timing guess, and concurrent activity only helps.
        SYNCHRONIZATION is enabled only for the fence so the session doesn't pay to
        record every sync between flushes."""
        if not sync:
            self._cupti.activity_flush_all(forced=forced)
            return
        added = self._begin_fence_kind()
        try:
            # _fence_sync_point marks a fence active (so the worker advances the
            # clock) BEFORE producing the sync record, then returns its timestamp.
            target = self._fence_sync_point()
            if target is None:
                # No CUDA available -> no GPU activity to fence; just flush.
                self._cupti.activity_flush_all(forced=True)
                return
            # Flush to deliver the sync-point's buffer, then block until the worker
            # decodes a sync record at/after it (the decoded clock reaching target).
            # The condition wakes the instant the worker advances the clock; the
            # short wait just re-drives a flush if CUPTI hasn't delivered the buffer
            # yet. The sync record is guaranteed to exist and be deliverable, so this
            # terminates; the deadline is only a backstop against an unexpected stall.
            deadline = time.time() + timeout_s
            while self._max_decoded_ns < target:
                self._cupti.activity_flush_all(forced=True)
                with self._decoded_cv:
                    if self._max_decoded_ns >= target:
                        return
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        logger.warning(
                            "CUPTI monitor flush(sync) did not reach its fence"
                        )
                        return
                    self._decoded_cv.wait(timeout=min(0.05, remaining))
        finally:
            with self._decoded_cv:
                self._fence_waiters -= 1
            self._end_fence_kind(added)

    def _begin_fence_kind(self) -> bool:
        """Enable + make decodable the SYNCHRONIZATION sync-point kind for the
        duration of a fence. Returns True if this call enabled it (so _end removes
        it); False if it was already enabled (an observer wanted it -- leave it)."""
        if _FENCE_KIND in self._enabled:
            return False
        if self.version >= 2:
            if self._subscriber is None:
                return False
            self._cupti.activity_enable_v2(self._subscriber, _FENCE_KIND, _FENCE_FIELDS)
            self._record_layouts = self._record_layouts + [
                (int(_FENCE_KIND), *record_layout(_FENCE_KIND, _FENCE_FIELDS))
            ]
        else:
            self._cupti.activity_enable(_FENCE_KIND)
        self._enabled = {**self._enabled, _FENCE_KIND: _FENCE_FIELDS}
        self._wanted = {**self._wanted, int(_FENCE_KIND): set(_FENCE_FIELDS)}
        return True

    def _end_fence_kind(self, added: bool) -> None:
        """Undo _begin_fence_kind (no-op if the kind was already enabled)."""
        if not added:
            return
        if self.version >= 2:
            if self._subscriber is not None:
                self._cupti.activity_disable_v2(self._subscriber, _FENCE_KIND)
            self._record_layouts = [
                layout
                for layout in self._record_layouts
                if layout[0] != int(_FENCE_KIND)
            ]
        else:
            self._cupti.activity_disable(_FENCE_KIND)
        self._enabled = {k: v for k, v in self._enabled.items() if k != _FENCE_KIND}
        self._wanted = {k: v for k, v in self._wanted.items() if k != int(_FENCE_KIND)}

    def _fence_sync_point(self) -> int | None:
        """Establish a deterministic fence point for ``flush(sync=True)``. Marks a
        fence active (so the worker advances the decoded clock, even if the
        background flush thread delivers the sync record first), captures CUPTI's
        clock, then device-syncs. The sync both drains outstanding GPU work and
        produces a SYNCHRONIZATION record with a timestamp past the captured point;
        the fence waits until that record is decoded. Unlike a tracer kernel, a sync
        adds no kernel, no cudaLaunchKernel, and no dispatcher op to the trace.
        Returns the timestamp, or None if no CUDA device is available -- in which
        case the caller must still balance the matching decrement."""
        with self._decoded_cv:
            self._fence_waiters += 1
        try:
            target = self._cupti.get_timestamp()
            torch.cuda.synchronize()
            return target
        except Exception:
            return None

    def _convert_time(self, value: int) -> int:
        if value == 0:
            return 0
        if self._time_converter is None:
            return value
        return self._time_converter.to_unix_ns(value)

    def convert_time(self, value: int) -> int:
        """Convert a CUPTI-clock timestamp to unix-epoch ns (public passthrough,
        used by observers). Identity until the monitor is started and the clock
        converter is calibrated."""
        return self._convert_time(value)

    def now_unix_ns(self) -> int:
        """Current time on the same unix-epoch clock as decoded record
        timestamps -- the approximate clock run through convert_time."""
        return self._convert_time(_PY_PROFILER._get_approximate_time())

    # --- observer registry (this monitor is the multiplexer) ---------------

    def register(
        self,
        activities: ActivitiesSpec,
        callback: Callable[..., None],
    ) -> _Observer:
        """Register an observer. ``activities`` is either an iterable of
        ``ActivityKind`` (meaning "all fields") or a field map ``{ActivityKind:
        iterable of field ids | "all"}`` selecting the fields per kind.

        ``callback(columns)`` fires from the worker thread once per completed
        buffer, with ``columns`` = ``{ActivityKind: {field_id: column}}`` -- the
        monitor demuxes every buffer to columns (whether v1 or v2) and slices them
        to this observer's selection, so the contract is identical for both
        versions (the observer never sees raw bytes or the decode strategy).

        Recomputes the enabled selection and starts the monitor on first
        registration."""
        kinds, fields = self._normalize_activities(activities)
        obs = _Observer(kinds, fields, callback)
        with self._lock:
            self._observers.append(obs)
            start_needed = not self._started
        if start_needed:
            self.start()
        self._apply_selection()
        return obs

    def unregister(self, obs: _Observer) -> None:
        """Unregister an observer; the layout manager drops kinds/fields no longer
        wanted by anyone, and the monitor stops once the last observer leaves.
        Idempotent."""
        with self._lock:
            if obs not in self._observers:
                return
            self._observers.remove(obs)
            empty = not self._observers
        if empty:
            self.stop()
        else:
            self._apply_selection()

    def _normalize_activities(
        self, activities: ActivitiesSpec
    ) -> tuple[frozenset[ActivityKind], dict[ActivityKind, frozenset[int]]]:
        """Resolve a registration request to ``(kinds, fields)``: the
        ``ActivityKind`` set plus, for v2, the per-activity field-id selection
        (``"all"``/``None`` -> the kind's full supported set; ``*_FIELD_KIND`` id 0
        is always included). v1 ignores fields (empty map)."""
        if isinstance(activities, Mapping):
            kinds: list[ActivityKind] = []
            fields: dict[ActivityKind, frozenset[int]] = {}
            for kind, sel in activities.items():
                k = ActivityKind(kind)
                kinds.append(k)
                fields[k] = self._resolve_fields(k, sel)
            return frozenset(kinds), fields
        kind_set = frozenset(ActivityKind(k) for k in activities)
        # Both versions resolve fields: the monitor always demuxes to columns, so
        # a bare kind list means "all fields of that kind".
        return kind_set, {k: self._resolve_fields(k, "all") for k in kind_set}

    @staticmethod
    def _resolve_fields(
        kind: ActivityKind, sel: Iterable[int] | str | None
    ) -> frozenset[int]:
        if sel is None or sel == "all":
            resolved = frozenset(int(f) for f in FIELD_REGISTRY.get(kind, frozenset()))
        else:
            resolved = frozenset(int(f) for f in sel)  # type: ignore[union-attr]
        return resolved | {0}  # FIELD_KIND (0) is required for enable + demux

    def _apply_selection(self) -> None:
        """Reconcile CUPTI's enabled selection (and the demux layout) to the current
        observer field union. Run only here -- when observers register/deregister or
        padding is toggled -- never per buffer; the layout is a function of
        registration. v1 enables whole kinds and decodes whole records via
        cupti-python; v2 enables a per-field selection and decodes columnar against
        a record layout computed from the field-size spec."""
        union = self._field_union()
        wanted = {int(k): set(v) for k, v in union.items()}
        if self.version >= 2:
            target = self._plan(wanted)
            if target != self._enabled:
                self._reconfigure_v2(target)
                self._enabled = target
            # Decode against the enabled (possibly padded) record layout, computed
            # from the spec; the demux extracts only the wanted fields from it.
            self._record_layouts = [
                (kind, *record_layout(kind, fields))
                for kind, fields in self._enabled.items()
            ]
        else:
            target_kinds = set(wanted)
            self._enable(target_kinds - self._enabled.keys())
            self._disable(self._enabled.keys() - target_kinds)
            self._enabled = {k: frozenset(wanted[k]) for k in target_kinds}
        self._wanted = wanted

    def _plan(self, wanted: dict[int, set[int]]) -> dict[int, frozenset[int]]:
        """The v2 field selection to enable: the wanted fields, padded to a uniform
        record size when uniform padding is on and achievable, else as-is."""
        union = {k: frozenset(v) for k, v in wanted.items()}
        return plan_padding(union) if self._pad_to_uniform else union

    def _reconfigure_v2(self, target: dict[int, frozenset[int]]) -> None:
        # Fence FIRST, while the current selection is still enabled and
        # _record_layouts still describes it, so every outstanding buffer is decoded
        # against the old layout before we switch. (The fence's tracer needs the old
        # kinds enabled to be recorded -- it can't run after a disable.) Then swap
        # the per-field selection on the subscriber.
        sub = self._subscriber
        if sub is None:
            return
        if self._enabled:
            # Drain buffers from the current selection (its kinds are enabled, so
            # the fence's tracer can be recorded). Nothing to drain on first enable.
            self.flush(forced=True, sync=True)
        for kind in self._enabled:
            self._cupti.activity_disable_v2(sub, kind)
        for kind, fields in target.items():
            self._cupti.activity_enable_v2(sub, kind, fields)

    def _enable(self, kinds: Iterable[int]) -> None:
        for kind in kinds:
            self._cupti.activity_enable(kind)
            self._apply_filters(kind)

    def _disable(self, kinds: Iterable[int]) -> None:
        if self.version >= 2:
            sub = self._subscriber
            if sub is not None:
                for kind in kinds:
                    self._cupti.activity_disable_v2(sub, kind)
        else:
            for kind in kinds:
                self._cupti.activity_disable(kind)

    def _apply_filters(self, kind: int) -> None:
        # v1: drop high-frequency, low-signal runtime/driver API callbacks. No-ops
        # when the per-cbid filter symbols are unavailable in the loaded libcupti.
        if kind == ActivityKind.RUNTIME:
            for cbid in cupti_python.disabled_runtime_cbids():
                self._cupti.activity_enable_runtime_api(cbid, False)
        if kind == ActivityKind.DRIVER:
            for cbid in cupti_python.disabled_driver_cbids():
                self._cupti.activity_enable_driver_api(cbid, False)

    def enable_uniform_padding(self, enabled: bool = True) -> None:
        """v2 only: pad the enabled selection so all kinds share one record size,
        letting multi-kind buffers take the vectorized stride decode (at the cost of
        extra buffer bandwidth). No-op on v1. Reconfigures if it changed."""
        if self.version < 2 or self._pad_to_uniform == enabled:
            return
        self._pad_to_uniform = enabled
        if self._started:
            self._apply_selection()

    def _field_union(self) -> dict[ActivityKind, frozenset[int]]:
        """The per-activity field selection wanted across all observers."""
        union: dict[ActivityKind, frozenset[int]] = {}
        with self._lock:
            for obs in self._observers:
                for kind, fset in obs.fields.items():
                    union[kind] = union.get(kind, frozenset()) | fset
        return union

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "version": self.version,
                "activities": list(self._enabled),
                "buffers_completed": self._buffers_completed,
                "buffers_allocated": _cupti_monitor_native.allocated_buffers()
                if self._started
                else self._final_allocated_buffers,
                "buffers_pending": _cupti_monitor_native.pending_buffers(),
                "valid_total_mb": self._valid_bytes / (1024 * 1024),
                "dropped_records": self._dropped_records,
                "observers": len(self._observers),
            }

    def push_external_correlation_id(self) -> int | None:
        """Allocate a process-unique external-correlation id and push it onto
        CUPTI's external-correlation stack. The stack is global (it spans all
        observers), so the monitor owns it; every CUDA activity recorded until the
        matching pop gets an EXTERNAL_CORRELATION record linking its
        correlation_id to this id. Returns the id, or None if not started. What
        the id *means* (e.g. a region name) is the caller's per-observer metadata,
        not the monitor's concern."""
        if not self._started:
            return None
        with self._lock:
            external_id = self._next_external_id
            self._next_external_id += 1
        if self.version >= 2:
            sub = self._subscriber
            if sub is None:
                return None
            pushed = self._cupti.activity_push_external_correlation_id_v2(
                sub, external_id
            )
        else:
            pushed = self._cupti.activity_push_external_correlation_id(external_id)
        return external_id if pushed else None

    def pop_external_correlation_id(self) -> int | None:
        """Pop the most recent external-correlation id off CUPTI's global stack.
        Returns the popped id, or None if not started/failed."""
        if not self._started:
            return None
        if self.version >= 2:
            sub = self._subscriber
            if sub is None:
                return None
            return self._cupti.activity_pop_external_correlation_id_v2(sub)
        return self._cupti.activity_pop_external_correlation_id()

    def session_info(self) -> dict[str, Any]:
        """Monitor/session metadata for consumers that need to describe the
        capture: versions, clock calibration, and buffer config. Call after
        start() so the clock fields are populated."""
        return {
            "cupti_version": self._cupti.get_version(),
            "cuda_version": _cuda_version_string(),
            "hes_enabled": is_hes_enabled(),
            "timestamp_mode": "approximate_clock",
            "session_start_unix_ns": self._session_start_unix_ns,
            "session_start_approx_ns": self._session_start_approx_ns,
            "session_start_calibrated_unix_ns": self._session_start_calibrated_unix_ns,
            "buffer_size": self.buffer_size,
            "flush_period_ns": int(self.flush_period_s * 1e9),
            "libcupti_path": cupti_python.find_cupti_library(),
        }

    def _flush_loop(self) -> None:
        try:
            while not self._flush_stop.wait(self.flush_period_s):
                if self._started:
                    self.flush(forced=False)
        except BaseException as exc:
            self._worker_error = exc
            self._worker_stop.set()

    def _worker_loop(self) -> None:
        try:
            while True:
                # Blocks with the GIL released until a buffer is ready; returns
                # None once stop() calls _cupti_monitor.shutdown_buffers().
                item = _cupti_monitor_native.get_completed()
                if item is None:
                    break
                # The native layout_epoch (5th field) is unused -- v2 computes its
                # own record layout from the spec rather than CUPTI's captured one.
                buffer_ptr, valid_size, ctx, stream_id, _layout_epoch = item
                with self._lock:
                    self._buffers_completed += 1
                    self._valid_bytes += valid_size
                try:
                    self._process_completed_buffer(
                        ctx, stream_id, buffer_ptr, valid_size
                    )
                finally:
                    _cupti_monitor_native.return_buffer(buffer_ptr)
                self._maybe_warn_backpressure()
        except BaseException as exc:
            self._worker_error = exc
            self._worker_stop.set()

    def _maybe_warn_backpressure(self) -> None:
        if self._outstanding_warned:
            return
        allocated = _cupti_monitor_native.allocated_buffers()
        if allocated >= _OUTSTANDING_WARN_THRESHOLD:
            self._outstanding_warned = True
            logger.warning(
                "CUPTI monitor allocated %d activity buffers; the processing "
                "worker is not keeping up with CUPTI, so memory use will grow. "
                "Reduce traced activity or buffer size.",
                allocated,
            )

    def _process_completed_buffer(
        self, ctx: int, stream_id: int, buffer_ptr: int, valid_size: int
    ) -> None:
        with self._lock:
            observers = list(self._observers)
        # The monitor owns parsing: demux the raw buffer into columns once, THEN
        # dispatch each observer the columns it selected. Observers never touch the
        # raw bytes or the decode strategy. The demux must finish before any
        # callback runs -- the buffer's bytes (and the strings its pointers
        # reference) are recycled once we return.
        decoded = self._demux(buffer_ptr, valid_size)
        if self._fence_waiters:
            self._advance_decoded_clock(decoded)
        self._dispatch_observers(decoded, observers)
        self._account_dropped_records(ctx, stream_id)

    def _advance_decoded_clock(self, decoded: dict[int, dict[int, Any]]) -> None:
        """Advance the decoded clock at sync points: track the max SYNCHRONIZATION
        END decoded so far (CUPTI clock) and wake any flush(sync=True) waiting for
        delivery to reach its fence point."""
        cols = decoded.get(_FENCE_KIND)
        if cols is None:
            return
        col = cols.get(_FENCE_END_FIELD)
        if col is None or not len(col):
            return
        newest = int(col.max())
        if newest > self._max_decoded_ns:
            with self._decoded_cv:
                self._max_decoded_ns = newest
                self._decoded_cv.notify_all()

    def _demux(self, buffer_ptr: int, valid_size: int) -> dict[int, dict[int, Any]]:
        """Demux a completed buffer into ``{kind: {field_id: column}}`` for the
        wanted fields. v2 decodes columnar against the record layout computed from
        the spec; v1 reads each record via cupti-python. Either way only the
        observers' selected fields are materialized, while the buffer is alive."""
        if not self._wanted:
            return {}
        if self.version >= 2:
            return decode_columns_v2(
                buffer_ptr, valid_size, self._record_layouts, self._wanted
            )
        return decode_columns_v1(
            self._iter_records, buffer_ptr, valid_size, self._wanted
        )

    def _iter_records(self, buffer_ptr: int, valid_size: int):
        # v1: walk the buffer yielding (kind, record_addr), threading CUPTI's in/out
        # record cursor (a fresh NULL each call would re-return the first record).
        prev_record: int | None = None
        while True:
            record_addr = self._cupti.activity_get_next_record(
                buffer_ptr, valid_size, prev_record
            )
            if record_addr is None:
                break
            yield self._cupti.read_activity_kind(record_addr), record_addr
            prev_record = record_addr

    def _dispatch_observers(
        self, decoded: dict[int, dict[int, Any]], observers: list[_Observer]
    ) -> None:
        """Hand each observer the already-demuxed columns sliced to its selection.
        Pure fan-out -- no buffer access, so observer callbacks need not finish
        before the buffer is recycled."""
        if not decoded:
            return
        for obs in observers:
            chunk: dict[ActivityKind, dict[int, Any]] = {}
            for kind, fields in obs.fields.items():
                kind_cols = decoded.get(int(kind))
                if not kind_cols:
                    continue
                sub = {f: kind_cols[f] for f in fields if f in kind_cols}
                if sub:
                    chunk[kind] = sub
            if chunk:
                obs.callback(chunk)

    def _account_dropped_records(self, ctx: int, stream_id: int) -> None:
        self._dropped_records += self._cupti.activity_get_num_dropped_records(
            ctx, stream_id
        )


_hes_enabled = False

_instance_lock = threading.Lock()
# At most one monitor per process; its activity-API version is fixed at first use
# from the $TORCH_CUPTI_MONITOR_USE_V2_API env var (see default_api_version).
_instance: CuptiMonitor | None = None


def enable_hes_early() -> None:
    global _hes_enabled
    if _hes_enabled:
        return
    if torch.cuda.is_initialized() or _has_active_cuda_context():
        raise RuntimeError(
            "enable_hes_early() must be called before CUDA context creation"
        )
    from cuda.bindings import driver as cuda_driver  # pyrefly: ignore[missing-import]

    rc = cuda_driver.cuInit(0)[0]
    if rc != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed with rc={rc}")

    # Use the direct libcupti call (self._cupti.activity_enable_hw_trace), not
    # cupti-python's activity_enable_hw_trace(): after torch is imported, the
    # latter makes subsequent cuptiActivityRegisterCallbacks() fail with
    # CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, while the direct call works.
    cupti_python.pylibcupti().activity_enable_hw_trace(True)
    _hes_enabled = True


def is_hes_enabled() -> bool:
    return _hes_enabled


# Env var selecting which CUPTI activity API the monitor uses by default: set it
# truthy to use v2 (user-defined records); unset/falsy uses v1 (classic activities).
# Only consulted when a version isn't passed explicitly.
USE_V2_ENV = "TORCH_CUPTI_MONITOR_USE_V2_API"


def default_api_version() -> int:
    """The CUPTI activity API version used when none is requested explicitly: 2 when
    ``$TORCH_CUPTI_MONITOR_USE_V2_API`` is truthy, else 1."""
    value = os.environ.get(USE_V2_ENV, "").strip().lower()
    return 2 if value in ("1", "true", "yes", "on") else 1


def get_monitor() -> CuptiMonitor | None:
    """The process-wide monitor singleton if it has been constructed, else None."""
    return _instance


def instance() -> CuptiMonitor:
    """The process-wide CUPTI monitor / multiplexer singleton, constructed on first
    use. Its activity-API version is fixed then from ``default_api_version()`` -- the
    ``$TORCH_CUPTI_MONITOR_USE_V2_API`` env var (1 = classic activities, 2 =
    user-defined records). Observers register with it via register()."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = CuptiMonitor(version=default_api_version())
        return _instance
