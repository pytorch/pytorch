# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

import torch

from . import cupti_python
from .cupti_python import ActivityKind
from .records import FIELD_REGISTRY, STRING_FIELDS, Sync


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
_FENCE_END_FIELD = Sync.END.id
_FENCE_FIELDS = frozenset({Sync.KIND.id, _FENCE_END_FIELD})


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


def _deref_cstr(ptr: int) -> str:
    if not ptr:
        return ""
    value = ctypes.cast(ptr, ctypes.c_char_p).value
    return value.decode(errors="replace") if value is not None else ""


class CuptiMonitorBuffer:
    """A completed CUPTI buffer (the item from ``_cupti_monitor.get_completed()``)
    plus the record layout CUPTI captured for it. Owns the buffer for its lifetime:
    it returns the buffer to the native pool on destruction (RAII), so the worker
    loop never has to. ``decode()`` demuxes its records columnar against the captured
    layout."""

    def __init__(self, item: tuple) -> None:
        # Bind _returned first so __del__ is safe even if unpacking fails.
        self._returned = False
        self.buffer_ptr, self.valid_size, self.ctx, self.stream, self.layouts = item

    def __del__(self) -> None:
        if not self._returned:
            self._returned = True
            _cupti_monitor_native.return_buffer(self.buffer_ptr)

    def decode(self) -> dict[int, dict[int, Any]]:
        """Demux this buffer into ``{kind: {field_id: column}}`` against the record
        layout CUPTI captured for it (``self.layouts``: ``[(kind, record_size,
        [(field_id, offset, size), ...]), ...]``). Every field in the layout is
        decoded -- the layout holds exactly the enabled selection (the observers'
        field union), so there is nothing extra to filter (the per-observer slice
        happens in dispatch).

        Records begin with *_FIELD_KIND (id 0, a 4-byte kind) at offset 0 and are
        sized by their kind's record_size. Three strategies, fastest first: one kind
        -> homogeneous stride; uniform size -> stride + dispatch by the KIND column;
        variable size -> per-record walk (CUPTI records aren't self-synchronizing).
        A bounds guard drops any trailing record that would run past valid_size."""
        buffer_ptr, valid_size, record_layouts = (
            self.buffer_ptr,
            self.valid_size,
            self.layouts,
        )
        # kind -> (record_size, {field_id: (offset, size)}).
        layouts: dict[int, tuple[int, dict[int, tuple[int, int]]]] = {}
        for kind, rsz, fields in record_layouts:
            if rsz > 0:
                layouts[kind] = (rsz, {fid: (off, sz) for fid, off, sz in fields})
        if not layouts or valid_size == 0:
            return {}

        raw = np.ctypeslib.as_array(
            (ctypes.c_uint8 * valid_size).from_address(buffer_ptr)
        )

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
                kinds_col = (
                    raw[starts[:, None] + np.arange(4)].copy().view("<u4").ravel()
                )
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
            positions = {
                k: np.array(v, dtype=np.int64) for k, v in pos_lists.items() if v
            }

        # Bounds guard: only decode records that fully fit in the valid region.
        for kind in list(positions):
            rsz = layouts[kind][0]
            fitted = positions[kind][positions[kind] + rsz <= valid_size]
            if len(fitted):
                positions[kind] = fitted
            else:
                del positions[kind]

        out: dict[int, dict[int, Any]] = {}
        for kind, pos_arr in positions.items():
            fields = layouts[kind][1]
            str_fields = STRING_FIELDS.get(kind, frozenset())
            cols: dict[int, Any] = {}
            for fid, (off, size) in fields.items():
                if fid in str_fields and size == 8:
                    # const char* field: deref each pointer to a str now.
                    ptrs = (
                        raw[pos_arr[:, None] + np.arange(off, off + 8)]
                        .copy()
                        .view("<u8")
                        .ravel()
                    )
                    cols[fid] = np.array(
                        [_deref_cstr(int(p)) for p in ptrs], dtype=object
                    )
                    continue
                if size not in (1, 2, 4, 8):
                    continue
                idx = pos_arr[:, None] + np.arange(off, off + size)
                cols[fid] = raw[idx].copy().view(f"<u{size}").ravel()
            if cols:
                out[kind] = cols
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
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        flush_period_s: float | None = None,
    ) -> None:
        # The monitor is the engine and the multiplexer: it owns the single CUPTI
        # subscription + buffer pool + decode worker, demuxes each completed buffer
        # into columns, and hands every observer the columns it selected. It reaches
        # CUPTI only through the self._cupti.activity_* wrappers -- no ctypes here.
        #
        # It uses CUPTI's v2 user-defined-record API: a subscriber + per-field
        # selection, decoded columnar against a record layout computed from the
        # field-size spec (no captured layout needed). This requires libcupti >= 13.2.
        self.buffer_size = buffer_size
        # Background-drain flush period (seconds). An explicit arg wins; otherwise it
        # comes from TORCH_CUPTI_MONITOR_FLUSH_PERIOD_S (default 1.0). Sign-encoded:
        #   > 0  -> background flush thread drains every flush_period_s.
        #    0   -> background flush thread drains continuously (no wait between flushes).
        #   < 0  -> NO background flush thread; the caller must drive flush() itself
        #           (e.g. at end of step). flush() semantics are unchanged -- the caller
        #           chooses sync=. This is the escape hatch for a libcupti/libnvperf HES
        #           thread-safety bug: the HW-trace decode the worker runs after
        #           cuptiActivityFlushAll can wild-write the host heap when it overlaps
        #           concurrent host activity (e.g. NCCL collective setup); draining only
        #           from the (quiescent) foreground avoids that race.
        if flush_period_s is None:
            flush_period_s = float(
                os.environ.get(
                    "TORCH_CUPTI_MONITOR_FLUSH_PERIOD_S", _DEFAULT_FLUSH_PERIOD_S
                )
            )
        self.flush_period_s = flush_period_s
        self._cupti = cupti_python.pylibcupti()
        # The CUPTI subscriber handle.
        self._subscriber: int | None = None
        # Layout state -- a function of registration, recomputed only when the
        # The fields enabled per kind on the subscriber (a function of the observer
        # field union, recomputed only on register/deregister, never per buffer). The
        # record byte layout is NOT tracked here -- each completed buffer carries
        # CUPTI's own captured layout (ppRecordLayouts) that records.decode reads.
        self._enabled: dict[int, frozenset[int]] = {}

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
        version = self._cupti.get_version()
        if version < cupti_python.LIBCUPTI_MIN_VERSION:
            raise RuntimeError(
                "CuptiMonitor requires libcupti >= "
                f"{cupti_python.LIBCUPTI_MIN_VERSION}; loaded "
                f"{cupti_python.find_cupti_library()} reports {version}"
            )
        native = _cupti_monitor_native
        request_addr = native.buffer_request_callback_address()
        complete_addr = native.buffer_complete_callback_address()
        # The activity API is subscription-scoped: subscribe, turn on user-defined
        # records, and register the v2 buffer callbacks. (A prior consumer that left
        # CUPTI attached -- e.g. Kineto -- can make cuptiSubscribe_v2 fail with
        # CUPTI_ERROR_MULTIPLE_SUBSCRIBERS; run such consumers with TEARDOWN_CUPTI=1
        # so they release CUPTI on teardown rather than us finalizing global state.)
        self._subscriber = self._cupti.subscribe()
        self._cupti.arm_user_defined_records(
            self._subscriber, request_addr, complete_addr
        )
        self._callbacks_registered = True

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CUPTI monitor is already started")
        _cupti_monitor_native.reset_buffers()
        _cupti_monitor_native.configure_buffers(self.buffer_size)
        self.register_callbacks()
        self._time_converter = _PY_PROFILER._ApproximateClockToUnixTimeConverter()
        # The approximate-clock timestamp callback is incompatible with the
        # user-defined-record subscriber (cuptiActivityRegisterTimestampCallback ->
        # CUPTI_ERROR_NOT_COMPATIBLE), so record timestamps stay in CUPTI's native
        # clock (durations are unaffected; absolute-time alignment is a follow-up).
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
        # Background drain when flush_period_s >= 0 (0 = drain continuously, no wait);
        # < 0 means no background thread -- the caller drives flush() itself.
        if self.flush_period_s >= 0:
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name="torch-cupti-monitor-flush",
                daemon=True,
            )
            self._flush_thread.start()
        # Kinds/fields are enabled by _apply_selection as observers register.
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Drain everything in flight (incl. CUPTI's async deliveries) before we
        # disable kinds and tear the worker down, so the final window is complete.
        self.flush(forced=True, sync=True)
        # Disable everything we enabled, then tear down the subscription.
        self._disable(self._enabled.keys())
        self._enabled = {}
        if self._subscriber is not None:
            # Release CUPTI without poisoning it for the next session: turn
            # user-defined-record mode back off (it changes CUPTI's record layout),
            # then unsubscribe. Crucially this does NOT call cuptiFinalize -- on this
            # libcupti a finalize poisons CUPTI for the rest of the process (a
            # subsequent monitor subscribe stops delivering buffers, and a classic
            # Kineto session records nothing), so disarm + unsubscribe is the only
            # clean teardown. This lets the monitor be started and stopped repeatedly
            # in one process. (Switching to a classic consumer after the monitor is a
            # separate libcupti limitation -- once the process has used UDR/v2 it
            # cannot downgrade without the poisonous finalize.)
            self._cupti.disarm_user_defined_records(self._subscriber)
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
        if _FENCE_KIND in self._enabled or self._subscriber is None:
            return False
        # Deliver records pending under the current selection before changing it:
        # without this, enabling the fence kind drops the still-buffered records
        # (e.g. kernels/launches) that the fence is about to flush for.
        self._cupti.activity_flush_all(forced=True)
        self._cupti.activity_enable(self._subscriber, _FENCE_KIND, _FENCE_FIELDS)
        self._enabled = {**self._enabled, _FENCE_KIND: _FENCE_FIELDS}
        return True

    def _end_fence_kind(self, added: bool) -> None:
        """Undo _begin_fence_kind (no-op if the kind was already enabled)."""
        if not added:
            return
        if self._subscriber is not None:
            # Flush before disabling so the records pending under the current
            # selection (incl. the fence's own sync record) are delivered rather
            # than dropped when the kind goes away.
            self._cupti.activity_flush_all(forced=True)
            self._cupti.activity_disable(self._subscriber, _FENCE_KIND)
        self._enabled = {k: v for k, v in self._enabled.items() if k != _FENCE_KIND}

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
            sub = self._subscriber
            if sub is None:
                return None
            # The subscriber-aware _v2 timestamp is required here: plain
            # cuptiGetTimestamp is CUPTI_ERROR_NOT_COMPATIBLE while the UDR subscriber
            # is active (13.3), which silently turned this fence into a no-op.
            target = self._cupti.get_timestamp(sub)
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
        monitor demuxes every buffer to columns and slices them to this observer's
        selection (the observer never sees raw bytes or the decode strategy).

        Recomputes the enabled selection and starts the monitor on first
        registration."""
        kinds, fields = self._normalize_activities(activities)
        obs = _Observer(kinds, fields, callback)
        with self._lock:
            self._observers.append(obs)
            start_needed = not self._started
        try:
            if start_needed:
                self.start()
            self._apply_selection()
        except Exception:
            # Don't leave a half-registered observer (or a half-started monitor) if
            # start/selection fails -- e.g. the CUPTI subscribe is rejected.
            with self._lock:
                if obs in self._observers:
                    self._observers.remove(obs)
            raise
        return obs

    def unregister(self, obs: _Observer) -> None:
        """Unregister an observer; drops kinds/fields no longer wanted by anyone,
        and the monitor stops once the last observer leaves. Idempotent."""
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
        ``ActivityKind`` set plus the per-activity field-id selection
        (``"all"``/``None`` -> the kind's full supported set; ``*_FIELD_KIND`` id 0
        is always included)."""
        if isinstance(activities, Mapping):
            kinds: list[ActivityKind] = []
            fields: dict[ActivityKind, frozenset[int]] = {}
            for kind, sel in activities.items():
                k = ActivityKind(kind)
                kinds.append(k)
                fields[k] = self._resolve_fields(k, sel)
            return frozenset(kinds), fields
        kind_set = frozenset(ActivityKind(k) for k in activities)
        # A bare kind list means "all fields of that kind".
        return kind_set, {k: self._resolve_fields(k, "all") for k in kind_set}

    @staticmethod
    def _resolve_fields(
        kind: ActivityKind, sel: Iterable[int] | str | None
    ) -> frozenset[int]:
        if sel is None or sel == "all":
            resolved = frozenset(f for f in FIELD_REGISTRY.get(kind, frozenset()))
        else:
            resolved = frozenset(int(f) for f in sel)  # type: ignore[union-attr]
        return resolved | {0}  # FIELD_KIND (0) is required for enable + demux

    def _apply_selection(self) -> None:
        """Reconcile CUPTI's enabled per-field selection to the current observer
        field union. Run only here -- when observers register/deregister -- never
        per buffer. No demux layout is computed: each completed buffer carries
        CUPTI's own captured layout (ppRecordLayouts), so this only sets which fields
        are enabled on the subscriber."""
        target = {int(k): frozenset(v) for k, v in self._field_union().items()}
        if target != self._enabled:
            self._reconfigure(target)
            self._enabled = target

    def _reconfigure(self, target: dict[int, frozenset[int]]) -> None:
        # Swap the per-field selection on the subscriber. No drain needed: each
        # completed buffer carries CUPTI's own captured layout, so buffers recorded
        # under the old selection still decode correctly against their own layout
        # even after the switch.
        sub = self._subscriber
        if sub is None:
            return
        for kind in self._enabled:
            self._cupti.activity_disable(sub, kind)
        # CUPTI requires a flush between disabling and (re-)enabling activities: a
        # kind re-enabled with a new field selection otherwise loses the records
        # pending under the old selection. Flush only when something was actually
        # disabled (nothing pending to lose on the first, empty-set configure).
        if self._enabled:
            self._cupti.activity_flush_all(forced=True)
        for kind, fields in target.items():
            self._cupti.activity_enable(sub, kind, fields)

    def _disable(self, kinds: Iterable[int]) -> None:
        sub = self._subscriber
        if sub is not None:
            for kind in kinds:
                self._cupti.activity_disable(sub, kind)

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
        if not self._started or self._subscriber is None:
            return None
        with self._lock:
            external_id = self._next_external_id
            self._next_external_id += 1
        # Pass the subscriber: the plain push returns NOT_COMPATIBLE under the UDR
        # subscriber, so the wrapper uses the subscriber-aware _v2 variant.
        pushed = self._cupti.activity_push_external_correlation_id(
            external_id, sub_handle=self._subscriber
        )
        return external_id if pushed else None

    def pop_external_correlation_id(self) -> int | None:
        """Pop the most recent external-correlation id off CUPTI's global stack.
        Returns the popped id, or None if not started/failed."""
        if not self._started or self._subscriber is None:
            return None
        return self._cupti.activity_pop_external_correlation_id(
            sub_handle=self._subscriber
        )

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
                buf = CuptiMonitorBuffer(item)
                try:
                    # decode() copies every field into fresh arrays / strs, so the
                    # returned columns hold no reference to the buffer's memory.
                    columns = buf.decode()
                    ctx, stream, valid_size = buf.ctx, buf.stream, buf.valid_size
                finally:
                    # Return the buffer to the pool now -- deterministically (not via
                    # GC) and before dispatch, so nothing references it and it never
                    # lingers in an exception traceback past stop()/reset_buffers().
                    del buf
                with self._lock:
                    self._buffers_completed += 1
                    self._valid_bytes += valid_size
                    observers = list(self._observers)
                if self._fence_waiters:
                    self._advance_decoded_clock(columns)
                self._dispatch_observers(columns, observers)
                self._account_dropped_records(ctx, stream)
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
# At most one monitor per process.
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


def get_monitor() -> CuptiMonitor | None:
    """The process-wide monitor singleton if it has been constructed, else None."""
    return _instance


def instance() -> CuptiMonitor:
    """The process-wide CUPTI monitor / multiplexer singleton, constructed on first
    use. It uses CUPTI's v2 user-defined-record API (requires libcupti >= 13.2).
    Observers register with it via register()."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = CuptiMonitor()
        return _instance
