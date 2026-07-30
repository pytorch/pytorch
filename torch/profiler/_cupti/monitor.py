# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast, TYPE_CHECKING

import numpy as np
from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

import torch

from . import cupti_python
from .records import Ctype, FIELD_CTYPE, FIELD_REGISTRY, Kernel, STRING_FIELDS, Sync


# A registration request: either a plain iterable of activity kinds (meaning "all
# fields"), or a field map {kind: iterable of field ids | "all"} selecting specific
# fields per kind. The monitor demuxes the selected fields to columns.
ActivitiesSpec = Mapping[ActivityKind, "Iterable[int] | str"] | Iterable[ActivityKind]


_PY_PROFILER = torch._C._profiler
# The native CUPTI buffer-pool / layout-capture module (C++ side of the monitor).
_cupti_monitor_native = _PY_PROFILER._cupti_monitor

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

# Buffers are a recycling pool bounded by peak concurrent demand, so the count
# only keeps climbing if the worker can't drain completed buffers as fast as
# CUPTI fills them. Warn once past this many outstanding buffers (1GB at the
# default 4MB size) as a sign of that backpressure.
_OUTSTANDING_WARN_THRESHOLD = 256

# flush(sync=True) fences at a SYNC point: it enables SYNCHRONIZATION, captures
# CUPTI's clock, device-syncs (which produces a SYNCHRONIZATION record at a
# timestamp past that point), waits until the native decoder reports a sync record
# that recent, then disables SYNCHRONIZATION again. CUPTI delivers buffers in fill
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


class _Observer:
    """A registered consumer of the monitor's records: the activity kinds it
    requested, its per-kind field selection (``{kind: frozenset(field_ids)}``), and
    its ``callback(columns)`` -- invoked once per drain (at flush time) with the
    demuxed columns sliced to its selection (see ``CuptiMonitor.register``)."""

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


class _SynchronizedClock:
    """Maps CUPTI record timestamps to unix-epoch ns on kineto's axis.

    CUPTI stamps records with cuptiGetTimestamp, which is CLOCK_REALTIME -- the very clock
    kineto's cpu_ops land on (kineto reads the approx clock for cpu_ops and converts it to
    unix == CLOCK_REALTIME at serialization). So a record's timestamp already IS its unix
    time: it is passed through unchanged (identity, offset 0). Records and cpu_ops then share
    the identical realtime clock, so a runtime call's start is its true realtime -- always >=
    the start of the cpu_op that issued it, with no clock estimate in the loop. calibrate()
    requires this and verifies it by bracketing one native read between two
    clock_gettime(CLOCK_REALTIME) reads; a host whose CUPTI clock is something else raises.

    The exception is the timestamp callback: when it engages CUPTI stamps records with the
    profiler's approx clock instead -- kineto's *source* clock -- so records arrive as approx
    ticks and are mapped to unix via the same _ApproximateClockToUnixTimeConverter kineto uses
    (a single-slope line -- median rate over 1001 samples -- so two evaluations recover its
    slope exactly; recovered once so the hot path never calls the converter per record). That
    path lifts the CLOCK_REALTIME requirement and is the ONLY one that touches the converter.

    calibrate() reads the native clock through an injected callable, so the conversion math
    runs (and is tested) without a live CUPTI session.
    """

    def __init__(self) -> None:
        self._callback_active = False
        self._native_is_realtime = False
        self._native_now: Callable[[], int] = lambda: 0
        # Session-start anchor. 0 until calibrated -> conversion is identity.
        self._native_ns = 0
        self._unix_ns = 0
        self._approx_ns = 0
        self._scale = 1.0

    def calibrate(
        self,
        *,
        callback_active: bool,
        native_now: Callable[[], int],
        converter: Any = None,
    ) -> None:
        self._callback_active = callback_active
        self._native_now = native_now
        # cuptiGetTimestamp is CLOCK_REALTIME iff it lands inside a clock_gettime bracket.
        rt_lo = time.clock_gettime_ns(time.CLOCK_REALTIME)
        cupti_now = native_now()
        rt_hi = time.clock_gettime_ns(time.CLOCK_REALTIME)
        self._native_is_realtime = cupti_now != 0 and rt_lo <= cupti_now <= rt_hi
        if callback_active:
            # Records are approx-clock ticks. Build the converter first (it only maps approx
            # reads taken after construction), then bracket the native read -- needed for PM
            # frames, which stay on the native clock -- between two approx reads and pair at
            # the midpoint; recover the converter's slope for vectorized column conversion.
            if converter is None:
                converter = _PY_PROFILER._ApproximateClockToUnixTimeConverter()
            approx_before = _PY_PROFILER._get_approximate_time()
            self._native_ns = (
                time.clock_gettime_ns(time.CLOCK_REALTIME)
                if self._native_is_realtime
                else native_now()
            )
            approx_after = _PY_PROFILER._get_approximate_time()
            self._approx_ns = (approx_before + approx_after) // 2
            self._unix_ns = converter.to_unix_ns(self._approx_ns)
            span = 10**12
            self._scale = (
                converter.to_unix_ns(self._approx_ns + span) - self._unix_ns
            ) / span
        else:
            if not self._native_is_realtime:
                raise RuntimeError(
                    "cupti_monitor requires cuptiGetTimestamp to be CLOCK_REALTIME so "
                    "record timestamps share kineto's unix clock; this host reports a "
                    "different clock source, unsupported until the CUPTI timestamp "
                    "callback is available"
                )
            # cuptiGetTimestamp == CLOCK_REALTIME == unix: records already carry unix time,
            # so pass them through (identity). No approx clock or converter needed.
            self._native_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
            self._approx_ns = 0
            self._unix_ns = self._native_ns
            self._scale = 1.0

    # A timestamp is in one of two source domains -- approx-clock ticks or CLOCK_REALTIME/unix --
    # and both map to unix; these are the two conversion primitives (0, and the uncalibrated
    # state, always map to itself). Which domain a record is in depends on the timestamp callback,
    # so the record dispatch lives in CuptiMonitor.convert_time[_array], not here.

    def convert_approx(self, value: int) -> int:
        # Approx-clock tick -> unix ns via the recovered converter slope.
        if value == 0 or self._native_ns == 0:
            return value
        return self._unix_ns + int((value - self._approx_ns) * self._scale)

    def convert_unix(self, value: int) -> int:
        # Native cuptiGetTimestamp clock (CLOCK_REALTIME) -> unix ns: a constant offset.
        if value == 0 or self._native_ns == 0:
            return value
        return value - self._native_ns + self._unix_ns

    def convert_approx_array(self, values: np.ndarray) -> np.ndarray:
        # Vectorized convert_approx. Keep the delta in float (small magnitude) and add the unix
        # anchor as int64 -- adding at the ~1e18 unix magnitude in float64 would lose ~us.
        out = values.astype(np.int64)
        if self._native_ns == 0:
            return out
        ticks = (out - self._approx_ns).astype(np.float64)
        delta = (ticks * self._scale).astype(np.int64)
        return np.where(out == 0, out, self._unix_ns + delta)

    def convert_unix_array(self, values: np.ndarray) -> np.ndarray:
        # Vectorized convert_unix.
        out = values.astype(np.int64)
        if self._native_ns == 0:
            return out
        offset = self._unix_ns - self._native_ns
        return np.where(out == 0, out, out + offset)

    def now_record_ns(self) -> int:
        # Current record-clock value: the approx clock when the callback is active, else the
        # native cuptiGetTimestamp clock (read cheaply via clock_gettime when it is realtime).
        # 0 before calibration.
        if self._native_ns == 0:
            return 0
        if self._callback_active:
            return _PY_PROFILER._get_approximate_time()
        if self._native_is_realtime:
            return time.clock_gettime_ns(time.CLOCK_REALTIME)
        return self._native_now()

    def reset(self) -> None:
        self._native_ns = 0
        self._native_now = lambda: 0

    @property
    def unix_anchor_ns(self) -> int:
        return self._unix_ns

    @property
    def approx_anchor_ns(self) -> int:
        return self._approx_ns


class CuptiMonitor:
    """Process-wide CUPTI monitor / multiplexer singleton. Like PmSampler, ``CuptiMonitor()``
    returns the one instance (constructed on first call); its settings are snapshotted from
    the class config -- set via ``cupti_monitor.configure()`` before first use -- not from
    constructor args."""

    _instance: CuptiMonitor | None = None
    _instance_lock = threading.Lock()
    # Process-wide settings the singleton snapshots when first constructed; set via
    # configure() (first-come-first-serve, no env var), defaults otherwise. Both cadences
    # default to -1 (no background flush / drain -- the caller drives flush()).
    #
    # use_approx_timestamps defaults OFF; when on, the per-subscriber timestamp callback
    # re-times HOST and DEVICE records onto the profiler's approx clock (kineto's timebase).
    # Opt-in via configure(use_approx_timestamps=True).
    _buffer_size: int = 4 * 1024 * 1024
    _background_flush_period_s: float = -1.0
    _background_drain_period_s: float = -1.0
    _use_approx_timestamps: bool = False
    _configured: bool = False

    def __new__(cls) -> Self:
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init()
                cls._instance = inst
                # the live monitor snapshotted the config; lock it
                cls._configured = True
            return cls._instance

    def _init(self) -> None:
        cls = type(self)
        # The monitor is the engine and the multiplexer: it owns the single CUPTI
        # subscription + buffer pool + native decode worker, which demuxes each
        # completed buffer into columns; the monitor drains those columns at flush
        # time and hands every observer the columns it selected. It reaches CUPTI
        # only through the self._cupti.activity_* wrappers -- no ctypes here.
        #
        # It uses CUPTI's v2 user-defined-record API: a subscriber + per-field
        # selection, decoded columnar against a record layout computed from the
        # field-size spec (no captured layout needed). This requires libcupti >= 13.2.
        #
        # Per-buffer pool size (bytes), default 4 MiB. Bigger buffers complete less often
        # (fewer worker wakeups, lower overhead) at the cost of more pinned host memory and
        # coarser delivery. Set process-wide via cupti_monitor.configure().
        self.buffer_size = cls._buffer_size
        # Two independent cadences (seconds); they control different things, so they are
        # separate knobs.
        #
        # background_flush_period_s (default -1) -- how often the native decode thread
        # self-drives cuptiActivityFlushAll, handing CUPTI's completed buffers to the
        # decoder. Sign-encoded:
        #   >= 0 -> the decode thread self-flushes on this cadence (0 = continuously).
        #   <  0 -> NO self-flush; the caller must drive flush() itself (e.g. at end of
        #           step). This is the default, and the escape hatch for a libcupti/libnvperf
        #           HES thread-safety bug: cuptiActivityFlushAll drives CUPTI's HW-trace
        #           processing against live collection state and can wild-write the host
        #           heap when it overlaps concurrent host activity (e.g. NCCL collective
        #           setup). The racy op is the flush, NOT the decode -- the decoder keeps
        #           decoding delivered buffers off-thread regardless. Driving flush() only
        #           from the quiescent foreground avoids the race.
        #
        # background_drain_period_s (default -1) -- how often the Python thread drains the
        # decoded columns and dispatches them to observers (GIL work). Same sign-encoding:
        #   >= 0 -> a background thread drains on this cadence (0 = continuously).
        #   <  0 -> NO background drain thread; the caller drives drain via flush().
        self.background_flush_period_s = cls._background_flush_period_s
        self.background_drain_period_s = cls._background_drain_period_s
        self._cupti = cupti_python.pylibcupti()
        # The CUPTI subscriber handle.
        self._subscriber: int | None = None
        self._latency_enabled = False
        # Layout state -- a function of registration, recomputed only when the
        # The fields enabled per kind on the subscriber (a function of the observer
        # field union, recomputed only on register/deregister, never per buffer). The
        # record byte layout is NOT tracked here -- each completed buffer carries
        # CUPTI's own captured layout (ppRecordLayouts) that records.decode reads.
        self._enabled: dict[int, frozenset[int]] = {}

        self._lock = threading.Lock()
        self._started = False
        self._callbacks_registered = False
        self._drain_stop = threading.Event()
        self._drain_and_dispatch_thread: threading.Thread | None = None
        # Serializes _drain_and_dispatch: the native decoder accumulates columns
        # GIL-free; Python drains them here. Only ever one driver at a time (the
        # foreground caller OR the background flush loop, never both), but the lock
        # keeps a stray concurrent drain from interleaving dispatch.
        self._drain_lock = threading.Lock()
        self._observers: list[_Observer] = []
        # {external_id: metadata blob} drained alongside the decoded records (see
        # drain_decoded). Accumulated here until an observer's external-correlation
        # join consumes it via take_external_metadata(); the blob is attached onto
        # the kernel event keyed by the same external id. Guarded by _lock.
        self._external_metadata: dict[int, str] = {}
        self._next_external_id = 1
        # All subsystems push external-correlation ids on ONE CUPTI kind, so CUPTI
        # inserts a single EXTERNAL_CORRELATION record (that kind's stack top) per op
        # -- i.e. it tags a kernel with only the *innermost* active id; the enclosing
        # ids are recovered from our own bookkeeping here. _id_chains[id] is the full
        # active stack (outermost..id) captured when `id` was pushed, so a consumer
        # maps a kernel's innermost id to every context active for that op (see
        # external_id_chain) and picks the one it owns (by membership in its own
        # name/metadata map). The live LIFO is per-thread (CUPTI's external-correlation
        # stacks are per-thread) in _push_tls.stack. A popped id's chain is kept a
        # couple of dispatch cycles -- its activity records arrive after the pop -- then
        # dropped via the _chains_gc_* generations. _id_chains/_chains_gc_* are guarded
        # by _lock; the per-thread stack needs no lock.
        self._push_tls = threading.local()
        self._id_chains: dict[int, tuple[int, ...]] = {}
        self._chains_gc_pending: list[int] = []
        self._chains_gc_ready: list[int] = []
        # Record-timestamp -> unix conversion lives in the clock; the monitor delegates
        # convert_time / now_record_ns to it and calibrates it in start(). Records normally
        # arrive on the native (realtime) clock; configure(use_approx_timestamps=True) puts
        # them directly on the approx clock via CUPTI's per-subscriber timestamp callback
        # (opt-in, sole-subscriber only).
        self._timestamp_callback_enabled = cls._use_approx_timestamps
        self._clock = _SynchronizedClock()
        self._timestamp_callback_active = False

        # Snapshot of the native pool size taken before stop() frees it, so
        # stats() stays meaningful after the monitor has been stopped.
        self._final_allocated_buffers = 0
        self._outstanding_warned = False
        self._dropped_records = 0

        # Opt-in PM sampling (true SM-active % / DRAM-throughput % counters): the monitor registers
        # each requesting observer as a consumer of the current device's per-device PmSampler
        # (only one PM session per device is possible), polls it on the flush cadence, and converts
        # each polled frame's raw CUPTI-clock timestamps into the trace clock before pushing to the
        # observer's sink (the sampler is clock-agnostic; conversion lives here, where the clock base
        # does). Each consumer brings its own metrics; the shared session samples their union, starts
        # on the first consumer, and disables after the last. self._pm_consumers maps an observer's
        # sink to its sampler handle so poll/release can address it.
        self._pm_consumers: dict[Callable[[dict[str, Any]], None], Any] = {}
        self._pm_sampler: Any = None
        # Serializes PM add/poll/remove so a flush-thread poll never decodes the collector while the
        # foreground is tearing it down (concurrent decode on one collector is unsafe).
        self._pm_lock = threading.Lock()

    def register_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        version = self._cupti.get_version()
        if version < cupti_python.LIBCUPTI_MIN_VERSION:
            raise RuntimeError(
                "CuptiMonitor requires libcupti >= "
                f"{cupti_python.LIBCUPTI_MIN_VERSION}; loaded "
                f"{cupti_python.LIBCUPTI_SONAME} reports {version}"
            )
        native = _cupti_monitor_native
        request_addr = native.buffer_request_callback_address()
        complete_addr = native.buffer_complete_callback_address()
        # The activity API is subscription-scoped: subscribe, turn on user-defined
        # records, and register the v2 buffer callbacks. (A prior consumer that left
        # CUPTI attached -- e.g. Kineto -- can make cuptiSubscribe_v2 fail with
        # CUPTI_ERROR_MULTIPLE_SUBSCRIBERS; run such consumers with TEARDOWN_CUPTI=1
        # so they release CUPTI on teardown rather than us finalizing global state.)
        # Subscribe solo only when the timestamp callback is opted in: CUPTI honors it only
        # while multiple subscribers are NOT allowed. Otherwise allow coexistence (default).
        try:
            self._subscriber = self._cupti.subscribe(
                allow_multiple=not self._timestamp_callback_enabled
            )
        except cupti_python.CuptiError as e:
            if self._timestamp_callback_enabled:
                # We requested sole-subscriber mode only because the approx-clock timestamp
                # callback needs it; another CUPTI consumer is likely attached. Point the user
                # at the opt-out so they can coexist (the callback is then off).
                raise RuntimeError(
                    "cupti_monitor could not subscribe as the sole subscriber, which the "
                    "opt-in approx-clock timestamp callback requires (another CUPTI consumer "
                    "is likely attached). Retry with use_approx_timestamps=False to allow "
                    f"coexisting subscribers: {e}"
                ) from e
            raise
        # Arm the per-subscriber approx-clock timestamp callback right after subscribe, before
        # arming UDR, so it is in effect before any user-defined record is produced.
        self._timestamp_callback_active = self._try_arm_approx_timestamp_callback(
            self._subscriber
        )
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
        # Put activity records on kineto's unix timeline via the clock (see _SynchronizedClock):
        # normally cuptiGetTimestamp == CLOCK_REALTIME == unix and records pass through, unless
        # register_callbacks armed the timestamp callback (approx clock). calibrate() reads the
        # native clock through this callable, which is valid for the life of the subscription.
        self._clock.calibrate(
            callback_active=self._timestamp_callback_active,
            native_now=lambda: self._cupti.get_timestamp(cast(int, self._subscriber)),
        )
        self._drain_stop.clear()
        # Hand the native decode worker the subscriber + the cuptiActivityGetNextRecord_v2
        # (and, for self-flush, cuptiActivityFlushAll) addresses, so it iterates records and
        # drives the periodic flush without a libcupti link, plus the fence kind/field so it
        # tracks the SYNCHRONIZATION-END clock for flush(sync). It pulls completed buffers and
        # decodes them GIL-free; Python drains the accumulated columns at flush time, so
        # per-buffer decode never contends with the training thread. When
        # background_flush_period_s >= 0 the decode thread also drives the periodic plain
        # cuptiActivityFlushAll itself (GIL-free) on that cadence, so there is no separate
        # flush thread; < 0 disables self-flush (the caller drives flush()).
        fn_addr = self._cupti.get_next_record_fn_address()
        if not fn_addr:
            raise RuntimeError(
                "libcupti is missing cuptiActivityGetNextRecord_v2 (need >= 13.2); "
                f"loaded {cupti_python.LIBCUPTI_SONAME}"
            )
        if self.background_flush_period_s >= 0:
            self_flush = True
            flush_period_ns = int(self.background_flush_period_s * 1e9)
            flush_fn = self._cupti.get_flush_fn_address()
            if not flush_fn:
                raise RuntimeError(
                    "libcupti is missing cuptiActivityFlushAll; "
                    f"loaded {cupti_python.LIBCUPTI_SONAME}"
                )
        else:
            self_flush = False
            flush_period_ns = 0
            flush_fn = 0
        _cupti_monitor_native.configure_decoder(
            cast(int, self._subscriber),
            fn_addr,
            int(_FENCE_KIND),
            _FENCE_END_FIELD,
            self_flush,
            flush_period_ns,
            flush_fn,
        )
        _cupti_monitor_native.start_decoder()
        # The decode thread self-flushes on background_flush_period_s (configured above); this
        # Python loop only pulls the decoded columns and dispatches them to observers,
        # which calls Python back and so must hold the GIL. No loop at background_drain_period_s < 0
        # -- the caller drives drain via flush() itself.
        if self.background_drain_period_s >= 0:
            self._drain_and_dispatch_thread = threading.Thread(
                target=self._drain_and_dispatch_loop,
                name="torch-cupti-monitor-drain",
                daemon=True,
            )
            self._drain_and_dispatch_thread.start()
        # Kinds/fields are enabled by _apply_selection as observers register.
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        # Stop the Python drain loop. The decode thread keeps running (and self-flushing
        # on its cadence) until stop_decoder() below, so it can still decode the fence's
        # sync record; its plain cadence flush and the fence's foreground flush are both
        # completed-buffers-only, so overlapping them is harmless.
        self._drain_stop.set()
        if self._drain_and_dispatch_thread is not None:
            self._drain_and_dispatch_thread.join(timeout=5.0)
            if self._drain_and_dispatch_thread.is_alive():
                logger.warning("CUPTI monitor drain thread did not stop within 5s")
            self._drain_and_dispatch_thread = None
        # Flush thread is down (no concurrent poll): final tail-drain + disable the PM sessions
        # while observers are still registered, so their last samples are delivered.
        self._stop_pm_sampler()
        # Drain everything in flight (incl. CUPTI's async deliveries) before we tear
        # the decoder down, so the final window is complete. Then stop the native
        # decode worker while the subscriber is STILL valid -- it may still decode a
        # few buffers the fence's trailing flush delivered, and it iterates records
        # via the subscriber, so it must not outlive the unsubscribe -- and dispatch
        # the residue.
        self.flush(sync=True)
        _cupti_monitor_native.stop_decoder()
        self._drain_and_dispatch()
        # Clear the timestamp callback (restore CUPTI's default timer) before unsubscribe.
        if self._timestamp_callback_active and self._subscriber is not None:
            self._cupti.disarm_approx_timestamp_callback(self._subscriber)
            self._timestamp_callback_active = False
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
        self._final_allocated_buffers = _cupti_monitor_native.allocated_buffers()
        _cupti_monitor_native.reset_buffers()
        self._clock.reset()

    def flush(self, *, sync: bool = False, timeout_s: float = 5.0) -> None:
        """Flush CUPTI's activity buffers to the processing worker.

        Both paths issue ``cuptiActivityFlushAll(0)`` -- which hands over COMPLETED
        records, never in-progress ones -- and end by draining the native decoder's
        accumulated columns and dispatching them to the observers. The monitor never
        FORCE-flushes (``CUPTI_ACTIVITY_FLAG_FLUSH_FORCED``): a forced flush hands back
        a still-running kernel's record with a zero end timestamp and consumes it, so
        CUPTI never re-delivers the real completion (it would strand a slow-but-healthy
        collective as a false hang in the comm watchdog), and forcing in-progress
        buffers over concurrent host activity is the flush race that corrupts the HES
        heap and freezes the decode worker.

        Plain (``sync=False``) flushes then drains -- the background flush loop and the
        per-step foreground driver. With ``sync=True`` it first blocks until the native
        decoder has processed every record up to the call, so the caller (drain,
        reconfigure, stop) sees a complete window.

        CUPTI invokes our buffer-complete callback on its own thread a beat *after*
        cuptiActivityFlushAll returns, so a single flush + idle-wait can race ahead
        of that async delivery and miss a just-flushed buffer. To fence
        deterministically we enable SYNCHRONIZATION just for this call, device-sync
        (which both completes outstanding GPU work -- so a plain flush now delivers
        everything -- and produces a SYNCHRONIZATION record past a captured CUPTI
        timestamp), then flush/poll until the native decoder reports a sync record that
        recent. CUPTI delivers buffers in fill order, so seeing it means everything
        before is delivered too -- no timing guess, and concurrent activity only helps.
        SYNCHRONIZATION is enabled only for the fence so the session doesn't pay to
        record every sync between flushes."""
        if not sync:
            self._cupti.activity_flush_all()
            self._account_dropped_records(0, 0)
            self._drain_and_dispatch()
            return
        added = self._begin_fence_kind()
        try:
            target = self._fence_sync_point()
            if target is None:
                # No CUDA available -> no GPU activity to fence; just flush.
                self._cupti.activity_flush_all()
                return
            # Flush to deliver the sync-point's buffer, then poll until the native
            # decoder has processed a sync record at/after it (its max-sync clock
            # reaching target). CUPTI delivers buffers in fill order, so seeing that
            # sync record means everything before it is delivered and decoded too.
            # The sync record is guaranteed to exist and be deliverable, so this
            # terminates; the deadline is only a backstop against an unexpected stall.
            deadline = time.time() + timeout_s
            while _cupti_monitor_native.decoder_max_sync_ns() < target:
                self._cupti.activity_flush_all()
                if _cupti_monitor_native.decoder_max_sync_ns() >= target:
                    break
                if time.time() >= deadline:
                    logger.warning("CUPTI monitor flush(sync) did not reach its fence")
                    break
                time.sleep(0.005)
        finally:
            self._end_fence_kind(added)
            # The fence guarantees everything up to the sync point is decoded; hand
            # the accumulated window to the observers now.
            self._drain_and_dispatch()

    def _begin_fence_kind(self) -> bool:
        """Enable + make decodable the SYNCHRONIZATION sync-point kind for the
        duration of a fence. Returns True if this call enabled it (so _end removes
        it); False if it was already enabled (an observer wanted it -- leave it)."""
        if _FENCE_KIND in self._enabled or self._subscriber is None:
            return False
        # Deliver records pending under the current selection before changing it:
        # without this, enabling the fence kind drops the still-buffered records
        # (e.g. kernels/launches) that the fence is about to flush for.
        self._cupti.activity_flush_all()
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
            self._cupti.activity_flush_all()
            self._cupti.activity_disable(self._subscriber, _FENCE_KIND)
        self._enabled = {k: v for k, v in self._enabled.items() if k != _FENCE_KIND}

    def _fence_sync_point(self) -> int | None:
        """Establish a deterministic fence point for ``flush(sync=True)``: capture
        CUPTI's clock, then device-sync. The sync both drains outstanding GPU work
        and produces a SYNCHRONIZATION record with a timestamp past the captured
        point; the fence waits until the native decoder reports that record. Unlike a
        tracer kernel, a sync adds no kernel, no cudaLaunchKernel, and no dispatcher
        op to the trace. Returns the timestamp, or None if no CUDA device is
        available. SYNCHRONIZATION is enabled only during the fence, so the decoder's
        max-sync clock only ever moves for an active fence."""
        sub = self._subscriber
        if sub is None:
            return None
        try:
            # The subscriber-aware _v2 timestamp is required here: plain
            # cuptiGetTimestamp is CUPTI_ERROR_NOT_COMPATIBLE while the UDR subscriber
            # is active (13.3), which silently turned this fence into a no-op.
            target = self._cupti.get_timestamp(sub)
            torch.cuda.synchronize()
            return target
        except Exception:
            return None

    def convert_time(self, value: int) -> int:
        """Convert a record-clock timestamp to unix-epoch ns. Records ride the approx clock
        while the timestamp callback is engaged, else CLOCK_REALTIME."""
        if self._timestamp_callback_active:
            return self._clock.convert_approx(value)
        return self._clock.convert_unix(value)

    def convert_time_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`convert_time` over a whole record column."""
        if self._timestamp_callback_active:
            return self._clock.convert_approx_array(values)
        return self._clock.convert_unix_array(values)

    def now_unix_ns(self) -> int:
        """Current time on the record clock, converted to unix-epoch ns."""
        return self.convert_time(self._clock.now_record_ns())

    def now_record_ns(self) -> int:
        """Current value of the record clock -- the unconverted timebase of decoded record
        START/END. Use this (not now_unix_ns) to stamp a window boundary compared against raw
        record timestamps. Returns 0 before the session is calibrated."""
        return self._clock.now_record_ns()

    def _try_arm_approx_timestamp_callback(self, sub_handle: int) -> bool:
        """Best-effort: hand CUPTI the profiler's approx-clock timestamp callback so it
        stamps activity records on kineto's exact timebase directly. Opt-in via
        configure(use_approx_timestamps=True) (and only as the sole subscriber); returns False --
        leaving records on the CLOCK_REALTIME pass-through -- when disabled or when CUPTI
        rejects it. Set as a per-subscriber attribute (CUPTI_ACTIVITY_ATTR_TIMESTAMP_CALLBACK),
        which coexists with the user-defined-record path -- unlike the global
        cuptiActivityRegisterTimestampCallback, which returns CUPTI_ERROR_NOT_COMPATIBLE."""
        if not self._timestamp_callback_enabled:
            return False
        # Older libcupti can't re-time device records when a context predates the subscriber, so
        # refuse rather than silently drop them; libcupti >= 130303 re-times regardless.
        if self._cupti.get_version() < 130303 and _has_active_cuda_context():
            logger.warning(
                "CUPTI monitor: use_approx_timestamps requested but a CUDA context already "
                "exists and this libcupti cannot re-time device records; falling back to the "
                "CLOCK_REALTIME pass-through."
            )
            return False
        addr = _cupti_monitor_native.approximate_time_callback_address()
        if self._cupti.arm_approx_timestamp_callback(sub_handle, addr):
            logger.info("CUPTI monitor: approx-clock timestamp callback engaged")
            return True
        logger.warning(
            "CUPTI monitor: timestamp callback rejected; using the cuptiGetTimestamp "
            "(CLOCK_REALTIME) pass-through"
        )
        return False

    # --- observer registry (this monitor is the multiplexer) ---------------

    def register(
        self,
        activities: ActivitiesSpec,
        callback: Callable[..., None],
    ) -> _Observer:
        """Register an observer. ``activities`` is either an iterable of
        ``ActivityKind`` (meaning "all fields") or a field map ``{ActivityKind:
        iterable of field ids | "all"}`` selecting the fields per kind.

        ``callback(columns)`` fires once per drain (at flush time), with ``columns``
        = ``{ActivityKind: {field_id: column}}`` -- the native decoder demuxes every
        buffer to columns and the drain slices them to this observer's selection (the
        observer never sees raw bytes or the decode strategy).

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

    # --- PM sampling (opt-in GPU utilization counters) -----------------------

    def request_pm_sampling(
        self, sink: Callable[[dict[str, Any]], None], metrics: Iterable[str]
    ) -> None:
        """Register ``sink`` as a PM-sampling consumer wanting ``metrics`` on the current device.
        The shared per-device session samples the union of all consumers' metrics; the first
        consumer starts it (see PmSampler.configure() for interval/look-back). Frames arrive on the
        flush thread with ``start_ns`` already converted into the trace clock. No-op if CUDA is
        unavailable, no metrics are given, or ``sink`` is already registered."""
        from torch.profiler._cupti.pm_sampling import PmSampler

        metrics = list(metrics)
        if not torch.cuda.is_available() or not metrics:
            return
        with self._pm_lock:
            if sink in self._pm_consumers:
                return
            sampler = PmSampler()  # per-device singleton for the current device
            try:
                handle = sampler.add_consumer(metrics)
            except Exception as e:
                logger.warning("PM sampling could not register consumer: %s", e)
                return
            self._pm_sampler = sampler
            self._pm_consumers[sink] = handle

    def _deliver_pm(
        self, sink: Callable[[dict[str, Any]], None], frame: dict[str, Any] | None
    ) -> None:
        # Convert the sampler's raw CUPTI-clock timestamps into the trace clock (the observer buckets
        # frames by trace-time start_ns), then push to the observer's sink. No-op on an empty poll.
        if frame is None:
            return
        frame = dict(frame)
        # PM samples are always on the unix (CLOCK_REALTIME) domain (the timestamp callback only
        # restamps activity records onto the approx clock), so convert on the unix domain, not the
        # record path.
        frame["start_ns"] = self._clock.convert_unix_array(frame["start_ns"])
        try:
            sink(frame)
        except Exception:
            logger.exception("PM sampling sink error")

    def release_pm_sampling(self, sink: Callable[[dict[str, Any]], None]) -> None:
        """Unregister a PM-sampling consumer; the session disables once the last one leaves.
        Idempotent. Polls the consumer one last time before removing so its final samples land
        (removal itself delivers nothing)."""
        with self._pm_lock:
            handle = self._pm_consumers.pop(sink, None)
            if handle is not None:
                self._deliver_pm(sink, handle.poll())  # final delivery
                handle.detach()
            if not self._pm_consumers:
                self._pm_sampler = None

    def _poll_pm_sampler(self) -> None:
        """Poll every PM consumer on the monitor's drain cadence (folded into
        _drain_and_dispatch) -- pulling the HW ring before it overflows. The final poll at
        release/stop catches the tail."""
        with self._pm_lock:
            for sink, handle in self._pm_consumers.items():
                self._deliver_pm(sink, handle.poll())

    def _stop_pm_sampler(self) -> None:
        """Poll then unregister the monitor's PM consumers (final delivery + disable when the last
        leaves). Called at monitor stop in case observers have not released yet."""
        with self._pm_lock:
            self._pm_sampler = None
            entries = list(self._pm_consumers.items())
            self._pm_consumers.clear()
            for sink, handle in entries:
                self._deliver_pm(sink, handle.poll())
                handle.detach()

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
        # A fence (flush(sync=True)) transiently enables SYNCHRONIZATION outside the
        # observer union; keep it in the target so a register/deregister mid-fence
        # doesn't strip it -- otherwise the fence never sees its sync record and
        # flush(sync) spins until it times out.
        fence = int(_FENCE_KIND)
        if fence in self._enabled:
            target[fence] = self._enabled[fence]
        if target != self._enabled:
            self._reconfigure(target)
            self._enabled = target
        # queued needs the per-subscriber latency-timestamp attribute (which also gates
        # submitted, not surfaced here). Enable it once, iff an observer selected the
        # QUEUED kernel field -- so the always-on timing path pays no latency overhead.
        if self._subscriber is not None and not self._latency_enabled:
            if Kernel.QUEUED.id in target.get(
                int(ActivityKind.CONCURRENT_KERNEL), frozenset()
            ):
                self._cupti.enable_kernel_latency_timestamps(self._subscriber, True)
                self._latency_enabled = True

    def _reconfigure(self, target: dict[int, frozenset[int]]) -> None:
        # Reconcile the per-field selection to ``target`` with a minimal diff: only
        # touch kinds that are being removed or whose field selection changed. Kinds
        # whose selection is unchanged stay enabled -- toggling them off/on is needless
        # churn and, for RUNTIME/DRIVER, breaks CUPTI's CUDA-graph kernel tracing (a
        # graph captured while those kinds were enabled stops emitting per-node kernel
        # records once they're disabled+re-enabled). Each completed buffer carries
        # CUPTI's own captured layout, so buffers from before a switch still decode.
        sub = self._subscriber
        if sub is None:
            return
        removed = [k for k in self._enabled if k not in target]
        changed = [
            k for k in target if k in self._enabled and self._enabled[k] != target[k]
        ]
        added = [k for k in target if k not in self._enabled]
        for kind in (*removed, *changed):
            self._cupti.activity_disable(sub, kind)
        # Flush between disabling and (re-)enabling a kind with a new field selection
        # so records pending under the old selection aren't lost. NON-forced: we only
        # force-flush while syncing (the fence). Forcing here would push in-progress
        # buffers concurrently with host activity -- the flush race that freezes the
        # decode worker.
        if removed or changed:
            self._cupti.activity_flush_all()
        for kind in (*added, *changed):
            self._cupti.activity_enable(sub, kind, target[kind])

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
        # CUPTI only emits CUDA_EVENT records when SYNCHRONIZATION is also enabled
        # (the two are joined via cudaEventSyncId), so couple it on whenever any
        # observer wants CUDA_EVENT. Enable the fence fields (KIND + END) so a
        # concurrent flush(sync) still finds its decodable sync-point record.
        if ActivityKind.CUDA_EVENT in union:
            union[_FENCE_KIND] = union.get(_FENCE_KIND, frozenset()) | _FENCE_FIELDS
        return union

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "activities": list(self._enabled),
                "buffers_completed": _cupti_monitor_native.decoder_buffers_decoded(),
                "buffers_allocated": _cupti_monitor_native.allocated_buffers()
                if self._started
                else self._final_allocated_buffers,
                "buffers_pending": _cupti_monitor_native.pending_buffers(),
                "valid_total_mb": _cupti_monitor_native.decoder_valid_bytes()
                / (1024 * 1024),
                "dropped_records": self._dropped_records,
                "observers": len(self._observers),
            }

    def _thread_push_stack(self) -> list[tuple[int, bool]]:
        """This thread's live LIFO of active external-correlation frames --
        ``(id, cupti_ok)``, where cupti_ok records whether CUPTI accepted that push --
        so pop unwinds CUPTI + the native mirror only for frames that actually took.
        CUPTI's external-correlation stacks are per-thread, so ours is too."""
        stack = getattr(self._push_tls, "stack", None)
        if stack is None:
            stack = self._push_tls.stack = []
        return stack

    def push_external_correlation_id(self) -> int | None:
        """Allocate a process-unique external-correlation id, record it as this
        thread's new innermost active context, and push it onto CUPTI's external-
        correlation stack. Every CUDA activity recorded until the matching pop gets an
        EXTERNAL_CORRELATION record linking its correlation_id to this id.

        All subsystems share ONE CUPTI kind, so CUPTI tags each kernel with only the
        innermost active id; we snapshot the full active stack here (``_id_chains``)
        so a consumer recovers every context active for an op from that innermost id
        (see :meth:`external_id_chain`) -- no per-subsystem kind, no shadowing, no
        kind-pool ceiling. Returns the id, or None if not started.

        The frame is recorded even when CUPTI rejects the push, so a matching (and
        possibly unconditional) pop stays balanced; the frame's cupti_ok flag tells
        pop not to unwind CUPTI/the mirror for it. A rejected push just leaves CUPTI
        tagging records with the prior id, so this id's chain goes unreferenced and is
        GC'd -- never an off-by-one unwind of the enclosing context."""
        if not self._started or self._subscriber is None:
            return None
        with self._lock:
            external_id = self._next_external_id
            self._next_external_id += 1
        # Pass the subscriber: the plain push returns NOT_COMPATIBLE under the UDR
        # subscriber, so the wrapper uses the subscriber-aware _v2 variant.
        cupti_ok = self._cupti.activity_push_external_correlation_id(
            external_id, sub_handle=self._subscriber
        )
        stack = self._thread_push_stack()
        stack.append((external_id, cupti_ok))
        chain = tuple(fid for fid, _ in stack)
        with self._lock:
            self._id_chains[external_id] = chain
        if cupti_ok:
            # Mirror into the native per-thread stack so the current (innermost) id is
            # readable without a CUPTI peek -- see current_external_correlation_id.
            _cupti_monitor_native.note_external_push(external_id)
        return external_id

    def pop_external_correlation_id(self) -> int | None:
        """Pop this thread's innermost active external-correlation frame (balances a
        push). Unwinds CUPTI + the native mirror only when that push was accepted by
        CUPTI, so a rejected push -- or an over-pop, which no-ops -- never unwinds the
        enclosing frame. Returns the popped id, or None if not started / nothing
        active. The id's chain snapshot is retired by a later drain (see
        :meth:`_gc_external_chains`)."""
        if not self._started or self._subscriber is None:
            return None
        stack = self._thread_push_stack()
        if not stack:
            return None
        external_id, cupti_ok = stack.pop()
        with self._lock:
            self._chains_gc_pending.append(external_id)
        if cupti_ok:
            self._cupti.activity_pop_external_correlation_id(
                sub_handle=self._subscriber
            )
            _cupti_monitor_native.note_external_pop()  # keep the mirror in sync
        return external_id

    def external_id_chain(self, innermost_id: int) -> tuple[int, ...]:
        """The full active-id stack (outermost..innermost) captured when
        ``innermost_id`` was pushed -- every external-correlation context active for
        an op CUPTI tagged with ``innermost_id``. A consumer maps a kernel's
        (innermost) external id through this and picks the id it owns (by membership
        in its own name/metadata map), recovering enclosing contexts the single-kind
        records don't carry. Resolve at parse/dispatch time: the snapshot is dropped a
        couple of drains after the id is popped. Falls back to ``(innermost_id,)``
        when there is no snapshot (already dropped, or an id we didn't push)."""
        with self._lock:
            return self._id_chains.get(innermost_id, (innermost_id,))

    def _gc_external_chains(self) -> None:
        """Advance the popped-chain GC one generation: drop chains popped two drains
        ago (their records are delivered + dispatched by now) and promote this cycle's
        popped ids to be dropped next. Called once per drain, so a popped id's chain
        survives the drains that dispatch its trailing records."""
        with self._lock:
            if not (self._chains_gc_ready or self._chains_gc_pending):
                return
            for retired in self._chains_gc_ready:
                self._id_chains.pop(retired, None)
            self._chains_gc_ready = self._chains_gc_pending
            self._chains_gc_pending = []

    def current_external_correlation_id(self) -> int | None:
        """The external-correlation id on top of THIS thread's stack (last pushed,
        not yet popped), or None. Reads the native host-side mirror of CUPTI's
        stack (CUPTI exposes push/pop but no peek). Lets a consumer on the same
        thread -- e.g. an in-process NCCL profiler plugin -- associate metadata with
        the annotation the caller already pushed, instead of pushing its own id."""
        cur = _cupti_monitor_native.current_external_id()
        return cur if cur else None

    def take_external_metadata(self) -> dict[int, str]:
        """Move out the {external_id: metadata blob} accumulated from drains since
        the last call. An observer's external-correlation join consumes this to
        attach the blob onto its kernel events (keyed by the same external id a
        producer pushed). Drained-and-reset so blobs aren't re-attached."""
        with self._lock:
            meta = self._external_metadata
            self._external_metadata = {}
            return meta

    def add_collective_metadata(self, **fields: Any) -> None:
        """Merge extra metadata into the CURRENT collective's entry (the most-recently-
        pushed external-correlation id on this thread), recursively (nested dicts
        combine; on a leaf conflict the later value wins). The seam for a backend to
        contribute schema fields the NCCL profiler plugin doesn't emit (e.g.
        ``process_group``, ``process_group_ranks``, ``input_sizes``) so the serializer
        plugins (FlightRecorder/clog) can fill them; the fields ride the same per-
        collective metadata as the plugin's descriptor.

        Call inside the comms wrapper's push/pop window (or after
        :meth:`push_external_correlation_id`). To attach metadata to a specific
        collective from outside its window, use the native
        ``metadata_put_external(blob, external_id)`` directly."""
        if fields:
            _cupti_monitor_native.metadata_put_external(json.dumps(fields))

    def session_info(self) -> dict[str, Any]:
        """Monitor/session metadata for consumers that need to describe the
        capture: versions, clock calibration, and buffer config. Call after
        start() so the clock fields are populated."""
        return {
            "cupti_version": self._cupti.get_version(),
            "cuda_version": _cuda_version_string(),
            "hes_enabled": is_hes_enabled(),
            "timestamp_mode": "approximate_clock",
            "session_start_unix_ns": self._clock.unix_anchor_ns,
            "session_start_approx_ns": self._clock.approx_anchor_ns,
            "buffer_size": self.buffer_size,
            "flush_period_ns": int(self.background_flush_period_s * 1e9),
            "drain_period_ns": int(self.background_drain_period_s * 1e9),
            "libcupti": cupti_python.LIBCUPTI_SONAME,
        }

    def _drain_and_dispatch_loop(self) -> None:
        # cuptiActivityFlushAll is driven off-thread by the native flusher; this loop
        # only drains the decoded columns + dispatches them (GIL work) -- the same work
        # flush(sync=False) does after its flush, minus the flush itself.
        # Floor the wait so a period of 0 ("continuously") doesn't busy-spin the GIL
        # (Event.wait(0) returns immediately); 1ms is well below any useful cadence.
        period = max(self.background_drain_period_s, 0.001)
        try:
            while not self._drain_stop.wait(period):
                if self._started:
                    self._account_dropped_records(0, 0)
                    self._drain_and_dispatch()
        except BaseException:
            logger.exception("CUPTI monitor drain thread died")

    def _drain_and_dispatch(self) -> None:
        """Drain the column groups the native decoder accumulated and fan them out
        to the observers. The native worker does the per-buffer decode GIL-free;
        this only views the drained bytes as their dtype and dispatches, so it is
        cheap and runs on whichever thread drives flush() (foreground or the flush
        loop).

        Native returns a LIST of ``(kind, {field_id: (size, bytes)})`` groups --
        one per distinct record layout, so within a group every field column is the
        same length. Groups are packed into frames (each frame holds at most one
        group per kind) so a dispatched ``{kind: cols}`` chunk always has
        length-consistent columns; at steady state every kind has a single layout,
        so this collapses to one frame -- the same multi-kind chunk as before."""
        with self._drain_lock:
            groups, ext_meta = _cupti_monitor_native.drain_decoded()
            if ext_meta:
                with self._lock:
                    self._external_metadata.update(ext_meta)
            if groups:
                frames: list[dict[int, dict[int, Any]]] = []
                for kind, fields in groups:
                    cols = self._columns_from_native(kind, fields)
                    if not cols:
                        continue
                    frame = next((f for f in frames if kind not in f), None)
                    if frame is None:
                        frame = {}
                        frames.append(frame)
                    frame[kind] = cols
                with self._lock:
                    observers = list(self._observers)
                for frame in frames:
                    self._dispatch_observers(frame, observers)
            # GC popped chains AFTER dispatch: a popped id's chain must survive the
            # drains that dispatch its trailing records (resolution reads it during
            # dispatch), so retire it a generation later, never before.
            self._gc_external_chains()
        # PM sampling shares the drain cadence: polling the HW ring is GIL work, so it
        # rides the drain path (foreground flush or the background drain loop) -- the
        # native self-flush thread must not touch it.
        self._poll_pm_sampler()
        self._maybe_warn_backpressure()

    def _columns_from_native(
        self, kind: int, fields: Mapping[int, tuple[int, bytes]]
    ) -> dict[int, Any]:
        """Turn one native group's ``{field_id: (field_size, bytes)}`` into
        ``{field_id: column}``: numeric fields are viewed per their :class:`Ctype`
        (unsigned/signed/float) at the captured width; const char* (string) fields are
        dereferenced to str."""
        str_fields = STRING_FIELDS.get(kind, frozenset())
        ctype_by_fid = FIELD_CTYPE.get(kind, {})
        cols: dict[int, Any] = {}
        for fid, (size, raw) in fields.items():
            if fid in str_fields and size == 8:
                ptrs = np.frombuffer(raw, dtype="<u8")
                cols[fid] = np.array([_deref_cstr(int(p)) for p in ptrs], dtype=object)
            elif size in (1, 2, 4, 8):
                # The width is the captured layout's; Ctype only picks the interpretation.
                # Fall back to unsigned for the cases numpy can't express -- a <f1 (there is
                # no 1-byte numpy float) or a CSTR that isn't an 8-byte pointer -- so a
                # mis-typed field can't crash the drain.
                ctype = ctype_by_fid.get(fid, Ctype.UINT)
                bad_float = ctype is Ctype.FLOAT and size not in (2, 4, 8)
                if ctype is Ctype.CSTR or bad_float:
                    ctype = Ctype.UINT
                # .copy() so the column is writable and owns its memory (the
                # frombuffer view is read-only over the transient bytes).
                cols[fid] = np.frombuffer(raw, dtype=ctype.numpy(size)).copy()
        return cols

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
                # A buffer missing any field this observer requested was recorded
                # before the observer's selection was enabled on the subscriber, so
                # skip the kind rather than hand over a partial chunk. A correctly
                # synced reconfigure makes this rare, but an in-flight buffer can
                # still straddle the field-selection change.
                if any(f not in kind_cols for f in fields):
                    continue
                chunk[kind] = {f: kind_cols[f] for f in fields}
            if chunk:
                obs.callback(chunk)

    def _account_dropped_records(self, ctx: int, stream_id: int) -> None:
        self._dropped_records += self._cupti.activity_get_num_dropped_records(
            ctx, stream_id
        )


_hes_enabled = False


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


def configure(
    *,
    buffer_size: int | None = None,
    background_flush_period_s: float | None = None,
    background_drain_period_s: float | None = None,
    use_approx_timestamps: bool | None = None,
) -> None:
    """Set the process-wide CUPTI monitor settings the singleton snapshots when first
    constructed (via CuptiMonitor()). First-come-first-serve (like
    PmSampler.configure): locked once this lands OR the singleton is built, so a later call
    is ignored with a warning -- pass all settings in one call, before first use. An unset
    (None) arg keeps its current value. The two cadences default to -1 (caller-driven; a
    non-negative value opts into background flush/drain at that period)."""
    with CuptiMonitor._instance_lock:
        if CuptiMonitor._configured:
            logger.warning(
                "cupti_monitor.configure() ignored: already configured "
                "(first-come-first-serve). Call it once before the first CuptiMonitor()."
            )
            return
        if buffer_size is not None:
            CuptiMonitor._buffer_size = buffer_size
        if background_flush_period_s is not None:
            CuptiMonitor._background_flush_period_s = background_flush_period_s
        if background_drain_period_s is not None:
            CuptiMonitor._background_drain_period_s = background_drain_period_s
        if use_approx_timestamps is not None:
            CuptiMonitor._use_approx_timestamps = use_approx_timestamps
        CuptiMonitor._configured = True


def get_config() -> dict[str, Any]:
    """The process-wide config the singleton will snapshot (or snapshotted): buffer_size,
    the two cadences, the approx-clock flag, and whether configure()/construction has pinned
    it (first-come-first-serve)."""
    return {
        "buffer_size": CuptiMonitor._buffer_size,
        "background_flush_period_s": CuptiMonitor._background_flush_period_s,
        "background_drain_period_s": CuptiMonitor._background_drain_period_s,
        "use_approx_timestamps": CuptiMonitor._use_approx_timestamps,
        "configured": CuptiMonitor._configured,
    }


def _reset_for_test() -> None:
    """Test-only: drop the singleton and reset the config to defaults so a test can build a
    freshly-configured monitor. Callers must have torn down any live session first."""
    with CuptiMonitor._instance_lock:
        CuptiMonitor._instance = None
        CuptiMonitor._buffer_size = 4 * 1024 * 1024
        CuptiMonitor._background_flush_period_s = -1.0
        CuptiMonitor._background_drain_period_s = -1.0
        CuptiMonitor._use_approx_timestamps = False
        CuptiMonitor._configured = False
