# mypy: allow-untyped-defs
"""CUPTI PM-sampling: continuous GPU performance-monitor sampling (SM-active %, DRAM-throughput
%) that runs concurrently with the activity monitor.

PM sampling reads dedicated on-chip performance-monitor units, so it has negligible GPU-side cost
and coexists with the activity subscriber, but it locks the GPU clocks while active (which can
shift absolute kernel durations) -- so it is opt-in via custom_profiler_config
{"enable_pm_sampling": true}, not always-on like the environment counters.

The HW units sample autonomously into a device-side ring (KEEP_LATEST) between start and decode, so
the engine spawns no thread of its own -- collection is passive and the caller drives
:meth:`~PmSampler.poll` on its own cadence (the cupti_monitor backend polls from its flush thread).
The first consumer opens the session; poll() drains the ring and a final tail-drain runs when the
last consumer leaves -- crucially *before* disabling, since a wrapped ring is only decodable while
sampling is active. A window that
fits the ring yields all its samples; one that exceeds it keeps the most recent (the trace's tail)
rather than erroring. ``decode`` drains the whole ring in one call and caps at its ``max_samples``
(a second call does not resume), so the host image is sized above the ring's sample capacity.
Samples are HW-timestamped in the CUPTI clock domain (raw); converting them to trace time is the
consumer's job (the sampler has no clock dependency), and doing so against the base the monitor's
record timestamps use aligns them with the activity records; they surface as GPU counter tracks
(siblings of the environment counters).

Each CUDA device has at most one :class:`PmSampler` -- a per-device singleton (``PmSampler(device)``
returns the one instance) -- so no device is ever driven by two collectors. Lifecycle follows
NVIDIA's cupti-python PM sampling sample: ``enable`` -> ``configure`` -> ``start`` -> drain ->
``stop`` -> release. ``enable`` initializes the *process-global* CUPTI profiler (idempotent), but
its deinitialize is NOT refcounted by CUPTI -- a second call segfaults -- so PmSampler refcounts its
live collectors and deinitializes exactly once, when the last is released. That makes sampling
several devices at once safe: one device's teardown never deinitializes a profiler another device is
still using."""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing_extensions import Self


logger = logging.getLogger(__name__)

# Interval + look-back are configured process-wide via env read at import (set them before importing
# torch); metrics are NOT -- each consumer brings its own via add_consumer(). The HW sampling
# interval (GPU_TIME_INTERVAL units = ns); TORCH_CUPTI_PM_SAMPLING_INTERVAL_MS, default 1 ms.
_SAMPLING_INTERVAL_NS = int(
    float(os.environ.get("TORCH_CUPTI_PM_SAMPLING_INTERVAL_MS", 1)) * 1_000_000
)
# Retained look-back window: the sampler is sized (max_samples + ring) to cover this much wall-clock
# at the interval. Kept modest because the host counter-data image is ~18 KiB/sample (see
# _counter_data_size), so a decode of window/interval samples allocates that many.
# TORCH_CUPTI_PM_SAMPLING_LOOKBACK_MS, default 10 s.
_DEFAULT_WINDOW_NS = int(
    float(os.environ.get("TORCH_CUPTI_PM_SAMPLING_LOOKBACK_MS", 10_000)) * 1_000_000
)
# Ceiling on max_samples (decode image + ring) so an extreme look-back/interval can't allocate an
# unbounded image; cap it and warn that the retained window is truncated. ~16k samples ~= 300 MB
# for 4 metrics.
_MAX_SAMPLES_CAP = 16384


def is_available() -> bool:
    try:
        import cupti.pm_sampling  # noqa: F401  # pyrefly: ignore[missing-import]
    except Exception:
        return False
    return True


def _counter_data_size(
    pm_sampling_object: int, metrics: list[str], max_samples: int
) -> int:
    """Exact byte size of the counter-data image that holds ``max_samples`` samples for ``metrics``
    (CUPTI's ``pm_sampling_get_counter_data_size``). Used to size the HW ring to the target window
    -- no per-sample byte estimate -- and it's the same size ``decode`` allocates for its image."""
    from cupti import cupti as c  # pyrefly: ignore[missing-import]
    from cupti.pm_sampling import metrics_to_c_array  # pyrefly: ignore[missing-import]

    _, metric_names_ptr = metrics_to_c_array(metrics)
    p = c.PmSampling_GetCounterDataSize_Params()
    p.struct_size = c.PM_SAMPLING_GET_COUNTER_DATA_SIZE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    p.p_metric_names = metric_names_ptr
    p.num_metrics = len(metrics)
    p.max_samples = max_samples
    c.pm_sampling_get_counter_data_size(p.ptr)
    return p.counter_data_size


@functools.cache
def _device_chip_name(device: int) -> str:
    """The CUPTI chip name (e.g. 'GB100') for a CUDA device index -- the key the profiler-host
    metric queries use. Cached: constant per device for the life of the process (queried on every
    add_consumer via the single-pass check)."""
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.Device_GetChipName_Params()
    p.struct_size = c.DEVICE_GET_CHIP_NAME_PARAMS_STRUCT_SIZE
    p.device_index = device
    c.device_get_chip_name(p.ptr)
    return p.p_chip_name


def _pm_sampling_disable(pm_sampling_object: int) -> None:
    """Free a device's PM sampling object -- the per-device half of ``Collector.disable`` -- WITHOUT
    the process-global ``profiler_deinitialize`` it bundles in, so PmSampler can refcount that deinit
    across devices (see ``PmSampler._active``)."""
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.PmSampling_Disable_Params()
    p.struct_size = c.PM_SAMPLING_DISABLE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    c.pm_sampling_disable(p.ptr)


def _profiler_deinitialize() -> None:
    """Deinitialize the process-global CUPTI profiler. CUPTI does NOT refcount this (a second call
    segfaults), so PmSampler calls it exactly once -- when its last live collector is torn down."""
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.Profiler_DeInitialize_Params()
    p.struct_size = c.PROFILER_DEINITIALIZE_PARAMS_STRUCT_SIZE
    c.profiler_deinitialize(p.ptr)


@functools.cache
def supported_metrics(*, with_sub_metrics: bool = False) -> frozenset[str]:
    """The PM-counter metric names CUPTI reports for the current CUDA device's chip -- the menu to
    pick ``add_consumer(metrics=...)`` metrics from. The chip name is resolved from the device via
    CUPTI (:func:`_device_chip_name`). ``with_sub_metrics`` expands each base metric to its
    rollup/suffix forms (e.g. ``sm__cycles_active.avg.pct_of_peak_sustained_elapsed``, the
    fully-qualified names ``configure`` accepts); otherwise the base names. Memoized (the
    profiler-host query is expensive and the chip is constant for a process); returns a frozenset,
    empty if PM sampling or the profiler host is unavailable. NOTE: not every returned metric is
    single-pass and a chosen set must all fit in one PM-sampling pass -- CUPTI validates that when a
    consumer is added."""
    if not is_available():
        return frozenset()
    try:
        from cupti import profiler_host as ph  # pyrefly: ignore[missing-import]
        from cupti.cupti import (  # pyrefly: ignore[missing-import]
            MetricType,
            ProfilerType,
        )
    except Exception:
        return frozenset()
    try:
        import torch

        chip_name = _device_chip_name(torch.cuda.current_device())
    except Exception as e:
        logger.warning("PM sampling could not resolve the chip name: %s", e)
        return frozenset()
    # PM_SAMPLING is the relevant type; fall back to RANGE_PROFILER (same base-metric DB) if its
    # host cannot initialize without a single-pass set name on this CUPTI version.
    for profiler_type in (ProfilerType.PM_SAMPLING, ProfilerType.RANGE_PROFILER):
        try:
            host = ph.ProfilerHost(chip_name, profiler_type)
            host.initialize()
        except Exception:
            continue
        try:
            names: set[str] = set()
            for mt in (MetricType.COUNTER, MetricType.RATIO, MetricType.THROUGHPUT):
                for base in host.get_base_metrics(mt):
                    names.update(
                        host.get_sub_metrics(base, mt) if with_sub_metrics else (base,)
                    )
            return frozenset(names)
        except Exception as e:
            logger.warning("PM sampling metric enumeration failed: %s", e)
            return frozenset()
        finally:
            host.deinitialize()
    logger.warning(
        "PM sampling profiler host could not initialize for chip %s", chip_name
    )
    return frozenset()


class _Consumer:
    """A registered PM-sampling consumer: the metric names it wants and the sink its frames go to.
    Returned by :meth:`PmSampler.add_consumer` as the opaque handle for
    :meth:`PmSampler.remove_consumer`."""

    __slots__ = ("metrics", "sink")

    def __init__(
        self, metrics: list[str], sink: Callable[[dict[str, Any]], None]
    ) -> None:
        self.metrics = metrics
        self.sink = sink


class PmSampler:
    """The single PM-sampling session on one CUDA device -- a per-device singleton
    (``PmSampler(device)`` returns the one instance, default the current device). Only one PM session
    per device is possible, so all users of a device share this one object. The process-global CUPTI
    profiler deinit is not refcounted by CUPTI, so PmSampler refcounts its live collectors
    (:attr:`_active`) and deinitializes once, when the last is torn down -- keeping concurrent
    sampling on multiple devices safe.

    Consumers register with :meth:`add_consumer` (the metrics they want + a sink) and unregister
    with :meth:`remove_consumer`. The session samples the *union* of all consumers' metrics and
    hands each consumer a frame sliced to just its own metrics. Frames carry RAW CUPTI-clock-ns
    timestamps in ``start_ns`` -- converting to trace/epoch time is the consumer's job, so the
    sampler has no clock dependency.

    The HW units sample autonomously into a device-side ring (KEEP_LATEST), so there is no
    background thread: the first consumer starts the session, a caller drains the ring with
    :meth:`poll` on its own cadence, and removing the last consumer drains the tail and disables.
    Add/remove reconfigure the live session to the new metric union; a union that cannot be
    collected in one PM-sampling pass is rejected (:meth:`add_consumer` raises), leaving the running
    session untouched."""

    # One sampler per CUDA device index; PmSampler(device) returns the per-device singleton (default
    # the current device), so PmSampler(0) is PmSampler(0). Construction routes through __new__.
    _instances: dict[int, PmSampler] = {}
    _instances_lock = threading.Lock()
    # Count of live collectors across the process, guarded by _instances_lock. The CUPTI profiler's
    # deinit is process-global and NOT refcounted by CUPTI (a second deinit segfaults), so we refcount
    # it: initialize is idempotent (each enable() may call it), but deinitialize runs exactly once,
    # when this drops to 0 -- which lets samplers on different devices coexist safely.
    _active = 0

    def __new__(cls, device: int | None = None) -> Self:
        import torch

        if device is None:
            device = torch.cuda.current_device()
        with cls._instances_lock:
            inst = cls._instances.get(device)
            if inst is None:
                inst = super().__new__(cls)
                inst._init(device)
                cls._instances[device] = inst
            return inst

    def _init(self, device: int) -> None:
        # One-time setup, run by __new__ the first time a device is seen. Deliberately not __init__:
        # __init__ would re-run on every PmSampler(device) call (Python calls it on the returned
        # singleton) and reset the live session.
        self._device = device
        self._sampling_interval_ns = _SAMPLING_INTERVAL_NS
        requested = max(1, _DEFAULT_WINDOW_NS // _SAMPLING_INTERVAL_NS)
        self._max_samples = min(requested, _MAX_SAMPLES_CAP)
        if requested > _MAX_SAMPLES_CAP:
            logger.warning(
                "PM sampling look-back (%d ns @ %d ns = %d samples) exceeds the %d-sample cap; "
                "retained window truncated to ~%d ns.",
                _DEFAULT_WINDOW_NS,
                _SAMPLING_INTERVAL_NS,
                requested,
                _MAX_SAMPLES_CAP,
                _MAX_SAMPLES_CAP * _SAMPLING_INTERVAL_NS,
            )
        # Registered consumers and the metric union currently sampled (empty until the first add).
        self._consumers: list[_Consumer] = []
        self._metric_names: list[str] = []
        self._col: Any = None
        # Guards the consumer list + collector ops (add/remove/poll may race across threads); an
        # RLock since remove_consumer drains (which the lock also guards) before reconfiguring.
        self._lock = threading.RLock()

    def add_consumer(
        self, metrics: Iterable[str], sink: Callable[[dict[str, Any]], None]
    ) -> _Consumer:
        """Register a consumer wanting ``metrics``, delivered to ``sink`` as raw-timestamp frames;
        returns a handle for :meth:`remove_consumer`. Starts the session (first consumer) or
        reconfigures it to the new metric union. Raises ValueError if ``metrics`` is empty or the
        resulting union needs more than one PM-sampling pass (the running session is left intact)."""
        metrics = list(metrics)
        if not metrics:
            raise ValueError("PM sampling requires a non-empty metric set")
        consumer = _Consumer(metrics, sink)
        with self._lock:
            # Validate the candidate union BEFORE committing, so a multi-pass add fails cleanly
            # without disturbing the running session.
            self._check_single_pass(self._union_of([*self._consumers, consumer]))
            # Flush ring samples (collected under the old metric union) to the EXISTING consumers
            # before the union may change; done pre-append so the drain's column set matches the old
            # union (the new consumer's metrics aren't sampled yet).
            self._drain()
            self._consumers.append(consumer)
            self._reconfigure()
        return consumer

    def remove_consumer(self, consumer: _Consumer) -> None:
        """Unregister a consumer. Drains pending samples to the current consumers (including the one
        leaving) first, then reconfigures to the smaller union -- or tears the session down when the
        last consumer leaves. Idempotent."""
        with self._lock:
            if consumer not in self._consumers:
                return
            self._drain()  # deliver in-flight samples to the leaver before it goes
            self._consumers.remove(consumer)
            self._reconfigure()

    @staticmethod
    def _union_of(consumers: list[_Consumer]) -> list[str]:
        seen: set[str] = set()
        union: list[str] = []
        for c in consumers:
            for m in c.metrics:
                if m not in seen:
                    seen.add(m)
                    union.append(m)
        return union

    def _reconfigure(self) -> None:
        # Bring the live session in line with the consumers' metric union (caller holds _lock): tear
        # it down when no consumers remain, else (re)build it whenever the union changed. A changed
        # union rebuilds with a clean enable/configure/start rather than an in-place
        # stop/configure/start -- decode after an in-place reconfigure errors on some CUPTI versions,
        # and union changes are rare (an observer joining/leaving). Callers drain the old ring first,
        # so no samples are lost across the rebuild.
        desired = self._union_of(self._consumers)
        if not desired:
            self._teardown()
        elif desired != self._metric_names:
            self._teardown()
            self._metric_names = desired
            self._start()

    @property
    def suggested_poll_interval_ns(self) -> int:
        """Recommended cadence for periodic :meth:`poll`. The KEEP_LATEST ring holds max_samples
        (~one look-back window) and each decode re-reads the whole ring, so polling much more often
        just re-decodes the same samples, while polling slower than the ring fills drops the samples
        between polls. Drain a little before the ring is full -- one window span minus a small buffer
        -- so each decode is productive without hitting the overflow cap; the final drain at teardown
        still catches the tail."""
        span = self._max_samples * self._sampling_interval_ns
        return max(self._sampling_interval_ns, span * 9 // 10)

    def poll(self) -> None:
        """Drain the ring into the consumers' sinks without disabling -- continuous flight-recorder
        decode; a caller polls on its own cadence (see :attr:`suggested_poll_interval_ns`). Safe to
        call repeatedly while sampling (the KEEP_LATEST ring is decodable while active). No-op until
        a consumer starts the session."""
        with self._lock:
            if self._col is None:
                return
            try:
                self._drain()
            except Exception:
                logger.exception("PM sampling poll decode error")

    def _start(self) -> None:
        # Enable + configure + start the collector for self._metric_names (the union). enable() also
        # initializes the process-global CUPTI profiler (idempotent); _teardown() releases this
        # collector's profiler ref. A failure after enable tears the partial session down so a later
        # retry is clean. Caller holds _lock.
        if self._col is not None or not is_available() or not self._metric_names:
            return
        from cupti import pm_sampling as pm  # pyrefly: ignore[missing-import]

        self._warn_unsupported()
        try:
            col = pm.Collector(device_index=self._device)
            col.enable()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            return
        # enable() succeeded -> this collector holds a process-global profiler ref; count it before
        # anything below can fail, so _teardown() balances the refcount.
        self._col = col
        with type(self)._instances_lock:
            type(self)._active += 1
        try:
            # Size the ring to hold the window's samples: the counter-data-image bytes for
            # max_samples (exact, via CUPTI) is >= the device records for the same count, so the ring
            # never wraps before the window is full.
            col.configure(
                metrics=self._metric_names,
                hardware_buffer_size=_counter_data_size(
                    col._pm_sampling_object, self._metric_names, self._max_samples
                ),
                sampling_interval=self._sampling_interval_ns,
                trigger_mode=pm.TriggerMode.GPU_TIME_INTERVAL,
                # KEEP_LATEST = ring buffer: an over-capacity window keeps the most recent samples
                # (the trace's tail) instead of erroring. Requires decoding before disabling.
                hw_buffer_append_mode=pm.HardwareBuffer_AppendMode.KEEP_LATEST,
            )
            col.start()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            self._teardown()

    def _teardown(self) -> None:
        # Release this device's collector: stop sampling, free its PM object, and drop our
        # process-global profiler ref -- deinitializing the profiler only when the last collector in
        # the process goes (Collector.disable() would deinit unconditionally, which is unsafe once a
        # second device is sampling). Detach the collector's GC finalizer so it can't later re-run
        # that bundled disable+deinit. A wrapped KEEP_LATEST ring must already be drained (callers
        # drain first). Caller holds _lock.
        col, self._col = self._col, None
        self._metric_names = []
        if col is None:
            return
        try:
            col.stop()
        except Exception:
            logger.exception("PM sampling stop error")
        try:
            col._finalizer.detach()
            _pm_sampling_disable(col._pm_sampling_object)
        except Exception:
            logger.exception("PM sampling disable error")
        with type(self)._instances_lock:
            type(self)._active -= 1
            last = type(self)._active == 0
        if last:
            try:
                _profiler_deinitialize()
            except Exception:
                logger.exception("PM sampling profiler deinit error")

    def _warn_unsupported(self) -> None:
        # Warn (don't fail) for any sampled metric the chip does not report; enable/configure is the
        # real gate. supported_metrics() may be empty (host query unavailable) -> skip. Compared by
        # base name (before the first '.') so a rollup/suffix doesn't cause a false warning.
        try:
            supported = supported_metrics()
        except Exception:
            return
        if not supported:
            return
        known = {s.split(".", 1)[0] for s in supported}
        unknown = [m for m in self._metric_names if m.split(".", 1)[0] not in known]
        if unknown:
            logger.warning(
                "PM sampling: metric(s) not reported by this chip, may fail to enable: %s",
                ", ".join(unknown),
            )

    def _check_single_pass(self, metrics: list[str]) -> None:
        """Raise ValueError if ``metrics`` can't be collected in one PM-sampling pass (PM sampling is
        single-pass only). Uses the profiler host to build the config image and count passes.
        Best-effort: if the host can't compute passes here, return and let configure() reject a
        multi-pass set (with a generic error)."""
        try:
            from cupti import profiler_host as ph  # pyrefly: ignore[missing-import]
            from cupti.cupti import ProfilerType  # pyrefly: ignore[missing-import]

            host = ph.ProfilerHost(
                _device_chip_name(self._device), ProfilerType.PM_SAMPLING
            )
            host.initialize()
            try:
                passes = ph.get_num_of_passes(host.create_config_image(metrics=metrics))
            finally:
                host.deinitialize()
        except Exception:
            return  # can't determine passes here; configure() still enforces single-pass
        if passes > 1:
            raise ValueError(
                f"PM sampling requires all metrics in a single pass, but the requested set needs "
                f"{passes} passes: {metrics}. Reduce it to a single-pass set (see "
                f"supported_metrics())."
            )

    def _drain(self) -> None:
        # A single decode drains the whole ring (see module docstring). With KEEP_LATEST a wrapped
        # ring just yields its most-recent samples, so no overflow error; the MemoryError guard is a
        # defensive backstop only. Caller holds _lock.
        col = self._col
        if col is None:
            return
        try:
            cd = col.decode(max_samples=self._max_samples)
        except MemoryError:
            logger.warning(
                "PM sampling HW buffer overflow during decode; samples dropped."
            )
            return
        except Exception:
            # A transient decode failure (e.g. right after a rebuild, before samples exist) must not
            # propagate through poll()/remove_consumer -- skip this drain, the next one recovers.
            logger.exception("PM sampling decode error")
            return
        n = cd.num_completed_samples
        if not n:
            return
        if n >= self._max_samples:
            # The image filled before the buffer drained; the newest samples past the cap were
            # dropped. Sized not to happen for a window that fit the buffer, so this is a backstop.
            logger.warning(
                "PM sampling decoded the maximum %d samples; some were dropped.",
                self._max_samples,
            )
        # One pass over the decoded ring: CounterData iterates exactly num_completed_samples and
        # evaluates each sample's metrics host-side on access (the decode cost). Stamp at the
        # sample's interval START: the value is the average over [start, end] (~1 ms), and the viewer
        # draws a counter as a step from each sample's ts to the next -- so start makes the step span
        # exactly its measurement window, lining the high-counter region up with the activity span
        # (end would lag a full interval; midpoint would offset the edges).
        ts = np.empty(n, dtype=np.int64)
        vals = np.empty((n, len(self._metric_names)), dtype=np.float64)
        for i, s in enumerate(cd):
            ts[i] = s.start_timestamp
            vals[i] = s.metric_values
        # Drop samples with a bogus start timestamp: the first sample of a session has an unset
        # interval-start -- 0, or a stale small value from a different clock domain (observed
        # ~7.5e13 vs the real ~1.78e18). Real samples all fall within the look-back of the newest,
        # so keep only ts in (max - look-back, max]; this also drops any stale pre-wrap remnant.
        keep = ts > 0
        if keep.any():
            keep &= ts >= int(ts[keep].max()) - _DEFAULT_WINDOW_NS
        if not keep.any():
            return
        ts = ts[keep]
        vals = vals[keep]
        device_col = np.full(len(ts), self._device, dtype=np.int64)
        col_index = {m: j for j, m in enumerate(self._metric_names)}
        # Frames share the ts/value arrays across consumers (read-only): a consumer that transforms
        # a column (e.g. converts start_ns) must produce a new array, not mutate in place.
        for consumer in self._consumers:
            frame: dict[str, Any] = {"start_ns": ts, "device_id": device_col}
            for m in consumer.metrics:
                frame[m] = vals[:, col_index[m]]
            consumer.sink(frame)
