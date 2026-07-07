# mypy: allow-untyped-defs
"""CUPTI PM-sampling: continuous GPU performance-monitor sampling (SM-active %, DRAM-throughput
%) that runs concurrently with the activity monitor.

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
import threading
from typing import Any, TYPE_CHECKING

import numpy as np
from cupti import (  # pyrefly: ignore[missing-import]
    cupti as _cupti,
    pm_sampling as _pm_sampling,
    profiler_host as _profiler_host,
)
from cupti.pm_sampling import (  # pyrefly: ignore[missing-import]
    metrics_to_c_array as _metrics_to_c_array,
)

import torch


if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing_extensions import Self


logger = logging.getLogger(__name__)

# Interval + look-back defaults; override process-wide (first-come-first-serve) via
# PmSampler.configure() -- there is no env var. Metrics are NOT process-wide -- each consumer
# brings its own via add_consumer(). The HW sampling interval (GPU_TIME_INTERVAL units = ns),
# default 1 ms.
_DEFAULT_SAMPLING_INTERVAL_NS = 1_000_000
# Retained look-back window: the sampler is sized (max_samples + ring) to cover this much wall-clock
# at the interval. Kept modest because the host counter-data image is ~18 KiB/sample (see
# _counter_data_size), so a decode of window/interval samples allocates that many. Default 10 s.
_DEFAULT_LOOKBACK_WINDOW_NS = 10_000 * 1_000_000


def _counter_data_size(
    pm_sampling_object: int, metrics: list[str], max_samples: int
) -> int:
    _, metric_names_ptr = _metrics_to_c_array(metrics)
    p = _cupti.PmSampling_GetCounterDataSize_Params()
    p.struct_size = _cupti.PM_SAMPLING_GET_COUNTER_DATA_SIZE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    p.p_metric_names = metric_names_ptr
    p.num_metrics = len(metrics)
    p.max_samples = max_samples
    _cupti.pm_sampling_get_counter_data_size(p.ptr)
    return p.counter_data_size


@functools.cache
def _device_chip_name(device: int) -> str:
    p = _cupti.Device_GetChipName_Params()
    p.struct_size = _cupti.DEVICE_GET_CHIP_NAME_PARAMS_STRUCT_SIZE
    p.device_index = device
    _cupti.device_get_chip_name(p.ptr)
    return p.p_chip_name


def _pm_sampling_disable(pm_sampling_object: int) -> None:
    p = _cupti.PmSampling_Disable_Params()
    p.struct_size = _cupti.PM_SAMPLING_DISABLE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    _cupti.pm_sampling_disable(p.ptr)


def _profiler_deinitialize() -> None:
    p = _cupti.Profiler_DeInitialize_Params()
    p.struct_size = _cupti.PROFILER_DEINITIALIZE_PARAMS_STRUCT_SIZE
    _cupti.profiler_deinitialize(p.ptr)


@functools.cache
def supported_metrics(*, with_sub_metrics: bool = False) -> frozenset[str]:
    try:
        chip_name = _device_chip_name(torch.cuda.current_device())
        host = _profiler_host.ProfilerHost(chip_name, _cupti.ProfilerType.PM_SAMPLING)
        host.initialize()
    except Exception as e:
        logger.warning("PM sampling could not initialize the profiler host: %s", e)
        return frozenset()
    try:
        names: set[str] = set()
        for mt in (
            _cupti.MetricType.COUNTER,
            _cupti.MetricType.RATIO,
            _cupti.MetricType.THROUGHPUT,
        ):
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


class _Consumer:
    """Internal per-consumer state held by the sampler: the metrics it wants and its cursor (the last
    start_ns returned to it). The caller gets a :class:`_Handle`, not this."""

    __slots__ = ("metrics", "cursor")

    def __init__(self, metrics: list[str]) -> None:
        self.metrics = metrics
        self.cursor = -1  # last start_ns returned; -1 = nothing yet


class _Handle:
    """The caller's handle to a registered consumer: :meth:`poll` for its samples, :meth:`detach` to
    unregister. Dropping the handle also detaches (a GC safety net), but prefer :meth:`detach` for
    deterministic teardown -- detaching the last consumer releases the GPU-clock lock and the CUPTI
    session. The sampler references only the :class:`_Consumer` record, not the handle, so a dropped
    handle is collectable and its :meth:`__del__` fires."""

    __slots__ = ("_sampler", "_consumer")

    def __init__(self, sampler: PmSampler, consumer: _Consumer) -> None:
        self._sampler = sampler
        self._consumer = consumer

    def poll(self) -> dict[str, Any] | None:
        """Drain the HW ring and return this consumer's samples newer than its cursor, sliced to its
        metrics, as a frame (``start_ns`` / ``device_id`` / one column per metric) -- or None if
        none. Poll on your own cadence, but before the KEEP_LATEST ring fills (span = max_samples *
        interval) and drops the oldest; samples another consumer hasn't polled stay retained until it
        does."""
        return self._sampler._poll_consumer(self._consumer)

    def detach(self) -> None:
        """Unregister; the session disables once the last consumer detaches. :meth:`poll` first for a
        final frame -- detaching returns nothing."""
        self._sampler._detach(self._consumer)

    def __del__(self) -> None:
        try:
            self.detach()
        except Exception:
            pass


class PmSampler:
    """The single PM-sampling session on one CUDA device -- a per-device singleton
    (``PmSampler(device)`` returns the one instance, default the current device). Only one PM session
    per device is possible, so all users of a device share this one object. The process-global CUPTI
    profiler deinit is not refcounted by CUPTI, so PmSampler refcounts its live collectors
    (:attr:`_active`) and deinitializes once, when the last is torn down -- keeping concurrent
    sampling on multiple devices safe.

    :meth:`add_consumer` (the metrics wanted) returns a handle; poll the handle for its samples and
    detach it -- or just drop it -- to unregister. The session samples the *union* of all consumers'
    metrics. Delivery is pull-based: ``handle.poll()`` drains the HW ring and returns the samples
    newer than the handle's
    cursor, sliced to its own metrics. decode drains, so a sample another consumer has not yet polled
    is retained (bounded to the look-back window) and GC'd once every consumer has passed it -- a
    lone, caught-up consumer keeps no copy. Frames carry RAW CUPTI-clock-ns timestamps in
    ``start_ns`` -- converting to trace/epoch time is the consumer's job, so the sampler has no clock
    dependency."""

    _instances: dict[int, PmSampler] = {}
    _instances_lock = threading.Lock()
    _active = 0
    # Process-wide sampling config; each device's singleton snapshots these when first created.
    # Set via configure() (first-come-first-serve, no env var); defaults otherwise.
    _sampling_interval_ns: int = _DEFAULT_SAMPLING_INTERVAL_NS
    _lookback_window_ns: int = _DEFAULT_LOOKBACK_WINDOW_NS
    _configured: bool = False

    @classmethod
    def configure(
        cls,
        *,
        sampling_interval_ms: float | None = None,
        lookback_window_ms: float | None = None,
    ) -> None:
        """Set the process-wide PM-sampling interval and look-back window (milliseconds).
        First-come-first-serve: locked once a call lands OR the first sampler is instantiated
        (whichever comes first) -- a live per-device session can't be resized -- so a later
        call is ignored with a warning. Call before the first consumer registers; an unset arg
        keeps its current value."""
        with cls._instances_lock:
            if cls._configured:
                logger.warning(
                    "PmSampler.configure() ignored: already configured (first-come-first-serve)"
                )
                return
            if sampling_interval_ms is not None:
                cls._sampling_interval_ns = int(sampling_interval_ms * 1_000_000)
            if lookback_window_ms is not None:
                cls._lookback_window_ns = int(lookback_window_ms * 1_000_000)
            cls._configured = True

    @classmethod
    def get_config(cls) -> dict[str, int | bool]:
        """The process-wide PM-sampling config a new device session will snapshot: the
        sampling interval and look-back window (nanoseconds), and whether configure() has
        pinned it (first-come-first-serve)."""
        return {
            "sampling_interval_ns": cls._sampling_interval_ns,
            "lookback_window_ns": cls._lookback_window_ns,
            "configured": cls._configured,
        }

    def __new__(cls, device: int | None = None) -> Self:
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
        self._device = device
        # Snapshot the process-wide config at creation and lock it: a live session can't be
        # resized, so from here configure() is a no-op (first-come-first-serve). Runs under
        # _instances_lock (held by __new__), the same lock configure() takes.
        cls = type(self)
        self._sampling_interval_ns = cls._sampling_interval_ns
        self._lookback_window_ns = cls._lookback_window_ns
        cls._configured = True
        samples = self._lookback_window_ns // self._sampling_interval_ns
        self._max_samples = max(1, samples)
        self._consumers: list[_Consumer] = []
        self._metric_names: list[str] = []
        self._col: Any = None
        # Samples drained from the HW ring but not yet consumed by every consumer (columnar, in the
        # union metric order). Held only while a consumer lags; GC'd by the min-cursor watermark.
        self._retained_ts = np.empty(0, dtype=np.int64)
        self._retained_vals = np.empty((0, 0), dtype=np.float64)
        self._lock = threading.RLock()

    def add_consumer(self, metrics: Iterable[str]) -> _Handle:
        """Register a consumer for ``metrics`` and return its handle (``handle.poll()`` /
        ``handle.detach()``; dropping the handle also detaches). Raises ValueError if ``metrics`` is
        empty or the resulting union can't be collected in one PM pass."""
        metrics = list(metrics)
        if not metrics:
            raise ValueError("PM sampling requires a non-empty metric set")
        consumer = _Consumer(metrics)
        with self._lock:
            self._check_single_pass(self._union_of([*self._consumers, consumer]))
            self._consumers.append(consumer)
            self._reconfigure()
        return _Handle(self, consumer)

    def _detach(self, consumer: _Consumer) -> None:
        with self._lock:
            if consumer not in self._consumers:
                return
            self._consumers.remove(consumer)
            self._reconfigure()
            self._gc()  # the leaver no longer pins the GC watermark

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
        desired = self._union_of(self._consumers)
        if not desired:
            self._teardown()
        elif desired != self._metric_names:
            # Rebuild, not in-place reconfigure -- a live stop/configure/start segfaults on decode.
            self._teardown()
            self._metric_names = desired
            self._start()

    def _poll_consumer(self, consumer: _Consumer) -> dict[str, Any] | None:
        # Implements _Handle.poll(): drain the HW ring, build `consumer`'s frame from the retained
        # samples newer than its cursor (sliced to its metrics) and advance its cursor, then GC.
        # Returns None if sampling is off, `consumer` isn't registered, or it has nothing new.
        with self._lock:
            if self._col is None or consumer not in self._consumers:
                return None
            self._pull_hw()
            frame: dict[str, Any] | None = None
            ts = self._retained_ts
            mask = ts > consumer.cursor if ts.size else None
            if mask is not None and mask.any():
                dts = ts[mask]
                dvals = self._retained_vals[mask]
                col_index = {m: j for j, m in enumerate(self._metric_names)}
                frame = {
                    "start_ns": dts,
                    "device_id": np.full(len(dts), self._device, dtype=np.int64),
                }
                for m in consumer.metrics:
                    frame[m] = dvals[:, col_index[m]]
                consumer.cursor = int(dts.max())
            self._gc()
            return frame

    def _start(self) -> None:
        if self._col is not None or not self._metric_names:
            return
        self._warn_unsupported()
        try:
            col = _pm_sampling.Collector(device_index=self._device)
            col.enable()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            return
        self._col = col
        with type(self)._instances_lock:
            type(self)._active += 1
        try:
            col.configure(
                metrics=self._metric_names,
                hardware_buffer_size=_counter_data_size(
                    col._pm_sampling_object, self._metric_names, self._max_samples
                ),
                sampling_interval=self._sampling_interval_ns,
                trigger_mode=_pm_sampling.TriggerMode.GPU_TIME_INTERVAL,
                hw_buffer_append_mode=_pm_sampling.HardwareBuffer_AppendMode.KEEP_LATEST,
            )
            col.start()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            self._teardown()

    def _teardown(self) -> None:
        col, self._col = self._col, None
        self._metric_names = []
        self._retained_ts = np.empty(0, dtype=np.int64)
        self._retained_vals = np.empty((0, 0), dtype=np.float64)
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
        try:
            host = _profiler_host.ProfilerHost(
                _device_chip_name(self._device), _cupti.ProfilerType.PM_SAMPLING
            )
            host.initialize()
            try:
                passes = _profiler_host.get_num_of_passes(
                    host.create_config_image(metrics=metrics)
                )
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

    def _pull_hw(self) -> None:
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
            logger.exception("PM sampling decode error")
            return
        n = cd.num_completed_samples
        if not n:
            return
        if n >= self._max_samples:
            logger.warning(
                "PM sampling decoded the maximum %d samples; some were dropped.",
                self._max_samples,
            )
        ts = np.empty(n, dtype=np.int64)
        vals = np.empty((n, len(self._metric_names)), dtype=np.float64)
        for i, s in enumerate(cd):
            ts[i] = s.start_timestamp
            vals[i] = s.metric_values
        # Drop an unset interval-start (0) or a stale small value from a
        # different clock domain (observed ~7.5e13 vs the real ~1.78e18).
        # Real samples fall within the look-back of the newest.
        keep = ts > 0
        if keep.any():
            keep &= ts >= int(ts[keep].max()) - self._lookback_window_ns
        if not keep.any():
            return
        ts = ts[keep]
        vals = vals[keep]
        if self._retained_ts.size:
            self._retained_ts = np.concatenate([self._retained_ts, ts])
            self._retained_vals = np.concatenate([self._retained_vals, vals])
        else:
            self._retained_ts = ts
            self._retained_vals = vals

    def _gc(self) -> None:
        # Drop retained samples every consumer has passed (<= the min cursor) or that have aged past
        # the look-back window (so a consumer that stops polling can't pin the buffer). A lone,
        # caught-up consumer leaves nothing retained. Caller holds _lock.
        ts = self._retained_ts
        if not ts.size:
            return
        newest = int(ts.max())
        watermark = max(
            min((c.cursor for c in self._consumers), default=newest),
            newest - self._lookback_window_ns,
        )
        keep = ts > watermark
        if keep.all():
            return
        self._retained_ts = ts[keep]
        self._retained_vals = self._retained_vals[keep]
