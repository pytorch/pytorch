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
# Periodic poll fires this long before the ring's fill boundary -- a fixed (not proportional) safety
# margin for scheduling jitter, so a large window can poll close to the boundary while a small one
# still keeps a proportional floor (see suggested_poll_interval_ns).
_POLL_SAFETY_BUFFER_NS = 500_000_000  # 500 ms


def is_available() -> bool:
    try:
        import cupti.pm_sampling  # noqa: F401  # pyrefly: ignore[missing-import]
    except Exception:
        return False
    return True


def _counter_data_size(
    pm_sampling_object: int, metrics: list[str], max_samples: int
) -> int:
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
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.Device_GetChipName_Params()
    p.struct_size = c.DEVICE_GET_CHIP_NAME_PARAMS_STRUCT_SIZE
    p.device_index = device
    c.device_get_chip_name(p.ptr)
    return p.p_chip_name


def _pm_sampling_disable(pm_sampling_object: int) -> None:
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.PmSampling_Disable_Params()
    p.struct_size = c.PM_SAMPLING_DISABLE_PARAMS_STRUCT_SIZE
    p.p_pm_sampling_object = pm_sampling_object
    c.pm_sampling_disable(p.ptr)


def _profiler_deinitialize() -> None:
    from cupti import cupti as c  # pyrefly: ignore[missing-import]

    p = c.Profiler_DeInitialize_Params()
    p.struct_size = c.PROFILER_DEINITIALIZE_PARAMS_STRUCT_SIZE
    c.profiler_deinitialize(p.ptr)


@functools.cache
def supported_metrics(*, with_sub_metrics: bool = False) -> frozenset[str]:
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
    sampler has no clock dependency."""

    _instances: dict[int, PmSampler] = {}
    _instances_lock = threading.Lock()
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
        self._device = device
        self._sampling_interval_ns = _SAMPLING_INTERVAL_NS
        self._max_samples = max(1, _DEFAULT_WINDOW_NS // _SAMPLING_INTERVAL_NS)
        self._consumers: list[_Consumer] = []
        self._metric_names: list[str] = []
        self._col: Any = None
        self._lock = threading.RLock()

    def add_consumer(
        self, metrics: Iterable[str], sink: Callable[[dict[str, Any]], None]
    ) -> _Consumer:
        metrics = list(metrics)
        if not metrics:
            raise ValueError("PM sampling requires a non-empty metric set")
        consumer = _Consumer(metrics, sink)
        with self._lock:
            self._check_single_pass(self._union_of([*self._consumers, consumer]))
            self._drain()
            self._consumers.append(consumer)
            self._reconfigure()
        return consumer

    def remove_consumer(self, consumer: _Consumer) -> None:
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
        desired = self._union_of(self._consumers)
        if not desired:
            self._teardown()
        elif desired != self._metric_names:
            self._teardown()
            self._metric_names = desired
            self._start()

    @property
    def suggested_poll_interval_ns(self) -> int:
        """Recommended cadence for periodic :meth:`poll`. decode drains the ring (each sample once),
        so total decode cost is independent of poll frequency -- the only hard constraint is polling
        before the KEEP_LATEST ring fills (its span = max_samples * interval), after which it drops
        the oldest samples. Poll a fixed buffer before that boundary (jitter is ~absolute, so a large
        window can poll close to the boundary), but never later than half the span -- otherwise a
        small window, where the fixed buffer would leave almost no time, would poll too tight. The
        final drain at teardown catches the tail."""
        span = self._max_samples * self._sampling_interval_ns
        return max(self._sampling_interval_ns, span // 2, span - _POLL_SAFETY_BUFFER_NS)

    def poll(self) -> None:
        with self._lock:
            if self._col is None:
                return
            try:
                self._drain()
            except Exception:
                logger.exception("PM sampling poll decode error")

    def _start(self) -> None:
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
                trigger_mode=pm.TriggerMode.GPU_TIME_INTERVAL,
                hw_buffer_append_mode=pm.HardwareBuffer_AppendMode.KEEP_LATEST,
            )
            col.start()
        except Exception as e:
            logger.warning("PM sampling could not start: %s", e)
            self._teardown()

    def _teardown(self) -> None:
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
        # the first sample of a session has an unset
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
        for consumer in self._consumers:
            frame: dict[str, Any] = {"start_ns": ts, "device_id": device_col}
            for m in consumer.metrics:
                frame[m] = vals[:, col_index[m]]
            consumer.sink(frame)
