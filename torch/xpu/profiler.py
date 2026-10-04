r"""Query the hardware metrics of XPU devices."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from ._utils import _get_device_index


if TYPE_CHECKING:
    from torch.types import Device

__all__ = ["available_metrics", "XpuMetric", "XpuMetricGroup"]


@dataclasses.dataclass(frozen=True)
class XpuMetric:
    r"""A hardware metric of an :class:`XpuMetricGroup`.

    ``metric_type`` is, for example, ``"duration"``, ``"event"`` or
    ``"throughput"``; ``value_type`` is, for example, ``"uint64"`` or ``"float32"``.
    """

    name: str
    description: str
    unit: str
    metric_type: str
    value_type: str


@dataclasses.dataclass(frozen=True)
class XpuMetricGroup:
    r"""A group of hardware metrics that the driver collects together.

    ``scope_compatible`` is ``True`` for event-based groups, the only kind
    whose metrics :mod:`torch.profiler` can collect per kernel. The driver may
    report a group name twice, once per sampling type.
    """

    name: str
    description: str
    scope_compatible: bool
    metrics: tuple[XpuMetric, ...]


@lru_cache(None)
def _available_metrics(device_index: int) -> tuple[XpuMetricGroup, ...]:
    uuid = bytes(torch.xpu.get_device_properties(device_index).uuid.bytes)
    return tuple(
        XpuMetricGroup(
            name=group["name"],
            description=group["description"],
            scope_compatible=group["scope_compatible"],
            metrics=tuple(XpuMetric(**metric) for metric in group["metrics"]),
        )
        for group in torch._C._profiler._xpu_available_metrics(uuid)
    )


def available_metrics(device: Device = None) -> list[XpuMetricGroup]:
    r"""Return the hardware metric groups the driver exposes for a device.

    ``ZET_ENABLE_METRICS=1`` must be set before XPU is first used. Without
    permission to access metrics, for example
    ``/proc/sys/dev/xe/observation_paranoid`` set to ``0``, some groups may be
    missing.

    Args:
        device (torch.device or int or str, optional): device to query. Uses
            :func:`~torch.xpu.current_device` if :attr:`device` is ``None``
            (default).

    Raises:
        RuntimeError: if the driver cannot report metrics for the device.

    Example::

        >>> # xdoctest: +SKIP("requires XPU hardware metrics")
        >>> from torch.profiler import profile, ProfilerActivity
        >>> groups = torch.xpu.profiler.available_metrics()
        >>> group = next(g for g in groups if g.scope_compatible)
        >>> names = [m.name for m in group.metrics[:2]]
        >>> activities = [ProfilerActivity.CPU, {ProfilerActivity.XPU: names}]
        >>> with profile(activities=activities):
        ...     torch.ones(8, device="xpu").sum()
    """
    return list(_available_metrics(_get_device_index(device, optional=True)))
