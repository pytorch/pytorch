r"""Query the hardware metrics of XPU devices."""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from ._utils import _get_device_index


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.types import Device

__all__ = ["available_scope_profiler_metrics"]


@lru_cache(None)
def _available_scope_profiler_metrics(
    device_index: int,
) -> Mapping[str, tuple[str, ...]]:
    if not hasattr(torch._C._profiler, "_xpu_available_scope_profiler_metrics"):
        raise RuntimeError(
            "XPU metric queries require a PyTorch build with XPU profiling support"
        )
    uuid = bytes(torch.xpu.get_device_properties(device_index).uuid.bytes)
    groups: dict[str, tuple[str, ...]] = {}
    for name, metrics in torch._C._profiler._xpu_available_scope_profiler_metrics(uuid):
        # PTI resolves scope groups by name and uses the first match.
        groups.setdefault(name, tuple(metrics))
    return MappingProxyType(groups)


def available_scope_profiler_metrics(
    device: Device = None,
) -> Mapping[str, tuple[str, ...]]:
    r"""Return the hardware metrics the XPU scope profiler can collect, by group.

    Maps each metric group's name to the names of its metrics. These names can
    be listed under ``ProfilerActivity.XPU`` in the ``activities`` of
    :func:`torch.profiler.profile`; names requested together must belong to
    one group.

    ``ZET_ENABLE_METRICS=1`` must be set before XPU is first used. Without
    permission to access metrics, for example
    ``/proc/sys/dev/xe/observation_paranoid`` set to ``0``, some groups may be
    missing.

    Args:
        device (torch.device or int or str, optional): device to query. Uses
            :func:`~torch.xpu.current_device` if :attr:`device` is ``None``
            (default).

    Returns:
        MappingProxyType: An immutable mapping from group names to tuples of
        metric names.

    Raises:
        RuntimeError: if the driver cannot report metrics for the device.

    Example::

        >>> # xdoctest: +SKIP("requires XPU hardware metrics")
        >>> from torch.profiler import profile, ProfilerActivity
        >>> names = list(
        ...     torch.xpu.profiler.available_scope_profiler_metrics()["ComputeBasic"][:2]
        ... )
        >>> activities = [ProfilerActivity.CPU, {ProfilerActivity.XPU: names}]
        >>> with profile(activities=activities):
        ...     torch.ones(8, device="xpu").sum()
    """
    return _available_scope_profiler_metrics(_get_device_index(device, optional=True))
