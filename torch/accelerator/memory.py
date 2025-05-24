import collections
from typing import Any

import torch

from ._utils import _device_t, _get_device_index


__all__ = [
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    torch._C._accelerator_emptyCache()


def memory_stats_as_nested_dict(device: _device_t = None, /) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.accelerator.memory_stats` as a nested dictionary.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    if not torch._C._accelerator_allocatorInitialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._accelerator_getDeviceStats(device)


def memory_stats(device: _device_t = None) -> dict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    stats = memory_stats_as_nested_dict(device=device)
    flat_stats = []

    def flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                nested_prefix = f"{prefix}.{k}" if prefix else k
                flatten(nested_prefix, v)
        else:
            flat_stats.append((prefix, value))

    flatten("", stats)
    flat_stats.sort()
    return collections.OrderedDict(flat_stats)


def memory_allocated(device: _device_t = None) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: _device_t = None) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: _device_t = None) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: _device_t = None) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device: _device_t = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the current :ref:`accelerator<accelerators>` memory allocator.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._accelerator_resetAccumulatedStats(device)


def reset_peak_memory_stats(device: _device_t = None) -> None:
    r"""Reset the "peak" stats tracked by the current :ref:`accelerator<accelerators>` memory allocator.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._accelerator_resetPeakStats(device)
