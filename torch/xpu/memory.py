import collections
from typing import Any, Dict, Union

import torch
from torch.types import Device

from . import _get_device_index, is_initialized


_device_t = Union[Device, str, int, None]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other XPU application.

    .. note::
        :func:`~torch.xpu.empty_cache` doesn't increase the amount of XPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of XPU memory in certain cases.
    """
    if is_initialized():
        torch._C._xpu_emptyCache()


def reset_peak_memory_stats(device: _device_t = None) -> None:
    r"""Reset the "peak" stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_resetPeakMemoryStats(device)


def reset_accumulated_memory_stats(device: _device_t = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the XPU memory allocator.

    See :func:`~torch.xpu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_resetAccumulatedMemoryStats(device)


def memory_stats_as_nested_dict(device: _device_t = None) -> Dict[str, Any]:
    r"""Return the result of :func:`~torch.xpu.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_memoryStats(device)


def memory_stats(device: _device_t = None) -> Dict[str, Any]:
    r"""Return a dictionary of XPU memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool (for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool (for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    result = []

    def _recurse_add_to_result(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_allocated(device: _device_t = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `xpu-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: _device_t = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.xpu.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: _device_t = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: _device_t = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.xpu.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


__all__ = [
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]
