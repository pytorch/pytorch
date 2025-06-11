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
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    torch._C._accelerator_emptyCache()


def memory_stats_as_nested_dict(device: _device_t = None, /) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.accelerator.memory_stats` as a nested dictionary.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._accelerator_getDeviceStats(device)


def memory_stats(device: _device_t = None, /) -> dict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from device memory allocation.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of June 2025, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of June 2025, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed device memory allocation calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of device memory allocation calls.
    - ``"num_device_free"``: number of device memory free calls.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    stats = memory_stats_as_nested_dict(device)
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


def memory_allocated(device: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: _device_t = None, /) -> int:
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
    return memory_stats(device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: _device_t = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.
    """
    return memory_stats(device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: _device_t = None, /) -> int:
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
    return memory_stats(device).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device: _device_t = None, /) -> None:
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


def reset_peak_memory_stats(device: _device_t = None, /) -> None:
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
