import collections
import ctypes
from typing import Any

import torch
from torch._utils import _dummy_type
from torch.types import Device
from . import _get_device_index, _is_compiled, _lazy_init, is_initialized


if not _is_compiled():
    # Define dummy base classes
    torch._C.__dict__["_xpu_XPUAllocator"] = _dummy_type("_xpu_XPUAllocator")


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


def reset_peak_memory_stats(device: Device = None) -> None:
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


def reset_accumulated_memory_stats(device: Device = None) -> None:
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


def memory_stats_as_nested_dict(device: Device = None) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.xpu.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_memoryStats(device)


def memory_stats(device: Device = None) -> dict[str, Any]:
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


def memory_allocated(device: Device = None) -> int:
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


def max_memory_allocated(device: Device = None) -> int:
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


def memory_reserved(device: Device = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Device = None) -> int:
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


def mem_get_info(device: Device = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        int: the memory available on the device in units of bytes.
        int: the total memory on the device in units of bytes
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_getMemoryInfo(device)


def get_per_process_memory_fraction(device: Device = None) -> float:
    r"""
    Retrieve the memory fraction currently set for a process on a given XPU device.
    This fraction represents the portion of the total device memory that
    the caching allocator is allowed to use. The allowed memory is calculated as:

    .. math:: \text{allowed\_memory} = \text{total\_memory} \times \text{fraction}

    Args:
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).

    Returns:
        float: The memory fraction in the range 0.0 to 1.0.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_getMemoryFraction(device)


def set_per_process_memory_fraction(fraction: float, device: Device = None) -> None:
    r"""
    Set the memory fraction for a single process on XPU device.
    This function limits the amount of memory that the caching allocator can allocate
    on the specified XPU device. The allowed memory is computed as:

    .. math:: \text{allowed\_memory} = \text{total\_memory} \times \text{fraction}

    If the process attempts to allocate more than this allowed memory,
    an out-of-memory error will be raised by the allocator.

    Arguments:
        fraction (float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).

    .. note:: In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    # pyrefly: ignore [missing-attribute]
    torch._C._xpu_setMemoryFraction(fraction, device)


class _XPUAllocator:
    r"""Wrapper over internal XPU memory allocators."""

    # pyrefly: ignore [missing-attribute]
    def __init__(self, allocator: torch._C._xpu_XPUAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class XPUPluggableAllocator(_XPUAllocator):
    r"""XPU memory allocator loaded from a shared library."""

    def __init__(self, path_to_lib_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""XPU memory allocator loaded dynamically from a shared library.

        This lets users provide custom allocation and free functions implemented
        in a separate shared library. The allocator is registered through
        ``torch._C._xpu_customAllocator`` and becomes available for use via
        ``torch.memory.xpu.change_current_allocator``.

        Arguments:
            path_to_lib_file (str):
                Filesystem path to the shared library file containing the allocation
                and free functions.
            alloc_fn_name (str):
                Name of the allocation function exported from the shared library.
                The function must have the signature:

                    ``void* alloc_fn(size_t size, int device, sycl::queue* queue);``

            free_fn_name (str):
                Name of the free function exported from the shared library.
                The function must have the signature:

                    ``void free_fn(void* ptr, size_t size, sycl::queue* queue);``
        """
        allocator_lib = ctypes.CDLL(path_to_lib_file)

        alloc_fn_ptr = getattr(allocator_lib, alloc_fn_name)
        free_fn_ptr = getattr(allocator_lib, free_fn_name)

        alloc_fn_addr = ctypes.cast(alloc_fn_ptr, ctypes.c_void_p).value
        free_fn_addr = ctypes.cast(free_fn_ptr, ctypes.c_void_p).value

        if alloc_fn_addr is None or free_fn_addr is None:
            raise RuntimeError(
                "Failed to load allocator symbols from the shared library."
            )

        # pyrefly: ignore [missing-attribute]
        self._allocator = torch._C._xpu_customAllocator(alloc_fn_addr, free_fn_addr)


def change_current_allocator(allocator: _XPUAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    .. note::
        If the current allocator has already been used/initialized, this function will error.

    Arguments:
        allocator (torch.xpu.memory._XPUAllocator): allocator to be set as the active one.
    """
    # pyrefly: ignore [missing-attribute]
    torch._C._xpu_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _XPUAllocator:
    r"""Return the allocator being currently used.

    Returns:
        _XPUAllocator: the allocator being currently used.
    """
    # pyrefly: ignore [missing-attribute]
    return _XPUAllocator(torch._C._xpu_getAllocator())


__all__ = [
    "XPUPluggableAllocator",
    "change_current_allocator",
    "empty_cache",
    "get_per_process_memory_fraction",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_per_process_memory_fraction",
]
