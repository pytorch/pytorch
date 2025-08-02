# mypy: allow-untyped-defs
r"""This package adds support for device memory management implemented in CUDA."""

import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Literal, Optional, TYPE_CHECKING
from typing_extensions import deprecated

import torch
from torch import _C
from torch._utils import _dummy_type

from . import (
    _get_amdsmi_device_index,
    _get_device_index,
    _get_nvml_device_index,
    _lazy_init,
    is_initialized,
)
from ._memory_viz import memory as _memory, segments as _segments


if TYPE_CHECKING:
    from torch.types import Device


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_enable",
    "get_per_process_memory_fraction",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "list_gpu_processes",
    "mem_get_info",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "use_mem_pool",
]


if not hasattr(torch._C, "_cuda_CUDAAllocator"):
    # Define dummy base classes
    torch._C.__dict__["_cuda_CUDAAllocator"] = _dummy_type("_cuda_CUDAAllocator")


if not hasattr(torch._C, "_MemPool"):
    # Define dummy base classes
    torch._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch._C.__dict__["_cuda_beginAllocateToPool"] = _dummy_type(
        "_cuda_beginAllocateToPool"
    )
    torch._C.__dict__["_cuda_beginAllocateCurrentThreadToPool"] = _dummy_type(
        "_cuda_beginAllocateCurrentThreadToPool"
    )
    torch._C.__dict__["_cuda_endAllocateToPool"] = _dummy_type(
        "_cuda_endAllocateToPool"
    )
    torch._C.__dict__["_cuda_releasePool"] = _dummy_type("_cuda_releasePool")

from torch._C import (  # noqa: F401
    _cuda_beginAllocateCurrentThreadToPool,
    _cuda_beginAllocateToPool,
    _cuda_CUDAAllocator,
    _cuda_endAllocateToPool,
    _cuda_releasePool,
    _MemPool,
)


def _host_allocator():
    _lazy_init()
    return torch._C._cuda_cudaHostAllocator()


@contextlib.contextmanager
def _free_mutex():
    torch._C._cuda_lock_mutex()
    try:
        yield
    finally:
        torch._C._cuda_unlock_mutex()


def caching_allocator_alloc(size, device: "Device" = None, stream=None):
    r"""Perform a memory allocation using the CUDA memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch.cuda.caching_allocator_delete`.

    Args:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
        stream (torch.cuda.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch.cuda.current_stream(device)
    if isinstance(stream, torch.cuda.streams.Stream):
        stream = stream.cuda_stream
    if not isinstance(stream, int):
        raise TypeError(
            "Invalid type for stream argument, must be "
            "`torch.cuda.Stream` or `int` representing a pointer "
            "to a existing stream"
        )
    with torch.cuda.device(device):
        return torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the CUDA memory allocator.

    Memory allocated with :func:`~torch.cuda.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Args:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    torch._C._cuda_cudaCachingAllocator_raw_delete(mem_ptr)


def caching_allocator_enable(value: bool = True) -> None:
    r"""Enable or disable the CUDA memory allocator. On by default."""
    if is_initialized():
        torch._C._cuda_cudaCachingAllocator_enable(value)


def set_per_process_memory_fraction(fraction, device: "Device" = None) -> None:
    r"""Set memory fraction for a process.

    The fraction is used to limit an caching allocator to allocated memory on a CUDA device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")

    torch._C._cuda_setMemoryFraction(fraction, device)


def get_per_process_memory_fraction(device: "Device" = None) -> float:
    r"""Get memory fraction for a process.

    Args:
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default CUDA device is used.
    Returns:
        memory fraction, in range 0~1. Allowed memory equals total_memory * fraction.
    """
    _lazy_init()
    if device is None:
        device = torch.cuda.current_device()
    device = _get_device_index(device)
    return torch._C._cuda_getMemoryFraction(device)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """
    if is_initialized():
        torch._C._cuda_emptyCache()


def memory_stats(device: "Device" = None) -> dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``cudaMalloc()``.
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
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed ``cudaMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of CUDA allocation calls. This includes both
      cuMemMap and cudaMalloc.
    - ``"num_device_free"``: number of CUDA free calls. This includes both cuMemUnmap
      and cudaFree.

    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory fragmentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:

    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.

    The caching allocator can be configured via ENV to round memory allocations in order
    to reduce fragmentation. Sometimes the overhead from rounding can be higher than
    the fragmentation it helps reduce. The following stat can be used to check if
    rounding adds too much overhead:

    - ``"requested_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      memory requested by client code, compare this with allocated_bytes to check if
      allocation rounding adds too much overhead.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.

    .. note::
        With :ref:`backend:cudaMallocAsync<cuda-memory-envvars>`, some stats are not
        meaningful, and are always reported as zero.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
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


def memory_stats_as_nested_dict(device: "Device" = None) -> dict[str, Any]:
    r"""Return the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryStats(device)


def reset_accumulated_memory_stats(device: "Device" = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device: "Device" = None) -> None:
    r"""Reset the "peak" stats tracked by the CUDA memory allocator.

    See :func:`~torch.cuda.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetPeakMemoryStats(device)


def host_memory_stats() -> dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

     The return value of this function is a dictionary of statistics, each of
     which is a non-negative integer.

     Core statistics:

     - ``"allocated.{current,peak,allocated,freed}"``:
       number of allocation requests received by the memory allocator.
     - ``"allocated_bytes.{current,peak,allocated,freed}"``:
       amount of allocated memory.
     - ``"segment.{current,peak,allocated,freed}"``:
       number of reserved segments from ``cudaMalloc()``.
     - ``"reserved_bytes.{current,peak,allocated,freed}"``:
       amount of reserved memory.

     For these core statistics, values are broken down as follows.

     Metric type:

     - ``current``: current value of this metric.
     - ``peak``: maximum value of this metric.
     - ``allocated``: historical total increase in this metric.
     - ``freed``: historical total decrease in this metric.

     In addition to the core statistics, we also provide some simple event
     counters:

     - ``"num_host_alloc"``: number of CUDA allocation calls. This includes both
       cudaHostAlloc and cudaHostRegister.
     - ``"num_host_free"``: number of CUDA free calls. This includes both cudaHostFree
       and cudaHostUnregister.

     Finally, we also provide some simple timing counters:

     - ``"host_alloc_time.{total,max,min,count,avg}"``:
       timing of allocation requests going through CUDA calls.
     - ``"host_free_time.{total,max,min,count,avg}"``:
       timing of free requests going through CUDA calls.

    For these timing statistics, values are broken down as follows.

     Metric type:

     - ``total``: total time spent.
     - ``max``: maximum value per call.
     - ``min``: minimum value per call.
     - ``count``: number of times it was called.
     - ``avg``: average time per call.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = host_memory_stats_as_nested_dict()
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def host_memory_stats_as_nested_dict() -> dict[str, Any]:
    r"""Return the result of :func:`~torch.cuda.host_memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    return torch._C._cuda_hostMemoryStats()


def reset_accumulated_host_memory_stats() -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict.
    """
    return torch._C._cuda_resetAccumulatedHostMemoryStats()


def reset_peak_host_memory_stats() -> None:
    r"""Reset the "peak" stats tracked by the host memory allocator.

    See :func:`~torch.cuda.host_memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.
    """
    return torch._C._cuda_resetPeakHostMemoryStats()


def reset_max_memory_allocated(device: "Device" = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: "Device" = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    warnings.warn(
        "torch.cuda.reset_max_memory_cached now calls torch.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: "Device" = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: "Device" = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: "Device" = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


@deprecated(
    "`torch.cuda.memory_cached` has been renamed to `torch.cuda.memory_reserved`",
    category=FutureWarning,
)
def memory_cached(device: "Device" = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.memory_reserved`."""
    return memory_reserved(device=device)


@deprecated(
    "`torch.cuda.max_memory_cached` has been renamed to `torch.cuda.max_memory_reserved`",
    category=FutureWarning,
)
def max_memory_cached(device: "Device" = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.max_memory_reserved`."""
    return max_memory_reserved(device=device)


def memory_snapshot(mempool_id=None):
    r"""Return a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return torch._C._cuda_memorySnapshot(mempool_id)["segments"]


def memory_summary(device: "Device" = None, abbreviated: bool = False) -> str:
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch CUDA memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} CUDA OOMs: {num_ooms:<12d} | {_:6} cudaMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} ",
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GPU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} ",
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def list_gpu_processes(device: "Device" = None) -> str:
    r"""Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if not torch.version.hip:
        try:
            import pynvml  # type: ignore[import]
        except ModuleNotFoundError:
            return "pynvml module not found, please install pynvml"
        from pynvml import NVMLError_DriverNotLoaded

        try:
            pynvml.nvmlInit()
        except NVMLError_DriverNotLoaded:
            return "cuda driver can't be loaded, is cuda enabled?"

        device = _get_nvml_device_index(device)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    else:
        try:
            import amdsmi  # type: ignore[import]
        except ModuleNotFoundError:
            return "amdsmi module not found, please install amdsmi"
        try:
            amdsmi.amdsmi_init()  # type: ignore[attr-defined]
        except amdsmi.AmdSmiException:  # type: ignore[attr-defined]
            return "amdsmi driver can't be loaded, is ROCm installed?"

        device = _get_amdsmi_device_index(device)

        try:
            handle = amdsmi.amdsmi_get_processor_handles()[device]  # type: ignore[attr-defined]
            procs = amdsmi.amdsmi_get_gpu_process_list(handle)  # type: ignore[attr-defined]
        except amdsmi.AmdSmiException:  # type: ignore[attr-defined]
            return "amdsmi cannot list processes from other users"

    lines = []
    lines.append(f"GPU:{device}")
    if len(procs) == 0:
        lines.append("no processes are running")
    for p in procs:
        if not torch.version.hip:
            mem = p.usedGpuMemory / (1024 * 1024)
            pid = p.pid
        else:
            try:
                proc_info = amdsmi.amdsmi_get_gpu_process_info(handle, p)  # type: ignore[possibly-undefined]
            except AttributeError:
                # https://github.com/ROCm/amdsmi/commit/c551c3caedbd903ba828e7fdffa5b56d475a15e7
                # is a BC-breaking change that removes amdsmi_get_gpu_process_info API from amdsmi
                proc_info = p
            mem = proc_info["memory_usage"]["vram_mem"] / (1024 * 1024)
            pid = proc_info["pid"]
        lines.append(f"process {pid:>10d} uses {mem:>12.3f} MB GPU memory")
    return "\n".join(lines)


def mem_get_info(device: "Device" = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device using cudaMemGetInfo.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default) or if the device index is not specified.

    .. note::
        See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    if device is None:
        device = torch.cuda.current_device()
    # optional=True allows `device = torch.device('cuda')` for which device.index is None
    device = _get_device_index(device, optional=True)
    return torch.cuda.cudart().cudaMemGetInfo(device)


def _record_memory_history_legacy(
    enabled: bool,
    record_context=True,
    trace_alloc_max_entries=1,
    trace_alloc_record_context=False,
    device: "Device" = None,
    record_context_cpp=False,
    clear_history=False,
    compile_context=False,
    global_record_annotations=False,
):
    _C._cuda_record_memory_history_legacy(  # type: ignore[call-arg]
        enabled,
        record_context,
        trace_alloc_max_entries,
        trace_alloc_record_context,
        record_context_cpp,
        clear_history,
        compile_context,
        global_record_annotations,
    )


def _record_memory_history(
    enabled: Literal[None, "state", "all"] = "all", *args, **kwargs
) -> None:
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.cuda.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.cuda.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries.

    C++ implementation for reference to ring buffer implementation:

    .. code-block:: cpp

        if (record_history) {
          if (alloc_trace->size() < alloc_trace_max_entries_) {
            alloc_trace->emplace_back(te);
          } else {
            (*alloc_trace)[alloc_trace_next++] = te;
            if (alloc_trace_next == alloc_trace_max_entries_) {
              alloc_trace_next = 0;
            }
          }
        }

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    if isinstance(enabled, bool):
        return _record_memory_history_legacy(enabled, *args, **kwargs)
    else:
        return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device: "Device" = None,
    clear_history: bool = False,
    compile_context: bool = False,
    global_record_annotations: bool = False,
):
    _C._cuda_record_memory_history(  # type: ignore[call-arg]
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        compile_context,
        global_record_annotations,
    )


_record_memory_history.__signature__ = signature(_record_memory_history_impl)  # type: ignore[attr-defined]


def _snapshot(device: "Device" = None):
    """Save a snapshot of CUDA memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # Segments are memory returned from a cudaMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int  #  cudaMalloc'd size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int  # size requested during malloc, may be smaller than
            # size due to rounding
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream to finish using
                # this, then it will become free
                "inactive",
            ]  # free for reuse
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str


        class TraceEntry(TypedDict):
            # When `torch.cuda.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # the allocated received a call to free memory
                "free_completed",  # the memory that was requested to be freed is now
                # able to be used in future allocation calls
                "segment_alloc",  # the caching allocator ask cudaMalloc for more memory
                # and added it as a segment in its cache
                "segment_free",  # the caching allocator called cudaFree to return memory
                # to cuda possibly trying free up memory to
                # allocate more segments or because empty_caches was called
                "oom",  # the allocator threw an OOM exception. 'size' is
                # the requested number of bytes that did not succeed
                "snapshot",  # the allocator generated a memory snapshot
                # useful to coorelate a previously taken
                # snapshot with this trace
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of
            # memory cuda still reports to be free

    Returns:
        The Snapshot dictionary object
    """
    return _C._cuda_memorySnapshot(None)


def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Snapshot file sizes scale with `max_entries` and stack trace depth per entry,
    with several KB per entry. These can easily be in the GB range for longer running
    workflows with large `max_entries`.

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = _snapshot()
    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with open(filename, "w") as f:
        f.write(_memory(snapshot))


def _set_allocator_settings(env: str):
    return torch._C._cuda_cudaCachingAllocator_set_allocator_settings(env)


def get_allocator_backend() -> str:
    r"""Return a string describing the active allocator backend as set by
    ``PYTORCH_CUDA_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator) and `cudaMallocAsync``
    (CUDA's built-in asynchronous allocator).

    .. note::
        See :ref:`cuda-memory-management` for details on choosing the allocator backend.
    """
    return torch._C._cuda_getAllocatorBackend()


class _CUDAAllocator:
    r"""Wrapper over internal CUDA memory allocators."""

    def __init__(self, allocator: torch._C._cuda_CUDAAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class CUDAPluggableAllocator(_CUDAAllocator):
    r"""CUDA memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch.memory.cuda.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, cudaStream_t stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, cudaStream_t stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`cuda-memory-management` for details on creating and using a custom allocator
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _CUDAAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.


    Args:
        allocator (torch.cuda.memory._CUDAAllocator): allocator to be set as the active one.
    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    torch._C._cuda_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _CUDAAllocator:
    r"""Return the allocator being currently used.

    .. note::
        See :ref:`cuda-memory-management` for details on creating and using a custom allocator
    """
    return _CUDAAllocator(torch._C._cuda_getAllocator())


class MemPool(_MemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the CUDACachingAllocator.

    Args:
        allocator(torch._C._cuda_CUDAAllocator, optional): a
            torch._C._cuda_CUDAAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the CUDACachingAllocator.
        use_on_oom(bool): a bool that indicates if this pool can be used
            as a last resort if a memory allocation outside of the pool fails due
            to Out Of Memory. This is False by default.
        symmetric(bool): a bool that indicates if this pool is symmetrical
            across ranks. This is False by default.
    """

    def __init__(
        self,
        allocator: Optional[_cuda_CUDAAllocator] = None,
        use_on_oom: bool = False,
        symmetric: bool = False,
    ):
        super().__init__(allocator, True, use_on_oom, symmetric)

    @property
    def id(self) -> tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    @property
    def is_symmetric(self) -> bool:
        r"""Returns whether this pool is used for NCCL's symmetric memory."""
        return super().is_symmetric

    @property
    def allocator(self) -> Optional[_cuda_CUDAAllocator]:
        r"""Returns the allocator this MemPool routes allocations to."""
        return super().allocator

    def use_count(self) -> int:
        r"""Returns the reference count of this pool."""
        return super().use_count()

    def snapshot(self):
        r"""Return a snapshot of the CUDA memory allocator pool state across all
        devices.

        Interpreting the output of this function requires familiarity with the
        memory allocator internals.

        .. note::
            See :ref:`cuda-memory-management` for more details about GPU memory
            management.
        """
        snapshot = torch.cuda.memory_snapshot(self.id)
        return snapshot


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device: "Device" = None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch.cuda.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This context manager makes only current thread's allocations route to
        the given pool. If a new thread is spawned inside the context manager
        (e.g. by calling backward) the allocations in that thread will not
        route to the given pool.
    """
    device_index = (
        torch.cuda.current_device() if device is None else _get_device_index(device)
    )
    _cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)
    try:
        yield
    finally:
        _cuda_endAllocateToPool(device_index, pool.id)
        _cuda_releasePool(device_index, pool.id)
