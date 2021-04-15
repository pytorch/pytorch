import collections
import contextlib
import warnings
from typing import Any, Dict, Union

import torch
from . import is_initialized, _get_device_index, _lazy_init
from torch.types import Device

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


def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    r"""Performs a memory allocation using the CUDA memory allocator.

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
        raise TypeError('Invalid type for stream argument, must be '
                        '`torch.cuda.Stream` or `int` representing a pointer '
                        'to a exisiting stream')
    with torch.cuda.device(device):
        return torch._C._cuda_cudaCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Deletes memory allocated using the CUDA memory allocator.

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


def set_per_process_memory_fraction(fraction, device: Union[Device, int] = None) -> None:
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
        raise TypeError('Invalid type for fraction argument, must be `float`')
    if fraction < 0 or fraction > 1:
        raise ValueError('Invalid fraction value: {}. '
                         'Allowed range: 0~1'.format(fraction))

    torch._C._cuda_setMemoryFraction(fraction, device)


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
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


def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns a dictionary of CUDA memory allocator statistics for a
    given device.

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

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
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


def memory_stats_as_nested_dict(device: Union[Device, int] = None) -> Dict[str, Any]:
    r"""Returns the result of :func:`~torch.cuda.memory_stats` as a nested dictionary."""
    if not is_initialized():
        return {}
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryStats(device)


def reset_accumulated_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "accumulated" (historical) stats tracked by the CUDA memory allocator.

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


def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    r"""Resets the "peak" stats tracked by the CUDA memory allocator.

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


def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum GPU memory occupied by
    tensors for a given device.

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
        FutureWarning)
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    r"""Resets the starting point in tracking maximum GPU memory managed by the
    caching allocator for a given device.

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
        FutureWarning)
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes for a given
    device.

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


def max_memory_allocated(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GPU memory occupied by tensors in bytes for a given
    device.

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


def memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the current GPU memory managed by the caching allocator in bytes
    for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Union[Device, int] = None) -> int:
    r"""Returns the maximum GPU memory managed by the caching allocator in bytes
    for a given device.

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


def memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.memory_reserved`."""
    warnings.warn(
        "torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved",
        FutureWarning)
    return memory_reserved(device=device)


def max_memory_cached(device: Union[Device, int] = None) -> int:
    r"""Deprecated; see :func:`~torch.cuda.max_memory_reserved`."""
    warnings.warn(
        "torch.cuda.max_memory_cached has been renamed to torch.cuda.max_memory_reserved",
        FutureWarning)
    return max_memory_reserved(device=device)


def memory_snapshot():
    r"""Returns a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    return torch._C._cuda_memorySnapshot()


def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.

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
        prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return "{:7d} {}".format(sz, prefix)

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return "{:7d} {} ".format(cnt, prefix)

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
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
    lines.append("  {_:9} CUDA OOMs: {num_ooms:<12d} | {_:6} cudaMalloc retries: {num_alloc_retries:<8d}  ")
    lines.append("=" * 75)
    lines.append("        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  ")

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = None, None, None, None

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

            lines.append(" {:<21} | {} | {} | {} | {} ".format(
                submetric_name,
                formatter(current, current_prefval),
                formatter(peak, peak_prefval),
                formatter(allocated, allocated_prefval),
                formatter(freed, freed_prefval)),
            )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def list_gpu_processes(device: Union[Device, int] = None) -> str:
    r"""Returns a human-readable printout of the running processes
    and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """

    try:
        import pynvml  # type: ignore
    except ModuleNotFoundError:
        return("pynvml module not found, please install pynvml")
    from pynvml import NVMLError_DriverNotLoaded
    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        return ("cuda driver can't be loaded, is cuda enabled?")
    device = _get_device_index(device, optional=True)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    lines = []
    lines.append(f"GPU:{device}")
    if len(procs) == 0:
        lines.append("no processes are running")
    for p in procs:
        mem = p.usedGpuMemory / (1024 * 1024)
        lines.append(f"process {p.pid:>10d} uses {mem:>12.3f} MB GPU memory")
    return "\n".join(lines)
