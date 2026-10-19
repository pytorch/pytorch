from collections import OrderedDict
from typing import Any

import torch
from torch.types import Device

from ._utils import _get_device_index


__all__ = [
    "empty_cache",
    "empty_host_cache",
    "get_allocator_settings",
    "get_memory_info",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_allocator_settings",
]


def get_allocator_settings() -> dict[str, Any]:
    r"""get_allocator_settings() -> dict

    Return the current caching allocator configuration, including defaults.

    Keys match the options in ``PYTORCH_ALLOC_CONF``. Sizes with an ``_mb``
    suffix are in MiB; ``None`` denotes an unlimited size or unset reserve
    override. ``roundup_power2_divisions`` maps power-of-two MiB boundaries to
    division counts. Backend-specific options are included when registered.

    The result is an independent dictionary that can be passed to
    :func:`set_allocator_settings` to restore the configuration. Neither function
    initializes the accelerator runtime.

    .. note::
        This returns configured options, not the state of existing allocations
        or pools. Some options are sampled during initialization; see
        :func:`set_allocator_settings`.

    Examples::

        >>> saved = torch.accelerator.memory.get_allocator_settings()
        >>> torch.accelerator.memory.set_allocator_settings(saved)
    """
    return torch._C._accelerator_getAllocatorSettingsDict()


def set_allocator_settings(settings: dict[str, Any]) -> None:
    r"""set_allocator_settings(settings) -> None

    Update supplied caching allocator options, leaving omitted options unchanged.

    All values are validated before any option is updated. Options apply
    process-wide. Call this only while other threads are not allocating or
    freeing memory. The legacy string getter continues to return its last
    string input; dictionary updates do not change that value.

    Args:
        settings (dict): options using the keys and value types returned by
            :func:`get_allocator_settings`. ``roundup_power2_divisions`` also
            accepts an integer applied to every interval, or a dictionary with
            power-of-two MiB boundaries and an optional ``">"`` entry for larger
            allocations. ``None`` restores an unlimited size or the default
            ``expandable_segments_reserve``. Reserve values are fractions of
            device memory or strings such as ``"40G"`` for GiB.

    .. note::
        Updating configuration does not reconstruct existing allocator resources.
        The backend is fixed at import. ``per_process_memory_fraction`` supplies
        the initial device memory limit and the early OOM check; use
        :func:`torch.cuda.set_per_process_memory_fraction` to change an initialized
        device's limit. ``pinned_reserve_segment_size_mb`` is sampled when the
        pinned reserve is first created. Disabling ``pinned_use_background_threads``
        does not stop a background thread that has already started.

    Examples::

        >>> saved = torch.accelerator.memory.get_allocator_settings()
        >>> torch.accelerator.memory.set_allocator_settings({"max_split_size_mb": 128})
        >>> torch.accelerator.memory.set_allocator_settings(saved)
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")
    torch._C._accelerator_setAllocatorSettingsDict(settings)


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other applications.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    torch._C._accelerator_emptyCache()


def empty_host_cache() -> None:
    r"""Release all unoccupied cached host (pinned) memory currently held by the host caching
    allocator so that it can be used by other applications.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    torch._C._accelerator_emptyHostCache()


def _flatten_stats(result: list[tuple[str, Any]], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_stats(result, nested_prefix, nested_value)
    else:
        result.append((prefix, value))


def memory_stats(device_index: Device = None, /) -> OrderedDict[str, Any]:
    r"""Return a dictionary of accelerator device memory allocator statistics for a given device index.

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
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        OrderedDict[str, Any]: an ordered dictionary mapping statistic names to their values.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return OrderedDict()
    device_index = _get_device_index(device_index, optional=True)
    stats = torch._C._accelerator_getDeviceStats(device_index)
    flat_stats = []

    _flatten_stats(flat_stats, "", stats)
    flat_stats.sort()
    # pyrefly: ignore [no-matching-overload]
    return OrderedDict(flat_stats)


def memory_allocated(device_index: Device = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device_index: Device = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device index.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory occupied by live tensors (in bytes) within the current process.
    """
    return memory_stats(device_index).get("allocated_bytes.all.peak", 0)


def memory_reserved(device_index: Device = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device_index: Device = None, /) -> int:
    r"""Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device index.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory reserved by PyTorch (in bytes) within the current process.
    """
    return memory_stats(device_index).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device_index: Device = None, /) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetAccumulatedStats(device_index)


def reset_peak_memory_stats(device_index: Device = None, /) -> None:
    r"""Reset the "peak" stats tracked by the current :ref:`accelerator<accelerators>`
    memory allocator for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """
    if not torch._C._accelerator_isAllocatorInitialized():
        return
    device_index = _get_device_index(device_index, optional=True)
    return torch._C._accelerator_resetPeakStats(device_index)


def get_memory_info(device_index: Device = None, /) -> tuple[int, int]:
    r"""Return the current device memory information for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        tuple[int, int]: a tuple of two integers (free_memory, total_memory) in bytes.
            The first value is the free memory on the device (available across all processes and applications),
            The second value is the device's total hardware memory capacity.
    """
    device_index = _get_device_index(device_index, optional=True)
    # pyrefly: ignore [missing-attribute]
    return torch._C._accelerator_getMemoryInfo(device_index)


def _snapshot(device=None, augment_with_fx_traces: bool = False):
    r"""Return a snapshot of the current :ref:`accelerator<accelerators>` memory allocator state.

    Requires :func:`_record_memory_history` on the appropriate device module
    (e.g., :func:`torch.cuda.memory._record_memory_history`) to have been called.

    Args:
        device: the device to snapshot. If not given, uses the current device.
        augment_with_fx_traces (bool, optional): if True, augment stack traces
            with FX graph information. Default: ``False``.

    Returns:
        dict: a dictionary containing memory allocator state information.
    """
    acc = torch.accelerator.current_accelerator()
    if acc is not None and acc.type == "xpu":
        return torch.xpu.memory._snapshot(
            device, augment_with_fx_traces=augment_with_fx_traces
        )
    return torch.cuda.memory._snapshot(
        device, augment_with_fx_traces=augment_with_fx_traces
    )
