import collections
import ctypes
import pickle
import sys
from typing import Any, Literal

import torch
from torch._utils import _augment_memory_snapshot_stack_traces, _dummy_type
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


def memory_snapshot(
    mempool_id: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    r"""
    Return a snapshot of the XPU memory allocator state across all devices.
    Provides detailed information for each memory segment managed by the allocator
    including its size, owning pool, associated stream, call stack traces, and other relevant attributes.

    Arguments:
        mempool_id (tuple[int, int] or None, optional): The memory pool id. If None, the default memory pool is used.

    Returns:
        list[dict[str, Any]]: List of memory segments and their attributes.
    """
    if not is_initialized():
        return []
    # pyrefly: ignore [missing-attribute]
    return torch._C._xpu_memorySnapshot(mempool_id)["segments"]


def _snapshot(device: Device = None, augment_with_fx_traces: bool = False):
    """
    Capture a snapshot of the XPU memory state at the time this function is called.

    The returned snapshot is a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments: List[Segment]
            device_traces: List[List[TraceEntry]]


        class Segment(TypedDict):
            # A Segment represents a contiguous memory region returned by the SYCL runtime.
            #
            # All reserved memory is composed of these segments. Segments are
            # cached and reused by the allocator. When allocations are smaller
            # than the segment, the segment may be split into multiple Blocks.
            #
            # Calling :func:`~torch.xpu.memory.empty_cache` releases segments that are entirely inactive.
            address: int
            total_size: int  #  total size of segment
            stream: int
            segment_type: Literal["small", "large"]  # 'large' (>1MB)
            allocated_size: int  # size of memory in use
            active_size: int  # size of memory in use or in active_awaiting_free state
            blocks: List[Block]


        class Block(TypedDict):
            # A sub-region of a Segment, either currently allocated or cached for reuse.
            size: int
            requested_size: int  # Original requested size (may be smaller than `size`)
            address: int
            state: Literal[
                "active_allocated",  # used by a tensor
                "active_awaiting_free",  # waiting for another stream synchronization, then become free
                "inactive",  # free for reuse
            ]
            frames: List[Frame]  # stack trace from where the allocation occurred


        class Frame(TypedDict):
            filename: str
            line: int
            name: str
            # Optional fields when `augment_with_fx_traces=True` and the frame
            # corresponds to FX-generated code.
            fx_node_op: str  # FX node operation type (e.g., 'call_function', 'output')
            fx_node_name: str  # FX node name (e.g., 'linear', 'relu_1')
            fx_original_trace: str  # Original model source code stack trace


        class TraceEntry(TypedDict):
            # Trace entries are recorded only when :func:`~torch.xpu.memory._record_memory_history` is enabled.
            action: Literal[
                "alloc"  # memory allocated
                "free_requested",  # received a call to free memory
                "free_completed",  # memory reclaimed and reusable
                "segment_alloc",  # ask SYCL runtime for more memory
                "segment_free",  # called SYCL runtime to return memory to XPU
                "segment_map",  # ask SYCL runtime to map memory
                "segment_unmap",  # called SYCL runtime to unmap memory
                "snapshot",  # snapshot taken
                "oom",  # threw an OOM exception
            ]
            addr: int  # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int  # only present for OOM, the amount of free memory reported by the device

    Arguments:
        device (torch.device or int or str, optional): selected device. It uses the current device,
            given by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None`` (default).
        augment_with_fx_traces (bool, optional): If True, augment stack trace frames with FX debug information
            that maps generated FX code back to original model source code. This adds the FX-related
            fields (fx_node_op, fx_node_name, fx_original_trace) to Frame objects. Default is ``False``.

    Returns:
        The Snapshot dictionary object
    """
    # pyrefly: ignore [missing-attribute]
    s = torch._C._xpu_memorySnapshot(None)
    if augment_with_fx_traces:
        s = _augment_memory_snapshot_stack_traces(s)  # type: ignore[assignment, arg-type]
    return s


def _dump_snapshot(
    filename: str = "dump_snapshot.pickle", augment_with_fx_traces: bool = False
) -> None:
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Snapshot file sizes scale with `max_entries` and stack trace depth per entry,
    with several KB per entry. These can easily be in the GB range for longer running
    workflows with large `max_entries`.

    Arguments:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
        augment_with_fx_traces (bool, optional): If True, augment the snapshot with FX debug information
            before dumping. This maps generated FX code stack traces back to original model
            source code. Defaults to ``False``.
    """
    s = _snapshot(augment_with_fx_traces=augment_with_fx_traces)

    with open(filename, "wb") as f:
        pickle.dump(s, f)


def _record_memory_history(
    enabled: Literal["state", "all"] | None = "all",
    context: Literal["state", "alloc", "all"] | None = "all",
    stacks: Literal["python", "all"] = "all",
    max_entries: int = sys.maxsize,
    clear_history: bool = False,
    skip_actions: list[str] | None = None,
) -> None:
    """
    Enable recording of stack traces associated with memory allocations, so you can
    tell what allocated any piece of memory in :func:`~torch.xpu.memory._snapshot()`.

    In addition to keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`~torch.xpu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    Buffer behavior
    ---------------

    This will store up to `max_entries` instances of `TraceEntry` when enabled.
    Python trace collection defaults to `sys.maxsize`, meaning long-running
    or indefinitely running jobs should set a reasonable limit to avoid excessive
    memory use. Expect each entry to be several KB.

    Longer running workflows or those with smaller `max_entries` values will only
    store the last accumulated `max_entries` entries, meaning new entries overwrite
    older entries, reference to ring buffer behavior.

    Latency impact
    --------------

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Arguments:
        enabled (Literal["state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currently allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal["state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks.
            `"all"`, additionally include C++ frames.
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
        clear_history (bool, optional): Clear history when enabling, defaults to ``False``.
        skip_actions (list[str], optional): List of action types to skip when recording
            memory history. This can be used to reduce memory overhead by excluding
            certain types of events from being recorded. Valid action types are:

            - `"alloc"`: Memory allocation events
            - `"free_requested"`: Free requests (memory marked for freeing)
            - `"free_completed"`: Completed free operations (memory actually freed)
            - `"segment_alloc"`: Segment allocation from SYCL runtime
            - `"segment_free"`: Segment freed back to XPU via SYCL runtime
            - `"segment_map"`: Segment map events
            - `"segment_unmap"`: Segment unmap events
            - `"snapshot"`: Memory snapshot generation events
            - `"oom"`: Out-of-memory exceptions

            For example, to skip recording free_requested events:
            `skip_actions=["free_requested"]`

            Defaults to ``None`` (record all actions).
    """
    # pyrefly: ignore [missing-attribute]
    torch._C._xpu_recordMemoryHistory(
        enabled,
        context,
        stacks,
        max_entries,
        clear_history,
        skip_actions if skip_actions is not None else [],
    )


class _XPUAllocator:
    r"""Wrapper over internal XPU memory allocators."""

    # pyrefly: ignore [missing-attribute]
    def __init__(self, allocator: torch._C._xpu_XPUAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class XPUPluggableAllocator(_XPUAllocator):
    r"""
    XPU memory allocator loaded dynamically from a shared library.

    This lets users provide custom allocation and free functions implemented
    in a separate shared library. The allocator is registered and could become
    available for use via :func:`~torch.xpu.memory.change_current_allocator`.

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

                ``void free_fn(void* ptr, size_t size, int device, sycl::queue* queue);``
    """

    def __init__(self, path_to_lib_file: str, alloc_fn_name: str, free_fn_name: str):
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
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "set_per_process_memory_fraction",
]
