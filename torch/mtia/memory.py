# pyre-strict

r"""This package adds support for device memory management implemented in MTIA."""

import pickle
from typing import Any, Callable, Optional

import torch

from . import _device_t, is_initialized
from ._utils import _get_device_index


def memory_stats(device: Optional[_device_t] = None) -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return {}
    return torch._C._mtia_memoryStats(_get_device_index(device, optional=True))


def max_memory_allocated(device: Optional[_device_t] = None) -> int:
    r"""Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("peak_bytes", 0)


def reset_peak_memory_stats(device: Optional[_device_t] = None) -> None:
    r"""Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return
    torch._C._mtia_resetPeakMemoryStats(_get_device_index(device, optional=True))


def record_memory_history(
    enabled: Optional[str] = "all", stacks: str = "python", max_entries: int = 0
) -> None:
    r"""Enable/Disable the memory profiler on MTIA allocator

    Args:
        enabled (all or state, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).

        stacks ("python" or "cpp", optional). Select the stack trace to record.

        max_entries (int, optional). Maximum number of entries to record.
    """
    if not is_initialized():
        return
    torch._C._mtia_recordMemoryHistory(enabled, stacks, max_entries)


def snapshot() -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator history"""

    return torch._C._mtia_memorySnapshot()


def dump_snapshot(filename: str = "dump_snapshot.pickle") -> None:
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = snapshot()
    with open(filename, "wb") as f:
        pickle.dump(s, f)


def attach_out_of_memory_observer(
    observer: Callable[[int, int, int, int], None]
) -> None:
    r"""Attach an out-of-memory observer to MTIA memory allocator"""
    torch._C._mtia_attachOutOfMemoryObserver(observer)


__all__ = [
    "memory_stats",
    "max_memory_allocated",
    "reset_peak_memory_stats",
    "dump_snapshot",
    "record_memory_history",
    "snapshot",
    "attach_out_of_memory_observer",
]
