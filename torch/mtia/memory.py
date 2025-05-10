# pyre-strict

r"""This package adds support for device memory management implemented in MTIA."""

from typing import Any, Optional

import torch

from . import is_initialized
from ._utils import _get_device_index


def memory_stats(device: Optional[torch.types.Device] = None) -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return {}
    return torch._C._mtia_memoryStats(_get_device_index(device, optional=True))


def max_memory_allocated(device: Optional[torch.types.Device] = None) -> int:
    r"""Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("peak_bytes", 0)


def reset_peak_memory_stats(device: Optional[torch.types.Device] = None) -> None:
    r"""Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return
    torch._C._mtia_resetPeakMemoryStats(_get_device_index(device, optional=True))


__all__ = [
    "memory_stats",
    "max_memory_allocated",
    "reset_peak_memory_stats",
]
