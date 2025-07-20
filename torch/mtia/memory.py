# pyre-strict

r"""This package adds support for device memory management implemented in MTIA."""

from typing import Any, Optional

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


def memory_allocated(device: Optional[_device_t] = None) -> int:
    r"""Return the current MTIA memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mtia.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("allocated_bytes", 0)


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


__all__ = [
    "memory_stats",
    "max_memory_allocated",
    "memory_allocated",
    "reset_peak_memory_stats",
]
