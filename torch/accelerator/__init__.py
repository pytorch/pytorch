# mypy: allow-untyped-defs
r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from typing import Any

import torch

from ._utils import _device_t, _get_device_index


def current_accelerator() -> str:
    r"""Return the device type of the current :ref:`accelerator<accelerators>`.

    Returns:
        str: the device type of the current accelerator. If no available accelerators,
            return cpu device type.

    Example::

        >>> if torch.acc.current_accelerator() == 'cuda':
        >>>     stream = torch.cuda.default_stream()
        >>> else:
        >>>     stream = torch.Stream()
    """
    return torch._C._accelerator_getAccelerator()


def device_count() -> int:
    r"""Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If no available accelerators, return 0.
    """
    return torch._C._accelerator_deviceCount()


def is_available() -> bool:
    r"""Check if there is an available :ref:`accelerator<accelerators>`.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    Example::

        >>> assert torch.acc.is_available() "No available accelerators detected."
    """
    return device_count() > 0


def current_device() -> int:
    r"""Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """
    return torch._C._accelerator_getDevice()


def set_device(device: _device_t) -> None:
    r"""Set the current device to a given device.

    Args:
        device (:class:`torch.device`, str, int): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. This function is a no-op if this argument is negative.
    """
    device_index = _get_device_index(device)
    torch._C._accelerator_setDevice(device_index)


def current_stream(device: _device_t = None) -> torch.Stream:
    r"""Return the currently selected stream for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given, use :func:`torch.acc.current_device` by default.
    Returns:
        torch.Stream: the currently selected stream for a given device.
    """
    device_index = _get_device_index(device, True)
    return torch._C._accelerator_getStream(device_index)


def set_stream(stream: torch.Stream) -> None:
    r"""Set the current stream to a given stream.

    Args:
        stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type.
            This function will set the current device to the device of the given stream.
    """
    torch._C._accelerator_setStream(stream)


def synchronize(device: _device_t = None) -> None:
    r"""Wait for all kernels in all streams on the given device to complete.

    Args:
        device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
            the current :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.acc.current_device` by default.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.acc.is_available() "No available accelerators detected."
        >>> start_event = torch.Event(enable_timing=True)
        >>> end_event = torch.Event(enable_timing=True)
        >>> start_event.record()
        >>> tensor = torch.randn(100, device=torch.acc.current_accelerator())
        >>> sum = torch.sum(tensor)
        >>> end_event.record()
        >>> torch.acc.synchronize()
        >>> elapsed_time_ms = start_event.elapsed_time(end_event)
    """
    device_index = _get_device_index(device, True)
    torch._C._accelerator_synchronizeDevice(device_index)


class DeviceGuard:
    r"""
    Instances of :class:`DeviceGuard` serve as context managers that allow regions of with statement to run
    in the given device index of the current :ref:`accelerator<accelerators>`. And switch back to the device
    that was originally selected upon invocation.

    Args:
        device_index (int): a given device index of the current :ref:`accelerator<accelerators>`.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
        >>> assert torch.acc.is_available() "No available accelerators detected."
        >>> assert torch.acc.device_count() > 1 "No multi-devices detected."
        >>> orig_device = 0
        >>> target_device = 1
        >>> with torch.acc.DeviceGuard(target_device):
        >>>     a = torch.randn(10, device=torch.acc.current_accelerator())
        >>>     sum = torch.sum(a)
        >>>     assert sum.device.index == target_device
        >>>     assert torch.acc.current_device() == target_device
        >>> assert torch.acc.current_device() == orig_device
        >>> sum = sum.to(device=torch.acc.current_accelerator())
        >>> assert sum.device.index == orig_device
    """

    def __init__(self, device_index: int):
        self.idx = device_index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch._C._accelerator_exchangeDevice(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch._C._accelerator_maybeExchangeDevice(self.prev_idx)
        return False


class StreamGuard:
    r"""
    Instances of :class:`StreamGuard` serve as context managers that allow regions of with statement to run
    in the given stream. And switch back to the stream that was originally selected upon invocation.

    Args:
        stream (:class:`torch.Stream`): a given stream that must match the current
            :ref:`accelerator<accelerators>` device type.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.acc.is_available() "No available accelerators detected."
        >>> s1 = torch.Stream()
        >>> s2 = torch.Stream()
        >>> torch.acc.set_stream(s1)
        >>> with torch.acc.StreamGuard(s2):
        >>>     a = torch.randn(10, device=torch.acc.current_accelerator())
        >>>     assert torch.acc.current_stream() == s2
        >>> s1.wait_stream(s2)
        >>> sum = torch.sum(a)
        >>> assert torch.acc.current_stream() == s1
    """

    def __init__(self, stream: torch.Stream):
        self.stream = stream
        self.src_prev_stream = None
        self.dst_prev_stream = None

    def __enter__(self):
        self.src_prev_stream = torch.acc.current_stream()  # type: ignore[assignment]

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != self.stream.device:  # type: ignore[attr-defined]
            with DeviceGuard(self.stream.device.index):
                self.dst_prev_stream = torch.acc.current_stream()  # type: ignore[assignment]
        torch.acc.set_stream(self.stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Reset the stream on the original device and destination device
        if self.src_prev_stream.device != self.stream.device:  # type: ignore[attr-defined]
            torch.acc.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.acc.set_stream(self.src_prev_stream)  # type: ignore[arg-type]
        return False


__all__ = [
    "DeviceGuard",
    "StreamGuard",
    "current_accelerator",
    "current_device",
    "current_stream",
    "device_count",
    "is_available",
    "set_device",
    "set_stream",
    "synchronize",
]
