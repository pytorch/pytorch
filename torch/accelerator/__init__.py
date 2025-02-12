r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from typing import Any, Optional
from typing_extensions import deprecated

import torch

from ._utils import _device_t, _get_device_index


__all__ = [
    "current_accelerator",
    "current_device_idx",  # deprecated
    "current_device_index",
    "current_stream",
    "device_count",
    "is_available",
    "set_device_idx",  # deprecated
    "set_device_index",
    "set_stream",
    "synchronize",
]


def device_count() -> int:
    r"""Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.
    """
    return torch._C._accelerator_deviceCount()


def is_available() -> bool:
    r"""Check if there is an available :ref:`accelerator<accelerators>`.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """
    return device_count() > 0


def current_accelerator() -> torch.device:
    r"""Return the device of the current :ref:`accelerator<accelerators>`.

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        And ensure to use :func:`torch.accelerator.is_available` to check if there is an available
        accelerator. If there is no available accelerator, this function will raise an exception.

    Example::

        >>> # xdoctest:
        >>> if torch.accelerator.is_available():
        >>>     current_device = torch.accelerator.current_accelerator()
        >>> else:
        >>>     current_device = torch.device("cpu")
        >>> if current_device.type == 'cuda':
        >>>     is_half_supported = torch.cuda.has_half
        >>> elif current_device.type == 'xpu':
        >>>     is_half_supported = torch.xpu.get_device_properties().has_fp16
        >>> elif current_device.type == 'cpu':
        >>>     is_half_supported = True
    """
    return torch._C._accelerator_getAccelerator()


def current_device_index() -> int:
    r"""Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """
    return torch._C._accelerator_getDeviceIndex()


current_device_idx = deprecated(
    "Use `current_device_index` instead.",
    category=FutureWarning,
)(current_device_index)


def set_device_index(device: _device_t, /) -> None:
    r"""Set the current device index to a given device.

    Args:
        device (:class:`torch.device`, str, int): a given device that must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if this device index is negative.
    """
    device_index = _get_device_index(device)
    torch._C._accelerator_setDeviceIndex(device_index)


set_device_idx = deprecated(
    "Use `set_device_index` instead.",
    category=FutureWarning,
)(set_device_index)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch.Stream"]

    def __init__(self, stream: Optional["torch.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if self.idx is None:
            self.idx = -1

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.accelerator.current_stream(None)

        # If the stream is not on the current device, then set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with torch.device(cur_stream.device):
                self.dst_prev_stream = torch.accelerator.current_stream(cur_stream.device)
        torch.accelerator.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device and destination device
        if self.src_prev_stream.device != cur_stream.device:
            torch.accelerator.set_stream(self.dst_prev_stream)
        torch.accelerator.set_stream(self.src_prev_stream)


def stream(stream: Optional["torch.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    return StreamContext(stream)


def current_stream(device: _device_t = None, /) -> torch.Stream:
    r"""Return the currently selected stream for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Stream: the currently selected stream for a given device.
    """
    device_index = _get_device_index(device, True)
    return torch._C._accelerator_getStream(device_index)


def set_stream(stream: torch.Stream) -> None:
    r"""Set the current stream to a given stream.

    Args:
        stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type.

    .. note:: This function will set the current device index to the device index of the given stream.
    """
    torch._C._accelerator_setStream(stream)


def synchronize(device: _device_t = None, /) -> None:
    r"""Wait for all kernels in all streams on the given device to complete.

    Args:
        device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
            the current :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the current :ref:`accelerator<accelerators>` is not initialized.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.accelerator.is_available() "No available accelerators detected."
        >>> start_event = torch.Event(enable_timing=True)
        >>> end_event = torch.Event(enable_timing=True)
        >>> start_event.record()
        >>> tensor = torch.randn(100, device=torch.accelerator.current_accelerator())
        >>> sum = torch.sum(tensor)
        >>> end_event.record()
        >>> torch.accelerator.synchronize()
        >>> elapsed_time_ms = start_event.elapsed_time(end_event)
    """
    device_index = _get_device_index(device, True)
    torch._C._accelerator_synchronizeDevice(device_index)
