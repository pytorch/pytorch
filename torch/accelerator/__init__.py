r"""
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

from typing import Literal, Optional
from typing_extensions import deprecated

import torch

from ._utils import _device_t, _get_device_index


__all__ = [
    "current_accelerator",
    "current_device_idx",  # deprecated
    "current_device_index",
    "current_stream",
    "device_count",
    "device_index",
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

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT posion fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """
    acc = current_accelerator()
    if acc is None:
        return 0

    mod = torch.get_device_module(acc)
    return mod.device_count()


def is_available() -> bool:
    r"""Check if the current accelerator is available at runtime: it was build, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """
    # Why not just check "device_count() > 0" like other is_available call?
    # Because device like CUDA have a python implementation of is_available that is
    # non-poisoning and some features like Dataloader rely on it.
    # So we are careful to delegate to the Python version of the accelerator here
    acc = current_accelerator()
    if acc is None:
        return False

    mod = torch.get_device_module(acc)
    return mod.is_available()


def current_accelerator(check_available: bool = False) -> Optional[torch.device]:
    r"""Return the device of the accelerator available at compilation time.
    If no accelerator were available at compilation time, returns None.
    See :ref:`accelerator<accelerators>` for details.

    Args:
        check_available (bool, optional): if True, will also do a runtime check to see
            if the device :func:`torch.accelerator.is_available` on top of the compile-time
            check.
            Default: ``False``

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        This API does NOT poison fork. For more details, see :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> # xdoctest:
        >>> # If an accelerator is available, sent the model to it
        >>> model = torch.nn.Linear(2, 2)
        >>> if (current_device := current_accelerator(check_available=True)) is not None:
        >>>     model.to(current_device)
    """
    if (acc := torch._C._accelerator_getAccelerator()) is not None:
        if (not check_available) or (check_available and is_available()):
            return acc
    return None


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


class device_index:
    r"""Context manager to set the current device index for the current :ref:`accelerator<accelerators>`.
    Temporarily changes the current device index to the specified value for the duration
    of the context, and automatically restores the previous device index when exiting
    the context.

    Args:
        device (Optional[int]): a given device index to temporarily set. If None,
            no device index switching occurs.

    Examples:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Set device 0 as the current device temporarily
        >>> with torch.accelerator.device_index(0):
        ...     # Code here runs with device 0 as the current device
        ...     pass
        >>> # Original device is now restored
        >>> # No-op when None is passed
        >>> with torch.accelerator.device_index(None):
        ...     # No device switching occurs
        ...     pass
    """

    def __init__(self, device: Optional[int], /) -> None:
        self.idx = device
        self.prev_idx = -1

    def __enter__(self) -> None:
        if self.idx is not None:
            self.prev_idx = torch._C._accelerator_exchangeDevice(self.idx)

    def __exit__(self, *args: object) -> Literal[False]:
        if self.idx is not None:
            torch._C._accelerator_maybeExchangeDevice(self.prev_idx)
        return False
