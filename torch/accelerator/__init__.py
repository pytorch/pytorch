from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch.types import Device

_device_t = Device | str | int | None


__all__ = [
    "current_accelerator",
    "current_device_index",
    "device_count",
    "device_index",
    "get_device_capability",
    "is_available",
    "set_device_index",
    "stream",
    "synchronize",
]


class device_index:
    """
    Context manager for temporarily switching accelerator devices.

    Example:
        >>> # xdoctest: +SKIP("requires accelerator runtime")
        >>> with device_index(0) as dev:
        ...     print(dev.idx)
    """

    def __init__(self, idx: int | None):
        self.idx = idx
        self.prev_idx: int = -1

    def __enter__(self) -> "device_index":
        """
        Enter the accelerator device context.

        Returns:
            device_index: current context manager instance.
        """

        if self.idx is not None:
            self.prev_idx = torch._C._accelerator_exchangeDevice(self.idx)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        """
        Restore the previous accelerator device.
        """

        if self.idx is not None:
            # Prevent invalid restoration if __enter__ failed
            # before prev_idx assignment.
            if self.prev_idx >= 0:
                torch._C._accelerator_maybeExchangeDevice(self.prev_idx)

        return None



def is_available() -> bool:
    """
    Returns whether an accelerator backend is available.

    Example:
        >>> assert torch.accelerator.is_available(), (
        ...     "No available accelerators detected."
        ... )
    """

    return torch._C._accelerator_isAvailable()



def current_accelerator(check_available: bool = False):
    """
    Returns the current accelerator backend.

    Args:
        check_available (bool):
            If True, validates backend availability first.

    Example:
        >>> # xdoctest: +SKIP("requires accelerator runtime")
        >>> current_accelerator()
    """

    if not check_available or is_available():
        return torch._C._accelerator_getAccelerator()

    return None



def current_device_index() -> int:
    """
    Returns the current accelerator device index.
    """

    return torch._C._accelerator_getDeviceIndex()



def set_device_index(device: int) -> None:
    """
    Sets the active accelerator device.
    """

    torch._C._accelerator_setDeviceIndex(device)



def device_count() -> int:
    """
    Returns number of available accelerator devices.
    """

    return torch._C._accelerator_deviceCount()



def synchronize() -> None:
    """
    Synchronize accelerator execution.

    Example:
        >>> assert torch.accelerator.is_available(), (
        ...     "No available accelerators detected."
        ... )
        >>> torch.accelerator.synchronize()
    """

    torch._C._accelerator_synchronizeDevice()


class stream:
    """
    Accelerator stream context manager.
    """

    def __init__(self, stream_obj: Any):
        self.stream_obj = stream_obj
        self.prev_stream = None

    def __enter__(self) -> "stream":
        self.prev_stream = torch._C._accelerator_getCurrentStream()
        torch._C._accelerator_setStream(self.stream_obj)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        if self.prev_stream is not None:
            torch._C._accelerator_setStream(self.prev_stream)

        return None


@lru_cache(maxsize=None)
def get_device_capability(device: _device_t = None, /) -> dict[str, Any]:
    """
    Returns accelerator device capability information.

    Args:
        device:
            Accelerator device identifier.

    Returns:
        dict[str, Any]:
            Capability information.

    Note:
        Tests using mocked hardware should clear cache:

            get_device_capability.cache_clear()
    """

    return torch._C._accelerator_getDeviceCapability(device)
