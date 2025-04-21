# mypy: allow-untyped-defs
r"""
This package implements abstractions found in ``torch.cuda``
to facilitate writing device-agnostic code.
"""

from contextlib import AbstractContextManager
from typing import Any, Optional, Union

import torch

from .. import device as _device
from . import amp


__all__ = [
    "is_available",
    "is_initialized",
    "synchronize",
    "current_device",
    "current_stream",
    "stream",
    "set_device",
    "device_count",
    "Stream",
    "StreamContext",
    "Event",
]

_device_t = Union[_device, str, int, None]


def _is_avx2_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX2."""
    return torch._C._cpu._is_avx2_supported()


def _is_avx512_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512."""
    return torch._C._cpu._is_avx512_supported()


def _is_avx512_bf16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AVX512_BF16."""
    return torch._C._cpu._is_avx512_bf16_supported()


def _is_vnni_supported() -> bool:
    r"""Returns a bool indicating if CPU supports VNNI."""
    # Note: Currently, it only checks avx512_vnni, will add the support of avx2_vnni later.
    return torch._C._cpu._is_avx512_vnni_supported()


def _is_amx_tile_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX_TILE."""
    return torch._C._cpu._is_amx_tile_supported()


def _is_amx_fp16_supported() -> bool:
    r"""Returns a bool indicating if CPU supports AMX FP16."""
    return torch._C._cpu._is_amx_fp16_supported()


def _init_amx() -> bool:
    r"""Initializes AMX instructions."""
    return torch._C._cpu._init_amx()


def is_available() -> bool:
    r"""Returns a bool indicating if CPU is currently available.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return True


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


class Stream:
    """
    N.B. This class only exists to facilitate device-agnostic code
    """

    def __init__(self, priority: int = -1) -> None:
        pass

    def wait_stream(self, stream) -> None:
        pass

    def record_event(self) -> None:
        pass

    def wait_event(self, event) -> None:
        pass


class Event:
    def query(self) -> bool:
        return True

    def record(self, stream=None) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(self, stream=None) -> None:
        pass


_default_cpu_stream = Stream()
_current_stream = _default_cpu_stream


def current_stream(device: _device_t = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """
    return _current_stream


class StreamContext(AbstractContextManager):
    r"""Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

    cur_stream: Optional[Stream]

    def __init__(self, stream):
        self.stream = stream
        self.prev_stream = _default_cpu_stream

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        self.prev_stream = _current_stream
        _current_stream = cur_stream

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream


def stream(stream: Stream) -> AbstractContextManager:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return StreamContext(stream)


def device_count() -> int:
    r"""Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return 1


def set_device(device: _device_t) -> None:
    r"""Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


def current_device() -> str:
    r"""Returns current device for cpu. Always 'cpu'.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return "cpu"


def is_initialized() -> bool:
    r"""Returns True if the CPU is initialized. Always True.

    N.B. This function only exists to facilitate device-agnostic code
    """
    return True
