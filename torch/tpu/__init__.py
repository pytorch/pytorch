r"""
This package introduces support for the TPU backend.

The TPU support is mainly provided by the torch_tpu package.
"""

import torch
from typing import Any


try:
    from torch_tpu import api as tpu_api  # type: ignore[import]
    from torch_tpu.api.streams import (  # type: ignore[import]
        TpuEvent as Event,
        TpuStream as Stream,
    )

    _device_mod = tpu_api._device_module._DeviceModule
except (ImportError, AttributeError):
    _device_mod = None
    Stream = None  # type: ignore[assignment, misc]
    Event = None  # type: ignore[assignment, misc]


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    if _device_mod is None:
        return 0
    return _device_mod.current_device()


def device_count() -> int:
    r"""Return the number of TPU devices available."""
    if _device_mod is None:
        return 0
    return _device_mod.device_count()


def is_available() -> bool:
    r"""Return whether TPU is available."""
    if _device_mod is None:
        return False
    return _device_mod.is_available()


def stream(stream: Stream | None) -> Any:  # type: ignore[arg-type]
    r"""Wrap around the context manager that selects a given stream.

    Args:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    if _device_mod is None:
        raise RuntimeError("torch_tpu is not available")
    return _device_mod.stream(stream)


def current_stream(device: torch.device = None) -> Stream:  # type: ignore[return-value]
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (int, optional): selected device. Returns the currently selected
            :class:`Stream` for the current device, given by
            :func:`~torch.tpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    if _device_mod is None:
        raise RuntimeError("torch_tpu is not available")
    return _device_mod.current_stream(device)


def set_stream(stream: Stream) -> None:  # type: ignore[arg-type]
    r"""Set the current stream.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if _device_mod is None:
        raise RuntimeError("torch_tpu is not available")
    _device_mod.set_stream(stream)
