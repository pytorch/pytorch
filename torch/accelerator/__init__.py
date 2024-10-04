from typing import Any
import torch
from ._utils import _device_t, _get_device_index

def current_accelerator() -> str:
    return torch._C._accelerator_getAccelerator()

def device_count() -> int:
    return torch._C._accelerator_deviceCount()

def is_available() -> bool:
    return device_count() > 0

def current_device() -> int:
    return torch._C._accelerator_getDevice()

def set_device(device: _device_t) -> None:
    device_index = _get_device_index(device)
    return torch._C._accelerator_setDevice(device_index)

def current_stream(device: _device_t = None) -> torch.Stream:
    device_index = _get_device_index(device, True)
    return torch._C._accelerator_getStream(device_index)

def set_stream(stream: torch.Stream) -> None:
    return torch._C._accelerator_setStream(stream)

def synchronize(device: _device_t = None) -> None:
    device_index = _get_device_index(device, True)
    return torch._C._accelerator_synchronizeDevice(device_index)

class DeviceGuard:
    def __init__(self, device_index: int):
        self.idx = device_index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch._C._accelerator_exchangeDevice(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch._C._accelerator_maybeExchangeDevice(self.prev_idx)
        return False


class StreamGuard:
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
    "current_accelerator",
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "current_stream",
    "set_stream",
    "synchronize",
    "DeviceGuard",
    "StreamGuard",
]
