import functools
from typing import Dict, Optional, Tuple, Union

import torch
from torch.cuda import _CudaDeviceProperties

# API to query cuda properties that will work in a triton compile process
# that cannot use the GPU APIs (due to processing fork() and initialization
# time issues). Properties are recorded in the main process before
# we fork the workers.

_compile_worker_current_device: Optional[int] = None


@functools.lru_cache(None)
def _properties() -> Dict[int, _CudaDeviceProperties]:
    if not torch.cuda.is_available():
        return {}
    try:
        return {
            i: torch.cuda.get_device_properties(i)
            for i in range(torch.cuda.device_count())
        }
    except RuntimeError:
        return {}


def set_compiler_worker_current_device(device: int) -> None:
    global _compile_worker_current_device
    _compile_worker_current_device = device


def current_device() -> int:
    if _compile_worker_current_device is not None:
        return _compile_worker_current_device
    return torch.cuda.current_device()


def _device(device: Optional[Union[torch.device, int]]) -> int:
    if device is not None:
        if isinstance(device, torch.device):
            assert device.type == "cuda"
            device = device.index
        return device
    return current_device()


def get_device_properties(
    device: Optional[Union[torch.device, int]] = None
) -> _CudaDeviceProperties:
    return _properties()[_device(device)]


def get_device_capability(
    device: Optional[Union[torch.device, int]] = None
) -> Tuple[int, int]:
    p = get_device_properties(device)
    return p.major, p.minor
