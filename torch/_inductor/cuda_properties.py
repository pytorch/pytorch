import functools

import torch


# API to query cuda properties that will work in a triton compile process
# that cannot use the GPU APIs (due to processing fork() and initialization
# time issues). Properties are recorded in the main process before
# we fork the workers.


@functools.lru_cache(None)
def _properties():
    r = {
        i: torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())
    }
    return r


_compile_worker_current_device = None


def set_compiler_worker_current_device(device):
    global _compile_worker_current_device
    _compile_worker_current_device = device


def current_device():
    if _compile_worker_current_device is not None:
        return _compile_worker_current_device
    return torch.cuda.current_device()


def _device(device):
    if device is not None:
        if isinstance(device, torch.device):
            assert device.type == "cuda"
            device = device.index
        return device
    return current_device()


def get_device_properties(device=None):
    return _properties()[_device(device)]


def get_device_capability(device=None):
    p = get_device_properties(device)
    return p.major, p.minor
