import functools

import torch

from .triton_backend import all_triton_backend_name, triton_backends


# API to query cuda properties that will work in a triton compile process
# that cannot use the GPU APIs (due to processing fork() and initialization
# time issues). Properties are recorded in the main process before
# we fork the workers.


@functools.lru_cache(None)
def _properties():
    device_properties = {}

    for _triton_backend in triton_backends:
        if not _triton_backend:
            continue

        try:
            device_properties[_triton_backend.name()] = {
                i: _triton_backend.get_device_properties(i)
                for i in range(_triton_backend.device_count())
            }
        except RuntimeError:
            return {}

    return device_properties


_compile_worker_current_device = None


def set_compiler_worker_current_device(device):
    global _compile_worker_current_device
    _compile_worker_current_device = device


def _device(device):
    assert device is not None
    assert isinstance(device, torch.device)
    assert device.type in all_triton_backend_name()
    return device.index


def get_device_properties(device: torch.device = None):
    assert device
    return _properties()[device.type][_device(device)]


def get_device_capability(device: torch.device):
    assert device
    p = get_device_properties(device)
    return p.major, p.minor
