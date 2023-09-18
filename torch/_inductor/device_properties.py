from typing import Any, Dict

import torch
from torch._dynamo.runtime import get_registered_device_runtimes, get_runtime_for_device

# API to query cuda properties that will work in a triton compile process
# that cannot use the GPU APIs (due to processing fork() and initialization
# time issues). Properties are recorded in the main process before
# we fork the workers.

_compile_worker_device_properties: Dict[str, Any] = {}
_compile_worker_current_devices: Dict[str, int] = {}


# All registered devices need to recored their properties in the main process before
# we fork the workers. We can NOT call this function only once because an out-of-tree
# package can register its runtime only when importing the package.
def _properties():
    for device, device_runtime in get_registered_device_runtimes():
        if device in _compile_worker_device_properties:
            continue
        # Recored the newly registered device properties.
        if device_runtime.is_available():
            device_prop = [
                device_runtime.get_device_properties(i)
                for i in range(device_runtime.device_count())
            ]
            _compile_worker_device_properties[device] = device_prop


def set_compiler_worker_current_device(device: torch.device) -> None:
    assert device.index is not None
    _compile_worker_current_devices[device.type] = device.index


def current_device(device: str) -> int:
    if device in _compile_worker_current_devices:
        return _compile_worker_current_devices[device]
    device_runtime = get_runtime_for_device(device)
    return device_runtime.current_device()


def _device(device: torch.device) -> int:
    assert device.type != "cpu"
    if device.index is not None:
        return device.index
    else:
        return current_device(device.type)


def get_device_properties(device: torch.device = torch.device("cuda")):
    _properties()
    if device.type not in _compile_worker_device_properties:
        return {}
    else:
        return _compile_worker_device_properties[device.type][_device(device)]
