# mypy: allow-untyped-defs
import threading
from typing import Any, Dict

import torch._C._lazy


class DeviceContext:
    _CONTEXTS: Dict[str, Any] = {}
    _CONTEXTS_LOCK = threading.Lock()

    def __init__(self, device):
        self.device = device


def get_device_context(device=None):
    if device is None:
        device = torch._C._lazy._get_default_device_type()
    else:
        device = str(device)
    with DeviceContext._CONTEXTS_LOCK:
        devctx = DeviceContext._CONTEXTS.get(device, None)
        if devctx is None:
            devctx = DeviceContext(device)
            DeviceContext._CONTEXTS[device] = devctx
        return devctx
