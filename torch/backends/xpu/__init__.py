r"""This is a dummy torch.xpu module aliased by torch._xpu; This module supports
two APIs torch._xpu.device_count() and torch._xpu.current_device() in JIT script.
In keeping with JIT script, the corresponding python native APIs also be supported."""
import torch._C
from functools import lru_cache as _lru_cache

@_lru_cache()
def is_available() -> bool:
    r"""Returns a bool indicating if XPU is currently available."""
    return torch._C._is_hooks_available("xpu")

def device_count() -> int:
    return torch._C._get_device_count("xpu")

def current_device() -> int:
    return torch._C._get_device("xpu")
