import torch
from functools import lru_cache as _lru_cache

def is_built() -> bool:
    r"""Returns whether PyTorch is built with MPS support. Note that this
    doesn't necessarily mean MPS is available; just that if this PyTorch
    binary were run a machine with working MPS drivers and devices, we
    would be able to use it."""
    return torch._C.has_mps

@_lru_cache()
def is_available() -> bool:
    r"""Returns a bool indicating if MPS is currently available."""
    return torch._C._is_mps_available()

@_lru_cache()
def is_macos13_or_newer() -> bool:
    r"""Returns a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._is_mps_on_macos_13_or_newer()
