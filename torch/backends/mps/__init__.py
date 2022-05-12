import sys
import torch

def is_built():
    r"""Returns whether PyTorch is built with MPS support. Note that this
    doesn't necessarily mean MPS is available; just that if this PyTorch
    binary were run a machine with working MPS drivers and devices, we
    would be able to use it."""
    return torch._C.has_mps

def is_available():
    r"""Returns a bool indicating if MPS is currently available."""
    return torch._C._is_mps_available
