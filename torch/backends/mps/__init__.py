import sys
import torch

def is_built():
    r"""Returns whether PyTorch is built with MPS support."""
    return torch._C.has_mps

def is_available():
    r"""Returns whether PyTorch is built with MPS support."""
    return torch._C.has_mps
