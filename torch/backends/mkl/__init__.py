import torch


def is_available():
    r"""Returns whether PyTorch is built with MKL support."""
    return torch._C.has_mkl
