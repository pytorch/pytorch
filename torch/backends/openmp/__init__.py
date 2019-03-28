import torch


def is_available():
    r"""Returns whether PyTorch is built with OpenMP support."""
    return torch._C.has_openmp
