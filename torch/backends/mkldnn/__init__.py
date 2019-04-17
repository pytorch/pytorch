import torch


def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn
