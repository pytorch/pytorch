import torch


def is_available():
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize PyTorch distributed support")


if is_available():
    from .distributed_c10d import *
