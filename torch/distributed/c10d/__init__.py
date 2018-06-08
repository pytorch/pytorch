import torch


def is_available():
    return hasattr(torch._C, '_c10d_init')

if is_available() and not torch._C._c10d_init():
    raise RuntimeError("c10d initialization failed")
