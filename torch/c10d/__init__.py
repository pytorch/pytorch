import torch

if not torch._C._c10d_init():
    raise RuntimeError("c10d initialization failed")
