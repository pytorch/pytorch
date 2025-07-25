import torch

if hasattr(torch._C, "GreenContext"):
    from torch._C import GreenContext
