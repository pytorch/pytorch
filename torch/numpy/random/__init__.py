import torch
import torch.numpy as np


def randn(size=0):
    if size==0:
        return torch.randn(1).item()
    else:
        return torch.randn(size)



