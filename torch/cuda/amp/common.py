import torch
from importlib.util import find_spec

__all__ = ["amp_definitely_not_available"]

def amp_definitely_not_available():
    return not (torch.cuda.is_available() or find_spec('torch_xla'))
