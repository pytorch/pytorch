# mypy: allow-untyped-defs
from importlib.util import find_spec

import torch


__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    return not (torch.cuda.is_available() or find_spec("torch_xla"))
