from enum import Enum

from torch.types import _bool, Tuple

# Defined in torch/csrc/cuda/shared/cusparselt.cpp
is_cuda: _bool

def getVersionInt() -> int: ...
