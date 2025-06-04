import torch
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    FakeTensor,
    FakeTensorMode,
    UnsupportedFakeTensorException,
)
from torch._subclasses.fake_utils import CrossRefFakeMode


__all__ = [
    "FakeTensor",
    "FakeTensorMode",
    "UnsupportedFakeTensorException",
    "DynamicOutputShapeException",
    "CrossRefFakeMode",
]
