import torch
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    FakeTensor,
    FakeTensorMode,
    make_fake_mode,
    UnsupportedFakeTensorException,
)
from torch._subclasses.fake_utils import CrossRefFakeMode


__all__ = [
    "FakeTensor",
    "FakeTensorMode",
    "make_fake_mode",
    "UnsupportedFakeTensorException",
    "DynamicOutputShapeException",
    "CrossRefFakeMode",
]
