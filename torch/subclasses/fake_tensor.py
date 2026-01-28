"""
torch.subclasses.fake_tensor
============================

This module provides FakeTensor and FakeTensorMode for tracing and compiling
PyTorch programs without requiring actual tensor data.

Fake tensors contain all the metadata of real tensors (shape, dtype, device,
strides, requires_grad) but no actual data storage. They are backed by meta
tensors internally.

Example::

    from torch.subclasses.fake_tensor import FakeTensorMode, FakeTensor

    with FakeTensorMode() as fake_mode:
        real = torch.randn(10, 20)
        fake = fake_mode.from_tensor(real)
        print(fake.shape)  # torch.Size([10, 20])
        print(isinstance(fake, FakeTensor))  # True

See Also:
    - :class:`FakeTensorMode`: Context manager for fake tensor operations
    - :class:`FakeTensor`: The tensor subclass representing fake tensors
"""

from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    FakeTensor,
    FakeTensorMode,
    UnsupportedFakeTensorException,
    unset_fake_temporarily,
)

__all__ = [
    "FakeTensor",
    "FakeTensorMode",
    "UnsupportedFakeTensorException",
    "DynamicOutputShapeException",
    "unset_fake_temporarily",
]
