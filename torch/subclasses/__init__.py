"""
torch.subclasses
================

This module provides public APIs for PyTorch tensor subclasses used in
compilation and tracing.

Fake Tensors
------------

Fake tensors are tensors that have all the metadata of a real tensor (shape,
dtype, device, strides) but no actual data storage. They are used by PyTorch's
compilation infrastructure for shape inference, memory analysis, and graph
tracing without requiring actual computation.

.. code-block:: python

    from torch.subclasses import FakeTensorMode

    with FakeTensorMode() as fake_mode:
        # Create a fake tensor from a real one
        real = torch.randn(10, 20)
        fake = fake_mode.from_tensor(real)

        # Operations produce fake tensors with correct metadata
        result = fake @ fake.T
        print(result.shape)  # torch.Size([10, 10])

See Also:
    - :class:`FakeTensorMode`: Context manager for fake tensor operations
    - :class:`FakeTensor`: The tensor subclass representing fake tensors
    - `FakeTensor User Guide <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_
"""

from torch._subclasses import (
    CrossRefFakeMode,
    DynamicOutputShapeException,
    FakeTensor,
    FakeTensorMode,
    UnsupportedFakeTensorException,
)

__all__ = [
    "FakeTensor",
    "FakeTensorMode",
    "UnsupportedFakeTensorException",
    "DynamicOutputShapeException",
    "CrossRefFakeMode",
]
