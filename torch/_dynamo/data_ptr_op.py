"""Data pointer primitive for torch.compile.

This module provides a traceable version of tensor.data_ptr() that works with
FakeTensors during compilation. It generates unique fake pointer values based
on storage identity, ensuring that runtime comparisons work correctly.
"""

from typing import Any
import weakref

import torch
from torch._prims import _make_prim, RETURN_TYPE
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode


# Fake pointer base value (high value to avoid collision with user constants)
_FAKE_PTR_BASE = 1_000_000

# Global state for generating unique fake pointers
_storage_to_fake_ptr: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_next_storage_id = 0


def _get_fake_data_ptr(tensor: torch.Tensor) -> int:
    """Generate a unique fake data pointer for a tensor.

    Args:
        tensor: The tensor to get a fake pointer for

    Returns:
        A unique fake pointer value (or 0 for empty tensors)
    """
    global _next_storage_id

    storage = tensor.untyped_storage()

    # Empty tensors always return pointer 0
    if storage.size() == 0:
        return 0

    # Assign a unique base pointer for each storage
    if storage not in _storage_to_fake_ptr:
        _next_storage_id += 1
        _storage_to_fake_ptr[storage] = _next_storage_id * _FAKE_PTR_BASE

    # data_ptr() points to the first element, accounting for storage offset
    base_ptr = _storage_to_fake_ptr[storage]
    offset = tensor.storage_offset() * tensor.element_size()

    return base_ptr + offset


_data_ptr = _make_prim(
    schema="_data_ptr(Tensor self) -> int",
    return_type=RETURN_TYPE.NEW,
    meta=_get_fake_data_ptr,
    impl_aten=lambda self: self.data_ptr(),
    doc="Traceable version of torch.Tensor.data_ptr()",
)


@_data_ptr.py_impl(FakeTensorMode)
def _data_ptr_fake_impl(fake_mode: FakeTensorMode, tensor: Any) -> int:
    """FakeTensor implementation - called during tracing."""
    return _get_fake_data_ptr(tensor)


@_data_ptr.py_impl(FunctionalTensorMode)
def _data_ptr_functional_impl(mode: FunctionalTensorMode, tensor: Any) -> int:
    """FunctionalTensor implementation - called during AOTAutograd."""
    return _get_fake_data_ptr(tensor)
