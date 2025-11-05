"""Data pointer operations for Dynamo tracing.

This module provides primitives for handling tensor.data_ptr() during tracing. It includes:
- A traceable version of data_ptr() that works with FakeTensors
- Comparison operations (eq, ne) that return Tensor instead of bool

The traceable data_ptr() generates unique fake pointer values based on storage identity,
ensuring that:
1. Tensors sharing storage have the same fake data_ptr
2. Tensors with different storage have different fake data_ptrs
3. Empty tensors return 0 (matching eager behavior)

The comparison operations use CompositeExplicitAutograd dispatch to decompose into
primitive operations at runtime while maintaining correct tracing behavior.
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

    if storage.size() == 0:
        return 0

    if storage not in _storage_to_fake_ptr:
        _next_storage_id += 1
        _storage_to_fake_ptr[storage] = _next_storage_id * _FAKE_PTR_BASE

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


# Data pointer comparison operations
#
# These ops return Tensor (scalar bool) instead of python bool to enable graph tracing.
# They use CompositeExplicitAutograd dispatch, which means:
# 1. At runtime, they decompose into primitive operations (prims._data_ptr)
# 2. During tracing, they remain as high-level ops in the graph
# 3. Autograd is handled explicitly via the decomposition
#
# This avoids issues with polyfill's cmp_ne doing `not cmp_eq(a, b)` which causes
# guard errors when the result is a tensor.
_data_ptr_cmp_lib = torch.library.Library("_dynamo_data_ptr", "DEF")

_data_ptr_cmp_lib.define("eq(Tensor self, int ptr_value) -> Tensor")
_data_ptr_cmp_lib.define("ne(Tensor self, int ptr_value) -> Tensor")


@torch.library.impl(_data_ptr_cmp_lib, "eq", "CompositeExplicitAutograd")
def _data_ptr_eq_impl(tensor: torch.Tensor, ptr_value: int) -> torch.Tensor:
    """CompositeExplicitAutograd implementation for data_ptr equality comparison.

    Decomposes into primitive operations at runtime while maintaining graph traceability.
    """
    actual_ptr = torch.ops.prims._data_ptr.default(tensor)
    result = actual_ptr == ptr_value
    return torch.tensor(result, dtype=torch.bool)


@torch.library.impl(_data_ptr_cmp_lib, "ne", "CompositeExplicitAutograd")
def _data_ptr_ne_impl(tensor: torch.Tensor, ptr_value: int) -> torch.Tensor:
    """CompositeExplicitAutograd implementation for data_ptr inequality comparison.

    Decomposes into primitive operations at runtime while maintaining graph traceability.
    """
    result = torch.ops.prims._data_ptr.default(tensor) != ptr_value
    return torch.tensor(result, dtype=torch.bool)


@torch.library.register_fake("_dynamo_data_ptr::eq")
def _data_ptr_eq_fake(tensor: torch.Tensor, ptr_value: int) -> torch.Tensor:
    """Fake implementation for data_ptr equality comparison.

    Returns an empty scalar bool tensor during tracing to indicate the result is
    data-dependent and cannot be determined statically. This prevents specialization
    on the comparison result.
    """
    return torch.empty((), dtype=torch.bool, device=tensor.device)


@torch.library.register_fake("_dynamo_data_ptr::ne")
def _data_ptr_ne_fake(tensor: torch.Tensor, ptr_value: int) -> torch.Tensor:
    """Fake implementation for data_ptr inequality comparison.

    Returns an empty scalar bool tensor during tracing to indicate the result is
    data-dependent and cannot be determined statically. This prevents specialization
    on the comparison result.
    """
    return torch.empty((), dtype=torch.bool, device=tensor.device)
