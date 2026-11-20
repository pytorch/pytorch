"""Tensor preparation shared by native DSL adapters."""

from __future__ import annotations

import math

import torch


def const_data_ptr(tensor: torch.Tensor) -> int:
    """Read a pointer without materializing COW or re-entering torch_function."""
    with torch._C.DisableTorchFunctionSubclass():
        return tensor.const_data_ptr()


def row_alignment(n: int, element_size: int, max_alignment: int = 16) -> int:
    """Byte alignment preserved between contiguous rows of n elements."""
    return math.gcd(n * element_size, max_alignment)


def reshape_contiguous(
    tensor: torch.Tensor, shape: tuple[int, ...], *, alignment: int = 1
) -> torch.Tensor:
    """Reshape for pointer-based kernels, copying only to resolve layout or alignment."""
    tensor = tensor.resolve_conj().resolve_neg()
    if tensor.shape != shape or not tensor.is_contiguous():
        tensor = tensor.reshape(shape).contiguous()
    # contiguous() preserves offsets on contiguous views, including misaligned ones.
    if alignment > 1 and const_data_ptr(tensor) % alignment:
        tensor = tensor.clone()
    return tensor
