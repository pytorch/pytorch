# mypy: allow-untyped-defs
"""
Operation handlers for StridedNestedTensor.

This module registers handlers for PyTorch operations on strided nested tensors,
enabling torch.compile support.

See https://github.com/pytorch/pytorch/issues/168307
"""
import functools
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor


__all__ = ["lookup_strided", "register_strided_func", "STRIDED_OPS_TABLE"]


STRIDED_OPS_TABLE: Dict[Any, Callable] = {}


def register_strided_func(aten_op):
    """
    Decorator to register a strided nested tensor operation handler.

    Usage:
        @register_strided_func(torch.ops.aten.some_op.default)
        def some_op_handler(func, *args, **kwargs):
            ...
    """

    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(aten_op, *args, **kwargs)

        STRIDED_OPS_TABLE[aten_op] = inner
        return func

    return wrapper


def lookup_strided(func, *args, **kwargs) -> Optional[Callable]:
    """Look up handler for a strided nested tensor operation."""
    return STRIDED_OPS_TABLE.get(func, None)


# =============================================================================
# Metadata accessors
# =============================================================================


@register_strided_func(torch.ops.aten._nested_tensor_size.default)
def _nested_tensor_size(func, self):
    """Return the nested sizes tensor."""
    return self._nested_sizes


@register_strided_func(torch.ops.aten._nested_tensor_strides.default)
def _nested_tensor_strides(func, self):
    """Return the nested strides tensor."""
    return self._nested_strides


@register_strided_func(torch.ops.aten._nested_tensor_storage_offsets.default)
def _nested_tensor_storage_offsets(func, self):
    """Return the storage offsets tensor."""
    return self._storage_offsets


# =============================================================================
# Basic tensor properties
# =============================================================================


@register_strided_func(torch.ops.aten.dim.default)
def dim(func, self):
    """Return the number of dimensions (1 for batch + ndim for elements)."""
    return 1 + self._nested_sizes.shape[1]


@register_strided_func(torch.ops.aten.numel.default)
def numel(func, self):
    """Return the total number of elements in the buffer."""
    return self._buffer.numel()


@register_strided_func(torch.ops.aten.sym_size.default)
def sym_size(func, self, dim=None):
    """Return the size of the tensor (or a specific dimension)."""
    # Use _outer_size which is always set during construction
    # Avoid calling self.shape which would recurse back into __torch_dispatch__
    outer_size = self._outer_size
    if dim is None:
        return list(outer_size)
    # Handle negative dimensions
    ndim = len(outer_size)
    if dim < 0:
        dim = ndim + dim
    return outer_size[dim]


@register_strided_func(torch.ops.aten.sym_stride.default)
def sym_stride(func, self, dim=None):
    """Return the stride of the tensor (or a specific dimension)."""
    # Use _outer_strides which is always set during construction
    # Avoid calling self.stride() which would recurse back into __torch_dispatch__
    outer_strides = self._outer_strides
    if dim is None:
        return list(outer_strides)
    # Handle negative dimensions
    ndim = len(outer_strides)
    if dim < 0:
        dim = ndim + dim
    return outer_strides[dim]


@register_strided_func(torch.ops.aten.sym_numel.default)
def sym_numel(func, self):
    """Return the total number of elements."""
    return self._buffer.numel()


@register_strided_func(torch.ops.aten.sym_storage_offset.default)
def sym_storage_offset(func, self):
    """Return the storage offset (always 0 for our wrapper)."""
    return 0


# =============================================================================
# Conversion operations
# =============================================================================


@register_strided_func(torch.ops.aten.nested_to_padded_tensor.default)
@register_strided_func(torch.ops.aten.to_padded_tensor.default)
def nested_to_padded_tensor(func, self, padding, output_size=None):
    """
    Convert strided nested tensor to a padded dense tensor.

    For uniform batch (all elements have same shape), this is a simple reshape.
    For our minimal implementation (tensor input), we always have uniform batch
    and can use the stored outer_size metadata to avoid data-dependent guards.
    """
    # Use stored outer_size for uniform batch (avoids data-dependent guards)
    # For tensor input, _outer_size is always set and represents (batch, *element_shape)
    outer_size = self._outer_size
    if outer_size:
        # Uniform batch from tensor input: simple reshape using stored metadata
        return self._buffer.view(outer_size)

    # Fallback for non-uniform batch (if ever constructed manually)
    batch_size = self._nested_sizes.shape[0]

    if batch_size == 0:
        # Empty batch - return empty tensor
        ndim = self._nested_sizes.shape[1]
        if output_size is not None:
            return torch.full(output_size, padding, dtype=self._buffer.dtype, device=self._buffer.device)
        return self._buffer.new_empty((0,) + (0,) * ndim)

    # Non-uniform batch: need to pad
    # Find max size in each dimension
    max_sizes = self._nested_sizes.max(dim=0).values
    max_shape = tuple(max_sizes.tolist())

    if output_size is not None:
        # Validate output_size
        if len(output_size) != 1 + len(max_shape):
            raise RuntimeError(
                f"output_size has wrong number of dimensions: {len(output_size)} vs expected {1 + len(max_shape)}"
            )
        if output_size[0] < batch_size:
            raise RuntimeError("output_size[0] must be >= batch_size")
        for i, (out_s, max_s) in enumerate(zip(output_size[1:], max_shape)):
            if out_s < max_s:
                raise RuntimeError(
                    f"output_size[{i+1}]={out_s} is smaller than max nested size {max_s}"
                )
        padded_shape = output_size
    else:
        padded_shape = (batch_size,) + max_shape

    # Create output tensor filled with padding value
    output = torch.full(
        padded_shape,
        padding,
        dtype=self._buffer.dtype,
        device=self._buffer.device,
    )

    # Copy each element into the output
    # This is a simple implementation; could be optimized with custom kernel
    for i in range(batch_size):
        element_sizes = tuple(self._nested_sizes[i].tolist())
        offset = int(self._storage_offsets[i].item())
        element_numel = 1
        for s in element_sizes:
            element_numel *= s

        element_data = self._buffer[offset : offset + element_numel].view(element_sizes)

        # Build slice for output
        slices = [i] + [slice(0, s) for s in element_sizes]
        output[tuple(slices)] = element_data

    return output


# =============================================================================
# Clone and copy operations
# =============================================================================


@register_strided_func(torch.ops.aten.clone.default)
def clone(func, self, *, memory_format=None):
    """Clone the strided nested tensor."""
    from .strided_nested_tensor import StridedNestedTensor

    return StridedNestedTensor(
        self._buffer.clone(),
        self._nested_sizes.clone(),
        self._nested_strides.clone(),
        self._storage_offsets.clone(),
        requires_grad=self.requires_grad,
        _outer_size=self._outer_size,
        _outer_strides=self._outer_strides,
    )


@register_strided_func(torch.ops.aten.detach.default)
def detach(func, self):
    """Detach the strided nested tensor from the computation graph."""
    from .strided_nested_tensor import StridedNestedTensor

    return StridedNestedTensor(
        self._buffer.detach(),
        self._nested_sizes,
        self._nested_strides,
        self._storage_offsets,
        requires_grad=False,
        _outer_size=self._outer_size,
        _outer_strides=self._outer_strides,
    )


# =============================================================================
# Device/dtype conversion
# =============================================================================


@register_strided_func(torch.ops.aten._to_copy.default)
def _to_copy(func, self, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    """Copy the tensor to a different device/dtype."""
    from .strided_nested_tensor import StridedNestedTensor

    new_buffer = self._buffer.to(
        dtype=dtype,
        device=device,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )

    new_device = new_buffer.device

    return StridedNestedTensor(
        new_buffer,
        self._nested_sizes.to(device=new_device),
        self._nested_strides.to(device=new_device),
        self._storage_offsets.to(device=new_device),
        requires_grad=self.requires_grad and (dtype is None or dtype.is_floating_point),
        _outer_size=self._outer_size,
        _outer_strides=self._outer_strides,
    )


# =============================================================================
# View operations
# =============================================================================


@register_strided_func(torch.ops.aten.view.default)
def view(func, self, size):
    """
    View operation for strided nested tensor.

    For uniform batch, supports viewing the element shape.
    """
    from .strided_nested_tensor import StridedNestedTensor

    batch_size = self._nested_sizes.shape[0]

    # Check if uniform batch
    if batch_size > 0 and not torch.all(self._nested_sizes == self._nested_sizes[0]):
        raise RuntimeError(
            "view is not supported for non-uniform strided nested tensors"
        )

    # For uniform batch, we can reshape
    if len(size) < 1:
        raise RuntimeError("view requires at least one dimension")

    # First dimension should match batch size or be -1
    if size[0] != batch_size and size[0] != -1:
        raise RuntimeError(
            f"view: first dimension must match batch size {batch_size} or be -1, got {size[0]}"
        )

    new_element_shape = size[1:] if size[0] != -1 else size[1:]

    # Check that the total elements match
    old_numel = self._buffer.numel() // batch_size if batch_size > 0 else 0
    new_numel = 1
    infer_dim = None
    for i, s in enumerate(new_element_shape):
        if s == -1:
            if infer_dim is not None:
                raise RuntimeError("only one dimension can be inferred")
            infer_dim = i
        else:
            new_numel *= s

    if infer_dim is not None:
        if new_numel == 0:
            raise RuntimeError("cannot infer dimension with zero-element tensor")
        inferred_size = old_numel // new_numel
        new_element_shape = list(new_element_shape)
        new_element_shape[infer_dim] = inferred_size
        new_element_shape = tuple(new_element_shape)
        new_numel = old_numel

    if new_numel != old_numel:
        raise RuntimeError(
            f"view: shape mismatch, old numel {old_numel} != new numel {new_numel}"
        )

    # Create new metadata
    ndim = len(new_element_shape)
    new_sizes = torch.tensor(
        new_element_shape, dtype=torch.int64, device=self._buffer.device
    )
    new_nested_sizes = new_sizes.unsqueeze(0).expand(batch_size, ndim).contiguous()

    # Compute new strides (contiguous)
    new_strides = []
    stride = 1
    for s in reversed(new_element_shape):
        new_strides.insert(0, stride)
        stride *= s
    new_strides_tensor = torch.tensor(
        new_strides, dtype=torch.int64, device=self._buffer.device
    )
    new_nested_strides = new_strides_tensor.unsqueeze(0).expand(batch_size, ndim).contiguous()

    return StridedNestedTensor(
        self._buffer,  # Share the same buffer
        new_nested_sizes,
        new_nested_strides,
        self._storage_offsets,
        requires_grad=self.requires_grad,
    )


# =============================================================================
# Contiguous
# =============================================================================


@register_strided_func(torch.ops.aten.is_contiguous.default)
@register_strided_func(torch.ops.aten.is_contiguous.memory_format)
def is_contiguous(func, self, memory_format=torch.contiguous_format):
    """Check if the nested tensor is contiguous."""
    # For now, we consider uniform batch nested tensors as contiguous
    # if the buffer is contiguous
    if memory_format != torch.contiguous_format:
        return False
    return self._buffer.is_contiguous()


@register_strided_func(torch.ops.aten.contiguous.default)
def contiguous(func, self, memory_format=torch.contiguous_format):
    """Return a contiguous tensor."""
    from .strided_nested_tensor import StridedNestedTensor

    if is_contiguous(func, self, memory_format):
        return self

    # Make buffer contiguous
    return StridedNestedTensor(
        self._buffer.contiguous(),
        self._nested_sizes.clone(),
        self._nested_strides.clone(),
        self._storage_offsets.clone(),
        requires_grad=self.requires_grad,
        _outer_size=self._outer_size,
        _outer_strides=self._outer_strides,
    )


# =============================================================================
# Prim operations (needed for torch.compile)
# =============================================================================


@register_strided_func(torch.ops.prim.layout.default)
def layout(func, self):
    """Return the layout of the tensor (always strided for StridedNestedTensor)."""
    return torch.strided
