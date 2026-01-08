# mypy: allow-untyped-defs
"""
Python tensor subclass for strided layout nested tensors.

This module provides torch.compile support for strided nested tensors by
implementing the __tensor_flatten__/__tensor_unflatten__ protocol required
by the FakeTensor system.

See https://github.com/pytorch/pytorch/issues/168307
"""
from typing import Any, Optional

import torch
from torch import Tensor
from torch._C import DispatchKey, DispatchKeySet


__all__ = ["StridedNestedTensor", "strided_nested_tensor_from_tensor"]


class StridedNestedTensor(torch.Tensor):
    """
    Python tensor subclass for strided layout nested tensors.

    This class wraps the metadata that would normally be stored in C++ NestedTensorImpl:
    - _buffer: 1D flattened data tensor
    - _nested_sizes: 2D tensor of shape (batch, ndim) with sizes of each element
    - _nested_strides: 2D tensor of shape (batch, ndim) with strides of each element
    - _storage_offsets: 1D tensor of shape (batch,) with buffer offsets

    For the minimal scope implementation, this supports uniform batch (all elements
    have identical shapes), which is the common case when converting a regular tensor
    to a nested tensor.
    """

    _buffer: Tensor
    _nested_sizes: Tensor
    _nested_strides: Tensor
    _storage_offsets: Tensor
    # Cache the outer shape/strides for uniform batch (avoids symbolic tensor access)
    _outer_size: tuple[int, ...]
    _outer_strides: tuple[int, ...]

    @staticmethod
    def __new__(
        cls,
        buffer: Tensor,
        nested_sizes: Tensor,
        nested_strides: Tensor,
        storage_offsets: Tensor,
        *,
        requires_grad: bool = False,
        # For uniform batch, pass the concrete element shape to avoid symbolic issues
        _outer_size: Optional[tuple[int, ...]] = None,
        _outer_strides: Optional[tuple[int, ...]] = None,
    ):
        # Validate inputs
        assert buffer.dim() == 1, "Buffer must be 1D"
        assert nested_sizes.dim() == 2, "nested_sizes must be 2D"
        assert nested_strides.dim() == 2, "nested_strides must be 2D"
        assert storage_offsets.dim() == 1, "storage_offsets must be 1D"

        batch_size = nested_sizes.shape[0]
        ndim = nested_sizes.shape[1]

        # Use provided outer_size/outer_strides if available (for torch.compile compatibility)
        # Otherwise fall back to computing from batch_size and ndim
        if _outer_size is not None:
            _size = _outer_size
            _strides = _outer_strides if _outer_strides is not None else (1,) * len(_outer_size)
        else:
            # Fallback: use batch_size and placeholder for element dims
            _size = (batch_size,) + (1,) * ndim
            _strides = (1,) * (1 + ndim)

        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            _size,
            _strides,
            0,  # storage_offset
            torch.contiguous_format,
            buffer.dtype,
            torch.strided,  # layout
            buffer.device,
            False,  # pin_memory
            requires_grad,
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
            storage_size=buffer.untyped_storage().size(),
        )
        return r

    def __init__(
        self,
        buffer: Tensor,
        nested_sizes: Tensor,
        nested_strides: Tensor,
        storage_offsets: Tensor,
        *,
        requires_grad: bool = False,
        _outer_size: Optional[tuple[int, ...]] = None,
        _outer_strides: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        self._buffer = buffer
        self._nested_sizes = nested_sizes
        self._nested_strides = nested_strides
        self._storage_offsets = storage_offsets
        # Store outer shape info for later use
        self._outer_size = _outer_size if _outer_size is not None else ()
        self._outer_strides = _outer_strides if _outer_strides is not None else ()

    def __repr__(self) -> str:
        batch_size = self._nested_sizes.shape[0]
        ndim = self._nested_sizes.shape[1]
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"StridedNestedTensor(batch={batch_size}, ndim={ndim}, buffer_size={self._buffer.numel()}{grad_str})"

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            "_outer_size": self._outer_size,
            "_outer_strides": self._outer_strides,
        }
        return ["_buffer", "_nested_sizes", "_nested_strides", "_storage_offsets"], ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: dict[str, Any],
        outer_size,
        outer_stride,
    ):
        return StridedNestedTensor(
            inner_tensors["_buffer"],
            inner_tensors["_nested_sizes"],
            inner_tensors["_nested_strides"],
            inner_tensors["_storage_offsets"],
            requires_grad=meta["requires_grad"],
            _outer_size=meta.get("_outer_size"),
            _outer_strides=meta.get("_outer_strides"),
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        from .strided_ops import lookup_strided

        fn = lookup_strided(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        raise NotImplementedError(
            f"StridedNestedTensor does not yet support {func}. "
            "Consider using layout=torch.jagged for better torch.compile support, "
            "or file an issue at https://github.com/pytorch/pytorch/issues"
        )

    # === Accessors matching C++ NestedTensorImpl interface ===

    def values(self) -> Tensor:
        """Return the underlying buffer tensor."""
        return self._buffer

    def _nested_tensor_size(self) -> Tensor:
        """Return nested sizes tensor of shape (batch, ndim)."""
        return self._nested_sizes

    def _nested_tensor_strides(self) -> Tensor:
        """Return nested strides tensor of shape (batch, ndim)."""
        return self._nested_strides

    def _nested_tensor_storage_offsets(self) -> Tensor:
        """Return storage offsets tensor of shape (batch,)."""
        return self._storage_offsets

    # === Utility methods ===

    def is_uniform(self) -> bool:
        """Check if all batch elements have the same shape (uniform batch)."""
        if self._nested_sizes.shape[0] <= 1:
            return True
        return bool(torch.all(self._nested_sizes == self._nested_sizes[0]).item())

    def get_uniform_shape(self) -> Optional[tuple[int, ...]]:
        """
        Get the uniform element shape if this is a uniform batch.
        Returns None if batch elements have different shapes.
        """
        if not self.is_uniform():
            return None
        if self._nested_sizes.shape[0] == 0:
            return ()
        return tuple(self._nested_sizes[0].tolist())


def strided_nested_tensor_from_tensor(
    tensor: Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> StridedNestedTensor:
    """
    Create a StridedNestedTensor from a regular tensor.

    For a tensor of shape (batch, d1, d2, ..., dn), creates a nested tensor
    where each batch element has shape (d1, d2, ..., dn).

    This is the uniform batch case - all elements have identical shapes.

    Args:
        tensor: Input tensor of shape (batch, d1, d2, ..., dn) with at least 2 dimensions
        dtype: Optional dtype for the nested tensor
        device: Optional device for the nested tensor

    Returns:
        StridedNestedTensor wrapping the input data
    """
    if tensor.dim() < 2:
        raise RuntimeError(
            "as_nested_tensor(): Expected tensor argument to have dim() > 1"
        )

    # Get dimensions - element_shape should be concrete even during tracing
    # because it's the non-batch dimensions
    batch_size = tensor.shape[0]
    element_shape = tensor.shape[1:]
    ndim = len(element_shape)

    # Convert element_shape to concrete Python ints (handles SymInt)
    # For uniform batch from a tensor, element dimensions should be concrete
    element_shape_concrete: tuple[int, ...] = tuple(int(s) for s in element_shape)

    # Determine target device
    target_device = device if device is not None else tensor.device

    # Create flattened buffer
    buffer = tensor.contiguous().view(-1)
    if dtype is not None or device is not None:
        buffer = buffer.to(dtype=dtype, device=device)

    # Create nested_sizes: (batch, ndim) - all rows identical for uniform batch
    element_sizes = torch.tensor(
        element_shape_concrete, dtype=torch.int64, device=target_device
    )
    nested_sizes = element_sizes.unsqueeze(0).expand(batch_size, ndim).contiguous()

    # Create nested_strides: (batch, ndim) - contiguous strides for each element
    element_strides_list: list[int] = []
    stride = 1
    for size in reversed(element_shape_concrete):
        element_strides_list.insert(0, stride)
        stride *= size
    element_strides_concrete: tuple[int, ...] = tuple(element_strides_list)
    element_strides_tensor = torch.tensor(
        element_strides_concrete, dtype=torch.int64, device=target_device
    )
    nested_strides = element_strides_tensor.unsqueeze(0).expand(batch_size, ndim).contiguous()

    # Create storage_offsets: evenly spaced for uniform batch
    element_numel = stride  # Product of all element dimensions
    storage_offsets = torch.arange(
        0,
        batch_size * element_numel,
        element_numel if element_numel > 0 else 1,
        dtype=torch.int64,
        device=target_device,
    )

    # Compute outer size and strides for the wrapper subclass
    # batch_size might be symbolic, but element_shape is concrete
    _outer_size: tuple[int, ...] = (batch_size,) + element_shape_concrete

    # Compute contiguous strides for the full outer shape
    _outer_strides_list: list[int] = []
    outer_stride = 1
    for s in reversed(_outer_size):
        _outer_strides_list.insert(0, outer_stride)
        outer_stride *= s
    _outer_strides: tuple[int, ...] = tuple(_outer_strides_list)

    return StridedNestedTensor(
        buffer,
        nested_sizes,
        nested_strides,
        storage_offsets,
        requires_grad=tensor.requires_grad,
        _outer_size=_outer_size,
        _outer_strides=_outer_strides,
    )
