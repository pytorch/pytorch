"""
Python implementation of function wrapping functionality for functorch.dim.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

import torch
from torch.utils._pytree import tree_map

from ._dim_entry import DimEntry
from ._enable_all_layers import EnableAllLayers
from ._tensor_info import TensorInfo


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Handle tensor conversion for torch function integration."""
    return tensor


class WrappedOperator:
    """
    This class wraps PyTorch operations to support first-class dimensions.
    """

    def __init__(
        self, orig: Callable, wrapper_implementation: Callable, dim_name: str = "dim"
    ):
        self.orig = orig
        self.wrapper_implementation = wrapper_implementation
        self.name = getattr(orig, "__name__", "")
        self.doc = getattr(orig, "__doc__", None)
        self.dim_name = dim_name

        self.is_pointwise = False
        self.dim_offset = 0
        self.keepdim_offset = 1
        self.single_dim = False
        self.reduce = True

        # Update docstring if we have a dim_name
        if self.doc and self.dim_name:
            self.doc = f"{self.doc}\nArgument '{self.dim_name}' can be either an integer or a torchdim.Dim object.\n"

    def function(self) -> Callable:
        """Create a wrapped function that calls our wrapper implementation."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            return self.wrapper_implementation(self, *args, **kwargs)

        # Copy metadata using functools.update_wrapper for just __name__ and __doc__
        functools.update_wrapper(
            wrapped_func, self.orig, assigned=("__name__",), updated=()
        )
        wrapped_func.__doc__ = self.doc

        return wrapped_func


def _wrap_dim(dim: Any, ndim: int, keepdim: bool = False) -> DimEntry:
    """Convert single dimension specification to DimEntry object."""
    from . import Dim

    if isinstance(dim, Dim):
        if keepdim:
            raise ValueError("cannot preserve first-class dimensions with keepdim=True")
        return DimEntry(dim)
    elif isinstance(dim, int):
        i = dim
        while i >= 0:
            i -= ndim
        return DimEntry(i)
    else:
        return DimEntry()


def _wrap_dims(dim: Any, ndim: int, keepdim: bool = False) -> list[DimEntry]:
    """Convert dimension specification to list of DimEntry objects."""
    de = _wrap_dim(dim, ndim, keepdim)
    result = []
    if not de.is_none():
        result.append(de)
    else:
        for d in dim:
            result.append(_wrap_dim(d, ndim, keepdim))
    return result


def patched_dim_method(wrapper: WrappedOperator, *args: Any, **kwargs: Any) -> Any:
    """
    This is the core method that handles dimension-aware operations.
    """
    if not args:
        raise ValueError("Expected at least one argument (self)")

    # Get dimension argument
    dim_arg = kwargs.get(wrapper.dim_name)
    if dim_arg is None and wrapper.dim_offset < len(args):
        # Try to get dim from positional args (accounting for self at index 0)
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(args):
            dim_arg = args[dim_idx]

    # If no dimension argument provided, fall back to standard functorch handling
    if dim_arg is None:
        info = TensorInfo.create(args[0], ensure_batched=True, ensure_present=False)
        if not info:
            return wrapper.orig(*args, **kwargs)

        with EnableAllLayers(info.levels) as guard:
            assert info.batchedtensor is not None
            guard.inplace_update_layers(info.batchedtensor, info.levels)
            new_args = list(args)
            new_args[0] = handle_from_tensor(info.batchedtensor)
            result = wrapper.orig(*new_args, **kwargs)
            return guard.from_batched(result, info.has_device)

    # Handle dimension-aware operation
    info = TensorInfo.create(args[0])
    if not info:
        return wrapper.orig(*args, **kwargs)

    # Check for keepdim parameter
    keepdim = False
    if wrapper.reduce:
        keepdim_arg = kwargs.get("keepdim")
        if keepdim_arg is None and wrapper.keepdim_offset < len(args):
            keepdim_idx = wrapper.keepdim_offset + 1
            if keepdim_idx < len(args):
                keepdim_arg = args[keepdim_idx]
        if keepdim_arg is not None:
            keepdim = bool(keepdim_arg)

    # Wrap dimensions
    ndim = info.ndim()
    dims = _wrap_dims(dim_arg, ndim, keepdim)

    # Convert dimensions to indices and validate
    dim_indices: list[int] = []
    seen = [False] * len(info.levels)

    for d in dims:
        midx = None
        for i, level in enumerate(info.levels):
            if level == d:
                midx = i
                break

        if midx is None:
            # Try to match by position/name more flexibly
            for i, level in enumerate(info.levels):
                if hasattr(level, "matches") and level.matches(d):
                    midx = i
                    break

            if midx is None:
                level_strs = [str(level) for level in info.levels]
                raise ValueError(
                    f"Tensor with dimensions {level_strs} does not contain {d}"
                )

        seen[midx] = True
        dim_indices.append(midx)

    # Determine new levels after reduction
    new_levels = []
    if wrapper.reduce and not keepdim:
        for i, level in enumerate(info.levels):
            if not seen[i]:
                new_levels.append(level)
    else:
        new_levels = info.levels[:]

    # Create dimension indices for the original function
    if len(dim_indices) == 1:
        py_indices: Any = dim_indices[0]
    else:
        py_indices = tuple(dim_indices)

    # Update arguments
    new_args = list(args)
    new_kwargs = kwargs.copy()
    assert info.tensor is not None
    new_args[0] = handle_from_tensor(info.tensor)

    # Update dimension argument
    if wrapper.dim_name in new_kwargs:
        new_kwargs[wrapper.dim_name] = py_indices
    else:
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(new_args):
            new_args = list(new_args)
            new_args[dim_idx] = py_indices

    # Call original function
    result = wrapper.orig(*new_args, **new_kwargs)

    # Wrap results
    def wrap_result(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            from . import Tensor

            return Tensor.from_positional(obj, new_levels, info.has_device)
        return obj

    return tree_map(wrap_result, result)


def _wrap(
    orig: Callable,
    dim_offset: Optional[int] = None,
    keepdim_offset: Optional[int] = None,
    dim_name: Optional[str] = None,
    single_dim: Optional[bool] = None,
    reduce: Optional[bool] = None,
) -> Callable:
    """
    Wrap a PyTorch function to support first-class dimensions.

    Args:
        orig: Original function to wrap
        dim_offset: Offset for dimension argument (default: 0)
        keepdim_offset: Offset for keepdim argument (default: 1)
        dim_name: Name of dimension parameter (default: "dim")
        single_dim: Whether function takes single dimension (default: False)
        reduce: Whether function reduces dimensions (default: True)
    """
    dim_name = dim_name or "dim"

    wrapper = WrappedOperator(orig, patched_dim_method, dim_name)

    if dim_offset is not None:
        wrapper.dim_offset = dim_offset
    if keepdim_offset is not None:
        wrapper.keepdim_offset = keepdim_offset
    if single_dim is not None:
        wrapper.single_dim = single_dim
    if reduce is not None:
        wrapper.reduce = reduce

    return wrapper.function()


def call_torch_function(
    wrapper: WrappedOperator,
    func: Callable,
    types: tuple,
    args: tuple = (),
    kwargs: Optional[dict] = None,
) -> Any:
    """
    Handle __torch_function__ calls for wrapped operators.
    """
    if kwargs is None:
        kwargs = {}

    # Import here to avoid circular imports
    from . import _Tensor

    # Use the torch function mechanism from _Tensor
    return _Tensor.__torch_function__(func, types, args, kwargs)
