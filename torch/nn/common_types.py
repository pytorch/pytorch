from typing import Optional, TypeVar, Union
from typing_extensions import TypeAlias as _TypeAlias

from torch import Tensor


# ruff: noqa: PYI042,PYI047

# Create some useful type aliases

# Template for arguments which can be supplied as a tuple, or which can be a scalar which PyTorch will internally
# broadcast to a tuple.
# Comes in several variants: A tuple of unknown size, and a fixed-size tuple for 1d, 2d, or 3d operations.
T = TypeVar("T")
_scalar_or_tuple_any_t: _TypeAlias = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t: _TypeAlias = Union[T, tuple[T]]
_scalar_or_tuple_2_t: _TypeAlias = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t: _TypeAlias = Union[T, tuple[T, T, T]]
_scalar_or_tuple_4_t: _TypeAlias = Union[T, tuple[T, T, T, T]]
_scalar_or_tuple_5_t: _TypeAlias = Union[T, tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t: _TypeAlias = Union[T, tuple[T, T, T, T, T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t: _TypeAlias = _scalar_or_tuple_any_t[int]
_size_1_t: _TypeAlias = _scalar_or_tuple_1_t[int]
_size_2_t: _TypeAlias = _scalar_or_tuple_2_t[int]
_size_3_t: _TypeAlias = _scalar_or_tuple_3_t[int]
_size_4_t: _TypeAlias = _scalar_or_tuple_4_t[int]
_size_5_t: _TypeAlias = _scalar_or_tuple_5_t[int]
_size_6_t: _TypeAlias = _scalar_or_tuple_6_t[int]

# For arguments which represent optional size parameters (eg, adaptive pool parameters)
_size_any_opt_t: _TypeAlias = _scalar_or_tuple_any_t[Optional[int]]
_size_2_opt_t: _TypeAlias = _scalar_or_tuple_2_t[Optional[int]]
_size_3_opt_t: _TypeAlias = _scalar_or_tuple_3_t[Optional[int]]

# For arguments that represent a ratio to adjust each dimension of an input with (eg, upsampling parameters)
_ratio_2_t: _TypeAlias = _scalar_or_tuple_2_t[float]
_ratio_3_t: _TypeAlias = _scalar_or_tuple_3_t[float]
_ratio_any_t: _TypeAlias = _scalar_or_tuple_any_t[float]

_tensor_list_t: _TypeAlias = _scalar_or_tuple_any_t[Tensor]

# For the return value of max pooling operations that may or may not return indices.
# With the proposed 'Literal' feature to Python typing, it might be possible to
# eventually eliminate this.
_maybe_indices_t: _TypeAlias = _scalar_or_tuple_2_t[Tensor]
