from typing import (
    Any,
    Iterable,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import Self

from torch import Tensor

from torch._prims_common import DeviceLikeType
from torch.types import _dtype

class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Optional[Tensor]
    unsorted_indices: Optional[Tensor]

def bind(optional: Any, fn: Any): ...

_T = TypeVar("_T")

class PackedSequence(PackedSequence_):
    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Optional[Tensor] = ...,
        sorted_indices: Optional[Tensor] = ...,
        unsorted_indices: Optional[Tensor] = ...,
    ) -> Self: ...
    def pin_memory(self: _T) -> _T: ...
    def cuda(self: _T, *args: Any, **kwargs: Any) -> _T: ...
    def cpu(self: _T) -> _T: ...
    def double(self: _T) -> _T: ...
    def float(self: _T) -> _T: ...
    def half(self: _T) -> _T: ...
    def long(self: _T) -> _T: ...
    def int(self: _T) -> _T: ...
    def short(self: _T) -> _T: ...
    def char(self: _T) -> _T: ...
    def byte(self: _T) -> _T: ...
    @overload
    def to(
        self: _T,
        dtype: _dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> _T: ...
    @overload
    def to(
        self: _T,
        device: Optional[DeviceLikeType] = None,
        dtype: Optional[_dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> _T: ...
    @overload
    def to(
        self: _T,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> _T: ...
    @property
    def is_cuda(self) -> bool: ...
    def is_pinned(self) -> bool: ...

def invert_permutation(permutation: Optional[Tensor]): ...
def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor,
    batch_first: bool = ...,
    enforce_sorted: bool = ...,
) -> PackedSequence: ...
def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = ...,
    padding_value: float = ...,
    total_length: Optional[int] = ...,
) -> Tuple[Tensor, ...]: ...
def pad_sequence(
    sequences: Union[Tensor, Iterable[Tensor]],
    batch_first: bool = False,
    padding_value: float = ...,
) -> Tensor: ...
def pack_sequence(
    sequences: Sequence[Tensor],
    enforce_sorted: bool = ...,
) -> PackedSequence: ...
def get_packed_sequence(
    data: Tensor,
    batch_sizes: Optional[Tensor],
    sorted_indices: Optional[Tensor],
    unsorted_indices: Optional[Tensor],
) -> PackedSequence: ...
