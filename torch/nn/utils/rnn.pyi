from typing import (
    Any,
    List,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from torch import Tensor
from torch.types import _device, _dtype

class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Optional[Tensor]
    unsorted_indices: Optional[Tensor]

def bind(optional: Any, fn: Any): ...

T = TypeVar("T")

class PackedSequence(PackedSequence_):
    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Optional[Tensor] = ...,
        sorted_indices: Optional[Tensor] = ...,
        unsorted_indices: Optional[Tensor] = ...,
    ) -> PackedSequence: ...
    def pin_memory(self: T) -> T: ...
    def cuda(self: T, *args: Any, **kwargs: Any) -> T: ...
    def cpu(self: T) -> T: ...
    def double(self: T) -> T: ...
    def float(self: T) -> T: ...
    def half(self: T) -> T: ...
    def long(self: T) -> T: ...
    def int(self: T) -> T: ...
    def short(self: T) -> T: ...
    def char(self: T) -> T: ...
    def byte(self: T) -> T: ...
    @overload
    def to(
        self: T,
        dtype: _dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> T: ...
    @overload
    def to(
        self: T,
        device: Optional[Union[_device, str]] = None,
        dtype: Optional[_dtype] = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> T: ...
    @overload
    def to(
        self: T,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> T: ...
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
    sequences: List[Tensor],
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
