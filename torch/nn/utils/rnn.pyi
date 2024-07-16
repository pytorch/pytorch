# mypy: allow-untyped-defs
from typing import Any, Iterable, NamedTuple, overload, Sequence
from typing_extensions import Self

from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.types import _dtype

class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Tensor | None
    unsorted_indices: Tensor | None

def bind(optional: Any, fn: Any): ...

class PackedSequence(PackedSequence_):
    def __new__(
        cls,
        data: Tensor,
        batch_sizes: Tensor | None = ...,
        sorted_indices: Tensor | None = ...,
        unsorted_indices: Tensor | None = ...,
    ) -> Self: ...
    def pin_memory(self: Self) -> Self: ...
    def cuda(self: Self, *args: Any, **kwargs: Any) -> Self: ...
    def cpu(self: Self) -> Self: ...
    def double(self: Self) -> Self: ...
    def float(self: Self) -> Self: ...
    def half(self: Self) -> Self: ...
    def long(self: Self) -> Self: ...
    def int(self: Self) -> Self: ...
    def short(self: Self) -> Self: ...
    def char(self: Self) -> Self: ...
    def byte(self: Self) -> Self: ...
    @overload
    def to(
        self: Self,
        dtype: _dtype,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...
    @overload
    def to(
        self: Self,
        device: DeviceLikeType | None = None,
        dtype: _dtype | None = None,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...
    @overload
    def to(
        self: Self,
        other: Tensor,
        non_blocking: bool = False,
        copy: bool = False,
    ) -> Self: ...
    @property
    def is_cuda(self) -> bool: ...
    def is_pinned(self) -> bool: ...

def invert_permutation(permutation: Tensor | None): ...
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
    total_length: int | None = ...,
) -> tuple[Tensor, ...]: ...
def pad_sequence(
    sequences: Tensor | Iterable[Tensor],
    batch_first: bool = False,
    padding_value: float = ...,
) -> Tensor: ...
def pack_sequence(
    sequences: Sequence[Tensor],
    enforce_sorted: bool = ...,
) -> PackedSequence: ...
def get_packed_sequence(
    data: Tensor,
    batch_sizes: Tensor | None,
    sorted_indices: Tensor | None,
    unsorted_indices: Tensor | None,
) -> PackedSequence: ...
