"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

from torch.distributed._pycute import (
    coalesce,
    complement,
    composition,
    flatten,
    is_tuple,
    Layout,
)


NestedIntTuple: TypeAlias = tuple["int | NestedIntTuple", ...]


@dataclass(frozen=True, init=True)
class _Layout(Layout):
    shape: NestedIntTuple
    stride: NestedIntTuple

    def __post_init__(self) -> None:
        if not is_tuple(self.shape):
            raise ValueError(f"shape must be a tuple, got {type(self.shape)}")
        if not is_tuple(self.stride):
            raise ValueError(f"stride must be a tuple, got {type(self.stride)}")
        if len(flatten(self.shape)) != len(flatten(self.stride)):
            raise ValueError(
                f"sizes {len(flatten(self.shape))} and "
                f"strides {len(flatten(self.stride))} must have the same length"
            )

    @property
    def sizes(self) -> NestedIntTuple:
        return self.shape

    @property
    def strides(self) -> NestedIntTuple:
        return self.stride

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        return zip(flatten(self.shape), flatten(self.stride))  # type: ignore[arg-type]

    def numel(self) -> int:
        return math.prod(flatten(self.shape))

    # operator []    (get-i like tuples)
    def __getitem__(self, i: int) -> "_Layout":
        size = self.sizes[i]
        stride = self.strides[i]
        if is_tuple(size) and is_tuple(stride):
            return _Layout(size, stride)  # type: ignore[arg-type]
        elif isinstance(size, int) and isinstance(stride, int):
            return _Layout((size,), (stride,))
        else:
            raise ValueError("size and stride must be either int or tuple")

    def coalesce(self) -> "_Layout":
        layout = coalesce(self)
        return _Layout(layout.shape, layout.stride)  # type: ignore[arg-type]

    def composition(self, layout: "_Layout") -> "_Layout":
        result = composition(self, layout)
        return _Layout(result.shape, result.stride)  # type: ignore[arg-type]

    def complement(self, max_idx: int) -> "_Layout":
        layout = complement(self, max_idx)
        return _Layout(layout.shape, layout.stride)  # type: ignore[arg-type]
