"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing_extensions import TypeAlias

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
class _MeshLayout(Layout):
    shape: NestedIntTuple
    stride: NestedIntTuple

    def __post_init__(self) -> None:
        if not is_tuple(self.shape):
            raise TypeError(f"shape must be a tuple, got {type(self.shape)}")
        if not is_tuple(self.stride):
            raise TypeError(f"stride must be a tuple, got {type(self.stride)}")
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
    def __getitem__(self, i: int) -> "_MeshLayout":
        size = self.sizes[i]
        stride = self.strides[i]
        if is_tuple(size) and is_tuple(stride):
            return _MeshLayout(size, stride)  # type: ignore[arg-type]
        elif isinstance(size, int) and isinstance(stride, int):
            return _MeshLayout((size,), (stride,))
        else:
            raise ValueError("size and stride must be either int or tuple")

    def coalesce(self) -> "_MeshLayout":
        layout = coalesce(self)
        return _MeshLayout(layout.shape, layout.stride)  # type: ignore[arg-type]

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        result = composition(self, layout)
        return _MeshLayout(result.shape, result.stride)  # type: ignore[arg-type]

    def complement(self, max_idx: int) -> "_MeshLayout":
        layout = complement(self, max_idx)
        return _MeshLayout(layout.shape, layout.stride)  # type: ignore[arg-type]
