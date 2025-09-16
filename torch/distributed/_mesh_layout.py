"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

from torch.distributed._pycute import (
    coalesce,
    complement,
    composition,
    flatten,
    IntTuple,
    is_int,
    is_tuple,
    Layout,
)


@dataclass(frozen=True, init=True)
class _MeshLayout(Layout):
    shape: IntTuple
    stride: IntTuple

    def __post_init__(self) -> None:
        if not is_tuple(self.shape) and not is_int(self.shape):
            raise TypeError(f"shape must be a tuple or int, got {type(self.shape)}")
        if not is_tuple(self.stride) and not is_int(self.stride):
            raise TypeError(f"stride must be a tuple or int, got {type(self.stride)}")
        if (
            is_tuple(self.shape)
            and is_tuple(self.stride)
            and len(flatten(self.shape)) != len(flatten(self.stride))
        ):
            raise ValueError(
                f"sizes {len(flatten(self.shape))} and "
                f"strides {len(flatten(self.stride))} must have the same length"
            )

    @property
    def sizes(self) -> IntTuple:
        return self.shape

    @property
    def strides(self) -> IntTuple:
        return self.stride

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        return zip(flatten(self.shape), flatten(self.stride))

    def numel(self) -> int:
        return math.prod(flatten(self.shape))

    # # operator []    (get-i like tuples)
    def __getitem__(self, i: int) -> "_MeshLayout":
        layout = super().__getitem__(i)
        return _MeshLayout(layout.shape, layout.stride)

    def coalesce(self) -> "_MeshLayout":
        layout = coalesce(self)
        return _MeshLayout(layout.shape, layout.stride)

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        result = composition(self, layout)
        return _MeshLayout(result.shape, result.stride)

    def complement(self, world_size: int) -> "_MeshLayout":
        layout = complement(self, world_size)
        return _MeshLayout(layout.shape, layout.stride)
