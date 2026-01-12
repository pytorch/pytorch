"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import NoReturn

import torch
from torch.distributed._pycute import (
    as_tuple,
    coalesce as pycute_coalesce,
    complement as pycute_complement,
    composition as pycute_composition,
    flatten,
    IntTuple,
    is_int,
    is_tuple,
    Layout,
    make_layout,
    match_structure,
    suffix_product,
)


@dataclass(frozen=True)
class _FlatLayout:
    """
    A canonical CuTe layout for a single dimension of a DeviceMesh

    This layout is _not_ itself subdivided into multiple dimensions. It might
    internally sometimes use multidimensional tuple to represent "irregular"
    layout (e.g., flattening non-adjacent dims), but this should be considered
    an opaque implementation detail.

    This class guarantees that all equivalent layouts are encoded as the same
    normalized representation, and thus compare equal. This is achieved by
    flattening and coalescing compatible adjacent dimensions (which includes
    removing all dimensions of size 1).

    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape: IntTuple, stride: IntTuple | None = None) -> None:
        """
        Create a _FlatLayout from shape and optional stride.

        Args:
            shape: Shape as an IntTuple (can be nested, will be normalized)
            stride: Stride as an IntTuple (can be nested, will be normalized).
                    If None, computes contiguous strides using suffix_product.
        """
        if not is_tuple(shape) and not is_int(shape):
            raise TypeError(f"shape must be a tuple or int, got {type(shape)}")
        stride = stride if stride is not None else suffix_product(shape)
        if not is_tuple(stride) and not is_int(stride):
            raise TypeError(f"stride must be a tuple or int, got {type(stride)}")
        if not match_structure(shape, stride):
            raise ValueError(f"sizes {shape} and strides {stride} don't match")

        coalesced_layout = pycute_coalesce(Layout(shape, stride))
        flat_shape = flatten(coalesced_layout.shape)
        flat_stride = flatten(coalesced_layout.stride)

        assert tuple(sorted(flat_stride, reverse=True)) == flat_stride, (
            "For the time being we don't support transposing mesh dimensions "
            "hence layouts should always remain sorted. If we choose to allow this, "
            "we first need to decide whether we consider [0, 1, 2, 3] and "
            "[0, 2, 1, 3] to be the same layout (and thus the same ProcessGroup)."
        )

        # TODO could we also assert that it's non-overlapping?

        # pycute will preserve a size=1 dim if it's the only remaining dim, but
        # we prefer to stick to the Tensor convention and make it 0-dimensional
        if flat_shape == (1,) and flat_stride == (0,):
            flat_shape = ()
            flat_stride = ()

        # Set attributes using object.__setattr__ since frozen=True
        object.__setattr__(self, "shape", flat_shape)
        object.__setattr__(self, "stride", flat_stride)

    def __len__(self) -> NoReturn:
        raise RuntimeError(
            "You should never need to know the length of the internal representation of a FlatLayout"
        )

    def __getitem__(self, i: int) -> NoReturn:
        raise RuntimeError(
            "You should never need to index into the internal representation of a FlatLayout"
        )

    def numel(self) -> int:
        return math.prod(self.shape)

    def codomain(self) -> list[int]:
        """
        This function computes the all ranks specified by the layout staring from zero.

        How it works:
        1. we enumerates every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            codomain = sum(coord[i] * strides[i] for i in range(ndim))

        Example A:
        sizes = (2, 3)        # 2 rows, 3 cols
        strides = (3, 1)        # row-major layout
        coords = (0,0) -> 0*3 + 0*1 = 0
                 (0,1) -> 0*3 + 1*1 = 1
                 (0,2) -> 0*3 + 2*1 = 2
                 (1,0) -> 1*3 + 0*1 = 3
                 (1,1) -> 1*3 + 1*1 = 4
                 (1,2) -> 1*3 + 2*1 = 5
        result = [0, 1, 2, 3, 4, 5]

        Example B:
        sizes = (2, 3)
        strides = (1, 2)        # non-standard / strided layout
        coords = (0,0) -> 0*1 + 0*2 = 0
                 (0,1) -> 0*1 + 1*2 = 2
                 (0,2) -> 0*1 + 2*2 = 4
                 (1,0) -> 1*1 + 0*2 = 1
                 (1,1) -> 1*1 + 1*2 = 3
                 (1,2) -> 1*1 + 2*2 = 5
        result = [0, 2, 4, 1, 3, 5]
        """
        return [
            sum(c * s for c, s in zip(coord, self.stride))
            for coord in product(*(range(s) for s in self.shape))
        ]

    def complement(self, world_size: int) -> "_FlatLayout":
        """
        Compute the "complement layout" relative to a given world_size.
        A complement layout fills in the "missing" factor so that: self repeat a layout of complement(self, world_size)
        will get a complete world_size. We use ⊗ to denote the repeat operation.

        Example:
          self = (4:1)   # size=4, stride=1
          world_size = 8
          Then:
            complete needed factor = 8 / 4 = 2
            complement(self, 8) = (2:1)

          Together they form:
            (4:1) ⊗ (2:1) = (4,2):(2,1)
          which has world_size = 4 * 2 = 8, as required.

        In distributed terms, complement() is often used to derive the "other"
        rank grouping when splitting processes into 2D meshes.

        For a visualized explanation, see https://x.com/ezyang/status/1962364978393981433/
        """
        result = pycute_complement(self.to_pycute(), world_size)
        return _FlatLayout(result.shape, result.stride)

    def composition(self, other: "_ListOfFlatLayouts") -> "_ListOfFlatLayouts":
        """
        By-dimension composition allows one layout to "select from" or "filter through" another layout.
        Think of it as function composition: (self ∘ layout)(input) = self(layout(input))
        between two layouts. This function is a wrapper of pycute's composition.

        The composition preserves the structure of the right operand (other), returning a
        _ListOfFlatLayouts with the same number of axes.

        Mental model about how to understand the composition logic:
        - The LEFT layout (self) defines the "output space" - what indices are possible
        - The RIGHT layout (other parameter) acts as a "selector" - which specific indices to pick
        - The composition only generates indices that the left layout could originally produce,
          but the right layout determines which indices to be picked.
        - The stride of the composition layout will not be smaller than the stride of the right layout,
          because when picking the indices the composition will at least follow the the right layout's stride
          to move forward.

        Example:
          self = (6,2):(2,1)      # sizes=(6,2), strides=(2,1)
          other = _ListOfFlatLayouts with 2 axes: [(3,):(2,), (2,):(1,)]
          self o other = _ListOfFlatLayouts with 2 axes preserving structure

        Args:
            other: A _ListOfFlatLayouts whose structure will be preserved in the result

        Returns:
            A _ListOfFlatLayouts with the same number of axes as other
        """
        result = pycute_composition(self.to_pycute(), other.to_pycute())
        result_axes = [
            _FlatLayout(shape, stride)
            for shape, stride in zip(as_tuple(result.shape), as_tuple(result.stride))
        ]
        return _ListOfFlatLayouts(result_axes)

    def to_pycute(self) -> Layout:
        """Convert to a pycute Layout for compatibility with pycute operations."""
        if not self.shape:
            return Layout(1, 0)
        return Layout(self.shape, self.stride)

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        """Iterate over (size, stride) pairs for each dimension."""
        return zip(self.shape, self.stride)

    def __str__(self) -> str:
        return f"{self.shape}:{self.stride}"


@dataclass(frozen=True)
class _ListOfFlatLayouts:
    """
    A list of normalized layouts, one per mesh dimension.

    This class represents the layout of a DeviceMesh, containing one _FlatLayout
    per mesh dimension. It provides operations for manipulating mesh layouts
    while maintaining the invariant that each axis is always normalized.

    The structure is always a flat tuple of _FlatLayout objects (one per mesh dimension).
    """

    axes: tuple[_FlatLayout, ...]

    def __init__(self, axes: Sequence[_FlatLayout]) -> None:
        """
        Create a _ListOfFlatLayouts from a sequence of _FlatLayout objects.

        Args:
            axes: Sequence of _FlatLayout, one per mesh dimension
        """
        object.__setattr__(self, "axes", tuple(axes))

    @classmethod
    def from_sizes_strides(
        cls, sizes: tuple[int, ...], strides: tuple[int, ...] | None = None
    ) -> "_ListOfFlatLayouts":
        """
        Create a _ListOfFlatLayouts from top-level sizes and optional strides.

        Each size/stride pair becomes a single-element _FlatLayout.

        Args:
            sizes: Tuple of sizes, one per mesh dimension
            strides: Tuple of strides, one per mesh dimension.
                     If None, computes contiguous strides using suffix_product.

        Returns:
            A new _ListOfFlatLayouts with one axis per dimension
        """
        if strides is None:
            strides = flatten(suffix_product(sizes))
        assert len(sizes) == len(strides)
        axes = tuple(_FlatLayout((s,), (d,)) for s, d in zip(sizes, strides))
        return cls(axes)

    def __len__(self) -> int:
        return len(self.axes)

    def __getitem__(self, i: int) -> _FlatLayout:
        return self.axes[i]

    def __iter__(self) -> Iterator[_FlatLayout]:
        return iter(self.axes)

    def to_pycute(self) -> Layout:
        if len(self.axes) == 0:
            return Layout(1, 0)
        return make_layout(*(axis.to_pycute() for axis in self.axes))

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        return tuple(axis.numel() for axis in self.axes)

    def numel(self) -> int:
        return math.prod(axis.numel() for axis in self.axes)

    def cosize(self) -> int:
        return self.to_pycute().cosize()

    def merge_axes_into_one(self) -> _FlatLayout:
        """
        Merge all axes into a single _FlatLayout.

        Combines all dimensions across all axes into one axis,
        with flattening and coalescing handled by the _FlatLayout constructor.
        """
        shapes = tuple(axis.shape for axis in self.axes)
        strides = tuple(axis.stride for axis in self.axes)
        return _FlatLayout(shapes, strides)

    def splice(
        self, start: int, end: int, layout: "_ListOfFlatLayouts"
    ) -> "_ListOfFlatLayouts":
        """
        Replace axes[start:end] with the axes from another layout.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            layout: The _ListOfFlatLayouts whose axes will be inserted

        Returns:
            A new _ListOfFlatLayouts with the specified axes replaced
        """
        new_axes = list(self.axes)
        new_axes[start:end] = list(layout.axes)
        return _ListOfFlatLayouts(new_axes)

    def check_non_overlap(self) -> bool:
        """
        Check if the layout has any overlap between the ranks it generates. If there is overlap,
        we return False, otherwise True.

        The layout is supposed to be injective i.e, aside from indice 0, indices from each
        dim of the layout must be non-overlapping.

        Example 1 - Valid (no overlap):
        Layout: sizes=(2,3), strides=(6,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=6, span=2*6=12, covers indices [0,6]
        → No overlap since 6 > 3

        Example 2 - Invalid (overlap):
        Layout: sizes=(2,3), strides=(2,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=2, span=2*2=4, covers indices [0,2]
        → Overlap! stride=2 < span=3, so indices [0,2] are duplicated

        Example 3 - Invalid (overlap):
        Layout: sizes=(4,2), strides=(1,1)
        - Dim 1: stride=1, span=4, covers indices [0,1,2,3]
        - Dim 0: stride=1, span=2, covers indices [0,1]
        → Overlap! stride is same for two dims, so indices [0,2] are duplicated

        Returns:
            bool: True if no overlap, False if overlap detected
        """
        ranks = self.merge_axes_into_one().codomain()
        return len(ranks) == len(set(ranks))

    def remap_to_tensor(self, rank_map: torch.Tensor) -> torch.Tensor:
        """
        Leverage layout as an index for mesh tensor that re-maps the indexes after layout
        transformation to actual device ranks.

        With this method, the cute layout serves as the backend of indices bookkeeping for the
        mesh tensor when it comes to flatten, unflatten and slicing operations. The actual mesh
        tensor still represents the actual device assignment and ranks. We need this function
        to specify device allocation and create backend for a mesh. Although any transform of mesh tensors
        can be treated as a view or subset of mesh tensor, we do need to use the actual view or
        sub-tensor for DeviceMesh and its backend creation.

        The shape of the `rank_map` must be 1D and contiguous.

        Examples:

        Case 1 - Consecutive ranks, full world:
            original_mesh_tensor = [[0,1],[2,3]]  # 2x2 mesh, ranks 0-3
            world_size = 4
            layout = Layout(2:2)
            Return: [[0,2],[1,3]]

        Case 2 - Non-consecutive ranks:
            original_mesh_tensor = [[10,20],[30,40]]  # custom rank assignment
            world_size = 4
            layout = Layout(2:2)
            Return: [[[10,30],[20,40]]]

        Args:
            rank_map: The concrete mesh tensor with actual device ranks

        Returns:
            torch.Tensor: A tensor representing the actual device allocation from rank_map
        """
        assert rank_map.ndim == 1
        assert rank_map.is_contiguous()
        assert rank_map.numel() >= self.cosize()

        complement_layout = self.merge_axes_into_one().complement(rank_map.numel())

        shapes: list[int] = [*complement_layout.shape]
        strides: list[int] = [*complement_layout.stride]
        for axis in self.axes:
            shapes.extend(axis.shape)
            strides.extend(axis.stride)

        return rank_map.as_strided(tuple(shapes), tuple(strides)).reshape(
            -1, *self.top_level_sizes
        )

    def __str__(self) -> str:
        axes_str = ", ".join(str(axis) for axis in self.axes)
        return f"[{axes_str}]"


# Alias for backwards compatibility
_MeshLayout = _ListOfFlatLayouts
