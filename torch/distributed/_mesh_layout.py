"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

from torch.distributed._pycute import (
    # coalesce,
    # complement,
    # composition,
    flatten,
    is_tuple,
    Layout,
    IntTuple,
)


NestedIntTuple: TypeAlias = tuple["int | NestedIntTuple", ...]


@dataclass(frozen=True, init=True)
class _Layout(Layout):
    """
    Utility class for representing an integer layout by borrowing ideas from CuTe Layout Algebra.
    See https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html for more details.

    Each layout is represented as a list of sizes and strides. We use it as a way for mechanical bookkeeping
    of the integers such as ranks in a SPMD mesh, and the transformation on top of it.

    Lots of methods of layout like coalesce, composition, complement, etc. are borrowed from pycute.
    https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/python/pycute/layout.py#L137,L257

    Note this is a CuTe-inspired layout, because CuTe uses co-lexicographic way in linearization while PyTorch
    is using lexicographic. So even though the CuTe documentation can still be referenced, the implementation will be
    different from that of PyCute's.
    """

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
    
    @staticmethod
    def ceil_div(n: int, m: int) -> int:
        return (n + m - 1) // m

    def coalesce(self) -> "_Layout":
        """
        A layout is represented by (sizes):(strides), e.g. (3,2):(4,2).
        Two consecutive dimensions can be "merged" into one if their
        strides are contiguous/multiplicative (i.e., the inner stride * inner size
        equals the next stride), we perform this kind of merge inside coalesce.

        Example 1 (simple): (3,2):(2,1)
        - inner dimension: has stride=1, size=2
        - outer dimension: stride = inner_stride * inner_size = 2
        → coalesced = (6:1)    # acts like a flat 1D array of length 6

        Example 2 (non-coalescible): (3,2):(4,1)
        - inner dimension: stride=1, size=2 → 2*1 = 2
        - outer dimension: stride=4, mismatch (≠ 2)
        → cannot merge; result stays (3,2):(4,1)
        """
        sizes: list[int] = []
        strides: list[int] = []
        for size, stride in self.sizes_and_strides:
            if size == 1:
                pass
            elif strides and strides[-1] == size * stride:
                strides.pop()
                prev_size = sizes.pop()
                sizes.append(size * prev_size)
                strides.append(stride)
            else:
                sizes.append(size)
                strides.append(stride)
        return _Layout(tuple(sizes), tuple(strides))

    def composition(self, layout: "_Layout") -> "_Layout":
        """
        Perform a by-dimension composition between this layout (self) and another layout (layout).

        Mental model about how to understand the composition logic:
        - Think of each dimension in this layout as a "slot" that can itself be
          refined by another layout.
        - Composition substitutes one of these slots with the provided `layout`, producing
          a new combined layout with updated sizes and strides.
        - For each (size, stride) pair in the layout, we expand the dimension
          according to the self layout's structure.

        Example:
          self = (6,2):(2,1)      # sizes=(6,2), strides=(2,1)
          layout = (3:2)          # sizes=(3,), stride=(2,)
          self o layout = (3:4)

        Returns:
          A list of composed layouts.
        """
        assert len(self.sizes) >= 1, "Layout for composition cannot be empty"
        # A layout can be expressed as the concatenation of its sublayouts.
        # When layout is injective (aka one-to-one), composition is left-distributive with concatenation.
        # We return a flattened list of list of self compose with each sublayout.
        if len(layout.sizes) > 1:
            layouts = (self.composition(layout_i) for layout_i in layout) # type: ignore[attr-defined]
            zip_res_sizes, zip_res_strides = zip(
                *((a.sizes, a.strides) for a in layouts)
            )
            return _Layout(tuple(zip_res_sizes), tuple(zip_res_strides))

        res_sizes: list[int] = []
        res_strides: list[int] = []
        # Since we now only compose with single-size sublayout, we can assume numel_so_far is always from strides[0].
        numel_so_far = layout.strides[0]
        sub_size = layout.sizes[0]
        assert isinstance(numel_so_far, int)
        assert isinstance(sub_size, int)
        sub_stride = numel_so_far
        numel_so_far *= sub_size

        # when self is multi-dimensional sublayout, aka, self = (a,b,...,c):(x,y,...,z), layout = s:d,
        # for integral s and d means that we want:
        # (1) “remove” the first d elements from self. (This will increase the stride.)
        # (2) “keep” the first s of those strided elements. (This does not affect the stride.)
        # For example, if self = (6,2):(2,1), layout = (3:2)
        # Step 1: remove the first 2 elements from self with stride increase, i.e., (6,2):(2,1) -> (3,2):(4,1)
        # Step 2: keep the first 3 of those strided elements, i.e., (3,2):(4,1) -> (3,1):(4,1)
        flattened_self = self.coalesce()
        for curr_size, curr_stride in zip(
            flatten(flattened_self.sizes)[:-1],
            flatten(flattened_self.sizes)[:-1],
        ):
            assert curr_size % sub_stride == 0 or sub_stride % curr_size == 0, (
                "Layouts do not meet stride divisibility condition"
            )
            new_size = min(max(1, curr_size // sub_stride), sub_size)
            if new_size != 1:
                res_sizes.append(new_size)
                res_strides.append(sub_stride * curr_stride)
            assert sub_size % new_size == 0, (
                "Layouts do not meet size divisibility condition"
            )
            sub_size = sub_size // new_size
            sub_stride = self.ceil_div(sub_stride, curr_size)

        # When self is integral and has single-size sublayout, aka, self = a:b, layout = s:d,
        # the result is rather trivial: self o layout = a:b o s:d = s:(b*d).
        # For example, if self = (6:2), layout = (3:2), the result is (3:(2*2)) = (3:4).
        if sub_size != 1 or len(res_sizes) == 0:
            res_sizes.append(sub_size)
            last_stride = flattened_self.strides[-1]
            assert isinstance(last_stride, int)
            res_strides.append(sub_stride * last_stride)

        return _Layout(tuple(res_sizes), tuple(res_strides))

    def complement(self, world_size: int) -> "_Layout":
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
        res_sizes: list[int] = []
        res_strides: list[int] = []
        current_idx = 1
        for size, stride in sorted(self.sizes_and_strides, key=lambda x: x[1]):
            if stride == 0 or size == 1:
                continue
            assert current_idx <= size * stride, (
                f"current_idx {current_idx} larger than numel so far {size * stride}."
            )

            res_sizes.append(stride // current_idx)
            res_strides.append(current_idx)
            current_idx = size * stride

        res_sizes.append(self.ceil_div(world_size, current_idx))
        res_strides.append(current_idx)
        # This is different from original pycute implementation, because we want to follow the lexicographic order here
        # where the right-most dimension is the innermost dimension (smallest stride).
        res_sizes.reverse()
        res_strides.reverse()

        return _Layout(tuple(res_sizes), tuple(res_strides)).coalesce()

    def local_ranks(self) -> list[int]:
        """
        This function computes the local rank specified by the layout.

        How it works:
        1. we enumerates every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            local_ranks = sum(coord[i] * strides[i] for i in range(ndim))

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
            sum(c * s for c, s in zip(coord, flatten(self.strides)))
            for coord in product(*(range(s) for s in flatten(self.sizes)))
        ]

    def global_ranks(self, world_size: int) -> list[list[int]]:
        """
        Build global ranks specified by the layout via two-level ranks composition.

        The nested list forms the Cartesian product of group ranks and group offset
        and the final global ranks are the addition of these two. The result is a
        list of lists: one sublist per group. This rank list will be used to build
        the communicator underlying the layout.

        Example:
        world_size = 16
        self.size = 4
        self.stride = 1
        group ranks = [0, 1, 2, 3]
        group offsets = [0, 4, 8, 12]
        result = [
            [0+0, 0+1, 0+2, 0+3],  # → [0, 1, 2, 3]
            [4+0, 4+1, 4+2, 4+3],  # → [4, 5, 6, 7]
            [8+0, 8+1, 8+2, 8+3],  # → [8, 9, 10,11]
            [12+0, 12+1, 12+2, 12+3],  # → [12,13,14,15]
        ]
        """
        return [
            [group_offset + group_rank for group_rank in self.local_ranks()]
            for group_offset in self.complement(world_size).local_ranks()
        ]


MeshLayoutType = tuple["_Layout", ...]


@dataclass(frozen=True)
class _MeshLayout:
    _layouts: MeshLayoutType

    @property
    def layouts(self) -> MeshLayoutType:
        return self._layouts

    # @staticmethod
    # def to_single_depth_layouts(
    #     mesh_size: IntTuple, mesh_stride: IntTuple
    # ) -> "_MeshLayout":
    #     """
    #     Convert a contiguous PyTorch mesh tensor's metadata (size, stride) into a list of layouts.
    #     If there are no transformation is being made to a device mesh, each dim of mesh is represented
    #     by a layout but this is not the case once the mesh has been transformed, unflatten for example.

    #     For each dimension of the input tensor, we extract its size (mesh.size(i)) and stride (mesh.stride(i)),
    #     and then wrap that (size, stride) pair into a 1D layout.

    #     The result is a list of independent layout, one per dimension.
    #     Each layout describes how indices in that dimension map to backend ranks.

    #     Example:
    #         Suppose mesh.shape = (3, 4), mesh.stride = (4, 1).
    #         Then:
    #         layouts[0] = (3:4)   # first dimension: size=3, stride=4
    #         layouts[1] = (4:1)   # second dimension: size=4, stride=1
    #     """
    #     if len(mesh_size) != len(mesh_stride):
    #         raise ValueError("mesh_size and mesh_stride must have the same length")
    #     return _MeshLayout(
    #         tuple(
    #             _Layout((size,), (stride,))
    #             for size, stride in zip(mesh_size, mesh_stride)
    #         )
    #     )
