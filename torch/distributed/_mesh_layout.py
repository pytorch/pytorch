"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

import torch
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
        layout = coalesce(self)
        return _MeshLayout(layout.shape, layout.stride)

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        """
        By-dimension composition allows one layout to "select from" or "filter through" another layout.
        Think of it as function composition: (self ∘ layout)(input) = self(layout(input))
        between two layouts. This function is a wrapper of pycute's composition.

        Mental model about how to understand the composition logic:
        - The LEFT layout (self) defines the "output space" - what indices are possible
        - The RIGHT layout (layout parameter) acts as a "selector" - which specific indices to pick
        - The composition only generates indices that the left layout could originally produce,
          but the right layout determines which indices to be picked.
        - The stride of the composition layout will not be smaller than the stride of the right layout,
          because when picking the indices the composition will at least follow the the right layout's stride
          to move forward.

        Example:
          self = (6,2):(2,1)      # sizes=(6,2), strides=(2,1)
          layout = (3:2)          # sizes=(3,), stride=(2,)
          self o layout = (3:2)

        Returns:
          Layout being composed.
        """
        result = composition(self, layout)
        return _MeshLayout(result.shape, result.stride)

    def complement(self, world_size: int) -> "_MeshLayout":
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
        layout = complement(self, world_size)
        return _MeshLayout(layout.shape, layout.stride)

    def all_ranks_from_zero(self) -> list[int]:
        """
        This function computes the all ranks specified by the layout staring from zero.

        How it works:
        1. we enumerates every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            all_ranks_from_zero = sum(coord[i] * strides[i] for i in range(ndim))

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

        The nested list forms the Cartesian product of all ranks for one layout and offset
        regarding filling up the world_size with the layout.
        The final global ranks are the addition of these two. The result is a
        list of lists: one sublist per layout. This rank list will be used to build
        the communicator underlying the layout and the given `world_size`.

        Example:
        world_size = 16
        self.size = 4
        self.stride = 1
        ranks = [0, 1, 2, 3]
        offsets = [0, 4, 8, 12]
        result = [
            [0+0, 0+1, 0+2, 0+3],  # → [0, 1, 2, 3]
            [4+0, 4+1, 4+2, 4+3],  # → [4, 5, 6, 7]
            [8+0, 8+1, 8+2, 8+3],  # → [8, 9, 10,11]
            [12+0, 12+1, 12+2, 12+3],  # → [12,13,14,15]
        ]
        """
        return [
            [offset + rank for rank in self.all_ranks_from_zero()]
            for offset in self.complement(world_size).all_ranks_from_zero()
        ]

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
        ranks = self.all_ranks_from_zero()
        return len(ranks) == len(set(ranks))

    def remap_to_tensor(
        self,
        mesh_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Leverage layout as an index for mesh tensor that re-maps the indexes after layout
        transformation to actual device ranks.

        With this method, the cute layout serves as the backend of indices bookkeeping for the
        mesh tensor when it comes to flatten, unflatten and slicing operations. The actual mesh
        tensor still represents the actual device assignment and ranks. We need this function
        to specify device allocation and create backend for a mesh. Although any transform of mesh tensors
        can be treated as a view or subset of mesh tensor, we do need to use the actual view or
        sub-tensor for DeviceMesh and its backend creation.

        The shape of the `mesh_tensor` can be any size because users can define a device mesh with any
        shapes. But we can further refactor the code so that internally we can only support 1D mesh tensor
        and reconstruct the mesh tensor with the shape of the layout when accessed by users.
        #TODO: Only support 1D mesh tensor stored internally and reconstruct the mesh tensor via layout.

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
            mesh_tensor: The concrete mesh tensor with actual device ranks

        Returns:
            torch.Tensor: A tensor representing the actual device allocation from mesh_tensor
        """
        complement_layout = self.complement(mesh_tensor.numel())

        return (
            mesh_tensor.flatten()
            .as_strided(
                flatten(complement_layout.sizes) + flatten(self.sizes),
                flatten(complement_layout.strides) + flatten(self.strides),
            )
            .reshape(-1, *(self[i].numel() for i in range(len(self))))
        )
