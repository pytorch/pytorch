#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

import math
from itertools import product


class _Layout:
    """
    Utility class for representing an integer layout by leveraging CuTe Layout Algebra.
    See https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html for more details.

    Each layout is represented as a list of (size, stride) pairs. We use it as a way for mechanical bookkeeping
    of the integers such as ranks in a SPMD mesh, and the transformation on top of it.

    Lots of methods of layout like coalesce, composition, complement, etc. are borrowed from pycute.
    https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/python/pycute/layout.py#L137,L257
    """

    def __init__(self, sizes_and_strides: tuple[tuple[int, int], ...]):
        self.sizes_and_strides = sizes_and_strides

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sizes_and_strides={self.sizes_and_strides!r})"
        )

    def __str__(self) -> str:
        return f"Sizes/Strides: {self.sizes_and_strides}"

    @property
    def sizes(self) -> list[int]:
        return [s for s, _ in self.sizes_and_strides]

    @property
    def strides(self) -> list[int]:
        return [s for _, s in self.sizes_and_strides]

    def numel(self) -> int:
        return math.prod(self.sizes)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Layout)
            and self.sizes_and_strides == other.sizes_and_strides
        )

    def __hash__(self) -> int:
        return hash(self.sizes_and_strides)

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
        res_sizes_and_strides: list[tuple[int, int]] = []
        for size, stride in self.sizes_and_strides:
            if size == 1:
                pass
            elif (
                res_sizes_and_strides and res_sizes_and_strides[-1][1] == size * stride
            ):
                prev_size, _ = res_sizes_and_strides.pop()
                res_sizes_and_strides.append((size * prev_size, stride))
            else:
                res_sizes_and_strides.append((size, stride))
        return _Layout(tuple(res_sizes_and_strides))

    def composition(self, layout: "_Layout") -> list["_Layout"]:
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
        assert len(self.sizes_and_strides) >= 1, (
            "Layout for composition cannot be empty"
        )
        # A layout can be expressed as the concatenation of its sublayouts.
        # When layout is injective (aka one-to-one), composition is left-distributive with concatenation.
        # We return a flattened list of list of self compose with each sublayout.
        if len(layout.sizes_and_strides) > 1:
            return [
                l
                for ss in layout.sizes_and_strides
                for l in self.composition(_Layout((ss,)))
            ]

        res: list[_Layout] = []
        # Since we now only compose with single-size sublayout, we can assume numel_so_far is always from strides[0].
        numel_so_far = layout.strides[0]
        for sub_size in layout.sizes:
            sub_stride = numel_so_far
            numel_so_far *= sub_size
            sub_res_sizes_and_strides = []

            # when self is multi-dimensional sublayout, aka, self = (a,b,...,c):(x,y,...,z), layout = s:d,
            # for integral s and d means that we want:
            # (1) “remove” the first d elements from self. (This will increase the stride.)
            # (2) “keep” the first s of those strided elements. (This does not affect the stride.)
            # For example, if self = (6,2):(2,1), layout = (3:2)
            # Step 1: remove the first 2 elements from self with strid increase, i.e., (6,2):(2,1) -> (3,2):(4,1)
            # Step 2: keep the first 3 of those strided elements, i.e., (3,2):(4,1) -> (3,1):(4,1)
            for curr_size, curr_stride in self.sizes_and_strides[:-1]:
                assert curr_size % sub_stride == 0 or sub_stride % curr_size == 0, (
                    "Layouts do not meet stride divisibility condition"
                )
                new_size = min(max(1, curr_size // sub_stride), sub_size)
                if new_size != 1:
                    sub_res_sizes_and_strides.append(
                        (new_size, sub_stride * curr_stride)
                    )
                assert sub_size % new_size == 0, (
                    "Layouts do not meet shape divisibility condition"
                )
                sub_size = sub_size // new_size
                sub_stride = self.ceil_div(sub_stride, curr_size)

            # When self is integral and has single-size sublayout, aka, self = a:b, layout = s:d,
            # the result is rather trivial: self o layout = a:b o s:d = s:(b*d).
            # For example, if self = (6:2), layout = (3:2), the result is (3:(2*2)) = (3:4).
            if sub_size != 1 or len(sub_res_sizes_and_strides) == 0:
                sub_res_sizes_and_strides.append(
                    (sub_size, sub_stride * self.strides[-1])
                )

            sub_res = _Layout(tuple(sub_res_sizes_and_strides))
            res.append(sub_res)
        return res

    def complement(self, world_size: int) -> "_Layout":
        """
        Compute the "complement layout" relative to a given world_size.
        A complement layout fills in the "missing" factor so that: self ⊗ complement(self, world_size)
        has a total cosize (product of sizes) = world_size.

        Example:
          self = (4:1)   # size=4, stride=1
          world_size = 8
          Then:
            cosize(self) = 4
            Needed factor = 8 / 4 = 2
            complement(self, 8) = (2:1)

          Together they form:
            (4:1) ⊗ (2:1) = (4,2):(2,1)
          which has cosize = 4 * 2 = 8, as required.

        In distributed terms, complement() is often used to derive the "other"
        rank grouping when splitting processes into 2D meshes.
        """
        res_sizes_and_strides = []
        current_idx = 1
        for size, stride in sorted(self.sizes_and_strides, key=lambda x: x[1]):
            if stride == 0 or size == 1:
                continue
            assert current_idx <= size * stride, (
                f"current_idx {current_idx} larger than numel so far {size * stride}."
            )

            res_sizes_and_strides.append((stride // current_idx, current_idx))
            current_idx = size * stride

        res_sizes_and_strides.append(
            (self.ceil_div(world_size, current_idx), current_idx)
        )
        # This is different from original pycute implementation, because in pytorch we usually
        # have the right-most dimension as the innermost dimension (smallest stride).
        res_sizes_and_strides.reverse()

        return _Layout(tuple(res_sizes_and_strides)).coalesce()

    def layout_to_group_ranks(self) -> list[int]:
        """
        This function computes the local group rank specified by the layout.

        How it works:
        1. we enumerates every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            group_rank = sum(coord[i] * strides[i] for i in range(ndim))

        Example A:
        sizes   = (2, 3)        # 2 rows, 3 cols
        strides = (3, 1)        # row-major layout
        coords  = (0,0) -> 0*3 + 0*1 = 0
                  (0,1) -> 0*3 + 1*1 = 1
                  (0,2) -> 0*3 + 2*1 = 2
                  (1,0) -> 1*3 + 0*1 = 3
                  (1,1) -> 1*3 + 1*1 = 4
                  (1,2) -> 1*3 + 2*1 = 5
        result = [0, 1, 2, 3, 4, 5]

        Example B:
        sizes   = (2, 3)
        strides = (1, 2)        # non-standard / strided layout
        coords  = (0,0) -> 0*1 + 0*2 = 0
                  (0,1) -> 0*1 + 1*2 = 2
                  (0,2) -> 0*1 + 2*2 = 4
                  (1,0) -> 1*1 + 0*2 = 1
                  (1,1) -> 1*1 + 1*2 = 3
                  (1,2) -> 1*1 + 2*2 = 5
        result = [0, 2, 4, 1, 3, 5]
        """
        return [
            sum(c * s for c, s in zip(coord, self.strides))
            for coord in product(*(range(s) for s in self.sizes))
        ]

    def layout_to_global_ranks(self, world_size: int) -> list[list[int]]:
        """
        Build global ranks specified by the layout via two-level ranks composition.

        The nested list forms the Cartesian product of group ranks and group offset
        and the final global ranks are the addition of these two. The result is a
        list of lists: one sublist per group. This rank list will be used to build
        the communicator underlying the layout.

        Example:
        group ranks    = [0, 1, 2, 3]
        group offsets  = [0, 4, 8, 12]
        result         = [
                            [0+0, 0+1, 0+2, 0+3],              # → [0, 1, 2, 3]
                            [4+0, 4+1, 4+2, 4+3],              # → [4, 5, 6, 7]
                            [8+0, 8+1, 8+2, 8+3],              # → [8, 9, 10,11]
                            [12+0, 12+1, 12+2, 12+3],          # → [12,13,14,15]
                        ]
        """
        return [
            [group_offset + group_rank for group_rank in self.layout_to_group_ranks()]
            for group_offset in self.complement(world_size).layout_to_group_ranks()
        ]


def init_layouts_from_mesh(
    mesh_size: tuple[int, ...], mesh_stride: tuple[int, ...]
) -> tuple["_Layout", ...]:
    """
    Convert a PyTorch mesh tensor's metadata (size, stride) into a list of CuTe-style layouts.

    For each dimension of the input tensor, we extract its size (mesh.size(i)) and stride (mesh.stride(i)),
    and then wrap that (size, stride) pair into a 1D CuTe layout.

    The result is a list of independent layout, one per dimension.
    Each layout describes how indices in that dimension map to backend ranks.

    Example:
        Suppose mesh.shape = (3, 4), mesh.stride = (4, 1).
        Then:
        layouts[0] = (3:4)   # first dimension: size=3, stride=4
        layouts[1] = (4:1)   # second dimension: size=4, stride=1
    """
    assert len(mesh_size) == len(mesh_stride), (
        "mesh_size and mesh_stride must have the same length"
    )
    return tuple(
        _Layout(((size, stride),)) for size, stride in zip(mesh_size, mesh_stride)
    )
