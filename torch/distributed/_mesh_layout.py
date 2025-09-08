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

"""
Definition of CuTe Layouts and functions to manipulate them
"""

from functools import reduce
from itertools import chain
import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

IntTuple = tuple[int, ...]

from typing import TypeAlias

NestedIntTuple: TypeAlias = int | tuple["NestedIntTuple", ...]

@dataclass(frozen=True)
class _Layout:
    _sizes: NestedIntTuple
    _strides: NestedIntTuple

    def __post_init__(self) -> None:
        if len(_Layout.flatten(self._sizes)) != len(_Layout.flatten(self._strides)):
            raise ValueError(
                f"sizes {len(_Layout.flatten(self._sizes))} and strides {len(_Layout.flatten(self._strides))} must have the same length"
            )

    @property
    def sizes(self) -> NestedIntTuple:
        return self._sizes

    @property
    def strides(self) -> NestedIntTuple:
        return self._strides

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        return zip(_Layout.flatten(self._sizes), _Layout.flatten(self._strides))

    def numel(self) -> int:
        return math.prod(_Layout.flatten(self.sizes))

    # operator len(L)  (len [rank] like tuples)
    def __len__(self):
        if _Layout.is_tuple(self._sizes):
            return len(_Layout.flatten(self._sizes))
        else:
            return 1

    @staticmethod
    def ceil_div(n: int, m: int) -> int:
        return (n + m - 1) // m

    @staticmethod
    def is_tuple(x) -> bool:
        return isinstance(x, tuple)

    @staticmethod
    def flatten(t: NestedIntTuple) -> IntTuple:
        if _Layout.is_tuple(t):
            if len(t) == 0:
                return ()
            else:
                return tuple(i for a in t for i in _Layout.flatten(a))
        else:
            assert isinstance(t, int)
            return (t,)

    # Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
    def coalesce(self, profile=None) -> "_Layout":
        sizes: list[int] = []
        strides: list[int] = []
        for shape, stride in self.sizes_and_strides:
            # skip their shape-1s
            if shape == 1:
                continue
            # replace our shape-1 with anything
            elif sizes[-1] == 1:
                sizes[-1] = shape
                strides[-1] = stride
            # merge modes if the shape*stride match
            elif sizes[-1] * strides[-1] == stride:
                sizes[-1] = sizes[-1] * shape
            # append a new mode
            else:
                sizes.append(shape)
                strides.append(stride)

        if len(sizes) == 1:
            return _Layout(sizes[0], strides[0])
        else:
            return _Layout(tuple(sizes), tuple(strides))

    # Layout composition
    # Use tuples-of-layouts to perform this operation by-mode and None as no-op
    def composition(self, layout: "_Layout") -> "_Layout":
        if _Layout.is_tuple(layout):
            assert len(self) >= len(layout)
            return make_layout(
                chain(
                    (composition(self[i], layout[i]) for i in range(0, len(layout))),
                    (self[i] for i in range(len(layout), len(self))),
                )
            )
        elif _Layout.is_tuple(layout.sizes):
            return make_layout(self.composition(layout_i) for layout_i in layout)

        if layout.strides == 0:
            return _Layout(layout.sizes, 0)
        else:
            result_shape = []
            result_stride = []
            rest_shape = layout.sizes
            rest_stride = layout.strides
            flat_A = self.coalesce()
            for curr_shape, curr_stride in zip(
                _Layout.flatten(flat_A.sizes)[:-1], _Layout.flatten(flat_A.sizes)[:-1]
            ):
                assert curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0
                new_shape = min(max(1, curr_shape // rest_stride), rest_shape)

                if new_shape != 1:
                    result_shape.append(new_shape)
                    result_stride.append(rest_stride * curr_stride)

                rest_shape = rest_shape // new_shape
                rest_stride = -(
                    -rest_stride // curr_shape
                )  # Python exclusive impl: "//" is always floor div so == ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)

            if rest_shape != 1 or len(result_shape) == 0:
                result_shape.append(rest_shape)
                result_stride.append(rest_stride * _Layout.flatten(flat_A.strides)[-1])

            if len(result_shape) == 1:
                return _Layout(result_shape[0], result_stride[0])
            else:
                return _Layout(tuple(result_shape), tuple(result_stride))

    # Layout complement
    def complement(self, max_idx=1) -> "_Layout":
        res_sizes: list[int] = []
        res_strides: list[int] = []
        current_idx = 1

        sorted_DS = sorted(self.sizes_and_strides)
        for stride, shape in sorted_DS:
            if stride == 0 or shape == 1:
                continue

            in_bound = current_idx <= shape * stride
            # To support symbolic value which can't be evaluated now
            assert (type(in_bound) is not bool) or in_bound

            res_sizes.append(stride // current_idx)
            res_strides.append(current_idx)
            current_idx = shape * stride

        res_sizes.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
        res_strides.append(current_idx)

        return _Layout(tuple(res_sizes), tuple(res_strides)).coalesce()
