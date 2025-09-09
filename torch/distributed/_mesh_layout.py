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

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing_extensions import TypeGuard


IntTuple = tuple[int, ...]

from typing import TypeAlias, Union


NestedIntTuple: TypeAlias = tuple["int | NestedIntTuple", ...]


@dataclass(frozen=True)
class _Layout:
    _sizes: NestedIntTuple
    _strides: NestedIntTuple

    def __post_init__(self) -> None:
        if len(_Layout.flatten(self._sizes)) != len(_Layout.flatten(self._strides)):
            raise ValueError(
                f"sizes {len(_Layout.flatten(self._sizes))} and "
                f"strides {len(_Layout.flatten(self._strides))} must have the same length"
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
    def __len__(self) -> int:
        return len(_Layout.flatten(self._sizes))

    # operator []    (get-i like tuples)
    def __getitem__(self, i: int) -> "_Layout":
        size = self.sizes[i]
        stride = self.strides[i]
        if _Layout.is_tuple(size) and _Layout.is_tuple(stride):
            return _Layout(size, stride)
        elif isinstance(size, int) and isinstance(stride, int):
            return _Layout((size,), (stride,))
        else:
            raise ValueError("size and stride must be either int or tuple")

    @staticmethod
    def ceil_div(n: int, m: int) -> int:
        return (n + m - 1) // m

    @staticmethod
    def is_tuple(x: Union[int, NestedIntTuple]) -> TypeGuard[NestedIntTuple]:
        return isinstance(x, tuple)

    @staticmethod
    def flatten(t: Union[int, NestedIntTuple]) -> IntTuple:
        if _Layout.is_tuple(t):
            if len(t) == 0:
                return ()
            else:
                return tuple(i for a in t for i in _Layout.flatten(a))
        else:
            assert isinstance(t, int)
            return (t,)

    # Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
    def coalesce(self) -> "_Layout":
        sizes: list[int] = []
        strides: list[int] = []
        for size, stride in self.sizes_and_strides:
            # skip their size-1s
            if size == 1:
                continue
            # replace our size-1 with anything
            elif sizes[-1] == 1:
                sizes[-1] = size
                strides[-1] = stride
            # merge modes if the size*stride match
            elif sizes[-1] * strides[-1] == stride:
                sizes[-1] = sizes[-1] * size
            # append a new mode
            else:
                sizes.append(size)
                strides.append(stride)

        return _Layout(tuple(sizes), tuple(strides))

    # Layout composition
    # Use tuples-of-layouts to perform this operation by-mode and None as no-op
    def composition(self, layout: "_Layout") -> "_Layout":
        if _Layout.is_tuple(layout.sizes):
            layouts = (self.composition(layout_i) for layout_i in layout)  # type: ignore[attr-defined]
            zip_res_sizes, zip_res_strides = zip(
                *((a.sizes, a.strides) for a in layouts)
            )
            return _Layout(tuple(zip_res_sizes), tuple(zip_res_strides))

        res_sizes: list[int] = []
        res_strides: list[int] = []
        rest_size = layout.sizes[0]
        rest_stride = layout.strides[0]
        assert isinstance(rest_size, int)
        assert isinstance(rest_stride, int)
        flat_layout = self.coalesce()
        for curr_size, curr_stride in zip(
            _Layout.flatten(flat_layout.sizes)[:-1],
            _Layout.flatten(flat_layout.sizes)[:-1],
        ):
            assert curr_size % rest_stride == 0 or rest_stride % curr_size == 0
            new_size = min(max(1, curr_size // rest_stride), rest_size)

            if new_size != 1:
                res_sizes.append(new_size)
                res_strides.append(rest_stride * curr_stride)

            rest_size = rest_size // new_size
            rest_stride = -(
                -rest_stride // curr_size
            )  # Python exclusive impl: "//" is always floor div so == ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)

        if rest_size != 1 or len(res_sizes) == 0:
            res_sizes.append(rest_size)
            res_strides.append(rest_stride * _Layout.flatten(flat_layout.strides)[-1])

        return _Layout(tuple(res_sizes), tuple(res_strides))

    # Layout complement
    def complement(self, max_idx: int) -> "_Layout":
        res_sizes: list[int] = []
        res_strides: list[int] = []
        current_idx = 1

        sorted_DS = sorted(self.sizes_and_strides)
        for stride, size in sorted_DS:
            if stride == 0 or size == 1:
                continue

            in_bound = current_idx <= size * stride
            # To support symbolic value which can't be evaluated now
            assert (type(in_bound) is not bool) or in_bound

            res_sizes.append(stride // current_idx)
            res_strides.append(current_idx)
            current_idx = size * stride

        res_sizes.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
        res_strides.append(current_idx)

        return _Layout(tuple(res_sizes), tuple(res_strides)).coalesce()
