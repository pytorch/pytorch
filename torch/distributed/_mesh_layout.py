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


class Integer:
    @classmethod
    def __subclasshook__(cls, c):  # type: ignore[no-untyped-def]
        if c in [bool, float]:
            return False

        return issubclass(c, int)


class LayoutBase:
    pass


def is_tuple(x):  # type: ignore[no-untyped-def]
    return isinstance(x, tuple)


def is_int(x):  # type: ignore[no-untyped-def]
    return isinstance(x, Integer)


def flatten(t):  # type: ignore[no-untyped-def]
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)


def product(a):  # type: ignore[no-untyped-def]
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a


# Exclusive prefix product with output congruent to input a
def prefix_product(a, init=1):  # type: ignore[no-untyped-def]
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            assert len(a) == len(init)
            return tuple(prefix_product(x, i) for x, i in zip(a, init))
        else:  # tuple "int"
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * product(v)
            return tuple(r)
    else:
        if is_tuple(init):  # "int" tuple
            raise AssertionError  # Error
        else:  # "int" "int"
            return init


def is_layout(x):  # type: ignore[no-untyped-def]
    return isinstance(x, LayoutBase)


class Layout(LayoutBase):  # type: ignore[no-untyped-def]
    def __init__(self, _shape, _stride=None):  # type: ignore[no-untyped-def]
        self.shape = _shape
        if _stride is None:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = _stride

    # operator ==
    def __eq__(self, other):  # type: ignore[no-untyped-def]
        return self.shape == other.shape and self.stride == other.stride

    # operator len(L)  (len [rank] like tuples)
    def __len__(self):  # type: ignore[no-untyped-def]
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    # operator []    (get-i like tuples)
    def __getitem__(self, i):  # type: ignore[no-untyped-def]
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])
        else:
            assert i == 0
            return Layout(self.shape, self.stride)

    # size(layout)   Size of the domain
    def size(self):  # type: ignore[no-untyped-def]
        return product(self.shape)

    # print and str
    def __str__(self):  # type: ignore[no-untyped-def]
        return f"{self.shape}:{self.stride}"

    # error msgs and representation
    def __repr__(self):  # type: ignore[no-untyped-def]
        return f"Layout({self.shape},{self.stride})"


# Make Layout from a list of layouts (each layout it's own mode in the result)
def make_layout(*layouts):  # type: ignore[no-untyped-def]
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]

    shape, stride = zip(*((a.shape, a.stride) for a in layouts))
    return Layout(shape, stride)


# Size of the domain
def size(layout):  # type: ignore[no-untyped-def]
    if is_layout(layout):
        return layout.size()
    return product(layout)


# Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
def coalesce(layout, profile=None):  # type: ignore[no-untyped-def]
    if is_tuple(profile):
        assert len(layout) >= len(profile)
        return make_layout(
            chain(
                (coalesce(layout[i], profile[i]) for i in range(0, len(profile))),
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = [1]
    result_stride = [0]
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        # merge modes if the shape*stride match
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(tuple(result_shape), tuple(result_stride))


# Layout composition
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def composition(layoutA, layoutB):  # type: ignore[no-untyped-def]
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return composition(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        assert len(layoutA) >= len(layoutB)
        return make_layout(
            chain(
                (composition(layoutA[i], layoutB[i]) for i in range(0, len(layoutB))),
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )
    elif is_tuple(layoutB.shape):
        return make_layout(composition(layoutA, layoutB_i) for layoutB_i in layoutB)

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        flat_A = coalesce(layoutA)
        for curr_shape, curr_stride in zip(
            flatten(flat_A.shape)[:-1], flatten(flat_A.stride)[:-1]
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
            result_stride.append(rest_stride * flatten(flat_A.stride)[-1])

        if len(result_shape) == 1:
            return Layout(result_shape[0], result_stride[0])
        else:
            return Layout(tuple(result_shape), tuple(result_stride))


# Layout complement
def complement(layout, max_idx=1):  # type: ignore[no-untyped-def]
    if is_int(layout):
        return complement(Layout(layout))

    result_shape = []
    result_stride = []
    current_idx = 1

    sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
    for stride, shape in sorted_DS:
        if stride == 0 or shape == 1:
            continue

        in_bound = current_idx <= shape * stride
        # To support symbolic value which can't be evaluated now
        assert (type(in_bound) is not bool) or in_bound

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))
