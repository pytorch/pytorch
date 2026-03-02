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
Functions for manipulating IntTuples
"""

from functools import reduce
from itertools import chain
from typing import TypeAlias
from typing_extensions import TypeIs

from .typing import Integer


# Type aliases for better readability
IntTuple: TypeAlias = int | tuple["IntTuple", ...]


def is_int(x: object) -> TypeIs[int]:
    return isinstance(x, Integer)


def is_tuple(x: object) -> TypeIs[tuple]:
    return isinstance(x, tuple)


def as_tuple(x: IntTuple) -> tuple[IntTuple, ...]:
    if is_int(x):
        return (x,)
    return x


def match_structure(a: IntTuple, b: IntTuple) -> bool:
    if is_int(a) and is_int(b):
        return True
    if is_tuple(a) and is_tuple(b):
        return len(a) == len(b) and all(match_structure(x, y) for x, y in zip(a, b))
    return False


def flatten(t: IntTuple) -> tuple[int, ...]:
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)


def signum(a: int) -> int:
    return bool(a > 0) - bool(a < 0)


def product(a: IntTuple) -> int:
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a


def inner_product(a: IntTuple, b: IntTuple) -> int:
    if is_tuple(a) and is_tuple(b):  # tuple tuple
        if len(a) != len(b):
            raise AssertionError
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:  # "int" "int"
        if is_tuple(a) or is_tuple(b):
            raise AssertionError
        return a * b


def tuple_max(a: IntTuple) -> int:
    if is_tuple(a):
        return max(tuple_max(x) for x in a)
    else:
        return a


def elem_scale(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                raise AssertionError
            return tuple(elem_scale(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            raise AssertionError("Invalid combination: tuple with int")
    else:
        if is_tuple(b):  # "int" tuple
            return elem_scale(a, product(b))
        else:  # "int" "int"
            return a * b


# Inclusive prefix ceil div with output congruent to input a
def shape_div(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                raise AssertionError
            return tuple(shape_div(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            # r = [shape_div(a[0],b)] + [shape_div(a[i],b := shape_div(b, product(a[i-1]))) for i in range(1,len(a))]
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))
            return tuple(r)
    else:
        if is_tuple(b):  # "int" tuple
            return shape_div(a, product(b))
        else:  # "int" "int"
            if not (a % b == 0 or b % a == 0):
                raise AssertionError
            return (a + b - 1) // b


# Exclusive suffix product with output congruent to input a (lexicographic)
def suffix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    # TODO: With all these length asserts, may want to create a zip_strict wrapper.
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            if len(a) != len(init):
                raise AssertionError
            return tuple(suffix_product(x, i) for x, i in zip(a, init))
        else:  # tuple "int"
            # Process from right to left for lexicographic ordering
            # r = [prefix_product(a[len(a)-1],init)] +
            # [prefix_product(a[i],init := init * product(a[i+1])) for i in range(len(a)-1,0)].reverse()
            r = []

            # Calculate products from right to left, appending to list
            for i in range(len(a) - 1, -1, -1):
                r.append(suffix_product(a[i], init))
                init = init * product(a[i])

            # Reverse to get correct lexicographic order
            r.reverse()
            return tuple(r)
    else:
        if is_tuple(init):  # "int" tuple
            raise AssertionError("Invalid combination: int with tuple init")
        else:  # "int" "int"
            return init


def idx2crd(idx: IntTuple, shape: IntTuple, stride: IntTuple | None = None) -> IntTuple:
    if stride is None:
        stride = suffix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape) and is_tuple(stride):  # tuple tuple tuple
            if not (len(idx) == len(shape) and len(stride) == len(shape)):
                raise AssertionError
            return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
        else:  # tuple "int" "int"
            raise AssertionError("Invalid combination: tuple with int stride")
    else:
        if is_tuple(shape) and is_tuple(stride):  # "int" tuple tuple
            if len(shape) != len(stride):
                raise AssertionError
            return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
        else:  # "int" "int" "int"
            if is_tuple(shape) or is_tuple(stride):
                raise AssertionError
            return (idx // stride) % shape  # all are ints after type checks


def crd2idx(
    crd: IntTuple | None, shape: IntTuple, stride: IntTuple | None = None
) -> int:
    if stride is None:
        stride = suffix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape) and is_tuple(stride):  # tuple tuple tuple
            if not (len(crd) == len(shape) and len(stride) == len(shape)):
                raise AssertionError
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        else:  # tuple "int" "int"
            raise AssertionError(f"Invalid combination: crd={crd}, shape={shape}")
    else:
        if crd is None:
            crd = 0

        if is_tuple(shape) and is_tuple(stride):  # "int" tuple tuple
            if len(shape) != len(stride):
                raise AssertionError
            result = 0
            # Process from right to left for lexicographic ordering
            for i in range(len(shape) - 1, 0, -1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            if len(shape) > 0:
                result += crd2idx(crd, shape[0], stride[0])
            return result
        else:  # "int" "int" "int"
            if is_tuple(shape) or is_tuple(stride):
                raise AssertionError
            return crd * stride  # all are ints after type checks


# Transform crd into the dst_shape's iteration space
def crd2crd(
    crd: IntTuple, dst_shape: IntTuple, src_shape: IntTuple | None = None
) -> IntTuple:
    if is_tuple(crd):
        if is_tuple(dst_shape):  # tuple tuple
            if len(crd) != len(dst_shape):
                raise AssertionError
            return tuple(crd2crd(x, y) for x, y in zip(crd, dst_shape))
        else:  # tuple "int"
            # Ambiguous unless we have src_shape
            if src_shape is None:
                raise AssertionError
            return crd2idx(crd, src_shape)
    else:
        if is_tuple(dst_shape):  # "int" tuple
            return idx2crd(crd, dst_shape)
        else:  # "int" "int"
            if crd >= dst_shape:
                raise AssertionError
            return crd


# Filter trg according to crd: keep only elements of trg that are paired with None
def slice_(crd: tuple | int | None, trg: tuple | int) -> tuple | int:
    if is_tuple(crd):
        if is_tuple(trg):  # tuple tuple
            if len(crd) != len(trg):
                raise AssertionError
            # match C++ behavior of `filter_tuple` using `tuple_cat(...)`
            return tuple(
                chain(
                    *filter(  # type: ignore[arg-type]  # filter returns Iterator which is compatible
                        lambda x: x != (),
                        [slice_(c, s) for c, s in zip(crd, trg)],
                    )
                )
            )
        else:
            raise AssertionError("Invalid combination: tuple crd with int trg")
    elif crd is None:
        # match C++ behavior `return cute::tuple<B>{b};`
        return (trg,)
    else:
        return ()


# Determine if None appears at any of an int_tuples' terminals
def has_none(a: tuple | int | None) -> bool:
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None
