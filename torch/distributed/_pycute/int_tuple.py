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
from typing import Optional, Union
from typing_extensions import TypeAlias, TypeGuard

from .typing import Integer


# Type aliases for better readability
IntTuple: TypeAlias = Union[int, tuple["IntTuple", ...]]


def is_int(x: object) -> TypeGuard[int]:
    return isinstance(x, Integer)


def is_tuple(x: object) -> TypeGuard[tuple]:
    return isinstance(x, tuple)


def flatten(t: IntTuple) -> tuple[int, ...]:
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)  # type: ignore[return-value]  # t is int, converted to tuple[int]


def signum(a: int) -> int:
    return bool(a > 0) - bool(a < 0)


def product(a: IntTuple) -> int:
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a  # type: ignore[return-value]  # a is int after is_tuple check


def inner_product(a: IntTuple, b: IntTuple) -> int:
    if is_tuple(a):  # tuple tuple
        assert len(a) == len(b)  # type: ignore[arg-type]
        return sum(inner_product(x, y) for x, y in zip(a, b))  # type: ignore[arg-type,misc]
    else:  # "int" "int"
        assert not is_tuple(b)
        return a * b  # type: ignore[operator,return-value]  # both are ints after type checks


def tuple_max(a: IntTuple) -> int:
    if is_tuple(a):
        return max(tuple_max(x) for x in a)
    else:
        return a  # type: ignore[return-value]  # a is int after is_tuple check


def elem_scale(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            assert len(a) == len(b)
            return tuple(elem_scale(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            raise AssertionError("Invalid combination: tuple with int")
    else:
        if is_tuple(b):  # "int" tuple
            return elem_scale(a, product(b))
        else:  # "int" "int"
            return a * b  # type: ignore[operator,return-value]  # a and b are ints after type checks


# Inclusive prefix ceil div with output congruent to input a
def shape_div(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            assert len(a) == len(b)
            return tuple(shape_div(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            # r = [shape_div(a[0],b)] + [shape_div(a[i],b := shape_div(b, product(a[i-1]))) for i in range(1,len(a))]
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))  # type: ignore[assignment]  # b is updated within loop
            return tuple(r)
    else:
        if is_tuple(b):  # "int" tuple
            return shape_div(a, product(b))
        else:  # "int" "int"
            assert a % b == 0 or b % a == 0  # type: ignore[operator]  # both are ints after type checks
            return (a + b - 1) // b  # type: ignore[operator,return-value]  # result is int, operators valid on ints


# Exclusive prefix product with output congruent to input a
def prefix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            assert len(a) == len(init)
            return tuple(prefix_product(x, i) for x, i in zip(a, init))
        else:  # tuple "int"
            # r = [prefix_product(a[0],init)] + [prefix_product(a[i],init := init * product(a[i-1])) for i in range(1,len(a))]
            r = []
            for v in a:
                r.append(prefix_product(v, init))
                init = init * product(v)
            return tuple(r)
    else:
        if is_tuple(init):  # "int" tuple
            raise AssertionError("Invalid combination: int with tuple init")
        else:  # "int" "int"
            return init


def idx2crd(
    idx: IntTuple, shape: IntTuple, stride: Optional[IntTuple] = None
) -> IntTuple:
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape):  # tuple tuple tuple
            assert len(idx) == len(shape) and len(stride) == len(shape)  # type: ignore[arg-type]
            return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))  # type: ignore[arg-type]
        else:  # tuple "int" "int"
            raise AssertionError("Invalid combination: tuple with int stride")
    else:
        if is_tuple(shape):  # "int" tuple tuple
            assert len(shape) == len(stride)  # type: ignore[arg-type]
            return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))  # type: ignore[arg-type]
        else:  # "int" "int" "int"
            return (idx // stride) % shape  # type: ignore[operator,return-value]  # all are ints after type checks


def crd2idx(
    crd: Optional[IntTuple], shape: IntTuple, stride: Optional[IntTuple] = None
) -> int:
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape):  # tuple tuple tuple
            assert len(crd) == len(shape) and len(stride) == len(shape)  # type: ignore[arg-type]
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))  # type: ignore[arg-type,misc]
        else:  # tuple "int" "int"
            raise AssertionError(f"Invalid combination: crd={crd}, shape={shape}")
    else:
        if crd is None:
            crd = 0

        if is_tuple(shape):  # "int" tuple tuple
            assert len(shape) == len(stride)  # type: ignore[arg-type]
            result = 0
            for i in range(len(shape) - 1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])  # type: ignore[operator,index,arg-type]
                crd = crd // product(shape[i])  # type: ignore[operator,index]
            return result + crd2idx(crd, shape[-1], stride[-1])  # type: ignore[index,arg-type]
        else:  # "int" "int" "int"
            return crd * stride  # type: ignore[operator,return-value]  # all are ints after type checks


# Transform crd into the dst_shape's iteration space
def crd2crd(
    crd: IntTuple, dst_shape: IntTuple, src_shape: Optional[IntTuple] = None
) -> IntTuple:
    if is_tuple(crd):
        if is_tuple(dst_shape):  # tuple tuple
            assert len(crd) == len(dst_shape)
            return tuple(crd2crd(x, y) for x, y in zip(crd, dst_shape))
        else:  # tuple "int"
            # Ambiguous unless we have src_shape
            assert src_shape is not None
            return crd2idx(crd, src_shape)  # type: ignore[return-value]
    else:
        if is_tuple(dst_shape):  # "int" tuple
            return idx2crd(crd, dst_shape)
        else:  # "int" "int"
            assert crd < dst_shape  # type: ignore[operator]  # both are ints after type checks
            return crd  # type: ignore[return-value]  # crd is int after type check


# Filter trg according to crd: keep only elements of trg that are paired with None
def slice_(crd: Union[None, tuple, int], trg: Union[tuple, int]) -> Union[tuple, int]:
    if is_tuple(crd):
        if is_tuple(trg):  # tuple tuple
            assert len(crd) == len(trg)
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
def has_none(a: Union[None, tuple, int]) -> bool:
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None
