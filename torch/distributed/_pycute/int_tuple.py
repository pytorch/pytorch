# mypy: ignore-errors
# flake8: noqa
# ruff: noqa: PGH004, B011
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
from typing import Union

from .typing import Integer


def is_int(x):
    return isinstance(x, Integer)


def is_tuple(x):
    return isinstance(x, tuple)


def flatten(t):
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)


def signum(a):
    return bool(a > 0) - bool(a < 0)


def product(a):
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a


def inner_product(a, b):
    if is_tuple(a):  # tuple tuple
        assert len(a) == len(b)
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:  # "int" "int"
        assert not is_tuple(b)
        return a * b


def tuple_max(a):
    if is_tuple(a):
        return max(tuple_max(x) for x in a)
    else:
        return a


def elem_scale(a, b):
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            assert len(a) == len(b)
            return tuple(elem_scale(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            assert False  # Error
    else:
        if is_tuple(b):  # "int" tuple
            return elem_scale(a, product(b))
        else:  # "int" "int"
            return a * b


# Inclusive prefix ceil div with output congruent to input a
def shape_div(a, b):
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            assert len(a) == len(b)
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
            assert a % b == 0 or b % a == 0
            return (a + b - 1) // b


# Exclusive prefix product with output congruent to input a
def prefix_product(a, init=1):
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
            assert False  # Error
        else:  # "int" "int"
            return init


def idx2crd(idx, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape):  # tuple tuple tuple
            assert len(idx) == len(shape) and len(idx) == len(stride)
            return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
        else:  # tuple "int" "int"
            assert False  # Error
    else:
        if is_tuple(shape):  # "int" tuple tuple
            assert len(shape) == len(stride)
            return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
        else:  # "int" "int" "int"
            return (idx // stride) % shape


def crd2idx(crd, shape, stride=None):
    if stride is None:
        stride = prefix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape):  # tuple tuple tuple
            assert len(crd) == len(shape) and len(crd) == len(stride)
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        else:  # tuple "int" "int"
            assert False, f"crd={crd}, shape={shape}"  # Error
    else:
        if crd is None:
            crd = 0

        if is_tuple(shape):  # "int" tuple tuple
            assert len(shape) == len(stride)
            result = 0
            for i in range(len(shape) - 1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            return result + crd2idx(crd, shape[-1], stride[-1])
        else:  # "int" "int" "int"
            return crd * stride


# Transform crd into the dst_shape's iteration space
def crd2crd(crd, dst_shape, src_shape=None):
    if is_tuple(crd):
        if is_tuple(dst_shape):  # tuple tuple
            assert len(crd) == len(dst_shape)
            return tuple(crd2crd(x, y) for x, y in zip(crd, dst_shape))
        else:  # tuple "int"
            # Ambiguous unless we have src_shape
            assert src_shape is not None
            return crd2idx(crd, src_shape)
    else:
        if is_tuple(dst_shape):  # "int" tuple
            return idx2crd(crd, dst_shape)
        else:  # "int" "int"
            assert crd < dst_shape
            return crd


# Filter trg according to crd: keep only elements of trg that are paired with None
def slice_(crd: Union[None, tuple, int], trg: Union[tuple, int]):
    if is_tuple(crd):
        if is_tuple(trg):  # tuple tuple
            assert len(crd) == len(trg)
            # match C++ behavior of `filter_tuple` using `tuple_cat(...)`
            return tuple(
                chain(
                    *filter(lambda x: x != (), [slice_(c, s) for c, s in zip(crd, trg)])
                )
            )
        else:
            assert False  # tuple "int" : Error
    elif crd is None:
        # match C++ behavior `return cute::tuple<B>{b};`
        return (trg,)
    else:
        return ()


# Determine if None appears at any of an int_tuples' terminals
def has_none(a: Union[None, tuple, int]):
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None
