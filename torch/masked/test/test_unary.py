# Copyright (c) Meta Platforms, Inc. and affiliates

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # type: ignore[import]
import torch
from common_utils import _compare_mt_t
from maskedtensor import masked_tensor
from maskedtensor.unary import NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS


def _get_test_data(fn_name):
    data = torch.randn(10, 10)
    mask = torch.rand(10, 10) > 0.5
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    if fn_name in ["log", "log10", "log1p", "log2", "sqrt"]:
        data = data.mul(0.5).abs()
    if fn_name in ["rsqrt"]:
        data = data.abs() + 1  # Void division by zero
    if fn_name in ["acos", "arccos", "asin", "arcsin", "logit"]:
        data = data.abs().mul(0.5).clamp(0, 1)
    if fn_name in ["atanh", "arctanh", "erfinv"]:
        data = data.mul(0.5).clamp(-1, 1)
    if fn_name in ["acosh", "arccosh"]:
        data = data.abs() + 1
    if fn_name in ["bitwise_not"]:
        data = data.mul(128).to(torch.int8)
    return data, mask


def _get_sample_kwargs(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    kwargs = {}
    if fn_name in ["clamp", "clip"]:
        kwargs["min"] = -0.5
        kwargs["max"] = 0.5
    return kwargs


def _get_sample_args(fn_name, data, mask):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    mt = masked_tensor(data, mask)
    t_args = [data]
    mt_args = [mt]
    if fn_name in ["pow"]:
        t_args += [2.0]
        mt_args += [2.0]
    return t_args, mt_args


@pytest.mark.parametrize("fn", NATIVE_UNARY_FNS)
def test_unary(fn):
    torch.random.manual_seed(0)
    fn_name = fn.__name__
    data, mask = _get_test_data(fn_name)
    kwargs = _get_sample_kwargs(fn_name)

    t_args, mt_args = _get_sample_args(fn_name, data, mask)

    mt_result = fn(*mt_args, **kwargs)
    t_result = fn(*t_args, **kwargs)
    _compare_mt_t(mt_result, t_result)


@pytest.mark.parametrize("fn", NATIVE_INPLACE_UNARY_FNS)
def test_inplace_unary(fn):
    torch.random.manual_seed(0)
    fn_name = fn.__name__
    data, mask = _get_test_data(fn_name)
    kwargs = _get_sample_kwargs(fn_name)

    t_args, mt_args = _get_sample_args(fn_name, data, mask)

    mt_result = fn(*mt_args, **kwargs)
    t_result = fn(*t_args, **kwargs)
    _compare_mt_t(mt_result, t_result)
