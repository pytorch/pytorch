# Copyright (c) Meta Platforms, Inc. and affiliates

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # type: ignore[import]
from typing import Dict
import torch
from common_utils import _compare_mt_t
from maskedtensor import masked_tensor
from maskedtensor.binary import NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS


def _get_test_data(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    data0 = torch.randn(10, 10)
    data1 = torch.randn(10, 10)
    mask = torch.rand(10, 10) > 0.5
    if fn_name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
        data0 = data0.mul(128).to(torch.int8)
        data1 = data1.mul(128).to(torch.int8)
    if fn_name in ["bitwise_left_shift", "bitwise_right_shift"]:
        data0 = data0.to(torch.int64)
        data1 = data1.to(torch.int64)
    return data0, data1, mask


def _get_sample_kwargs(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    kwargs = {}   # type: Dict[str, str]
    return kwargs


def _yield_sample_args(fn_name, data0, data1, mask):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    mt0 = masked_tensor(data0, mask)
    mt1 = masked_tensor(data1, mask)

    t_args = [data0, data1]
    mt_args = [mt0, mt1]
    yield t_args, mt_args

    t_args = [data0, data1]
    mt_args = [mt0, data1]
    yield t_args, mt_args


@pytest.mark.parametrize("fn", NATIVE_BINARY_FNS)
def test_binary(fn):
    torch.random.manual_seed(0)
    fn_name = fn.__name__
    data0, data1, mask = _get_test_data(fn_name)
    kwargs = _get_sample_kwargs(fn_name)

    for (t_args, mt_args) in _yield_sample_args(fn_name, data0, data1, mask):
        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)


@pytest.mark.parametrize("fn", NATIVE_INPLACE_BINARY_FNS)
def test_inplace_binary(fn):
    torch.random.manual_seed(0)
    fn_name = fn.__name__
    data0, data1, mask = _get_test_data(fn_name)
    kwargs = _get_sample_kwargs(fn_name)

    for (t_args, mt_args) in _yield_sample_args(fn_name, data0, data1, mask):
        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)


@pytest.mark.parametrize("fn_name", ["add", "add_"])
def test_masks_match(fn_name):
    torch.random.manual_seed(0)
    fn = getattr(torch.ops.aten, fn_name)
    data0, data1, mask = _get_test_data(fn_name)
    mask0 = mask
    mask1 = torch.rand(mask.size()) > 0.5
    mt0 = masked_tensor(data0, mask0)
    mt1 = masked_tensor(data1, mask1)
    try:
        fn(mt0, mt1)
        raise AssertionError()
    except ValueError as e:
        assert (
            "Input masks must match. If you need support for this, please open an issue on Github."
            == str(e)
        )
