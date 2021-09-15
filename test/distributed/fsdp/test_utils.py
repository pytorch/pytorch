# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test utility functions from torch/distributed/fsdp/utils.py. """

import pytest

import torch

from torch.distributed._fsdp.utils import (
    replace_by_prefix_,
    apply_to_tensors,
    chunk_and_pad
)

from collections import OrderedDict
import random


def test_replace_by_prefix():
    state_dict = {"layer.a": torch.tensor(1), "abc.layer.def": torch.tensor(2), "layer.b": torch.tensor(3)}
    replace_by_prefix_(state_dict, "layer.", "module.layer.")
    assert state_dict == {
        "module.layer.a": torch.tensor(1),
        "abc.layer.def": torch.tensor(2),
        "module.layer.b": torch.tensor(3),
    }


@pytest.mark.parametrize("devices", [["cpu"], ["cuda"], ["cpu", "cuda"]])
def test_apply_to_tensors(devices):
    """Test apply_to_tensors for both cpu & gpu"""
    if "cuda" in devices and not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("Skipped due to lack of GPU")
    expected = 0

    def get_a_tensor():
        """Return a random tensor on random device."""
        dev = random.choice(devices)
        shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
        t = torch.rand(shape).to(dev)
        nonlocal expected
        expected += t.numel()
        return t

    # create a mixed bag of data.
    data = [1, "str"]
    data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
    data.insert(0, set(["x", get_a_tensor(), get_a_tensor()]))
    data.append(([1], get_a_tensor(), (1), [get_a_tensor()], set((1, 2))))
    od = OrderedDict()
    od["k"] = "value"
    data.append(od)

    total = 0

    def fn(t):
        nonlocal total
        total += t.numel()
        return t

    new_data = apply_to_tensors(fn, data)
    assert total == expected, f"{total} vs. {expected}"
    for i, v in enumerate(data):
        assert type(new_data[i]) == type(v), f"expected type {type(v)} got {type(new_data[i])}"


@pytest.mark.parametrize("num_chunks", list(num_chunks for num_chunks in range(1, 33)))
def test_chunk_and_pad(num_chunks):
    max_tensor_size = 256
    tensor = torch.zeros(max_tensor_size)
    for tensor_size in range(1, max_tensor_size + 1):
        tensor_i = tensor[:tensor_size]
        chunks = chunk_and_pad(tensor_i, num_chunks)
        assert len(chunks) == num_chunks
        assert all(len(chunks[0]) == len(chunk) for chunk in chunks)
