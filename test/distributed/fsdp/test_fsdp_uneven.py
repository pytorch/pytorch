# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with uneven parameter shards. """

import tempfile

import pytest
import torch
from torch import Tensor
import torch.multiprocessing as mp
from torch.nn import Linear, Sequential
from torch.optim import SGD

from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._fsdp.fully_sharded_data_parallel import TrainingState
from torch.testing._internal.common_fsdp import dist_init, teardown


def _test_func(rank, world_size, model, fsdp_config, tempfile_name, unused, test_case):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    my_lr = 0.1

    device = torch.device("cuda")
    if fsdp_config.get("mixed_precision", False):
        dtype = torch.float16
        fsdp_config["fp32_reduce_scatter"] = True
    else:
        dtype = torch.float32

    if test_case["assert_ref_out"]:
        with torch.no_grad():
            # Compute one iteration local output.
            fp32_weight = model.weight.T.clone().to(device)
            weight = fp32_weight.to(dtype)
            v = torch.Tensor(test_case["inputs"][0][rank]).to(device, dtype)
            ref_forward_output_my_rank = torch.matmul(v, weight)
            # Compute one iteration global weight update.
            v = torch.Tensor(test_case["inputs"][0][:world_size]).to(device, dtype)
            grad = v.float().sum(0).repeat(weight.shape[0], 1).div(world_size)
            ref_weight_out = fp32_weight - grad.T * my_lr
            assert ref_weight_out.dtype == torch.float32
    model.to(device)  # not dtype, since FSDP will manage mixed precision internally
    assert isinstance(fsdp_config, dict), str(fsdp_config)
    model = FSDP(model, **fsdp_config)
    optim = SGD(model.parameters(), lr=my_lr)
    inputs = test_case["inputs"]
    assert len(inputs) == 1 or not test_case["assert_ref_out"]
    assert len(inputs[0]) >= world_size
    for in_data in inputs:
        in_data = Tensor(in_data[rank]).to(device, dtype)
        out = model(in_data)
        out.float().sum().backward()
        optim.step()
        optim.zero_grad()
        if test_case["assert_ref_out"]:
            with model.summon_full_params():
                weight_out = model.module.weight.data.T.clone()
            # make sure we can do more fwd/bwd
            loss = model(in_data)
            loss.sum().backward()

    if test_case["assert_ref_out"]:
        torch.testing.assert_allclose(ref_forward_output_my_rank, out)
        torch.testing.assert_allclose(ref_weight_out, weight_out)

    model.assert_state(TrainingState.IDLE)
    teardown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
@pytest.mark.parametrize("test_case", [{"inputs": [torch.rand(8, 3)], "assert_ref_out": True}])
@pytest.mark.parametrize(
    "fsdp_config", [{}, {"flatten_parameters": False}, {"mixed_precision": True}],
)
@pytest.mark.parametrize("world_size", list(range(2, 9)))
def test_one_iteration(world_size, test_case, fsdp_config):
    """Test FSDP with uneven divide of parameter shards."""
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs.")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    # TODO (Min): we may want to extend this to a simple 2 layer model so that it covers
    #             more cases in FSDP. Also, assert_ref_out can be extended to multiple
    #             iterations. This could be a good bootcamp task. I should file a github
    #             issue once we merge.
    model = Linear(3, 3, bias=False)
    mp.spawn(
        _test_func,
        args=(world_size, model, fsdp_config, temp_file_name, unused, test_case),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
@pytest.mark.parametrize("test_case", [{"inputs": [torch.rand(8, 3), torch.rand(8, 3)], "assert_ref_out": False}])
@pytest.mark.parametrize("fsdp_config", [{}, {"flatten_parameters": False}])
@pytest.mark.parametrize("world_size", list(range(2, 9)))
def test_smaller_than_world_size(world_size, test_case, fsdp_config):
    """Test FSDP with uneven divide of parameter shards."""
    if world_size > torch.cuda.device_count():
        pytest.skip("Not enough GPUs.")

    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    model = Sequential(
        Linear(3, 3, bias=False),
        Linear(3, 4, bias=False),
        Linear(4, 5, bias=False),
        Linear(5, 4, bias=False),
        Linear(4, 3, bias=False),
        Linear(3, 1, bias=False),
        Linear(1, 1, bias=False),  # param here is smaller than world_size if unflattened.
    )
    mp.spawn(
        _test_func,
        args=(world_size, model, fsdp_config, temp_file_name, unused, test_case),
        nprocs=world_size,
        join=True,
    )
