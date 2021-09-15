# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with different input types. """

import tempfile

import pytest
import torch
from torch.nn import Linear, Module
from torch.optim import SGD

from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._fsdp.fully_sharded_data_parallel import TrainingState
from torch.testing._internal.common_fsdp import dist_init, rmf, teardown


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    num = 2  # dist_init needs 2 files
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    yield tuple(files)

    # temp files could have been removed, so we use rmf.
    for name in files:
        rmf(name)


# We only test on GPU since mix-precision only works on GPU.
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1, reason="CUDA required"
)
@pytest.mark.parametrize(
    "fsdp_config", [{}, {"mixed_precision": True}],
)
@pytest.mark.parametrize("input_cls", [dict, list])
def test_input_type(temp_files, fsdp_config, input_cls):
    """Test FSDP with input being a list or a dict, only single GPU."""
    result = dist_init(rank=0, world_size=1, filename=temp_files[0], filename_rpc=temp_files[1])
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.layer = Linear(4, 4)

        def forward(self, input):
            if isinstance(input, list):
                input = input[0]
            else:
                assert isinstance(input, dict), input
                input = input["in"]
            return self.layer(input)

    model = FSDP(Model(), **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for _ in range(5):
        in_data = torch.rand(64, 4).cuda()
        in_data.requires_grad = True
        if input_cls is list:
            in_data = [in_data]
        else:
            assert input_cls is dict
            in_data = {"in": in_data}

        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    model.assert_state(TrainingState.IDLE)

    teardown()
