# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with different multiple forward of the same module. """

import tempfile

import pytest
import torch
import torch.multiprocessing as mp
from torch.nn import Linear, Module
from torch.optim import SGD

from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._fsdp.fully_sharded_data_parallel import TrainingState
from torch.testing._internal.common_fsdp import dist_init, teardown


def _test_func(rank, world_size, fsdp_config, tempfile_name, unused):
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    assert isinstance(fsdp_config, dict), str(fsdp_config)

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.inner = FSDP(Linear(4, 4), **fsdp_config)
            self.outer = Linear(4, 5)

        def forward(self, x):
            # Forward twice.
            i = self.inner(x)
            j = self.inner(x)
            return self.outer(i + j)

    model = FSDP(Model(), **fsdp_config).cuda()
    optim = SGD(model.parameters(), lr=0.1)

    for _ in range(3):
        in_data = torch.rand(64, 4).cuda()
        in_data.requires_grad = True
        out = model(in_data)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

    model.assert_state(TrainingState.IDLE)
    teardown()


# We use strings for precision and flatten instead of bool to
# make the pytest output more readable.
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
@pytest.mark.parametrize("precision", ["full", "mixed"])
@pytest.mark.parametrize("flatten", ["flatten", "no_flatten"])
def test1(precision, flatten):
    temp_file_name = tempfile.mkstemp()[1]
    unused = tempfile.mkstemp()[1]

    fsdp_config = {}
    fsdp_config["mixed_precision"] = precision == "mixed"
    fsdp_config["flatten_parameters"] = flatten == "flatten"

    # Some bugs only show up when we are in world_size > 1 due to sharding changing
    # the tensor dimensions.
    world_size = 2
    mp.spawn(
        _test_func, args=(world_size, fsdp_config, temp_file_name, unused), nprocs=world_size, join=True,
    )
