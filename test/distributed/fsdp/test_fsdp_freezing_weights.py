# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

""" Test FSDP with some params frozen. """


from enum import Enum
from itertools import product
import tempfile

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim

from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_fsdp import dist_init, objects_are_equal, rmf, teardown


class FreezeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)

        self.trunk = FSDP(self.trunk)

    def forward(self, x):
        return self.head(self.trunk(x))


def _freeze_distributed_worker(
    gpu_id, world_size, tempfile_name, unused,
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    # The use case for this test is where the weights in the submodule
    # are not frozen but the leftover weights or those contained by the
    # root module are frozen. Refer to issue #758 for a real world example.
    model = FreezeModel()
    model = model.cuda()

    for param in model.head.parameters():
        param.requires_grad = False

    model = FSDP(model)

    if gpu_id == 0:
        print(model)

    target = torch.tensor([0, 1], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for iteration in range(3):
        out = model(batch)
        fake_loss = criterion(out, target)
        print("Loss", iteration, ":", fake_loss.item())
        optimizer.zero_grad()
        fake_loss.backward()
        optimizer.step()

    teardown()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
def test_submodule_freezing_weights(temp_files):
    world_size = 2
    mp.spawn(
        _freeze_distributed_worker, (world_size, temp_files[0], temp_files[1]), nprocs=world_size,
    )


class Model(nn.Module):
    def __init__(self, with_fsdp, freeze_after_wrap_fsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))


class NestedTrunkModel(nn.Module):
    def __init__(self, with_fsdp, freeze_after_wrap_fsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp),
            self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(64, 10),)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))

    def _create_block(self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp):
        block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU(inplace=True),)
        return block


def _create_model(with_fsdp, with_nested_trunk, freeze_after_wrap_fsdp):
    if with_nested_trunk:
        model = NestedTrunkModel(with_fsdp, freeze_after_wrap_fsdp)
    else:
        model = Model(with_fsdp, freeze_after_wrap_fsdp)
    return model


class FreezingMethod(str, Enum):
    GradToNone = "grad_to_none"
    RequiresGrad = "requires_grad"


def _distributed_worker(
    gpu_id,
    world_size,
    with_fsdp,
    with_nested_trunk,
    freezing_method,
    freeze_after_wrap_fsdp,
    tempfile_name,
    unused,
    rank_0_output,
    expected_state,
):
    torch.cuda.set_device(gpu_id)

    rank = gpu_id
    result = dist_init(rank, world_size, tempfile_name, unused)
    assert result, "Dist init failed"

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    batch = torch.randn(size=(2, 3, 224, 224)).cuda()

    model = _create_model(with_fsdp, with_nested_trunk, freeze_after_wrap_fsdp)
    model = model.cuda()

    # freezing the trunk using requires_grad.
    if freezing_method == FreezingMethod.RequiresGrad:
        for param in model.trunk.parameters():
            param.requires_grad = False

    if with_fsdp:
        if not freeze_after_wrap_fsdp:
            model.fsdp_wrap()
        model = FSDP(model)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu_id])

    if gpu_id == 0:
        print(model)

    target = torch.tensor([0, 1], dtype=torch.long).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for iteration in range(3):
        out = model(batch)
        fake_loss = criterion(out, target)
        print("Loss", iteration, ":", fake_loss.item())
        optimizer.zero_grad()
        fake_loss.backward()
        if freezing_method == FreezingMethod.GradToNone:
            for param in model.trunk.parameters():
                param.grad = None
        optimizer.step()

    if with_fsdp:
        fsdp_state = model.state_dict()
        # Move tensors to CPU to compare numerics.
        for k, v in fsdp_state.items():
            fsdp_state[k] = v.cpu()
        assert objects_are_equal(expected_state, fsdp_state, raise_exception=True)
    elif rank == 0:
        state_after = model.module.cpu().state_dict()
        torch.save(state_after, rank_0_output)

    teardown()


# A fixture to get tempfiles and ensure they are cleaned up.
@pytest.fixture()
def temp_files():
    num = 15  # 1 DDP and 4 FSDP cases each needs 3 files.
    files = [tempfile.mkstemp()[1] for _ in range(num)]

    yield tuple(files)

    # temp files could have been removed, so we use rmf.
    for name in files:
        rmf(name)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="multiple GPUs required"
)
@pytest.mark.parametrize("nested_trunk", ["nested_trunk", "simple_trunk"])
def test_freezing_weights(temp_files, nested_trunk):
    with_nested_trunk = nested_trunk == "nested_trunk"

    world_size = 2
    # DDP
    with_fsdp = False
    freezing_method = FreezingMethod.RequiresGrad
    mp.spawn(
        _distributed_worker,
        (world_size, with_fsdp, with_nested_trunk, freezing_method, True) + temp_files[0:3] + (None,),
        nprocs=world_size,
    )
    # FSDP, case 1 and 2.
    with_fsdp = True
    expected_state = torch.load(temp_files[2])
    temp_file_idx = 3
    for freezing_method, freeze_after_wrap_fsdp in product(
        [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone], [True, False]
    ):
        print(f"Testing FSDP with freezing method {freezing_method}")
        mp.spawn(
            _distributed_worker,
            (world_size, with_fsdp, with_nested_trunk, freezing_method, freeze_after_wrap_fsdp)
            + temp_files[temp_file_idx : temp_file_idx + 3]
            + (expected_state,),
            nprocs=world_size,
        )
        temp_file_idx += 3
