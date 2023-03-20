# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import time

import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync._balance import balance_by_size, balance_by_time, blockpartition
from torch.distributed.pipeline.sync._balance.profile import layerwise_sandbox
from torch.testing._internal.common_utils import run_tests

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


def test_blockpartition():
    assert blockpartition.solve([1, 2, 3, 4, 5, 6], partitions=2) == [[1, 2, 3, 4], [5, 6]]


def test_blockpartition_zeros():
    assert blockpartition.solve([0, 0], partitions=2) == [[0], [0]]


def test_blockpartition_non_positive_partitions():
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=0)
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=-1)


def test_blockpartition_short_sequence():
    with pytest.raises(ValueError):
        blockpartition.solve([], partitions=1)
    with pytest.raises(ValueError):
        blockpartition.solve([42], partitions=2)


@pytest.mark.parametrize("device", devices)
@pytest.mark.skip(reason="Flaky due to time.sleep()")
def test_balance_by_time(device):
    class Delay(nn.Module):
        def __init__(self, seconds):
            super().__init__()
            self.seconds = seconds

        def forward(self, x):
            time.sleep(self.seconds)
            return x

    model = nn.Sequential(*[Delay(i / 10) for i in [1, 2, 3, 4, 5, 6]])
    sample = torch.rand(1)
    balance = balance_by_time(2, model, sample, device=device)
    assert balance == [4, 2]


def test_balance_by_time_loop_resets_input():
    # nn.Flatten was introduced at PyTorch 1.2.0.
    class Flatten(nn.Module):
        def forward(self, x):
            return x.flatten(1)

    model = nn.Sequential(nn.Conv2d(3, 2, 1), Flatten(), nn.Linear(128, 10))
    sample = torch.rand(10, 3, 8, 8)
    balance = balance_by_time(2, model, sample, device="cpu")
    assert balance == [1, 2]


@skip_if_no_cuda
def test_balance_by_size_latent():
    class Expand(nn.Module):
        def __init__(self, times):
            super().__init__()
            self.times = times

        def forward(self, x):
            for i in range(self.times):
                x = x + torch.rand_like(x, requires_grad=True)
            return x

    sample = torch.rand(10, 100, 100)

    model = nn.Sequential(*[Expand(i) for i in [1, 2, 3, 4, 5, 6]])
    balance = balance_by_size(2, model, sample)
    assert balance == [4, 2]

    model = nn.Sequential(*[Expand(i) for i in [6, 5, 4, 3, 2, 1]])
    balance = balance_by_size(2, model, sample)
    assert balance == [2, 4]


@skip_if_no_cuda
def test_balance_by_size_param():
    model = nn.Sequential(*[nn.Linear(i + 1, i + 2) for i in range(6)])
    sample = torch.rand(7, 1)
    balance = balance_by_size(2, model, sample, param_scale=100)
    assert balance == [4, 2]

    model = nn.Sequential(*[nn.Linear(i + 2, i + 1) for i in reversed(range(6))])
    sample = torch.rand(1, 7)
    balance = balance_by_size(2, model, sample, param_scale=100)
    assert balance == [2, 4]


@skip_if_no_cuda
def test_balance_by_size_param_scale():
    class Tradeoff(nn.Module):
        def __init__(self, param_size, latent_size):
            super().__init__()
            self.fc = nn.Linear(param_size, param_size)
            self.latent_size = latent_size

        def forward(self, x):
            for i in range(self.latent_size):
                x = x + torch.rand_like(x, requires_grad=True)
            return x

    model = nn.Sequential(
        Tradeoff(param_size=1, latent_size=6),
        Tradeoff(param_size=2, latent_size=5),
        Tradeoff(param_size=3, latent_size=4),
        Tradeoff(param_size=4, latent_size=3),
        Tradeoff(param_size=5, latent_size=2),
        Tradeoff(param_size=6, latent_size=1),
    )

    sample = torch.rand(1, requires_grad=True)

    balance = balance_by_size(2, model, sample, param_scale=0)
    assert balance == [2, 4]

    balance = balance_by_size(2, model, sample, param_scale=100)
    assert balance == [4, 2]


@pytest.mark.parametrize("device", devices)
def test_layerwise_sandbox(device):
    model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
    model.eval()

    for layer in layerwise_sandbox(model, torch.device(device)):
        assert layer.training
        assert all(p.device.type == device for p in layer.parameters())

    assert all(not l.training for l in model)
    assert all(p.device.type == "cpu" for p in model.parameters())


@pytest.mark.parametrize("device", devices)
def test_sandbox_during_profiling(device):
    model = nn.Sequential(nn.BatchNorm2d(3))

    before = {k: v.clone() for k, v in model.state_dict().items()}

    sample = torch.rand(1, 3, 10, 10)
    balance_by_time(1, model, sample, device=device)

    after = model.state_dict()

    assert before.keys() == after.keys()
    for key, value in before.items():
        assert torch.allclose(after[key], value), key


def test_not_training():
    class AssertTraining(nn.Module):
        def forward(self, x):
            assert self.training
            return x

    model = nn.Sequential(AssertTraining())

    model.eval()
    assert not model.training

    sample = torch.rand(1)
    balance_by_time(1, model, sample, device="cpu")

    assert not model.training


def test_balance_by_time_tuple():
    class Twin(nn.Module):
        def forward(self, x):
            return x, x.detach()

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    model = nn.Sequential(Twin(), Add())
    sample = torch.rand(1, requires_grad=True)
    balance_by_time(1, model, sample, device="cpu")


@skip_if_no_cuda
def test_balance_by_size_tuple():
    class Twin(nn.Module):
        def forward(self, x):
            return x, x.detach()

    class Add(nn.Module):
        def forward(self, a, b):
            return a + b

    model = nn.Sequential(Twin(), Add())
    sample = torch.rand(1, requires_grad=True)
    balance_by_size(1, model, sample)


def test_already_has_grad():
    model = nn.Sequential(nn.Conv2d(3, 3, 1))
    sample = torch.rand(1, 3, 32, 32)
    model(sample).norm().backward()

    with pytest.raises(ValueError, match="some parameter already has gradient"):
        balance_by_time(1, model, sample, device="cpu")


if __name__ == "__main__":
    run_tests()
