# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync import Pipe
from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from torch.distributed.pipeline.sync.skip.portal import PortalBlue, PortalCopy, PortalOrange
from torch.distributed.pipeline.sync.utils import partition_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("balance", [[3], [1, 2], [2, 1], [1, 1, 1]], ids=["3", "1:2", "2:1", "1:1:1"])
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
def test_1to3(balance, checkpoint, setup_rpc):
    if torch.cuda.device_count() < len(balance):
        pytest.skip("at least %d cuda devices required" % len(balance))

    @skippable(stash=["1to3"])
    class Layer1(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            yield stash("1to3", input)
            output = self.conv(input)
            return output  # noqa: B901

    class Layer2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            output = self.conv(input)
            return output

    @skippable(pop=["1to3"])
    class Layer3(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

        def forward(self, input):
            skip_1to3 = yield pop("1to3")
            output = self.conv(input) + skip_1to3
            return output

    model = nn.Sequential(Layer1(), Layer2(), Layer3())
    model = partition_model(model, balance)
    model = Pipe(model, chunks=3, checkpoint=checkpoint)

    in_device = model.devices[0]
    out_device = model.devices[-1]

    input = torch.rand(30, 3, 224, 224, device=in_device, requires_grad=True)
    output = model(input)
    loss = output.local_value().mean()
    loss.backward()

    assert torch.allclose(output.local_value().norm(), torch.tensor(1039.0, device=out_device), atol=6e-1)
    assert torch.allclose(input.grad.norm(), torch.tensor(0.0004533053, device=in_device))


def test_none_skip(setup_rpc):
    @skippable(stash=["none"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("none", None)
            return input  # noqa: B901

    @skippable(pop=["none"])
    class Pop(nn.Module):
        def forward(self, input):
            none = yield pop("none")
            assert none is None
            return input

    model = nn.Sequential(Stash(), Pop())
    model = Pipe(model, chunks=5)

    input = torch.rand(10, requires_grad=True)
    output = model(input)

    def assert_grad_fn_is_not_portal(grad_fn, visited=None):
        if visited is None:
            visited = set()
        if grad_fn in visited or grad_fn is None:
            return

        assert not isinstance(grad_fn, PortalBlue._backward_cls)
        assert not isinstance(grad_fn, PortalCopy._backward_cls)
        assert not isinstance(grad_fn, PortalOrange._backward_cls)

        visited.add(grad_fn)
        for next_grad_fn, _ in grad_fn.next_functions:
            assert_grad_fn_is_not_portal(next_grad_fn, visited)

    assert_grad_fn_is_not_portal(output.local_value().grad_fn)

    output.local_value().sum().backward()
    assert input.grad.mean().item() == 1
