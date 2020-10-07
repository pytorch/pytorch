# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import pytest
import torch
from torch import nn

from torch.distributed._pipeline.sync import Pipe
from torch.distributed._pipeline.sync.skip import pop, skippable, stash
from torch.distributed._pipeline.sync.skip.portal import PortalBlue, PortalCopy, PortalOrange


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
@pytest.mark.parametrize("balance", [[3], [1, 2], [2, 1], [1, 1, 1]], ids=["3", "1:2", "2:1", "1:1:1"])
@pytest.mark.parametrize("checkpoint", ["never", "always", "except_last"])
def test_1to3(balance, checkpoint):
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
            return output # noqa

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
    model = Pipe(model, balance, chunks=3, checkpoint=checkpoint)

    in_device = model.devices[0]
    out_device = model.devices[-1]

    input = torch.rand(30, 3, 224, 224, device=in_device, requires_grad=True)
    output = model(input)
    loss = output.mean()
    loss.backward()

    assert torch.allclose(output.norm(), torch.tensor(1039.0, device=out_device), atol=6e-1)
    assert torch.allclose(input.grad.norm(), torch.tensor(0.0004533053, device=in_device))


def test_none_skip():
    @skippable(stash=["none"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("none", None)
            return input # noqa

    @skippable(pop=["none"])
    class Pop(nn.Module):
        def forward(self, input):
            none = yield pop("none")
            assert none is None
            return input

    model = nn.Sequential(Stash(), Pop())
    model = Pipe(model, [1, 1], devices=["cpu", "cpu"], chunks=5)

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

    assert_grad_fn_is_not_portal(output.grad_fn)

    output.sum().backward()
    assert input.grad.mean().item() == 1
