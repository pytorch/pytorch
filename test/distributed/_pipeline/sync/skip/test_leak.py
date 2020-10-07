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

from torch.distributed._pipeline.sync import Pipe, is_checkpointing, is_recomputing
from torch.distributed._pipeline.sync.skip import pop, skippable, stash
from torch.distributed._pipeline.sync.skip.tracker import current_skip_tracker


@skippable(stash=["skip"])
class Stash(nn.Module):
    def forward(self, input):
        yield stash("skip", input)
        return input # noqa


@skippable(pop=["skip"])
class Pop(nn.Module):
    def forward(self, input):
        skip = yield pop("skip")
        return input + skip


@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("checkpoint", ["always", "except_last", "never"])
def test_delete_portal_tensor(train, checkpoint):
    # Without checkpointing:
    # +- Stash --+  +--- Pop ----+ - - - layers
    # | 2,blue,1 |--| 1,orange,0 | - - - tensor_life and portal function
    # +----------+  +------------+
    #
    # With checkpointing:
    # +- Stash --+  +--- Pop ----+  +--- Pop'----+  +- Stash'--+
    # | 3,blue,2 |--| 2,orange,1 |--| 1,orange,0 |--| 1,blue,0 |
    # +----------+  +------------+  +------------+  +----------+

    def portal_tensor_life_is(tensor_life, skip_tracker=None):
        if skip_tracker is None:
            skip_tracker = current_skip_tracker()

        # Get the current portal.
        portal = list(skip_tracker.portals.values())[0]

        if tensor_life == 0:
            return portal.tensor_life == 0 and portal.tensor is None
        else:
            return portal.tensor_life == tensor_life and portal.tensor is not None

    # Check the portal tensor after 'Stash'.
    stash_ = Stash()

    @stash_.register_forward_hook
    def check_portal_tensor_after_stash(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(2)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(1)

    pop_ = Pop()

    @pop_.register_forward_hook
    def check_portal_tensor_after_pop(*_):
        if is_checkpointing():
            assert portal_tensor_life_is(1)
        elif is_recomputing():
            assert portal_tensor_life_is(0)
        else:
            assert portal_tensor_life_is(0)

    class NoPortalTensorAtBackward(nn.Module):
        class F(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.skip_tracker = current_skip_tracker()
                return input.detach()

            @staticmethod
            def backward(ctx, grad):
                assert portal_tensor_life_is(0, skip_tracker=ctx.skip_tracker)
                return grad

        def forward(self, input):
            return self.F.apply(input)

    model = nn.Sequential(NoPortalTensorAtBackward(), stash_, pop_)
    model = Pipe(model, balance=[2, 1], devices=["cpu", "cpu"], chunks=2, checkpoint=checkpoint)

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        output = model(input)
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)


@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
def test_no_portal_without_pipe(train, monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("tried to create Portal without Pipe")

    monkeypatch.setattr("torch.distributed._pipeline.sync.skip.portal.Portal.__init__", deny)

    model = nn.Sequential(Stash(), Pop())

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        output = model(input)
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)
