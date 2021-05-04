# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync import Pipe, is_checkpointing, is_recomputing
from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from torch.distributed.pipeline.sync.skip.tracker import current_skip_tracker


@skippable(stash=["skip"])
class Stash(nn.Module):
    def forward(self, input):
        yield stash("skip", input)
        return input  # noqa: B901


@skippable(pop=["skip"])
class Pop(nn.Module):
    def forward(self, input):
        skip = yield pop("skip")
        return input + skip


@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("checkpoint", ["always", "except_last", "never"])
def test_delete_portal_tensor(train, checkpoint, setup_rpc):
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
    model = Pipe(model, chunks=2, checkpoint=checkpoint)

    input = torch.rand(10, requires_grad=True)

    if train:
        model.train()
        output = model(input).local_value()
        output.norm().backward()
    else:
        model.eval()
        with torch.no_grad():
            model(input)


@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
def test_no_portal_without_pipe(train, monkeypatch, setup_rpc):
    def deny(*args, **kwargs):
        raise AssertionError("tried to create Portal without Pipe")

    monkeypatch.setattr("torch.distributed.pipeline.sync.skip.portal.Portal.__init__", deny)

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
