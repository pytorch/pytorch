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

from torch.distributed._pipeline.sync.skip import pop, skippable, stash
from torch.distributed._pipeline.sync.skip.tracker import SkipTracker, use_skip_tracker


@pytest.fixture(autouse=True)
def skip_tracker():
    skip_tracker = SkipTracker()
    with use_skip_tracker(skip_tracker):
        yield skip_tracker


def test_stash(skip_tracker):
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2 # noqa

    l1 = Stash()

    assert len(skip_tracker.tensors) == 0

    with use_skip_tracker(skip_tracker):
        l1(torch.tensor(42))

    assert len(skip_tracker.tensors) == 1


def test_pop():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2 # noqa

    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo # noqa

    l1 = Stash()
    l2 = Pop()

    output = l2(l1(torch.tensor(42)))

    assert output.item() == 42


def test_declare_but_not_use():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            return input * 2

    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            return input * 3

    l1 = Stash()
    l2 = Pop()

    with pytest.raises(RuntimeError):
        l1(torch.tensor(42))

    with pytest.raises(RuntimeError):
        l2(torch.tensor(42))


def test_stash_not_declared():
    @skippable()
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2 # noqa

    l1 = Stash()

    with pytest.raises(RuntimeError):
        l1(torch.tensor(42))


def test_pop_not_declared():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2 # noqa

    @skippable()
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo # noqa

    l1 = Stash()
    l2 = Pop()

    latent = l1(torch.tensor(42))

    with pytest.raises(RuntimeError):
        l2(latent)


def test_pop_not_stashed():
    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            yield pop("foo")

    l1 = Pop()

    with pytest.raises(RuntimeError):
        l1(torch.tensor(42))


def test_stash_none():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", None)
            return input * 2 # noqa

    l1 = Stash()
    l1(torch.tensor(42))
