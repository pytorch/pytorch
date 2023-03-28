# Owner(s): ["oncall: distributed"]

# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch
from torch import nn

from torch.distributed.pipeline.sync.skip import pop, skippable, stash
from torch.distributed.pipeline.sync.skip.tracker import SkipTracker, use_skip_tracker
from torch.testing._internal.common_utils import run_tests


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
            return input * 2  # noqa: B901

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
            return input * 2  # noqa: B901

    @skippable(pop=["foo"])
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo

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
            return input * 2  # noqa: B901

    l1 = Stash()

    with pytest.raises(RuntimeError):
        l1(torch.tensor(42))


def test_pop_not_declared():
    @skippable(stash=["foo"])
    class Stash(nn.Module):
        def forward(self, input):
            yield stash("foo", input)
            return input * 2  # noqa: B901

    @skippable()
    class Pop(nn.Module):
        def forward(self, input):
            foo = yield pop("foo")
            return foo

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
            return input * 2  # noqa: B901

    l1 = Stash()
    l1(torch.tensor(42))


if __name__ == "__main__":
    run_tests()
