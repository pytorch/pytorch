# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from torch import nn

from torch.distributed.pipeline.sync.skip import Namespace, pop, skippable, stash
from torch.distributed.pipeline.sync.skip.layout import inspect_skip_layout


class Pass(nn.Module):
    def forward(self, input):
        return input


@skippable(stash=["foo"])
class StashFoo(nn.Module):
    def forward(self, input):
        yield stash("foo", input)
        return input # noqa


@skippable(pop=["foo"])
class PopFoo(nn.Module):
    def forward(self, input):
        foo = yield stash("foo")
        return input + foo


@skippable(stash=["bar"])
class StashBar(nn.Module):
    def forward(self, input):
        yield stash("bar", input)
        return input # noqa


@skippable(pop=["bar"])
class PopBar(nn.Module):
    def forward(self, input):
        bar = yield pop("bar")
        return input + bar


def test_no_skippables():
    p1 = nn.Sequential(Pass())
    p2 = nn.Sequential(Pass())

    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]

    assert policy == [[], []]


def test_inner_partition():
    p1 = nn.Sequential(StashFoo(), PopFoo())
    p2 = nn.Sequential(Pass())

    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]

    assert policy == [[], []]


def test_adjoining_partitions():
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(PopFoo())

    layout = inspect_skip_layout([p1, p2])
    policy = [list(layout.copy_policy(i)) for i in range(2)]

    assert policy == [[], [(0, None, "foo")]]


def test_far_partitions():
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(Pass())
    p3 = nn.Sequential(PopFoo())

    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]

    assert policy == [[], [], [(0, None, "foo")]]


def test_pop_2_from_different_partitions():
    p1 = nn.Sequential(StashFoo())
    p2 = nn.Sequential(StashBar())
    p3 = nn.Sequential(PopBar(), PopFoo())

    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]

    # p3 pops 'bar' before 'foo', but the plan is sorted by source partition index.
    assert policy == [[], [], [(0, None, "foo"), (1, None, "bar")]]


def test_namespace():
    ns1 = Namespace()
    ns2 = Namespace()

    p1 = nn.Sequential(StashFoo().isolate(ns1))
    p2 = nn.Sequential(StashFoo().isolate(ns2))
    p3 = nn.Sequential(PopFoo().isolate(ns2), PopFoo().isolate(ns1))

    layout = inspect_skip_layout([p1, p2, p3])
    policy = [list(layout.copy_policy(i)) for i in range(3)]

    # p3 pops 'bar' before 'foo', but the plan is sorted by source partition index.
    assert policy == [[], [], [(0, ns1, "foo"), (1, ns2, "foo")]]
