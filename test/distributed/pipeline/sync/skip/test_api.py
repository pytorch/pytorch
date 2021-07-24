# Copyright 2019 Kakao Brain
#
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
import copy

from torch import nn

from torch.distributed.pipeline.sync.skip import Namespace, skippable, stash


def test_namespace_difference():
    ns1 = Namespace()
    ns2 = Namespace()
    assert ns1 != ns2


def test_namespace_copy():
    ns = Namespace()
    assert copy.copy(ns) == ns
    assert copy.copy(ns) is not ns


def test_skippable_repr():
    @skippable(stash=["hello"])
    class Hello(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)

        def forward(self, x):
            yield stash("hello", x)
            return self.conv(x)  # noqa: B901

    m = Hello()
    assert (
        repr(m)
        == """
@skippable(Hello(
  (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
))
""".strip()
    )
