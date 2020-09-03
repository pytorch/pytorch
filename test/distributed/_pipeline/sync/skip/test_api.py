# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2019 Kakao Brain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from torch import nn

from torch.distributed._pipeline.sync.skip import Namespace, skippable, stash


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
            return self.conv(x) # noqa

    m = Hello()
    assert (
        repr(m)
        == """
@skippable(Hello(
  (conv): Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1))
))
""".strip()
    )
