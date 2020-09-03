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

from torch.distributed._pipeline.sync.pipeline import clock_cycles


def test_clock_cycles():
    assert list(clock_cycles(1, 1)) == [[(0, 0)]]
    assert list(clock_cycles(1, 3)) == [[(0, 0)], [(0, 1)], [(0, 2)]]
    assert list(clock_cycles(3, 1)) == [[(0, 0)], [(1, 0)], [(2, 0)]]

    assert list(clock_cycles(3, 3)) == [  # noqa
        [(0, 0)],
        [(1, 0), (0, 1)],
        [(2, 0), (1, 1), (0, 2)],
        [(2, 1), (1, 2)],
        [(2, 2)],
    ]

    assert list(clock_cycles(4, 2)) == [  # noqa
        [(0, 0)],
        [(1, 0), (0, 1)],
        [(2, 0), (1, 1)],
        [(3, 0), (2, 1)],
        [(3, 1)],
    ]
