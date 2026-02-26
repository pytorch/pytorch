# ruff: noqa: PGH004, G004, F403
# flake8: noqa
# Owner(s): ["oncall: distributed"]
#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Unit tests for _pycute.coalesce
"""

import logging

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


_LOGGER = logging.getLogger(__name__)


class TestCoalesce(TestCase):
    def helper_test_coalesce(self, layout, expected_coalesced_layout):
        layoutR = coalesce(layout)

        _LOGGER.debug(f"{layout}  =>  {layoutR}")

        self.assertEqual(expected_coalesced_layout.shape, layoutR.shape)
        self.assertEqual(expected_coalesced_layout.stride, layoutR.stride)
        self.assertEqual(size(layoutR), size(layout))

        for i in range(size(layout)):
            self.assertEqual(layoutR(i), layout(i))

    def test_coalesce(self):
        layout = Layout(1, 0)
        expected_coalesced_layout = Layout(1, 0)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout(1, 1)
        expected_coalesced_layout = Layout(1, 0)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((1, 1), (42, 3))
        expected_coalesced_layout = Layout(1, 0)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((1, 1, 1), (17, 4, 3))
        expected_coalesced_layout = Layout(1, 0)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 4))
        expected_coalesced_layout = Layout(8, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 4, 6))
        expected_coalesced_layout = Layout(48, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 4, 6), (1, 6, 2))
        expected_coalesced_layout = Layout((2, 4, 6), (1, 6, 2))
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 1, 6), (1, 7, 2))
        expected_coalesced_layout = Layout((2, 6), (1, 2))
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 1, 6), (4, 7, 8))
        expected_coalesced_layout = Layout((2, 6), (4, 8))
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, (4, 6)))
        expected_coalesced_layout = Layout(48, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((1, 2), (8, 1))
        expected_coalesced_layout = Layout(2, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 4), (4, 1))
        expected_coalesced_layout = Layout(8, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 4, 6), (24, 6, 1))
        expected_coalesced_layout = Layout(48, 1)
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout((2, 1, 3), (2, 4, 4))
        expected_coalesced_layout = Layout((2, 3), (2, 4))
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
        expected_coalesced_layout = Layout((2, 2, 2, 2), (1, 4, 8, 32))
        self.helper_test_coalesce(layout, expected_coalesced_layout)

        layout = Layout(((2, 2), (2, 2)), ((32, 8), (4, 1)))
        expected_coalesced_layout = Layout((2, 4, 2), (32, 4, 1))
        self.helper_test_coalesce(layout, expected_coalesced_layout)


if __name__ == "__main__":
    run_tests()
