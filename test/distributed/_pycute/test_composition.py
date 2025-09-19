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
Unit tests for _pycute.composition
"""

import logging

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


_LOGGER = logging.getLogger(__name__)


class TestComposition(TestCase):
    def helper_test_composition(self, layoutA, layoutB):
        layoutR = composition(layoutA, layoutB)

        _LOGGER.debug(f"{layoutA} o {layoutB}  =>  {layoutR}")

        # True post-condition: Every coordinate c of layoutB with L1D(c) < size(layoutR) is a coordinate of layoutR.

        # Test that R(c) = A(B(c)) for all coordinates c in layoutR
        for i in range(size(layoutR)):
            self.assertEqual(layoutR(i), layoutA(layoutB(i)))

    def test_composition(self):
        layoutA = Layout(1, 0)
        layoutB = Layout(1, 0)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(1, 0)
        layoutB = Layout(1, 1)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(1, 1)
        layoutB = Layout(1, 0)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(1, 1)
        layoutB = Layout(1, 1)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout(4)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4), (2))
        layoutB = Layout(4)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout((4), (2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4), (0))
        layoutB = Layout(4)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout((4), (0))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((1), (0))
        layoutB = Layout(4)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout((1), (0))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout(2)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4), (2))
        layoutB = Layout(2)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(4)
        layoutB = Layout((2), (2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4), (2))
        layoutB = Layout((2), (2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(12)
        layoutB = Layout((4, 3))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((12), (2))
        layoutB = Layout((4, 3))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(12)
        layoutB = Layout((4, 3), (3, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((12), (2))
        layoutB = Layout((4, 3), (3, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(12)
        layoutB = Layout((2, 3), (2, 4))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3))
        layoutB = Layout((4, 3))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3))
        layoutB = Layout(12)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3))
        layoutB = Layout((6), (2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3))
        layoutB = Layout((6, 2), (2, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3), (3, 1))
        layoutB = Layout((4, 3))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3), (3, 1))
        layoutB = Layout(12)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3), (3, 1))
        layoutB = Layout((6), (2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 3), (3, 1))
        layoutB = Layout((6, 2), (2, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((8, 8))
        layoutB = Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((8, 8), (8, 1))
        layoutB = Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
        layoutB = Layout(8, 4)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout(((4, 2)), ((1, 16)))
        layoutB = Layout((4, 2), (2, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((2, 2), (2, 1))
        layoutB = Layout((2, 2), (2, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 8, 2))
        layoutB = Layout((2, 2, 2), (2, 8, 1))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 8, 2), (2, 8, 1))
        layoutB = Layout((2, 2, 2), (1, 8, 2))
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4, 8, 2), (2, 8, 1))
        layoutB = Layout((4, 2, 2), (2, 8, 1))
        self.helper_test_composition(layoutA, layoutB)

        # Pre-coalesced LHS
        layoutA = Layout((4, 6, 8), (1, 4, 7))
        layoutB = Layout((6), (1))
        self.helper_test_composition(layoutA, layoutB)

        # Pre-coalesced RHS
        layoutA = Layout((8, 6, 4), (7, 4, 1))
        layoutB = Layout((6), (1))
        self.helper_test_composition(layoutA, layoutB)

        # Case when not meet stride divisibility condition
        with self.assertRaises(AssertionError):
            layoutA = Layout((4, 6, 8, 10), (2, 3, 5, 7))
            layoutB = Layout(6, 12)
            self.helper_test_composition(layoutA, layoutB)

        # Mid-layout truncation
        layoutA = Layout((10, 8, 6, 4), (7, 5, 3, 2))
        layoutB = Layout(6, 12)
        self.helper_test_composition(layoutA, layoutB)

        layoutA = Layout((4,), (3,))
        layoutB = Layout((6,), (2,))
        self.helper_test_composition(layoutA, layoutB)


if __name__ == "__main__":
    run_tests()
