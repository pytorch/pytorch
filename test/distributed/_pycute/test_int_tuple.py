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
Unit tests for _pycute.int_tuple
"""

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


class TestIntTuple(TestCase):
    def test_product(self):
        self.assertEqual(product(2), 2)

        self.assertEqual(product((3, 2)), 6)

        self.assertEqual(product(product(((2, 3), 4))), 24)

    def test_inner_product(self):
        self.assertEqual(inner_product(2, 3), 6)

        self.assertEqual(inner_product((1, 2), (3, 2)), 7)

        self.assertEqual(inner_product(((2, 3), 4), ((2, 1), 2)), 15)

    def test_shape_div(self):
        self.assertEqual(shape_div((3, 4), 6), (1, 2))

        self.assertEqual(shape_div((3, 4), 12), (1, 1))

        self.assertEqual(shape_div((3, 4), 36), (1, 1))

        self.assertEqual(shape_div(((3, 4), 6), 36), ((1, 1), 2))

        self.assertEqual(shape_div((6, (3, 4)), 36), (1, (1, 2)))

    def test_prefix_product(self):
        self.assertEqual(prefix_product(2), 1)

        self.assertEqual(prefix_product((3, 2)), (1, 3))

        self.assertEqual(prefix_product((3, 2, 4)), (1, 3, 6))

        self.assertEqual(prefix_product(((2, 3), 4)), ((1, 2), 6))

        self.assertEqual(
            prefix_product(((2, 3), (2, 1, 2), (5, 2, 1))),
            ((1, 2), (6, 12, 12), (24, 120, 240)),
        )


if __name__ == "__main__":
    run_tests()
