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

    def test_suffix_product(self):
        self.assertEqual(suffix_product(2), 1)

        self.assertEqual(suffix_product((3, 2)), (2, 1))

        self.assertEqual(suffix_product((3, 2, 4)), (8, 4, 1))

        self.assertEqual(suffix_product(((2, 3), 4)), ((12, 4), 1))

        self.assertEqual(
            suffix_product(((2, 3), (2, 1, 2), (5, 2, 1))),
            ((120, 40), (20, 20, 10), (2, 1, 1)),
        )

    def test_crd2idx_basic(self):
        # Test basic int/int case
        self.assertEqual(crd2idx(2, 5, 1), 2)
        self.assertEqual(crd2idx(0, 5, 1), 0)
        self.assertEqual(crd2idx(4, 5, 1), 4)

        # Test with custom stride
        self.assertEqual(crd2idx(2, 5, 3), 6)
        self.assertEqual(crd2idx(1, 5, 3), 3)

    def test_crd2idx_tuple(self):
        # Test tuple coordinates with default stride
        self.assertEqual(crd2idx((1, 2), (3, 4)), 6)  # 1*4 + 2*1 = 6
        self.assertEqual(crd2idx((0, 0), (3, 4)), 0)
        self.assertEqual(crd2idx((2, 3), (3, 4)), 11)  # 2*4 + 3*1 = 11

        # Test with custom stride
        self.assertEqual(crd2idx((1, 2), (3, 4), (8, 2)), 12)  # 1*8 + 2*2 = 12

        # Test 3D case
        self.assertEqual(crd2idx((1, 0, 2), (2, 3, 4)), 14)  # 1*12 + 0*4 + 2*1 = 14

    def test_crd2idx_none(self):
        # Test None coordinate (should default to 0)
        self.assertEqual(crd2idx(None, 5), 0)
        self.assertEqual(crd2idx(None, (3, 4)), 0)

    def test_crd2idx_int_with_tuple_shape(self):
        # Test single integer coordinate with multi-dimensional shape and stride
        # When crd is int and shape is tuple, it converts the int to multi-dim coordinate first
        self.assertEqual(crd2idx(0, (2, 2), (2, 1)), 0)  # 0 -> (0,0) -> 0*2 + 0*1 = 0
        self.assertEqual(crd2idx(1, (2, 2), (2, 1)), 1)  # 1 -> (0,1) -> 0*2 + 1*1 = 1
        self.assertEqual(crd2idx(2, (2, 2), (2, 1)), 2)  # 2 -> (1,0) -> 1*2 + 0*1 = 2
        self.assertEqual(crd2idx(3, (2, 2), (2, 1)), 3)  # 3 -> (1,1) -> 1*2 + 1*1 = 3

        # Test with non-trivial strides
        self.assertEqual(crd2idx(0, (2, 3), (6, 2)), 0)  # 0 -> (0,0) -> 0*6 + 0*2 = 0
        self.assertEqual(crd2idx(1, (2, 3), (6, 2)), 2)  # 1 -> (0,1) -> 0*6 + 1*2 = 2
        self.assertEqual(crd2idx(2, (2, 3), (6, 2)), 4)  # 2 -> (0,2) -> 0*6 + 2*2 = 4
        self.assertEqual(crd2idx(3, (2, 3), (6, 2)), 6)  # 3 -> (1,0) -> 1*6 + 0*2 = 6
        self.assertEqual(crd2idx(4, (2, 3), (6, 2)), 8)  # 4 -> (1,1) -> 1*6 + 1*2 = 8
        self.assertEqual(crd2idx(5, (2, 3), (6, 2)), 10)  # 5 -> (1,2) -> 1*6 + 2*2 = 10

        # Test with larger strides
        self.assertEqual(crd2idx(0, (3, 2), (10, 5)), 0)  # 0 -> (0,0) -> 0*10 + 0*5 = 0
        self.assertEqual(crd2idx(1, (3, 2), (10, 5)), 5)  # 1 -> (0,1) -> 0*10 + 1*5 = 5
        self.assertEqual(
            crd2idx(2, (3, 2), (10, 5)), 10
        )  # 2 -> (1,0) -> 1*10 + 0*5 = 10
        self.assertEqual(
            crd2idx(3, (3, 2), (10, 5)), 15
        )  # 3 -> (1,1) -> 1*10 + 1*5 = 15
        self.assertEqual(
            crd2idx(4, (3, 2), (10, 5)), 20
        )  # 4 -> (2,0) -> 2*10 + 0*5 = 20
        self.assertEqual(
            crd2idx(5, (3, 2), (10, 5)), 25
        )  # 5 -> (2,1) -> 2*10 + 1*5 = 25

        # Test with 3D shape and various strides
        self.assertEqual(
            crd2idx(0, (2, 2, 2), (8, 4, 2)), 0
        )  # 0 -> (0,0,0) -> 0*8 + 0*4 + 0*2 = 0
        self.assertEqual(
            crd2idx(1, (2, 2, 2), (8, 4, 2)), 2
        )  # 1 -> (0,0,1) -> 0*8 + 0*4 + 1*2 = 2
        self.assertEqual(
            crd2idx(2, (2, 2, 2), (8, 4, 2)), 4
        )  # 2 -> (0,1,0) -> 0*8 + 1*4 + 0*2 = 4
        self.assertEqual(
            crd2idx(3, (2, 2, 2), (8, 4, 2)), 6
        )  # 3 -> (0,1,1) -> 0*8 + 1*4 + 1*2 = 6
        self.assertEqual(
            crd2idx(4, (2, 2, 2), (8, 4, 2)), 8
        )  # 4 -> (1,0,0) -> 1*8 + 0*4 + 0*2 = 8
        self.assertEqual(
            crd2idx(7, (2, 2, 2), (8, 4, 2)), 14
        )  # 7 -> (1,1,1) -> 1*8 + 1*4 + 1*2 = 14

        self.assertEqual(
            crd2idx(4, ((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32))), 8
        )  # 4 -> (1,0,0) -> 1*8 = 8

    def test_idx2crd_basic(self):
        # Test basic int/int case
        self.assertEqual(idx2crd(2, 5, 1), 2)
        self.assertEqual(idx2crd(0, 5, 1), 0)
        self.assertEqual(idx2crd(4, 5, 1), 4)

        # Test with custom stride
        self.assertEqual(idx2crd(6, 5, 3), 2)  # (6 // 3) % 5 = 2
        self.assertEqual(idx2crd(3, 5, 3), 1)  # (3 // 3) % 5 = 1

    def test_idx2crd_tuple(self):
        # Test tuple shape with default stride
        self.assertEqual(idx2crd(6, (3, 4)), (1, 2))  # 6 -> (1, 2)
        self.assertEqual(idx2crd(0, (3, 4)), (0, 0))
        self.assertEqual(idx2crd(11, (3, 4)), (2, 3))

        # Test 3D case
        self.assertEqual(idx2crd(14, (2, 3, 4)), (1, 0, 2))

    def test_crd2idx_idx2crd_roundtrip(self):
        # Test that crd2idx and idx2crd are inverse operations
        shapes = [
            5,
            (3, 4),
            (2, 3, 4),
            (2, 2, 2, 2),
        ]

        for shape in shapes:
            size = product(shape)
            for idx in range(size):
                crd = idx2crd(idx, shape)
                recovered_idx = crd2idx(crd, shape)
                self.assertEqual(
                    recovered_idx, idx, f"Failed roundtrip for shape {shape}, idx {idx}"
                )

    def test_idx2crd_crd2idx_roundtrip(self):
        # Test roundtrip starting from coordinates
        test_cases = [
            (0, 5),
            (4, 5),
            ((0, 0), (3, 4)),
            ((1, 2), (3, 4)),
            ((2, 3), (3, 4)),
            ((0, 0, 0), (2, 3, 4)),
            ((1, 2, 3), (2, 3, 4)),
        ]

        for crd, shape in test_cases:
            idx = crd2idx(crd, shape)
            recovered_crd = idx2crd(idx, shape)
            self.assertEqual(
                recovered_crd, crd, f"Failed roundtrip for crd {crd}, shape {shape}"
            )


if __name__ == "__main__":
    run_tests()
