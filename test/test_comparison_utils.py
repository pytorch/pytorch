#!/usr/bin/env python3
# Owner(s): ["module: internals"]

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestComparisonUtils(TestCase):
    def test_all_equal_no_assert(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    def test_all_equal_no_assert_nones(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    def test_assert_strides(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    def test_assert_sizes(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_assert_device(self):
        t = torch.tensor([0.5], device="cpu")

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, device="cuda")

    def test_assert_layout(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, layout=torch.sparse_coo)


if __name__ == "__main__":
    run_tests()
