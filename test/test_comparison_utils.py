#!/usr/bin/env python3
# Owner(s): ["module: internals"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class TestComparisonUtils(TestCase):
    def test_all_equal_no_assert(self, device):
        t = torch.tensor([0.5], device=device)
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    def test_all_equal_no_assert_nones(self, device):
        t = torch.tensor([0.5], device=device)
        torch._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self, device):
        t = torch.tensor([0.5], device=device)

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    def test_assert_strides(self, device):
        t = torch.tensor([0.5], device=device)

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    def test_assert_sizes(self, device):
        t = torch.tensor([0.5], device=device)

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)

    def test_assert_device(self, device):
        t = torch.tensor([0.5], device="cpu")

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, device=device)

    def test_assert_layout(self, device):
        t = torch.tensor([0.5], device=device)

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, layout=torch.sparse_coo)


instantiate_device_type_tests(
    TestComparisonUtils,
    globals(),
    except_for=("meta", "cpu"),
    allow_mps=True,
    allow_xpu=True,
)

if __name__ == "__main__":
    run_tests()
