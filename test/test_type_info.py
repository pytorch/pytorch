# Owner(s): ["module: typing"]

from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY, load_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

import torch
import unittest

if TEST_NUMPY:
    import numpy as np


class TestDTypeInfo(TestCase):

    def test_invalid_input(self):
        for dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.complex64, torch.complex128, torch.bool]:
            with self.assertRaises(TypeError):
                _ = torch.iinfo(dtype)

        for dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool]:
            with self.assertRaises(TypeError):
                _ = torch.finfo(dtype)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_iinfo(self):
        for dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.iinfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.iinfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_finfo(self):
        initial_default_type = torch.get_default_dtype()
        for dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.finfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.finfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.eps, xninfo.eps)
            self.assertEqual(xinfo.tiny, xninfo.tiny)
            self.assertEqual(xinfo.resolution, xninfo.resolution)
            self.assertEqual(xinfo.dtype, xninfo.dtype)
            if not dtype.is_complex:
                torch.set_default_dtype(dtype)
                self.assertEqual(torch.finfo(dtype), torch.finfo())

        # Special test case for BFloat16 type
        x = torch.zeros((2, 2), dtype=torch.bfloat16)
        xinfo = torch.finfo(x.dtype)
        self.assertEqual(xinfo.bits, 16)
        self.assertEqual(xinfo.max, 3.38953e+38)
        self.assertEqual(xinfo.min, -3.38953e+38)
        self.assertEqual(xinfo.eps, 0.0078125)
        self.assertEqual(xinfo.tiny, 1.17549e-38)
        self.assertEqual(xinfo.tiny, xinfo.smallest_normal)
        self.assertEqual(xinfo.resolution, 0.01)
        self.assertEqual(xinfo.dtype, "bfloat16")
        torch.set_default_dtype(x.dtype)
        self.assertEqual(torch.finfo(x.dtype), torch.finfo())

        # Restore the default type to ensure that the test has no side effect
        torch.set_default_dtype(initial_default_type)

if __name__ == '__main__':
    run_tests()
