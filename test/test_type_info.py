# mypy: allow-untyped-defs
# Owner(s): ["module: typing"]

from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    set_default_dtype,
    TEST_NUMPY,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

import sys
import unittest

import torch


if TEST_NUMPY:
    import numpy as np


class TestDTypeInfo(TestCase):
    def test_invalid_input(self):
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.iinfo(dtype)

        for dtype in [
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.finfo(dtype)
            with self.assertRaises(RuntimeError):
                dtype.to_complex()

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
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]:
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
                with set_default_dtype(dtype):
                    self.assertEqual(torch.finfo(dtype), torch.finfo())

        # Special test case for BFloat16 type
        x = torch.zeros((2, 2), dtype=torch.bfloat16)
        xinfo = torch.finfo(x.dtype)
        self.assertEqual(xinfo.bits, 16)
        self.assertEqual(xinfo.max, 3.38953e38)
        self.assertEqual(xinfo.min, -3.38953e38)
        self.assertEqual(xinfo.eps, 0.0078125)
        self.assertEqual(xinfo.tiny, 1.17549e-38)
        self.assertEqual(xinfo.tiny, xinfo.smallest_normal)
        self.assertEqual(xinfo.resolution, 0.01)
        self.assertEqual(xinfo.dtype, "bfloat16")
        with set_default_dtype(x.dtype):
            self.assertEqual(torch.finfo(x.dtype), torch.finfo())

        # Special test case for Float8_E5M2
        xinfo = torch.finfo(torch.float8_e5m2)
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 57344.0)
        self.assertEqual(xinfo.min, -57344.0)
        self.assertEqual(xinfo.eps, 0.25)
        self.assertEqual(xinfo.tiny, 6.10352e-05)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e5m2")

        # Special test case for Float8_E4M3FN
        xinfo = torch.finfo(torch.float8_e4m3fn)
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 448.0)
        self.assertEqual(xinfo.min, -448.0)
        self.assertEqual(xinfo.eps, 0.125)
        self.assertEqual(xinfo.tiny, 0.015625)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e4m3fn")

    def test_to_complex(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/124868
        # If reference count is leaked this would be a set of 10 elements
        ref_cnt = {sys.getrefcount(torch.float32.to_complex()) for _ in range(10)}
        # pyrefly: ignore  # missing-attribute
        self.assertLess(len(ref_cnt), 3)

        self.assertEqual(torch.float64.to_complex(), torch.complex128)
        self.assertEqual(torch.float32.to_complex(), torch.complex64)
        self.assertEqual(torch.float16.to_complex(), torch.complex32)

    def test_to_real(self):
        # Regression test for https://github.com/pytorch/pytorch/issues/124868
        # If reference count is leaked this would be a set of 10 elements
        ref_cnt = {sys.getrefcount(torch.cfloat.to_real()) for _ in range(10)}
        # pyrefly: ignore  # missing-attribute
        self.assertLess(len(ref_cnt), 3)

        self.assertEqual(torch.complex128.to_real(), torch.double)
        self.assertEqual(torch.complex64.to_real(), torch.float32)
        self.assertEqual(torch.complex32.to_real(), torch.float16)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
