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
        for dtype in [torch.float32, torch.float64]:
            with self.assertRaises(TypeError):
                _ = torch.iinfo(dtype)

        for dtype in [torch.int64, torch.int32, torch.int16, torch.uint8]:
            with self.assertRaises(TypeError):
                _ = torch.finfo(dtype)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_iinfo(self):
        for dtype in [torch.int64, torch.int32, torch.int16, torch.uint8]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.iinfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.iinfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_finfo(self):
        initial_default_type = torch.get_default_dtype()
        for dtype in [torch.float32, torch.float64]:
            x = torch.zeros((2, 2), dtype=dtype)
            xinfo = torch.finfo(x.dtype)
            xn = x.cpu().numpy()
            xninfo = np.finfo(xn.dtype)
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.eps, xninfo.eps)
            self.assertEqual(xinfo.tiny, xninfo.tiny)
            torch.set_default_dtype(dtype)
            self.assertEqual(torch.finfo(dtype), torch.finfo())
        # Restore the default type to ensure that the test has no side effect
        torch.set_default_dtype(initial_default_type)

if __name__ == '__main__':
    run_tests()
