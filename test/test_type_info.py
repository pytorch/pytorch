from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY, load_tests, torch_to_numpy_dtype_dict

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
            iinfo = torch.iinfo(dtype)
            niinfo = np.iinfo(torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(iinfo.bits, niinfo.bits)
            self.assertEqual(iinfo.max, niinfo.max)
            self.assertEqual(iinfo.min, niinfo.min)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_finfo(self):
        initial_default_type = torch.get_default_dtype()
        for dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            finfo = torch.finfo(dtype)
            np_finfo = np.finfo(torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(finfo.bits, np_finfo.bits)
            self.assertEqual(finfo.max, np_finfo.max)
            self.assertEqual(finfo.min, np_finfo.min)
            self.assertEqual(finfo.eps, np_finfo.eps)
            self.assertEqual(finfo.tiny, np_finfo.tiny)
            if not dtype.is_complex:
                torch.set_default_dtype(dtype)
                self.assertEqual(torch.finfo(dtype), torch.finfo())
        # Restore the default type to ensure that the test has no side effect
        torch.set_default_dtype(initial_default_type)

if __name__ == '__main__':
    run_tests()
