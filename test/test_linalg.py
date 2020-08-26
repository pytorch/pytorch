import torch
import unittest

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfNoMagma, skipCPUIfNoLapack)

if TEST_NUMPY:
    import numpy as np

class TestLinalg(TestCase):
    exact_dtype = True

    # TODO: test out variant
    # Tests torch.ger, and its alias, torch.outer, vs. NumPy
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.float)
    def test_outer(self, device, dtype):
        a = torch.randn(50, device=device, dtype=dtype)
        b = torch.randn(50, device=device, dtype=dtype)

        ops = (torch.ger, torch.Tensor.ger,
               torch.outer, torch.Tensor.outer)

        expected = np.outer(a.cpu().numpy(), b.cpu().numpy())
        for op in ops:
            actual = op(a, b)
            self.assertEqual(actual, expected)

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @dtypes(torch.double)
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))


        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(IndexError):
            op(t)


instantiate_device_type_tests(TestLinalg, globals())

if __name__ == '__main__':
    run_tests()
