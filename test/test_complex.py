import math
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
import unittest

if TEST_NUMPY:
    import numpy as np

devices = (torch.device('cpu'), torch.device('cuda:0'))


class TestComplexTensor(TestCase):
    def test_to_list_with_complex_64(self):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), dtype=torch.complex64).tolist(), [[0j, 0j], [0j, 0j]])

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_exp(self):
        def exp_fn(dtype):
            a = torch.tensor(1j, dtype=dtype) * torch.arange(18) / 3 * math.pi
            expected = np.exp(a.numpy())
            actual = torch.exp(a)
            self.assertEqual(actual, torch.from_numpy(expected))

        exp_fn(torch.complex64)
        exp_fn(torch.complex128)

    def test_dtype_inference(self):
        # issue: https://github.com/pytorch/pytorch/issues/36834
        torch.set_default_dtype(torch.double)
        x = torch.tensor([3., 3. + 5.j])
        self.assertEqual(x.dtype, torch.cdouble)

    def test_torch_complex(self):
        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        z = torch.complex(real, imag)
        self.assertEqual(torch.tensor([1.0+3.0j, 2.0+4.0j]), z)

    def test_torch_complex_polar(self):
        abs = torch.tensor([1, 2], dtype=torch.float64)
        angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64)
        z = torch.complex_polar(abs, angle)
        self.assertEqual(torch.tensor([0+1.0j, -1.41421356237-1.41421356237j],
                                      dtype=torch.complex128),
                         z, atol=1e-5, rtol=1e-5)

    def test_torch_complex_error(self):
        real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float64)
        error = ("Expected object of scalar type Float but got scalar type "
                 "Double for argument 'imag'")
        with self.assertRaisesRegex(RuntimeError, error):
            z = torch.complex(real, imag)

        abs = torch.tensor([1, 2])
        angle = torch.tensor([3, 4])
        error = ("\"complex_polar_cpu\" not implemented for 'Long'")
        with self.assertRaisesRegex(RuntimeError, error):
            z = torch.complex_polar(abs, angle)

if __name__ == '__main__':
    run_tests()
