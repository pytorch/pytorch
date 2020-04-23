import math
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU
import unittest

if TEST_NUMPY:
    import numpy as np

devices = (torch.device('cpu'), torch.device('cuda:0'))


class TestComplexTensor(TestCase):

    @onlyCPU
    def test_to_list_with_complex_64(self, device):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), dtype=torch.complex64).tolist(), [[0j, 0j], [0j, 0j]])

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @onlyCPU
    def test_exp(self, device):
        def exp_fn(dtype):
            a = torch.tensor(1j, dtype=dtype) * torch.arange(18) / 3 * math.pi
            expected = np.exp(a.numpy())
            actual = torch.exp(a)
            self.assertEqual(actual, torch.from_numpy(expected))

        exp_fn(torch.complex64)
        exp_fn(torch.complex128)

    @onlyCPU
    def test_copy_real_imag_methods(self, device):
        real = torch.randn(4)
        imag = torch.randn(4)
        complex_tensor = real + 1j * imag
        self.assertEqual(complex_tensor.copy_real(), real)
        self.assertEqual(complex_tensor.copy_imag(), imag)

instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == '__main__':
    run_tests()
