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

if __name__ == '__main__':
    run_tests()
