import math
import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import unittest

TEST_NUMPY = True
try:
    import numpy as np
except ImportError:
    TEST_NUMPY = False

devices = (torch.device('cpu'), torch.device('cuda:0'))


class TestComplexTensor(TestCase):
    def test_to_list_with_complex_64(self):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), dtype=torch.complex64).tolist(), [[0j, 0j], [0j, 0j]])

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_exp(self):
        def exp_fn(dtype):
            a = (1j * torch.tensor([0, 1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype) /
                 3 * math.pi)
            expected = np.exp(a.numpy())
            actual = torch.exp(a)
            self.assertEqual(actual, torch.from_numpy(expected))
        
        exp_fn(torch.complex64)
        exp_fn(torch.complex128)


if __name__ == '__main__':
    run_tests()
