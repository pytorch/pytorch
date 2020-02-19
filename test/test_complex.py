import unittest

import torch
import numpy as np

from numpy.testing import *

devices = (torch.device('cpu'), torch.device('cuda:0'))

class TestComplexTensor(unittest.TestCase):
    def test_to_list_with_complex_64(self):
        # test that the complex float tensor is being made with the expected values and there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2,2), dtype=torch.complex64).tolist(), [[0j, 0j], [0j, 0j]])

if __name__ == '__main__':
    unittest.main()
