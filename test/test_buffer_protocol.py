# This test suit will be merged to test_torch.py.
import sys
import math
import random
import torch
import torch.cuda
import tempfile
import unittest
from itertools import product, chain
from functools import wraps
from common import TestCase, iter_indices, TEST_NUMPY

if TEST_NUMPY:
    import numpy as np

class TestTorch(TestCase):

    def test_sanity_tensor_1(self):
        x = torch.CharTensor(2, 5)
        z = memoryview(x)

        x = torch.CharTensor(4, 5)
        z = memoryview(x)

    def test_sanity_storage_2(self):
        x = torch.CharStorage(5)
        z = memoryview(x)

        x = torch.CharStorage(4)
        z = memoryview(x)

    def test_sanity_issue_14_1(self):
        x = torch.DoubleStorage(12)
        z = memoryview(x)
        len(z)
        z.ndim
        z.tolist()
        z.tobytes()

    def test_sanity_issue_14_2(self):
        memoryview(torch.DoubleStorage(12)).tobytes()

if __name__ == '__main__':
    unittest.main()
