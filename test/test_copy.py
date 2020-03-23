import unittest
import torch

class TestCopy(unittest.TestCase):

    def test_copy(self):
        a = torch.randn(2,3,dtype=torch.complex64).cuda()
        b = a.to(torch.complex128)
