# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.linear import LinearAPoT
import unittest

class TestNonUniformObserver(unittest.TestCase):
    """
        Test linear_APoT_fn by calling forward method
    """
    def test_linear_APoT_fn(self):
        linear = LinearAPoT(torch.tensor([[0.13245, 55.23234, 53.1234, 24.1345], [0.124, 32.12432, 24.3156, 37.1235], [0.13245, 55.23234, 53.1234, 24.1345], [0.13245, 55.23234, 53.1234, 24.1345]]))

        linear_result = linear.forward(torch.tensor([[1, 3, 5, 7], [0, 3, 2, 9], [5, 3, 1, 10], [11, 2, 4, 8]]))

        # calculate expected results



if __name__ == '__main__':
    unittest.main()
