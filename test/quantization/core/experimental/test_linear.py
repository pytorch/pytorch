# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.experimental.linear import LinearAPoT
import unittest

class TestNonUniformObserver(unittest.TestCase):
    """
        Test linear_APoT_fn by calling forward method
    """
    def test_linear_APoT_fn(self):
        weight = 1000 * torch.rand(4, 4)
        activation2quantize = 1000 * torch.rand(4, 4)

        uniform_observer = MinMaxObserver()
        uniform_observer(activation2quantize)
        scale, zero_point = uniform_observer.calculate_qparams()

        activation = quantize_per_tensor(input=activation2quantize,
                                         scale=scale,
                                         zero_point=zero_point,
                                         dtype=torch.quint8).int_repr()
        # activation = activation.int()

        # calculate result from calling linear forward method
        linear = LinearAPoT(weight)
        linear_result = linear.forward(activation)

        # calculate expected results
        levels_decomposed = linear.decompose_APoT()

        expected_result = torch.zeros(weight.shape[0], activation.shape[1])

        for i in range(weight.shape[0]):
            for j in range(activation.shape[1]):
                for k in range(activation.shape[0]):
                    ele1 = levels_decomposed[i][k]
                    r = int(activation[k][j])

                    for x in ele1:
                        curr_result = x * r
                        expected_result[i][j] += float(curr_result)

        expected_result = expected_result * linear.gamma

        self.assertTrue(torch.equal(linear_result, expected_result))

if __name__ == '__main__':
    unittest.main()
