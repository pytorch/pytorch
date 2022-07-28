# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.linear import LinearAPoT
from torch.nn.modules.linear import Linear
import unittest

class TestNonUniformObserver(unittest.TestCase):
    """
        Test linear_APoT_fn by comparing to uniform linear
        for 2d tensors with size (4,4)
    """
    def test_linear_APoT_fn(self):
        # weight: fp tensor
        weight = 1000 * torch.rand(4, 4)

        # activtion: fp32 tensor with ~ integer values
        activation = torch.randint(low=0, high=255, size=(4, 4), dtype=torch.float)

        # calculate result from calling linear forward method
        apot_linear = LinearAPoT(weight, 8, 1)
        apot_linear_result = apot_linear(activation)

        # calculate expected results
        fp_linear = Linear(4, 4, bias=False)

        # set weight for fp linear
        apot_quantized_weight_float = apot_linear.weight.type(torch.FloatTensor)
        fp_linear_weight = torch.nn.parameter.Parameter(data=apot_quantized_weight_float)
        fp_linear.weight = fp_linear_weight

        fp_linear_result = fp_linear(activation).data

        self.assertTrue(torch.equal(apot_linear_result, fp_linear_result))

if __name__ == '__main__':
    unittest.main()
