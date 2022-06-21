# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
from torch.ao.quantization.experimental.apot_utils import float_to_apot
import unittest
import random
quantize_APoT = APoTQuantizer.quantize_APoT
dequantize = APoTQuantizer.dequantize

class TestQuantizer(unittest.TestCase):
    r""" Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_rand_1d(self):
        # generate random size of tensor2dequantize between 1 -> 20
        size = random.randint(1, 20)

        # generate tensor with random fp values
        tensor2quantize = torch.rand(size, dtype=torch.float)

        qtensor = APoTQuantizer(4, 1, False)

        # get quantized tensor result
        qtensor = qtensor.quantize_APoT(tensor2quantize=tensor2quantize)

        # convert quantized tensor to int repr
        qtensor_int_repr = qtensor.apply_(lambda x: float_to_apot(x, qtensor.quantization_levels, qtensor.level_indices))

        # get uniform quantization quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        qtensor_data = torch.tensor(qtensor_int_repr.data).type(torch.uint8)
        uniform_quantized_tensor = uniform_quantized.data

        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))

    r""" Tests quantize_APoT result on random 2-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_rand_2d(self):
        # generate random size of tensor2dequantize between 1 -> 20
        size = random.randint(1, 20)

        # generate tensor with random fp values
        tensor2quantize = torch.rand((size, size), dtype=torch.float)

        qtensor = APoTQuantizer(4, 1, False)

        # get quantized tensor result
        qtensor = qtensor.quantize_APoT(tensor2quantize=tensor2quantize)

        # convert quantized tensor to int repr
        qtensor_int_repr = qtensor.apply_(lambda x: float_to_apot(x, qtensor.quantization_levels, qtensor.level_indices))

        # get uniform quantization quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        qtensor_data = torch.tensor(qtensor_int_repr.data).type(torch.uint8)
        uniform_quantized_tensor = uniform_quantized.data

        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))

    r""" Tests quantize_APoT for k != 1.
        Tests quantize_APoT result on random 1-dim tensor and
        hardcoded values for b=4, k=2 by comparing results to
        hand-calculated error bound (+/- 0.25 between reduced precision fp representation
        and original input tensor values because max difference between quantization levels
        for b=4, k=2 is 0.25).
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_quantize_APoT_k2(self):
        # generate random size of tensor2dequantize between 1 -> 20
        size = random.randint(1, 20)

        # generate tensor with random fp values
        tensor2quantize = torch.rand((size), dtype=torch.float)

        qtensor = APoTQuantizer(4, 2, False)

        # get apot quantized tensor result
        qtensor = qtensor.quantize_APoT(tensor2quantize=tensor2quantize)
        qtensor_data = torch.tensor(qtensor.data).type(torch.float)

        expectedResult = True

        for result, orig in zip(qtensor_data, tensor2quantize):
            if abs(result - orig) > 0.25:
                expectedResult = False

        self.assertTrue(expectedResult)

    def test_dequantize(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.dequantize(self)


    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
