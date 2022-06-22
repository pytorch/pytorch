# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
import unittest
import random

class TestQuantizer(unittest.TestCase):
    r""" Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform quantization
        (non-uniform quantization reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_rand_k1(self):
        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # generate tensor with random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        quantizer = APoTQuantizer(4, 1, torch.max(tensor2quantize), False)

        # get apot quantized tensor result
        qtensor = quantizer.quantize_APoT(tensor2quantize=tensor2quantize)

        # get uniform quantization quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        qtensor_data = torch.tensor(qtensor).type(torch.uint8)
        uniform_quantized_tensor = uniform_quantized.data

        self.assertTrue(torch.equal(qtensor_data, uniform_quantized_tensor))

    r""" Tests quantize_APoT for k != 1.
        Tests quantize_APoT result on random 1-dim tensor and hardcoded values for
        b=4, k=2 by comparing results to hand-calculated results from APoT paper
        https://arxiv.org/pdf/1909.13144.pdf
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_quantize_APoT_k2(self):
        r"""
        given b = 4, k = 2, alpha = 1.0, we know:
        (from APoT paper example: https://arxiv.org/pdf/1909.13144.pdf)

        quantization_levels = tensor([0.0000, 0.0208, 0.0417, 0.0625, 0.0833, 0.1250, 0.1667,
        0.1875, 0.2500, 0.3333, 0.3750, 0.5000, 0.6667, 0.6875, 0.7500, 1.0000])

        level_indices = tensor([ 0, 3, 12, 15,  2, 14,  8, 11, 10, 1, 13,  9,  4,  7,  6,  5]))
        """

        # generate tensor with random fp values between 0 -> 1000
        tensor2quantize = torch.tensor([0.0215, 0.1692, 0.385, 0.0391])

        quantizer = APoTQuantizer(4, 2, 1.0, False)

        # get apot quantized tensor result
        qtensor = quantizer.quantize_APoT(tensor2quantize=tensor2quantize)
        qtensor_data = torch.tensor(qtensor).type(torch.uint8)

        # expected qtensor values calculated based on
        # corresponding level_indices to nearest quantization level
        # for each fp value in tensor2quantize
        # e.g.
        # 0.0215 in tensor2quantize nearest 0.0208 in quantization_levels -> 3 in level_indices
        expected_qtensor = torch.tensor([3, 8, 13, 12], dtype=torch.uint8)

        self.assertTrue(torch.equal(qtensor_data, expected_qtensor))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_dequantize_quantize_rand_b4(self):
        # generate random size of float2apot between 1->20
        size = random.randint(1, 20)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random values between 0 -> 2**4 = 16
        # because there are 2**b = 2**4 quantization levels total
        float2apot = 16 * torch.rand(size)
        quantizer = APoTQuantizer(4, 2, 1.0, False)
        float2apot = float2apot.int()
        orig_input = torch.clone(float2apot)

        dequantized_result = quantizer.dequantize(float2apot)

        quantized_result = quantizer.quantize_APoT(tensor2quantize=dequantized_result)

        quantized_result = quantized_result.int()

        self.assertTrue(torch.equal(quantized_result, orig_input))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_dequantize_quantize_rand_b6(self):
        # generate random size of float2apot
        size = random.randint(1, 20)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random values between 0 -> 2**6 = 64
        # because there are 2**b = 2**6 quantization levels total
        float2apot = 64 * torch.rand(size)
        quantizer = APoTQuantizer(6, 2, 1.0, False)
        float2apot = float2apot.int()
        orig_input = torch.clone(float2apot)

        dequantized_result = quantizer.dequantize(float2apot)

        quantized_result = quantizer.quantize_APoT(tensor2quantize=dequantized_result)

        quantized_result = quantized_result.int()

        self.assertTrue(torch.equal(quantized_result, orig_input))

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
