# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
import unittest
import random
quantize_APoT = TensorAPoT.quantize_APoT
dequantize = TensorAPoT.dequantize

class TestQuantizedTensor(unittest.TestCase):
    """ Tests quantize_APoT result on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform observer
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_quantize_APoT_rand_1d(self):
        # generate random size of tensor2dequantize between 1 -> 16
        # because there are 2**b = 2**4 quantization levels total
        size = random.randint(1, 16)

        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size)

        apot_quantized = TensorAPoT(4, 2, False)

        # get apot quantized tensor result
        apot_quantized = apot_quantized.quantize_APoT(tensor2quantize=tensor2quantize)

        # get uniform observer quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        apot_quantized_tens = torch.tensor(apot_quantized.data).type(torch.uint8)
        uniform_quantized_tens = uniform_quantized.data

        self.assertTrue(torch.equal(apot_quantized_tens, uniform_quantized_tens))

    """ Tests quantize_APoT result on random 2-dim tensor
        and hardcoded values for b, k by comparing to uniform observer
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_quantize_APoT_rand_2d(self):
        # generate random size of tensor2dequantize between 1 -> 64
        # because there are 2**b = 2**6 quantization levels total
        size = random.randint(1, 64)

        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size, size)

        apot_quantized = TensorAPoT(6, 2, False)

        # get apot quantized tensor result
        apot_quantized = apot_quantized.quantize_APoT(tensor2quantize=tensor2quantize)

        # get uniform observer quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        apot_quantized_tens = torch.tensor(apot_quantized.data).type(torch.uint8)
        uniform_quantized_tens = uniform_quantized.data

        self.assertTrue(torch.equal(apot_quantized_tens, uniform_quantized_tens))

    def test_dequantize(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.dequantize(self)


    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
