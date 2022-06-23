# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.quantizer import APoTQuantizer, quantize_APoT, dequantize_APoT
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
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

        observer = APoTObserver(b=4, k=1)
        observer.forward(tensor2quantize)
        qparams = observer.calculate_qparams(signed=False)

        quantizer = APoTQuantizer(alpha=observer.alpha, gamma=qparams[0], quantization_levels=qparams[1], level_indices=qparams[2])

        # get apot quantized tensor result
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                quantization_levels=qparams[1],
                                level_indices=qparams[2])

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
        tensor2quantize = torch.tensor([0, 0.0215, 0.1692, 0.385, 1, 0.0391])

        min_val = torch.min(tensor2quantize)
        max_val = torch.max(tensor2quantize)

        observer = APoTObserver(b=4, k=2)
        observer.forward(tensor2quantize)
        qparams = observer.calculate_qparams(signed=False)

        quantizer = APoTQuantizer(alpha=observer.alpha, gamma=qparams[0], quantization_levels=qparams[1], level_indices=qparams[2])

        # get apot quantized tensor result
        qtensor = quantize_APoT(tensor2quantize=tensor2quantize,
                                quantization_levels=qparams[1],
                                level_indices=qparams[2])
        qtensor_data = torch.tensor(qtensor).type(torch.uint8)

        # expected qtensor values calculated based on
        # corresponding level_indices to nearest quantization level
        # for each fp value in tensor2quantize
        # e.g.
        # 0.0215 in tensor2quantize nearest 0.0208 in quantization_levels -> 3 in level_indices
        expected_qtensor = torch.tensor([0, 3, 8, 13, 5, 12], dtype=torch.uint8)

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
        # make observer
        observer = APoTObserver(4, 2)

        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # make tensor2quantize: random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        observer.forward(tensor2quantize)

        qparams = observer.calculate_qparams(signed=False)

        # make quantizer
        quantizer = APoTQuantizer(observer.alpha, qparams[0], qparams[1], qparams[2])

        # make apot_tensor
        apot_tensor = TensorAPoT(quantizer, tensor2quantize)
        original_input = torch.clone(apot_tensor.data)

        # dequantize apot_tensor
        dequantize_result = dequantize_APoT(apot_tensor)

        # quantize apot_tensor
        quantize_result = quantize_APoT(dequantize_result, qparams[1], qparams[2])

        self.assertTrue(torch.equal(original_input, quantize_result))

    r""" Tests dequantize_apot result on random 1-dim tensor
        and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 12
        * k: 4
    """
    def test_dequantize_quantize_rand_b6(self):
        # make observer
        observer = APoTObserver(12, 4)

        # generate random size of tensor2quantize between 1 -> 20
        size = random.randint(1, 20)

        # make tensor2quantize: random fp values between 0 -> 1000
        tensor2quantize = 1000 * torch.rand(size, dtype=torch.float)

        observer.forward(tensor2quantize)

        qparams = observer.calculate_qparams(signed=False)

        # make quantizer
        quantizer = APoTQuantizer(observer.alpha, qparams[0], qparams[1], qparams[2])

        # make apot_tensor
        apot_tensor = TensorAPoT(quantizer, tensor2quantize)
        original_input = torch.clone(apot_tensor.data)

        # dequantize apot_tensor
        dequantize_result = dequantize_APoT(apot_tensor)

        # quantize apot_tensor
        quantize_result = quantize_APoT(dequantize_result, qparams[1], qparams[2])

        self.assertTrue(torch.equal(original_input, quantize_result))

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
