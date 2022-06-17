# Owner(s): ["oncall: quantization"]

import torch
from torch import quantize_per_tensor
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT, APoTRepr
import unittest
import random
quantize_APoT = TensorAPoT.quantize_APoT
dequantize = TensorAPoT.dequantize

class TestQuantizedTensor(unittest.TestCase):
    r""" Tests quantize_APoT result (int representation) on random 1-dim tensor
        and hardcoded values for b, k by comparing to uniform observer
        (non-uniform observer reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_rand_1d(self):
        # generate random size of tensor2dequantize between 1 -> 16
        # because there are 2**b = 2**4 quantization levels total
        size = random.randint(1, 16)

        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size)

        apot_tens = TensorAPoT(4, 1, False)

        # get apot quantized tensor result
        apot_tens = apot_tens.quantize_APoT(tensor2quantize=tensor2quantize, apot_repr=APoTRepr.level_indices)

        # get uniform observer quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        apot_tens_tens = torch.tensor(apot_tens.data).type(torch.uint8)
        uniform_quantized_tens = uniform_quantized.data

        self.assertTrue(torch.equal(apot_tens_tens, uniform_quantized_tens))

    r""" Tests quantize_APoT result (int representation) on random 2-dim tensor
        and hardcoded values for b, k by comparing to uniform observer
        (non-uniform observer reduces to uniform for k = 1)
        quantized tensor (https://pytorch.org/docs/stable/generated/torch.quantize_per_tensor.html)
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_rand_2d(self):
        # generate random size of tensor2dequantize between 1 -> 16
        # because there are 2**b = 2**4 quantization levels total
        size = random.randint(1, 16)

        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size, size)

        apot_tens = TensorAPoT(4, 1, False)

        # get apot quantized tensor result
        apot_tens = apot_tens.quantize_APoT(tensor2quantize=tensor2quantize, apot_repr=APoTRepr.level_indices)

        # get uniform observer quantized tensor result
        uniform_quantized = quantize_per_tensor(input=tensor2quantize, scale=1.0, zero_point=0, dtype=torch.quint8).int_repr()

        apot_tens_tens = torch.tensor(apot_tens.data).type(torch.uint8)
        uniform_quantized_tens = uniform_quantized.data

        self.assertTrue(torch.equal(apot_tens_tens, uniform_quantized_tens))

    r""" Tests quantize_APoT result (reduced precision fp representation) on random 1-dim tensor
        and hardcoded values for b, k by comparing to int representation
        * tensor2quantize: Tensor
        * b: 4
        * k: 1
    """
    def test_quantize_APoT_reduced_precision(self):
        # generate random size of tensor2dequantize between 1 -> 16
        # because there are 2**b = 2**4 quantization levels total
        size = random.randint(1, 16)

        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size)

        apot_tens = TensorAPoT(4, 2, False)

        # get apot reduced precision fp quantized tensor result
        apot_tens_red_prec = torch.clone(apot_tens.quantize_APoT(tensor2quantize=tensor2quantize,
                                                                 apot_repr=APoTRepr.reduced_precision_fp))
        reduced_precision_lst = list(apot_tens_red_prec)

        # get apot int representation quantized tensor result
        apot_tens_int_rep = torch.clone(apot_tens.quantize_APoT(tensor2quantize=tensor2quantize, apot_repr=APoTRepr.level_indices))
        int_rep_lst = list(apot_tens_int_rep)

        # get quantization levels and level indices
        quant_levels_lst = list(apot_tens.quantization_levels)
        level_indices_lst = list(apot_tens.level_indices)

        # compare with quantized int representation to verify result
        expectedResult = True
        for ele, i in zip(reduced_precision_lst, int_rep_lst):
            reduced_prec_idx = quant_levels_lst.index(ele)
            int_rep_idx = level_indices_lst.index(i)
            if int_rep_idx != reduced_prec_idx:
                expectedResult = False

        self.assertTrue(expectedResult)

    r""" Tests dequantize_apot result on random 1-dim tensor
        (int repr) and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_dequantize_quantize_rand_1d(self):
        # generate random size of tensor2dequantize
        size = random.randint(1, 16)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random values between 0 -> 2**4 = 16
        # because there are 2**b = 2**4 quantization levels total
        tensor2dequantize = 16 * torch.rand(size)
        apot_tens = TensorAPoT(4, 2, False)
        apot_tens.data = tensor2dequantize.int()
        apot_tens.apot_repr = APoTRepr.level_indices
        orig_input = torch.clone(apot_tens.data)

        dequantized_result = apot_tens.dequantize()

        quantized_result = apot_tens.quantize_APoT(tensor2quantize=dequantized_result, apot_repr=APoTRepr.level_indices)

        self.assertTrue(torch.equal(dequantized_result, quantized_result))

    r""" Tests dequantize_apot result on random 2-dim tensor
        (int repr) and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_dequantize_quantize_rand_2d(self):
        # generate random size of tensor2dequantize
        size = random.randint(1, 64)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random values between 0 -> 2**6 = 64
        # because there are 2**b = 2**6 quantization levels total
        tensor2dequantize = 64 * torch.rand(size, size)
        apot_tens = TensorAPoT(6, 2, False)
        apot_tens.data = tensor2dequantize.int()
        apot_tens.apot_repr = APoTRepr.level_indices
        orig_input = torch.clone(apot_tens.data)

        dequantized_result = apot_tens.dequantize()

        quantized_result = apot_tens.quantize_APoT(tensor2quantize=dequantized_result, apot_repr=APoTRepr.level_indices)

        self.assertTrue(torch.equal(dequantized_result, quantized_result))

    r""" Tests dequantize_apot result on random 3-dim tensor
        (int repr) and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_dequantize_quantize_rand_3d(self):
        # generate random size of tensor2dequantize
        size = random.randint(1, 64)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random values between 0 -> 2**6 = 64
        # because there are 2**b = 2**6 quantization levels total
        tensor2dequantize = 64 * torch.rand(size, size, size)
        apot_tens = TensorAPoT(6, 2, False)
        apot_tens.data = tensor2dequantize.int()
        apot_tens.apot_repr = APoTRepr.level_indices
        orig_input = torch.clone(apot_tens.data)

        dequantized_result = apot_tens.dequantize()

        quantized_result = apot_tens.quantize_APoT(tensor2quantize=dequantized_result, apot_repr=APoTRepr.level_indices)

        self.assertTrue(torch.equal(dequantized_result, quantized_result))

    r""" Tests dequantize_apot result on random 1-dim tensor
        (reduced precision fp repr) and hardcoded values for b, k.
        Dequant -> quant an input tensor and verify that
        result is equivalent to input
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_dequantize_quantize_fp_rand_1d(self):
        # generate random size of tensor2dequantize between 1 -> 16
        # because there are 2**b = 2**4 quantization levels total
        size = random.randint(1, 16)

        # initialize quantize APoT tensor to dequantize:
        # generate tensor with random fp values between 0 -> 1
        tensor2quantize = torch.rand(size)
        apot_tens = TensorAPoT(4, 2, False)
        apot_tens.apot_repr = APoTRepr.reduced_precision_fp
        orig_input = torch.clone(apot_tens.data)

        dequantized_result = apot_tens.dequantize()

        quantized_result = apot_tens.quantize_APoT(tensor2quantize=dequantized_result, apot_repr=APoTRepr.reduced_precision_fp)

        self.assertTrue(torch.equal(dequantized_result, quantized_result))

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
