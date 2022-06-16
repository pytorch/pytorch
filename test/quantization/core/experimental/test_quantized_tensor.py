# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
import unittest
import random
quantize_APoT = TensorAPoT.quantize_APoT
dequantize = TensorAPoT.dequantize

class TestQuantizedTensor(unittest.TestCase):
    """ Tests quantize_apot result on simple 1-dim tensor
        and hardcoded values for b and k
        * tensor2quantize: torch.tensor([0, 1])
        * b: 4
        * k: 2
    """
    def test_dequantize_quantize_APoT(self):
        # generate random size of tensor2dequantize
        size = random.randint(1, 16)

        # generate tensor with random ints between 0 -> 16
        t = 16 * torch.rand(size)

        t = t.int()

        orig_input = t.clone()

        print(orig_input)

        dequant = dequantize(t, 4, 2)

        print(dequant)

        quant_result = quantize_APoT(dequant, 4, 2)

        print(quant_result)

    """ Tests quantize_apot result on simple 1-dim tensor
        and hardcoded values for b and k
        * tensor2quantize: torch.tensor([0, 1])
        * b: 4
        * k: 2
    """
    # NOTE: bound the error of resulting fp tensor (maybe with max diff between quant levels??)
    """
    def test_quantize_dequantize_APoT(self):
        # generate tensor with random floats between 0 -> 1
        t = torch.tensor([0.7206, 0.7905, 0.5051, 0.9202, 0.6865, 0.7989, 0.9824, 0.2253, 0.5745,
        0.8093, 0.3702, 0.0543, 0.4141, 0.1152, 0.3168])

        orig_input = t.clone()

        print(orig_input)

        quant = quantize_APoT(t, 4, 2)

        print(quant)

        dequant_result = dequantize(quant, 4, 2)

        print(dequant_result)
    """

        # # generate random size of tensor2dequantize
        # size = random.randint(1, 16)

        # # generate tensor with random floats between 0 -> 1
        # t = torch.rand(size, dtype=float)

        # orig_input = t.clone()

        # print(orig_input)

        # quant = quantize_APoT(t, 4, 2)

        # print(quant)

        # dequant_result = dequantize(quant, 4, 2)

        # print(dequant_result)

    """ Tests quantize_apot result on random 1-dim tensor
        and hardcoded values for b, k
        * tensor2quantize: Tensor
        * b: 4
        * k: 2
    """
    def test_dequantize_APoT_ramd_1d(self):
        # generate random size of tensor2dequantize
        size = random.randint(1, 16)

        # generate tensor with random values between 0 -> 2**4 = 16
        # because there are 2**b = 2**4 quantization levels total
        tensor2dequantize = 16 * torch.rand(size)

        tensor2dequantize = tensor2dequantize.int()

        orig_input = tensor2dequantize.clone()

        max_val = torch.max(orig_input)

        tensor2dequantize = dequantize(tensor2dequantize, 4, 2)

        # make observer
        obs = APoTObserver(max_val=max_val, b=4, k=2)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        input_arr = list(orig_input)
        result_arr = list(tensor2dequantize)

        zipped_result = zip(input_arr, result_arr)

        expected_result = True

        for ele, res in zipped_result:
            idx = list(level_indices).index(ele)
            if res != quantized_levels[idx]:
                expected_result = False

        self.assertTrue(expected_result)

    """ Tests quantize_apot result on random 2-dim tensor
        and hardcoded values for b, k
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_dequantize_APoT_ramd_2d(self):
        # generate random size of tensor2dequantize
        size1 = random.randint(1, 64)
        size2 = random.randint(1, 64)

        # generate tensor with random values between 0 -> 2**6 = 64
        # because there are 2**b = 2**6 quantization levels total
        tensor2dequantize = 64 * torch.rand(size1, size2)

        tensor2dequantize = tensor2dequantize.int()

        orig_input = tensor2dequantize.clone()

        max_val = torch.max(orig_input)

        tensor2dequantize = dequantize(tensor2dequantize, 6, 2)

        # make observer
        obs = APoTObserver(max_val=max_val, b=6, k=2)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        input_arr = list(orig_input.flatten())
        result_arr = list(tensor2dequantize.flatten())

        zipped_result = zip(input_arr, result_arr)

        expected_result = True

        for ele, res in zipped_result:
            idx = list(level_indices).index(ele)
            if res != quantized_levels[idx]:
                expected_result = False

        self.assertTrue(expected_result)

    """ Tests quantize_apot result on random 3-dim tensor
        and hardcoded values for b, k
        * tensor2quantize: Tensor
        * b: 6
        * k: 2
    """
    def test_dequantize_APoT_ramd_3d(self):
        # generate random size of tensor2dequantize
        size1 = random.randint(1, 64)
        size2 = random.randint(1, 64)
        size3 = random.randint(1, 64)

        # generate tensor with random values between 0 -> 2**6 = 64
        # because there are 2**b = 2**6 quantization levels total
        tensor2dequantize = 64 * torch.rand(size1, size2, size3)

        tensor2dequantize = tensor2dequantize.int()

        orig_input = tensor2dequantize.clone()

        max_val = torch.max(orig_input)

        tensor2dequantize = dequantize(tensor2dequantize, 6, 2)

        # make observer
        obs = APoTObserver(max_val=max_val, b=6, k=2)
        obs_result = obs.calculate_qparams(signed=False)

        quantized_levels = obs_result[1]
        level_indices = obs_result[2]

        input_arr = list(orig_input.flatten())
        result_arr = list(tensor2dequantize.flatten())

        zipped_result = zip(input_arr, result_arr)

        expected_result = True

        for ele, res in zipped_result:
            idx = list(level_indices).index(ele)
            if res != quantized_levels[idx]:
                expected_result = False

        self.assertTrue(expected_result)

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
