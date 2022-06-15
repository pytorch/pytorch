# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
import unittest
quantize_APoT = TensorAPoT.quantize_APoT

class TestQuantizedTensor(unittest.TestCase):
    def test_quantize_APoT(self):
        t = torch.Tensor()
        with self.assertRaises(NotImplementedError):
            TensorAPoT.quantize_APoT(t)

    def test_dequantize(self):
        t = TensorAPoT()

        input = torch.tensor([0, 1])

        input_quantized = quantize_APoT(input, 4, 2)
        output = t.dequantize(input_quantized, 4, 2)

        self.assertEqual(input, output)

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
