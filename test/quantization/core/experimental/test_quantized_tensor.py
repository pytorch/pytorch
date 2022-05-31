# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.APoT_tensor import TensorAPoT
import unittest

class TestQuantizedTensor(unittest.TestCase):
    def test_quantize_APoT(self):
        t = torch.Tensor()
        with self.assertRaises(NotImplementedError):
            TensorAPoT.quantize_APoT(t)

    def test_dequantize(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.dequantize(self)

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            TensorAPoT.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
