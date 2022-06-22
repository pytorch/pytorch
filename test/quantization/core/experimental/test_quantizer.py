# Owner(s): ["oncall: quantization"]

import torch
from torch.ao.quantization.experimental.quantizer import APoTQuantizer
import unittest

class TestQuantizer(unittest.TestCase):
    def test_quantize_APoT(self):
        t = torch.Tensor()
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.quantize_APoT(t)

    def test_dequantize(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.dequantize(self)

    def test_q_apot_alpha(self):
        with self.assertRaises(NotImplementedError):
            APoTQuantizer.q_apot_alpha(self)

if __name__ == '__main__':
    unittest.main()
