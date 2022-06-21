# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.quantizer import APoTQuantizer
import unittest
import random
quantize_APoT = TensorAPoT.quantize_APoT
dequantize = TensorAPoT.dequantize

class TestQuantizedTensor(unittest.TestCase):
    def test_int_repr(self):
        quantizer = APoTQuantizer()
        with self.assertRaises(NotImplementedError):
            quantizer.int_repr()

if __name__ == '__main__':
    unittest.main()
