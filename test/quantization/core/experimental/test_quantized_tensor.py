# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.quantizer import APoTQuantizer
import unittest

class TestQuantizedTensor(unittest.TestCase):
    def test_int_repr(self):
        quantizer = APoTQuantizer()
        with self.assertRaises(NotImplementedError):
            quantizer.int_repr()

if __name__ == '__main__':
    unittest.main()
