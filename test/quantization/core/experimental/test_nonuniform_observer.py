# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        t = torch.Tensor()
        obs = APoTObserver(t, t, t, 0, 0)

        with self.assertRaises(AssertionError):
            obs1_result = obs1.calculate_qparams(signed=False)

    def test_override_calculate_qparams(self):
        t = torch.Tensor()
        obs = APoTObserver(t, t, t, 0, 0)

        with self.assertRaises(NotImplementedError):
            obs._calculate_qparams()

if __name__ == '__main__':
    unittest.main()
