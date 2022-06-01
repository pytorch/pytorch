# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        # t = torch.Tensor()
        # obs = APoTObserver(t, t, t, 0, 0)

        # with self.assertRaises(NotImplementedError):
        #     obs.calculate_qparams()
        t = torch.Tensor()
        obs = APoTObserver(t, t, t, 0, 0)

        raised = False
        try:
            obs.calculate_qparams()
        except Exception as e:
            raised = True
            print(e)
        self.assertFalse(raised, 'Exception raised')

    def test_override_calculate_qparams(self):
        t = torch.Tensor()
        obs = APoTObserver(t, t, t, 0, 0)

        raised = False
        try:
            obs._calculate_qparams(t, t)
        except Exception:
            raised = True
        self.assertFalse(raised, 'Exception raised')

if __name__ == '__main__':
    unittest.main()
