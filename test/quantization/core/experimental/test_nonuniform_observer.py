# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest
import torch

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        """
        cases to test:
        1. works for empty observer
        2. works for normal observer (vasiliy playground example)
        3. alpha is gotten correctly -> test after checking w charlie about computation implementation
        4. works for normal observer of larger size
        """

        obs1 = APoTObserver()

        """
        vasiliy playground example
        Assume hardcoded parameters:
        * b = 4 (total number of bits across all terms)
        * k = 2 (base bitwidth, i.e. bitwidth of every term)
        * n = 2 (number of additive terms)
        * note: b = k * n
        """
        obs2 = APoTObserver(b = 2, k = 2)

        ObserverList = [obs1, obs2]

        # t = torch.Tensor()
        # obs = APoTObserver(t, t, t, 0, 0)

        # with self.assertRaises(NotImplementedError):
        #     obs.calculate_qparams()

        # t = torch.Tensor()
        # obs = APoTObserver(t, t, t, 0, 0)

        # raised = False
        # try:
        #     obs.calculate_qparams()
        # except Exception as e:
        #     raised = True
        #     print(e)
        # self.assertFalse(raised, 'Exception raised')

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
