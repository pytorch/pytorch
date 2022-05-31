# Owner(s): ["oncall: quantization"]

from torch.ao.quantization.experimental.observer import APoTObserver
import unittest

class TestNonUniformObserver(unittest.TestCase):
    def test_calculate_qparams(self):
        raised = False
        try:
            APoTObserver.calculate_qparams()
        except Exception:
            raised = True
        self.assertFalse(raised, 'Exception raised')

    def test_override_calculate_qparams(self):
        with self.assertRaises(NotImplementedError):
            APoTObserver._calculate_qparams()

if __name__ == '__main__':
    unittest.main()
