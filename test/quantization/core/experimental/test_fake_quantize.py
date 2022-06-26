# Owner(s): ["oncall: quantization"]

import torch
import unittest
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize
from torch.ao.quantization.experimental.apot_utils import float_to_reduced_precision

class TestFakeQuantize(unittest.TestCase):
    r""" Tests fake quantize calculate_qparams() method
         by comparing with result from observer calculate_qparams.
         Uses hard-coded values: alpha=1.0, b=4, k=2.
    """
    def test_fake_calc_qparams(self):
        observer = APoTObserver(b=4, k=2)

        apot_fake = APoTFakeQuantize(observer)

        min_val = torch.tensor([0.0])
        max_val = torch.tensor([1.0])

        alpha, gamma, quantization_levels, level_indices = apot_fake.calculate_qparams(signed=False,
                                                                                       min_val=min_val,
                                                                                       max_val=max_val)

        qparams_expected = observer.calculate_qparams(signed=False, min_val=min_val, max_val=max_val)

        self.assertEqual(qparams[0], qparams_expected[0])
        self.assertTrue(torch.equal(qparams[1], qparams_expected[1]))
        self.assertTrue(torch.equal(qparams[2], qparams_expected[2]))

    r""" Tests fake quantize forward() method
         by comparing result with expected
         float_to_reduced_precision mapping of input tensor.
         Uses input tensor with random values from 0 -> 1000
         and APoT observer with hard-coded values b=4, k=2
    """
    def test_forward(self):
        # generate a tensor of size 20 with random values
        # between 0 -> 1000 to quantize -> dequantize
        X = 1000 * torch.rand(20)

        min_val, max_val = torch.aminmax(X)

        observer = APoTObserver(b=4, k=2)
        alpha, gamma, quantization_levels, level_indices = observer.calculate_qparams(signed=False,
                                                                                      min_val=min_val,
                                                                                      max_val=max_val)
        quantization_levels = qparams[1]
        level_indices = qparams[2]

        apot_fake = APoTFakeQuantize(observer)

        X_reduced_precision_fp = apot_fake.forward(torch.clone(X), False)

        X_expected = X.apply_(lambda x: float_to_reduced_precision(x, quantization_levels, level_indices))

        self.assertTrue(torch.equal(X_reduced_precision_fp, X_expected))

if __name__ == '__main__':
    unittest.main()
