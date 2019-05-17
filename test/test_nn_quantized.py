import torch
import torch.nn.quantized.functional as F

import numpy as np
from common_utils import TestCase, run_tests

def _quantize(x, scale, zero_point, qmin=0, qmax=255):
    """Quantizes a numpy array."""
    qx = np.round(x / scale + zero_point)
    qx = np.clip(qx, qmin, qmax).astype(np.uint8)
    return qx

class FunctionalAPITest(TestCase):
    def test_functional_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        Y = X.numpy().copy()
        Y[Y < 0] = 0
        qY = _quantize(Y, scale, zero_point)
        qX = X.quantize_linear(scale=scale, zero_point=zero_point, dtype=torch.qint8)
        qY_hat = F.relu(qX)
        np.testing.assert_equal(qY, qY_hat.int_repr())

if __name__ == '__main__':
    run_tests()
