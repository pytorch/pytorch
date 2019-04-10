#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
import unittest


class TestQuantizedRelu(unittest.TestCase):
    """Tests the correctness of the quantized::relu op."""
    def test_passthrough_qrelu(self):
        relu = torch.ops.quantized.relu

        # If using `torch.ops.c10.quantized_relu`, dtype should be uint8.
        X_tensor = np.arange(0, 10, dtype=np.uint8)
        scale = 255.0
        zero_point = 5

        Y_tensor = X_tensor.copy()
        Y_tensor[X_tensor < zero_point] = zero_point

        X = (torch.from_numpy(X_tensor), scale, zero_point)

        Y_hat = relu(*X)
        Y_hat_tensor = Y_hat[0].numpy()

        np.testing.assert_equal(Y_tensor, Y_hat_tensor)


if __name__ == "__main__":
    unittest.main()
