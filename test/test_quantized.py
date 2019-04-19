from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.jit
import numpy as np
import unittest
from common_utils import TestCase, run_tests, skipIfNotRegistered

def canonical(graph):
    return str(torch._C._jit_pass_canonicalize(graph))


# Reference method for quantizing a tensor.
def _fake_quantize_reference(X, scale, zero_point, num_bits):
    quant_min, quant_max = 0, 2 ** num_bits - 1
    res = (np.clip(np.round(X / scale) + zero_point, quant_min, quant_max) - zero_point) * scale
    res = res.reshape(X.shape)
    return res


# Reference method for the gradient of the quantizer.
def _fake_quantize_grad_reference(X, dY, scale, zero_point, num_bits):
    Xq = np.round(X / scale) + zero_point
    quant_min, quant_max = 0, 2 ** num_bits - 1
    mask = np.logical_and(Xq >= quant_min, Xq <= quant_max)
    res = dY[mask].reshape(dY.shape)
    return res


@skipIfNotRegistered("Relu_ENGINE_DNNLOWP",
                     "fbgemm-based Caffe2 ops are not linked")
class TestQuantized(TestCase):
    def test_relu(self):
        a = (torch.tensor([4, 6, 1, 10], dtype=torch.uint8), 0.01, 5)
        r = torch.ops.c10.quantized_relu(a)
        np.testing.assert_equal(r[0].numpy(), torch.tensor([5, 6, 5, 10], dtype=torch.uint8).numpy())
        np.testing.assert_almost_equal(0.01, r[1])
        self.assertEqual(5, r[2])

    def test_quantize(self):
        a = (torch.tensor([4, 6, 1, 10], dtype=torch.uint8), 0.01, 5)
        r = torch.ops.c10.dequantize(a)
        np.testing.assert_almost_equal(r.numpy(), [-0.01, 0.01, -0.04, 0.05])
        # default args
        q_def = torch.ops.c10.quantize(r)
        # specified
        q = torch.ops.c10.quantize(r, scale=0.01, zero_point=5)
        np.testing.assert_equal(q[0].numpy(), a[0].numpy())
        np.testing.assert_almost_equal(q[1], a[1])
        self.assertEqual(q[2], a[2])

    def test_script(self):
        @torch.jit.script
        def foo(x):
            # type: (Tuple[Tensor, float, int]) -> Tuple[Tensor, float, int]
            return torch.ops.c10.quantized_relu(x)
        self.assertExpectedInline(canonical(foo.graph), '''\
graph(%x : (Tensor, float, int)):
  %1 : (Tensor, float, int) = c10::quantized_relu(%x)
  return (%1)
''')


class TestQuantizedRelu(unittest.TestCase):
    """Tests the correctness of the quantized::relu op."""
    def test_qrelu(self):
        relu = torch.ops.quantized.relu

        X_tensor = np.arange(0, 10, dtype=np.uint8)
        scale = 255.0
        zero_point = 5

        Y_tensor = X_tensor.copy()
        Y_tensor[X_tensor < zero_point] = zero_point

        X = (torch.from_numpy(X_tensor), scale, zero_point)

        Y_hat = relu(*X)
        Y_hat_tensor = Y_hat[0].numpy()

        np.testing.assert_equal(Y_tensor, Y_hat_tensor)


class TestFakeQuantize(unittest.TestCase):
    """Tests the forward path of the fake quantization minMax args op."""
    def test_forward(self):
        fake_quantize_forward = torch.ops.quantized.fake_quantize_forward

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        Y = _fake_quantize_reference(X, scale, zero_point, num_bits)
        X_torch = torch.from_numpy(X)
        Y_prime = fake_quantize_forward(
            X=X_torch, scale=scale, zero_point=zero_point, num_bits=num_bits,
            quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    def test_backward(self):
        fake_quantize_backward = torch.ops.quantized.fake_quantize_backward

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        Y = _fake_quantize_reference(X, scale, zero_point, num_bits)
        dY = Y - X  # Fake gradient
        dX = _fake_quantize_grad_reference(X, dY, scale, zero_point, num_bits)
        X_torch = torch.from_numpy(X)
        dY_torch = torch.from_numpy(dY)
        dX_prime = fake_quantize_backward(
            X=X_torch, dY=dY_torch, scale=scale, zero_point=zero_point,
            num_bits=num_bits, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(dX, dX_prime, rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    run_tests()
