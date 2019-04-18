from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.jit
import numpy as np
import unittest
# from caffe2.python import core
from common_utils import TestCase, run_tests


def canonical(graph):
    return str(torch._C._jit_pass_canonicalize(graph))


# Equivalent of the `nudge` method in the `::fake_quantize_ops.cpp`.
def _nudge(min, max, num_bits, narrow_range):
    quant_min = float(narrow_range)
    quant_max = float(1 << num_bits) - 1.0
    scale = (max - min) / (quant_max - quant_min)

    zero_point_from_min = np.round(quant_min - min / scale)
    nudged_zero_point = np.int(
        np.fmin(
            np.fmax(zero_point_from_min, quant_min),
            quant_max))
    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale

    return (nudged_min, nudged_max, scale)


# Reference method for quantizing a tensor.
def _quantize_min_max_args_reference(X, min, max, num_bits, narrow_range):
    nudge_min, nudge_max, scale = _nudge(min, max, num_bits, narrow_range)
    res = (np.round((np.clip(X, nudge_min, nudge_max) - nudge_min) / scale)
           * scale + nudge_min)
    res = res.reshape(X.shape)
    return res


# Reference method for the gradient of the quantizer.
def _quantize_min_max_args_grad_reference(X, dY, min, max, num_bits, narrow_range):
    nudge_min, nudge_max, scale = _nudge(min, max, num_bits, narrow_range)
    mask = np.logical_and(X >= nudge_min, X <= nudge_max)
    res = dY[mask].reshape(dY.shape)
    return res


# Adjusts the input if too close to min or max.
def _adjust_input_tensor(X, min, max, epsilon=1e-3):
    def adjust(x):
        if abs(x - min) < epsilon:
            return min
        if abs(x - max) < epsilon:
            return max
        return x

    return np.array(
        [adjust(x) for x in X.flatten()],
        dtype=np.float32
    ).reshape(X.shape)

@unittest.skip("Skipping due to the protobuf dependency in the CI's")
# @unittest.skipIf("Relu_ENGINE_DNNLOWP" not in core._REGISTERED_OPERATORS, "fbgemm-based Caffe2 ops are not linked")
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


class TestFakeQuantizeMinMax(unittest.TestCase):
    """Tests the forward path of the fake quantization minMax args op."""
    def test_forward(self):
        fake_quantize_minmax_args_forward = torch.ops.quantized.fake_quantize_minmax_forward

        min = -10
        max = 10
        num_bits = 8
        narrow_range = False
        X = np.random.rand(20, 20) * (max - min) + min
        X = _adjust_input_tensor(X, min, max)
        Y = _quantize_min_max_args_reference(X, min, max, num_bits,
                                             narrow_range)
        X_torch = torch.from_numpy(X)
        Y_prime = fake_quantize_minmax_args_forward(
            X=X_torch, min=min, max=max, num_bits=num_bits,
            quant_delay=0, iter=0, narrow_range=narrow_range)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    def test_backward(self):
        fake_quantize_minmax_args_backward = torch.ops.quantized.fake_quantize_minmax_backward

        min = -10
        max = 10
        num_bits = 8
        narrow_range = False
        X = np.random.rand(2, 2) * (max - min) + min
        X = _adjust_input_tensor(X, min, max)
        Y = _quantize_min_max_args_reference(X, min, max, num_bits,
                                             narrow_range)
        dY = Y - X  # Fake gradient
        dX = _quantize_min_max_args_grad_reference(X, dY, min, max, num_bits,
                                                   narrow_range)
        X_torch = torch.from_numpy(X)
        dY_torch = torch.from_numpy(dY)
        dX_prime = fake_quantize_minmax_args_backward(
            X=X_torch, dY=dY_torch, min=min, max=max,
            num_bits=num_bits, quant_delay=0, iter=0, narrow_range=narrow_range)
        tolerance = 1e-6
        np.testing.assert_allclose(dX, dX_prime, rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    run_tests()
