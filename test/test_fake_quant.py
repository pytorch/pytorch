from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.cuda
import torch.jit
import numpy as np
import unittest
from common_utils import run_tests
import torch.nn._intrinsic as _intrinsic


# Reference method for quantizing a tensor.
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, num_bits):
    quant_min, quant_max = 0, 2 ** num_bits - 1
    res = (np.clip(np.round(X / scale) + zero_point, quant_min, quant_max) - zero_point) * scale
    res = res.reshape(X.shape)
    return res


# Reference method for the gradient of the quantizer.
def _fake_quantize_per_tensor_affine_grad_reference(X, dY, scale, zero_point, num_bits):
    Xq = np.round(X / scale) + zero_point
    quant_min, quant_max = 0, 2 ** num_bits - 1
    mask = np.logical_and(Xq >= quant_min, Xq <= quant_max)
    res = dY[mask].reshape(dY.shape)
    return res

NP_RANDOM_SEED = 19


class TestFakeQuantizePerTensorAffine(unittest.TestCase):
    """Tests the forward path of the FakeQuantizePerTensorAffine op."""
    def test_forward(self):
        np.random.seed(NP_RANDOM_SEED)

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).float()
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, num_bits)
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch, scale=scale, zero_point=zero_point, num_bits=num_bits,
            quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    def test_backward(self):
        np.random.seed(NP_RANDOM_SEED)
        fake_quantize_per_tensor_affine_backward = torch.ops.quantized.fake_quantize_per_tensor_affine_backward

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, num_bits)
        dY = Y - X  # Fake gradient
        dX = _fake_quantize_per_tensor_affine_grad_reference(X, dY, scale, zero_point, num_bits)
        X_torch = torch.from_numpy(X).float()
        dY_torch = torch.from_numpy(dY).float()
        dX_prime = _intrinsic.fq_per_tensor_affine_backward(
            X=X_torch, dY=dY_torch, scale=scale, zero_point=zero_point,
            num_bits=num_bits, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(dX, dX_prime, rtol=tolerance, atol=tolerance)

    def test_numerical_consistency(self):
        '''
        Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        '''
        np.random.seed(NP_RANDOM_SEED)
        fake_quantize_per_tensor_affine_forward = torch.ops.quantized.fake_quantize_per_tensor_affine_forward

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).float()
        Y = torch.dequantize(torch.quantize_linear(X_torch, scale, zero_point, torch.qint8))
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch, scale=scale, zero_point=zero_point, num_bits=num_bits,
            quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the forward path of the FakeQuantizePerTensorAffine CUDA op."""
    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_forward_cuda(self):
        np.random.seed(NP_RANDOM_SEED)
        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).float().cuda()
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, num_bits)
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch, scale=scale, zero_point=zero_point, num_bits=num_bits,
            quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_backward_cuda(self):
        np.random.seed(NP_RANDOM_SEED)

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, num_bits)
        dY = Y - X  # Fake gradient
        dX = _fake_quantize_per_tensor_affine_grad_reference(X, dY, scale, zero_point, num_bits)
        X_torch = torch.from_numpy(X).float().cuda()
        dY_torch = torch.from_numpy(dY).float().cuda()
        dX_prime = _intrinsic.fq_per_tensor_affine_backward(
            X=X_torch, dY=dY_torch, scale=scale, zero_point=zero_point,
            num_bits=num_bits, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(dX, dX_prime.cpu(), rtol=tolerance, atol=tolerance)

    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_numerical_consistency_cuda(self):
        '''
        Comparing numerical consistency between CPU quantize/dequantize op and the CUDA fake quantize op
        '''
        np.random.seed(NP_RANDOM_SEED)

        scale = 3
        zero_point = 2
        num_bits = 8
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).float()
        Y = torch.dequantize(torch.quantize_linear(X_torch, scale, zero_point, torch.qint8))
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch.cuda(), scale=scale, zero_point=zero_point, num_bits=num_bits,
            quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

if __name__ == '__main__':
    run_tests()
