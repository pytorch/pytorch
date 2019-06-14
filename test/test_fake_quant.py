from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.cuda
import torch.jit
import numpy as np
import unittest
from hypothesis import assume, given
from hypothesis import strategies as st
from common_utils import run_tests
import torch.nn._intrinsic as _intrinsic


# Reference method for quantizing a tensor.
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    res = (np.clip(np.round(X / scale) + zero_point, quant_min, quant_max) - zero_point) * scale
    res = res.reshape(X.shape)
    return res


# Reference method for the gradient of the quantizer.
def _fake_quantize_per_tensor_affine_grad_reference(X, dY, scale, zero_point, quant_min, quant_max):
    Xq = np.round(X / scale) + zero_point
    mask = np.logical_and(Xq >= quant_min, Xq <= quant_max)
    res = np.zeros_like(dY)
    res[mask] = dY[mask]
    return res

NP_RANDOM_SEED = 19


class TestFakeQuantizePerTensorAffine(unittest.TestCase):
    """Tests the forward path of the FakeQuantizePerTensorAffine op."""
    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_forward(self, device):
        np.random.seed(NP_RANDOM_SEED)

        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).float()
        if device == 'cuda':
            X_torch = X_torch.cuda()
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max)
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch, scale=scale, zero_point=zero_point, quant_min=quant_min,
            quant_max=quant_max, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_backward(self, device):
        np.random.seed(NP_RANDOM_SEED)
        fake_quantize_per_tensor_affine_backward = torch.ops.quantized.fake_quantize_per_tensor_affine_backward

        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = np.random.rand(20, 20) * 125
        Y = _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max)
        dY = Y - X  # Fake gradient
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            X, dY, scale, zero_point, quant_min, quant_max)
        X_torch = torch.from_numpy(X).to(device=torch.device(device), dtype=torch.float32)
        dY_torch = torch.from_numpy(dY).to(device=torch.device(device), dtype=torch.float32)
        dX_prime = _intrinsic.fq_per_tensor_affine_backward(
            X=X_torch, dY=dY_torch, scale=scale, zero_point=zero_point,
            quant_min=quant_min, quant_max=quant_max, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(dX, dX_prime, rtol=tolerance, atol=tolerance)

    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_numerical_consistency(self, device):
        '''
        Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        '''
        np.random.seed(NP_RANDOM_SEED)
        fake_quantize_per_tensor_affine_forward = torch.ops.quantized.fake_quantize_per_tensor_affine_forward

        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).to(device=torch.device(device), dtype=torch.float32)
        Y = torch.dequantize(torch.quantize_linear(X_torch, scale, zero_point, torch.qint8))
        Y_prime = _intrinsic.fq_per_tensor_affine_forward(
            X=X_torch, scale=scale, zero_point=zero_point, quant_min=quant_min,
            quant_max=quant_max, quant_delay=0, iter=0)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_fq_module(self, device):
        np.random.seed(NP_RANDOM_SEED)

        quant_min, quant_max = 0, 255
        X = np.random.rand(20, 20) * 125
        X_torch = torch.from_numpy(X).to(device=torch.device(device), dtype=torch.float32)
        X_torch.requires_grad_(True)
        qconfig = {
            'qscheme': None, # TODO: change after qscheme diff is landed
            'qmin': quant_min,
            'qmax': quant_max,
        }
        fq_module = _intrinsic.FakeQuantize(qconfig)
        print('before calling fq_module, X_torch:', X_torch.requires_grad)
        Y_prime = fq_module(X_torch)
        print("after calling fq_modeule, X_troch, Y_prime:", X_torch.requires_grad, Y_prime.requires_grad)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_tensor_affine_reference(X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        tolerance = 1e-6
        np.testing.assert_allclose(Y, Y_prime.detach().numpy(), rtol=tolerance, atol=tolerance)
        Y_prime.backward(torch.from_numpy(X).float())
        dY = X
        dX = _fake_quantize_per_tensor_affine_grad_reference(X, dY, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        print(type(X_torch.grad))
        print(dX)
        print(X_torch.grad)
        np.testing.assert_allclose(dX, X_torch.grad, rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    run_tests()
