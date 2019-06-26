from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.cuda
import torch.jit
import numpy as np
import unittest
from hypothesis import assume, given
from hypothesis import strategies as st
from common_utils import run_tests
from torch.quantization.fake_quantize import FakeQuantize


# Reference method for quantizing a tensor.
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    res = (torch.clamp(torch.round(X / scale) + zero_point, quant_min, quant_max) - zero_point) * scale
    res = res.reshape(X.shape)
    return res


# Reference method for the gradient of the quantizer.
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    Xq = torch.round(X / scale) + zero_point
    mask = (Xq >= quant_min) & (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res

NP_RANDOM_SEED = 19
tolerance = 1e-6

class TestFakeQuantizePerTensorAffine(unittest.TestCase):
    """Tests the forward path of the FakeQuantizePerTensorAffine op."""
    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_forward(self, device):
        np.random.seed(NP_RANDOM_SEED)

        def to_tensor(X):
            return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)
        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = to_tensor(np.random.rand(20, 20) * 125)
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max, 0, 0)
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    """Tests the backward method. Note that this runs the reference quantization
    and thus the errors might be originating there."""
    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_backward(self, device):
        np.random.seed(NP_RANDOM_SEED)

        def to_tensor(X):
            return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)
        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = to_tensor(np.random.rand(20, 20) * 125)
        X.requires_grad_(True)
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max, 0, 0)
        dY = X  # Fake gradient
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dY, X, scale, zero_point, quant_min, quant_max)
        Y_prime.backward(X)
        np.testing.assert_allclose(dX.detach().numpy(), X.grad.detach().numpy(), rtol=tolerance, atol=tolerance)

    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_numerical_consistency(self, device):
        '''
        Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        '''
        np.random.seed(NP_RANDOM_SEED)

        def to_tensor(X):
            return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)
        scale = 3
        zero_point = 2
        quant_min, quant_max = 0, 255
        X = to_tensor(np.random.rand(20, 20) * 125)
        Y = torch.dequantize(torch.quantize_linear(X, scale, zero_point, torch.qint8))
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min,
            quant_max, 0, 0)
        np.testing.assert_allclose(Y, Y_prime, rtol=tolerance, atol=tolerance)

    @given(device = st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']))
    def test_fq_module(self, device):
        np.random.seed(NP_RANDOM_SEED)

        def to_tensor(X):
            return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)

        quant_min, quant_max = 0, 255
        X = to_tensor(np.random.rand(20, 20) * 125)
        X.requires_grad_(True)
        qconfig = {
            'qscheme': torch.per_tensor_affine,
            'quant_min': quant_min,
            'quant_max': quant_max,
        }
        fq_module = FakeQuantize(qconfig)
        Y_prime = fq_module(X)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y.detach().numpy(), Y_prime.detach().numpy(), rtol=tolerance, atol=tolerance)
        Y_prime.backward(X.float())
        dY = X
        dX = _fake_quantize_per_tensor_affine_grad_reference(dY, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(dX.detach().numpy(), X.grad.detach().numpy(), rtol=tolerance, atol=tolerance)


if __name__ == '__main__':
    run_tests()
