import torch
import torch.cuda
import torch.jit
import numpy as np
from hypothesis import given
from hypothesis import strategies as st
import hypothesis_utils as hu
from hypothesis_utils import no_deadline
from common_utils import run_tests, TestCase
from torch.quantization import FakeQuantize
from torch.quantization import default_observer
import io
# Reference method for fake quantize
def _fake_quantize_per_tensor_affine_reference(X, scale, zero_point, quant_min, quant_max):
    res = (torch.clamp(torch.round(X * (1.0 / scale) + zero_point), quant_min, quant_max) - zero_point) * scale
    return res


# Reference method for the gradient of the fake quantize operator
def _fake_quantize_per_tensor_affine_grad_reference(dY, X, scale, zero_point, quant_min, quant_max):
    Xq = torch.round(X * (1.0 / scale) + zero_point)
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res

NP_RANDOM_SEED = 19
tolerance = 1e-6

class TestFakeQuantizePerTensorAffine(TestCase):
    # NOTE: Tests in this class are decorated with no_deadline
    # to prevent spurious failures due to cuda runtime initialization.

    def to_tensor(self, X, device):
        return torch.tensor(X).to(device=torch.device(device), dtype=torch.float32)

    @no_deadline
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_forward(self, device, X):
        r"""Tests the forward path of the FakeQuantizePerTensorAffine op.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = torch.tensor(X).to(dtype=torch.float, device=device)
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @no_deadline
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_backward(self, device, X):
        r"""Tests the backward method. Note that this runs the reference quantization
        and thus the errors might be originating there.
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = torch.tensor(X).to(dtype=torch.float, device=device)
        X.requires_grad_()
        Y = _fake_quantize_per_tensor_affine_reference(X.cpu(), scale, zero_point, quant_min, quant_max)
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        dout = torch.rand(X.shape, dtype=torch.float).to(device)
        dX = _fake_quantize_per_tensor_affine_grad_reference(
            dout, X, scale, zero_point, quant_min, quant_max)
        Y_prime.backward(dout)
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    @no_deadline
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_numerical_consistency(self, device, X):
        r"""Comparing numerical consistency between CPU quantize/dequantize op and the CPU fake quantize op
        """
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = torch.tensor(X).to(dtype=torch.float, device=device)
        # quantize_linear and dequantize are only implemented in CPU
        Y = torch.dequantize(torch.quantize_linear(X.cpu(), scale, zero_point, torch_type))
        Y_prime = torch.fake_quantize_per_tensor_affine(
            X, scale, zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y, Y_prime.cpu(), rtol=tolerance, atol=tolerance)

    @no_deadline
    @given(device=st.sampled_from(['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']),
           X=hu.tensor(shapes=hu.array_shapes(1, 5,),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_fq_module(self, device, X):
        np.random.seed(NP_RANDOM_SEED)
        X, (scale, zero_point, torch_type) = X
        quant_min = torch.iinfo(torch_type).min
        quant_max = torch.iinfo(torch_type).max

        X = torch.tensor(X).to(dtype=torch.float, device=device)
        X.requires_grad_()
        fq_module = FakeQuantize(default_observer(), quant_min, quant_max)
        Y_prime = fq_module(X)
        assert fq_module.scale is not None
        assert fq_module.zero_point is not None
        Y = _fake_quantize_per_tensor_affine_reference(X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(Y.cpu().detach().numpy(), Y_prime.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

        # Test backward
        dout = torch.rand(X.shape, dtype=torch.float, device=device)
        Y_prime.backward(dout)
        dX = _fake_quantize_per_tensor_affine_grad_reference(dout, X, fq_module.scale, fq_module.zero_point, quant_min, quant_max)
        np.testing.assert_allclose(dX.cpu(), X.grad.cpu().detach().numpy(), rtol=tolerance, atol=tolerance)

    def test_fq_serializable(self):
        observer = default_observer()
        quant_min = 0
        quant_max = 255
        fq_module = FakeQuantize(observer, quant_min, quant_max)
        X = torch.tensor([-5, -3.5, -2, 0, 3, 5, 7], dtype=torch.float32)
        y_ref = fq_module(X)
        state_dict = fq_module.state_dict()

        self.assertEqual(state_dict['fake_quant_enabled'], True)
        self.assertEqual(state_dict['observer_enabled'], True)
        self.assertEqual(state_dict['scale'], 0.094488)
        self.assertEqual(state_dict['zero_point'], 53)
        b = io.BytesIO()
        torch.save(state_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in state_dict:
            self.assertEqual(state_dict[key], loaded_dict[key])

if __name__ == '__main__':
    run_tests()
