from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized as nnq
import torch.nn.quantized.functional as qF
from torch.nn.quantized.modules import Conv2d
from common_utils import TestCase, run_tests, tempfile
from hypothesis import given
from hypothesis import strategies as st


'''
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `caffe2/test/test_quantized.py`.
'''


class FunctionalAPITest(TestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.ops.quantized.relu(qX)
        qY_hat = qF.relu(qX)
        self.assertEqual(qY, qY_hat)


class ModuleAPITest(TestCase):
    def test_dynamic_linear_api(self):
        """test API functionality for nn.quantized.DynamicLinear"""
        batch_size = 1
        in_features = 2
        out_features = 2
        W = torch.rand(out_features, in_features).float()

        # W_int8, col_offsets, scale, zero_point = torch.fbgemm_linear_quantize_weight(W)
        # W_pack = torch.fbgemm_pack_quantized_matrix(W_int8.clone())

        # max_min_ref
        qmin, qmax = -128, 127
        n_levels = 255.0
        min_val = torch.min(W).item()
        max_val = torch.max(W).item()
        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / n_levels
            scale = max(scale, torch.finfo(torch.float32).eps)
            zero_point = qmin - round(min_val / scale)
            zero_point = max(qmin, zero_point)
            zero_point = min(qmax, zero_point)
        # scale = 
        # zero_point = 
        W_q = torch.quantize_linear(W, scale, zero_point, torch.qint8)
        W_pack_col_offset = torch.ops.quantized.fbgemm_linear_prepack(W_q)

        X = torch.rand(batch_size, in_features).float()
        B = torch.rand(out_features).float()
        qlinear = nnq.DynamicLinear(in_features, out_features)
        qlinear._packed_weight = W_pack_col_offset
        qlinear.bias = B
        # qlinear.col_offsets = col_offsets
        # qlinear.scale = scale
        # qlinear.zero_point = zero_point
        Z_dq = qlinear(X)

        # Z_ref = torch.fbgemm_linear_int8_weight_fp32_activation(
        #     X, W, W_pack, col_offsets,
        #     scale, zero_point, B)
        Z_ref = torch.ops.quantized.fbgemm_linear_dynamic(X, W_pack_col_offset, B)

        self.assertEqual(Z_ref, Z_dq)

    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
    )
    def test_linear_api(self, batch_size, in_features, out_features, use_bias):
        """test API functionality for nn.quantized.linear"""
        W = torch.rand(out_features, in_features).float()
        W_q = torch.quantize_linear(W, 0.1, 4, torch.qint8)
        W_pack = torch.ops.quantized.fbgemm_linear_prepack(W_q)
        X = torch.rand(batch_size, in_features).float()
        X_q = torch.quantize_linear(X, 0.2, 10, torch.quint8)
        B = torch.rand(out_features).float() if use_bias else None
        B_q = torch.quantize_linear(B, W_q.q_scale() * X_q.q_scale(), 0, torch.qint32) if use_bias else None
        out_scale = 0.5
        out_zero_point = 3
        qlinear = nnq.Linear(in_features, out_features)
        qlinear._packed_weight = W_pack
        qlinear.bias = B_q if use_bias else None
        qlinear.out_scale = torch.tensor([out_scale])
        qlinear.out_zero_point = torch.tensor([out_zero_point])
        Z_q = qlinear(X_q)
        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.fbgemm_linear(X_q, W_pack, B_q, out_scale, out_zero_point)
        self.assertEqual(Z_ref, Z_q)

        # Test serialization of quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        if use_bias:
            self.assertEqual(model_dict['bias'], B_q)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model_dict, f)
            f.seek(0)
            loaded_dict = torch.load(f)
        for key in model_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qlinear = nnq.Linear(in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)

        linear_unpack = torch.ops.quantized.fbgemm_linear_unpack
        self.assertEqual(linear_unpack(qlinear._packed_weight),
                         linear_unpack(loaded_qlinear._packed_weight))
        if use_bias:
            self.assertEqual(qlinear.bias, loaded_qlinear.bias)
        self.assertEqual(qlinear.out_scale, loaded_qlinear.out_scale)
        self.assertEqual(qlinear.out_zero_point, loaded_qlinear.out_zero_point)
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertTrue(hasattr(qlinear, '_packed_weight'))
        self.assertTrue(hasattr(loaded_qlinear, '_packed_weight'))
        self.assertTrue(hasattr(qlinear, 'weight'))
        self.assertTrue(hasattr(loaded_qlinear, 'weight'))
        self.assertEqual(qlinear.weight, loaded_qlinear.weight)
        self.assertEqual(qlinear.weight, torch.ops.quantized.fbgemm_linear_unpack(qlinear._packed_weight))
        Z_q2 = qlinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        # test serialization of module directly - will add this later
        # with tempfile.NamedTemporaryFile() as f:
        #     torch.save(qLinear, f)
        #     f.seek(0)
        #     loaded = torch.load(f)
        # state = qLinear.__getstate__()
        # compareUnpackedWeight(qLinear._packed_weight, loaded._packed_weight)
        # self.assertEqual(qLinear.bias, loaded.bias)
        # self.assertEqual(qLinear.out_scale, loaded.out_scale)
        # self.assertEqual(qLinear.out_zero_point, loaded.out_zero_point)

    def test_quant_dequant_api(self):
        r = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float)
        scale, zero_point, dtype = 1.0, 2, torch.qint8
        # testing Quantize API
        qr = torch.quantize_linear(r, scale, zero_point, dtype)
        quant_m = nnq.Quantize(scale, zero_point, dtype)
        qr2 = quant_m(r)
        self.assertEqual(qr, qr2)
        # testing Dequantize API
        rqr = qr.dequantize()
        dequant_m = nnq.DeQuantize()
        rqr2 = dequant_m(qr2)
        self.assertEqual(rqr, rqr2)

    def test_conv_api(self):
        """Tests the correctness of the conv module.

        The correctness is defined against the functional implementation.
        """

        N, iC, H, W = 10, 10, 10, 3
        oC, g, kH, kW = 16, 1, 3, 3
        scale, zero_point = 1.0 / 255, 128

        X = torch.randn(N, iC, H, W, dtype=torch.float32)
        X = X.permute([0, 2, 3, 1]).contiguous()
        qX = torch.quantize_linear(X, scale=scale, zero_point=128, dtype=torch.quint8)

        w = torch.randn(oC, iC // g, kH, kW, dtype=torch.float32)
        w = w.permute([0, 2, 3, 1]).contiguous()
        qw = torch.quantize_linear(w, scale=scale, zero_point=0, dtype=torch.qint8)

        b = torch.randn(oC, dtype=torch.float32)
        qb = torch.quantize_linear(b, scale=1.0 / 1024, zero_point=0, dtype=torch.qint32)

        conv_under_test = Conv2d(in_channels=iC,
                                 out_channels=oC,
                                 kernel_size=(kH, kW),
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 groups=g,
                                 bias=True,
                                 padding_mode='zeros')
        conv_under_test.weight = qw
        conv_under_test.bias = qb
        conv_under_test.scale = scale
        conv_under_test.zero_point = zero_point

        # Test members
        self.assertTrue(hasattr(conv_under_test, '_packed_weight'))
        self.assertTrue(hasattr(conv_under_test, '_scale'))
        self.assertTrue(hasattr(conv_under_test, '_zero_point'))

        # Test properties
        # self.assertEqual(qw, conv_under_test.weight)
        self.assertEqual(qb, conv_under_test.bias)
        self.assertEqual(scale, conv_under_test.scale)
        self.assertEqual(zero_point, conv_under_test.zero_point)

        # Test forward
        result_under_test = conv_under_test(qX)
        result_reference = qF.conv2d(qX, qw, bias=qb,
                                     scale=scale, zero_point=zero_point,
                                     stride=1, padding=0,
                                     dilation=1, groups=g,
                                     prepacked=False, dtype=torch.quint8)

        self.assertEqual(result_reference, result_under_test,
                         message="Tensors are not equal.")


if __name__ == '__main__':
    run_tests()
