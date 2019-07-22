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
        scale = 0.5
        zero_point = 3
        qlinear = nnq.Linear(in_features, out_features)
        qlinear._packed_weight = W_pack
        qlinear.bias = B_q if use_bias else None
        qlinear.scale = torch.tensor([scale], dtype=torch.double)
        qlinear.zero_point = torch.tensor([zero_point], dtype=torch.long)
        Z_q = qlinear(X_q)
        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.fbgemm_linear(X_q, W_pack, B_q, scale, zero_point)
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
        self.assertEqual(qlinear.scale, loaded_qlinear.scale)
        self.assertEqual(qlinear.zero_point, loaded_qlinear.zero_point)
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
        # self.assertEqual(qLinear.scale, loaded.scale)
        # self.assertEqual(qLinear.zero_point, loaded.zero_point)

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
        conv_under_test.scale = torch.tensor([scale], dtype=torch.double)
        conv_under_test.zero_point = torch.tensor([zero_point], dtype=torch.long)

        # Test members
        self.assertTrue(hasattr(conv_under_test, '_packed_weight'))
        self.assertTrue(hasattr(conv_under_test, 'scale'))
        self.assertTrue(hasattr(conv_under_test, 'zero_point'))

        # Test properties
        self.assertEqual(qw, conv_under_test.weight)
        self.assertEqual(qb, conv_under_test.bias)
        self.assertEqual(scale, conv_under_test.scale)
        self.assertEqual(zero_point, conv_under_test.zero_point)

        # Test forward
        result_under_test = conv_under_test(qX)
        result_reference = qF.conv2d(qX.permute([0, 2, 3, 1]), qw.permute([0, 2, 3, 1]), bias=qb,
                                     scale=scale, zero_point=zero_point,
                                     stride=1, padding=0,
                                     dilation=1, groups=g,
                                     prepacked=False, dtype=torch.quint8).permute([0, 3, 1, 2])

        self.assertEqual(result_reference, result_under_test,
                         message="Tensors are not equal.")

    def test_pool_api(self):
        """Tests the correctness of the pool module.

        The correctness is defined against the functional implementation.
        """
        N, C, H, W = 10, 10, 10, 3
        kwargs = {
            'kernel_size': 2,
            'stride': None,
            'padding': 0,
            'dilation': 1
        }

        scale, zero_point = 1.0 / 255, 128

        X = torch.randn(N, C, H, W, dtype=torch.float32)
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point,
                                   dtype=torch.quint8)
        qX_expect = torch.nn.functional.max_pool2d(qX, **kwargs)

        pool_under_test = torch.nn.quantized.MaxPool2d(**kwargs)
        qX_hat = pool_under_test(qX)
        self.assertEqual(qX_expect, qX_hat)

if __name__ == '__main__':
    run_tests()
