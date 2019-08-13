from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized as nnq
import torch.nn._intrinsic.quantized as nnq_fused
import torch.nn.quantized.functional as qF
from torch.nn.quantized.modules import Conv2d
from torch.nn._intrinsic.quantized import ConvReLU2d
from common_utils import run_tests, tempfile
from common_quantization import QuantizationTestCase, no_deadline
from hypothesis import given
from hypothesis import strategies as st
import unittest


'''
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `caffe2/test/test_quantized.py`.
'''


class FunctionalAPITest(QuantizationTestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.ops.quantized.relu(qX)
        qY_hat = qF.relu(qX)
        self.assertEqual(qY, qY_hat)


@unittest.skipIf(not torch.fbgemm_is_cpu_supported(),
                 'Quantized RNN requires FBGEMM. FBGEMM is only optimized for CPUs'
                 ' with instruction set support avx2 or newer.')
class ModuleAPITest(QuantizationTestCase):
    @no_deadline
    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
        use_fused=st.booleans(),
    )
    def test_linear_api(self, batch_size, in_features, out_features, use_bias, use_fused):
        """test API functionality for nn.quantized.linear and nn._intrinsic.quantized.linear_relu"""
        W = torch.rand(out_features, in_features).float()
        W_q = torch.quantize_linear(W, 0.1, 4, torch.qint8)
        X = torch.rand(batch_size, in_features).float()
        X_q = torch.quantize_linear(X, 0.2, 10, torch.quint8)
        B = torch.rand(out_features).float() if use_bias else None
        B_q = torch.quantize_linear(B, W_q.q_scale() * X_q.q_scale(), 0, torch.qint32) if use_bias else None
        scale = 0.5
        zero_point = 3
        if use_fused:
            qlinear = nnq_fused.LinearReLU(in_features, out_features)
        else:
            qlinear = nnq.Linear(in_features, out_features)
        qlinear.set_weight(W_q)
        # Simple round-trip test to ensure weight()/set_weight() API
        self.assertEqual(qlinear.weight(), W_q)
        W_pack = qlinear._packed_weight
        qlinear.bias = B_q if use_bias else None

        qlinear.scale = float(scale)
        qlinear.zero_point = int(zero_point)
        Z_q = qlinear(X_q)
        # Check if the module implementation matches calling the
        # ops directly
        if use_fused:
            Z_ref = torch.ops.quantized.fbgemm_linear_relu(X_q, W_pack, B_q, scale, zero_point)
        else:
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
        if use_fused:
            loaded_qlinear = nnq_fused.LinearReLU(in_features, out_features)
        else:
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
        self.assertEqual(qlinear.weight(), loaded_qlinear.weight())
        self.assertEqual(qlinear.weight(), torch.ops.quantized.fbgemm_linear_unpack(qlinear._packed_weight))
        Z_q2 = loaded_qlinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        # test serialization of module directly
        with tempfile.NamedTemporaryFile() as f:
            torch.save(qlinear, f)
            f.seek(0)
            loaded = torch.load(f)
        # This check is disabled pending an issue in PyTorch serialization:
        # https://github.com/pytorch/pytorch/issues/24045
        # self.assertEqual(qlinear.weight(), loaded.weight())
        self.assertEqual(qlinear.bias, loaded.bias)
        self.assertEqual(qlinear.scale, loaded.scale)
        self.assertEqual(qlinear.zero_point, loaded.zero_point)

        # Test JIT
        self.checkScriptable(qlinear, list(zip([X_q], [Z_ref])), check_save_load=True)

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

    @no_deadline
    @given(
        use_bias=st.booleans(),
        use_fused=st.booleans(),
    )
    def test_conv_api(self, use_bias, use_fused):
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

        b = torch.randn(oC, dtype=torch.float32) if use_bias else None
        qb = torch.quantize_linear(b, scale=1.0 / 1024, zero_point=0, dtype=torch.qint32) if use_bias else None

        if use_fused:
            conv_under_test = ConvReLU2d(in_channels=iC,
                                         out_channels=oC,
                                         kernel_size=(kH, kW),
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=g,
                                         bias=use_bias,
                                         padding_mode='zeros')
        else:
            conv_under_test = Conv2d(in_channels=iC,
                                     out_channels=oC,
                                     kernel_size=(kH, kW),
                                     stride=1,
                                     padding=0,
                                     dilation=1,
                                     groups=g,
                                     bias=use_bias,
                                     padding_mode='zeros')
        conv_under_test.set_weight(qw)
        conv_under_test.bias = qb
        conv_under_test.scale = scale
        conv_under_test.zero_point = zero_point

        # Test members
        self.assertTrue(hasattr(conv_under_test, '_packed_weight'))
        self.assertTrue(hasattr(conv_under_test, 'scale'))
        self.assertTrue(hasattr(conv_under_test, 'zero_point'))

        # Test properties
        self.assertEqual(qw, conv_under_test.weight())
        self.assertEqual(qb, conv_under_test.bias)
        self.assertEqual(scale, conv_under_test.scale)
        self.assertEqual(zero_point, conv_under_test.zero_point)

        # Test forward
        result_under_test = conv_under_test(qX)
        result_reference = qF.conv2d(qX, qw, bias=qb,
                                     scale=scale, zero_point=zero_point,
                                     stride=1, padding=0,
                                     dilation=1, groups=g, dtype=torch.quint8
                                     )
        if use_fused:
            # result_reference < zero_point doesn't work for qtensor yet
            # result_reference[result_reference < zero_point] = zero_point
            MB, OC, OH, OW = result_reference.size()
            for i in range(MB):
                for j in range(OC):
                    for h in range(OH):
                        for w in range(OW):
                            if result_reference[i][j][h][w].int_repr() < zero_point:
                                # assign 0. that gets converted to zero_point
                                result_reference[i][j][h][w] = 0.

        self.assertEqual(result_reference, result_under_test,
                         message="Tensors are not equal.")

        # Test serialization of quantized Conv Module using state_dict
        model_dict = conv_under_test.state_dict()
        self.assertEqual(model_dict['weight'], qw)
        if use_bias:
            self.assertEqual(model_dict['bias'], qb)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(model_dict, f)
            f.seek(0)
            loaded_dict = torch.load(f)
        for key in model_dict:
            self.assertEqual(loaded_dict[key], model_dict[key])
        if use_fused:
            loaded_conv_under_test = ConvReLU2d(in_channels=iC,
                                                out_channels=oC,
                                                kernel_size=(kH, kW),
                                                stride=1,
                                                padding=0,
                                                dilation=1,
                                                groups=g,
                                                bias=use_bias,
                                                padding_mode='zeros')
        else:
            loaded_conv_under_test = Conv2d(in_channels=iC,
                                            out_channels=oC,
                                            kernel_size=(kH, kW),
                                            stride=1,
                                            padding=0,
                                            dilation=1,
                                            groups=g,
                                            bias=use_bias,
                                            padding_mode='zeros')
        loaded_conv_under_test.load_state_dict(loaded_dict)
        self.assertEqual(loaded_conv_under_test.weight(), conv_under_test.weight())
        if use_bias:
            self.assertEqual(loaded_conv_under_test.bias, conv_under_test.bias)
        self.assertEqual(loaded_conv_under_test.scale, conv_under_test.scale)
        self.assertEqual(loaded_conv_under_test.zero_point, conv_under_test.zero_point)
        self.assertTrue(dir(loaded_conv_under_test) == dir(conv_under_test))
        self.assertTrue(hasattr(conv_under_test, '_packed_weight'))
        self.assertTrue(hasattr(loaded_conv_under_test, '_packed_weight'))
        self.assertTrue(hasattr(conv_under_test, 'weight'))
        self.assertTrue(hasattr(loaded_conv_under_test, 'weight'))
        self.assertEqual(loaded_conv_under_test.weight(), conv_under_test.weight())
        self.assertEqual(loaded_conv_under_test.weight(), qw)
        loaded_result = loaded_conv_under_test(qX)
        self.assertEqual(loaded_result, result_reference)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(conv_under_test, f)
            f.seek(0)
            loaded_conv = torch.load(f)

        self.assertEqual(conv_under_test.bias, loaded_conv.bias)
        self.assertEqual(conv_under_test.scale, loaded_conv.scale)
        self.assertEqual(conv_under_test.zero_point, loaded_conv.zero_point)

        # JIT testing
        self.checkScriptable(conv_under_test, list(zip([qX], [result_reference])), check_save_load=True)

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

        # JIT Testing
        self.checkScriptable(pool_under_test, list(zip([X], [qX_expect])))

if __name__ == '__main__':
    run_tests()
