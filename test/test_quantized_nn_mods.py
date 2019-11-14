import torch
import torch.nn.functional as F
import torch.nn.intrinsic.quantized as nnq_fused
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.quantized.functional as qF
import torch.quantization

from torch.nn.quantized.modules import Conv2d
from torch.nn.intrinsic.quantized import ConvReLU2d

from common_quantization import QuantizationTestCase, prepare_dynamic
from common_quantized import _calculate_dynamic_qparams, override_quantized_engine
from common_utils import run_tests, IS_PPC, TEST_WITH_UBSAN
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis_utils import no_deadline

import io
import numpy as np
import unittest

'''
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `caffe2/test/test_quantized.py`.
'''

def _make_conv_test_input(
    batch_size, input_channels_per_group, input_feature_map_size,
    output_channels_per_group, groups, kernel_size, X_scale, X_zero_point,
    W_scale, W_zero_point, use_bias, use_channelwise,
):
    input_channels = input_channels_per_group * groups
    output_channels = output_channels_per_group * groups

    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min, X_value_max,
        (batch_size, input_channels,) + input_feature_map_size)
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)

    W_scale = W_scale * output_channels
    W_zero_point = W_zero_point * output_channels
    # Resize W_scale and W_zero_points arrays equal to output_channels
    W_scale = W_scale[:output_channels]
    W_zero_point = W_zero_point[:output_channels]
    # For testing, we use small values for weights and for activations so that
    # no overflow occurs in vpmaddubsw instruction. If the overflow occurs in
    # qconv implementation and if there is no overflow.
    # In reference we can't exactly match the results with reference.
    # Please see the comment in qconv implementation file
    #   aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
    (W_value_min, W_value_max) = (-5, 5)
    # The operator expects them in the format
    # (output_channels, input_channels/groups,) + kernel_size
    W_init = torch.randint(
        W_value_min, W_value_max,
        (output_channels, input_channels_per_group,) + kernel_size)
    b_init = torch.randint(0, 10, (output_channels,))

    if use_channelwise:
        W_shape = (-1, 1) + (1,) * len(kernel_size)
        W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
        W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
        W = W_scales_tensor.reshape(*W_shape) * (
            W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
        b = X_scale * W_scales_tensor * b_init.float()
        W_q = torch.quantize_per_channel(
            W, W_scales_tensor, W_zero_points_tensor.long(), 0,
            dtype=torch.qint8)
    else:
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        b = X_scale * W_scale[0] * b_init.float()
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8)

    return (X, X_q, W, W_q, b if use_bias else None)


class FunctionalAPITest(QuantizationTestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.relu(qX)
        qY_hat = qF.relu(qX)
        self.assertEqual(qY, qY_hat)

    def _test_conv_api_impl(
        self, qconv_fn, conv_fn, batch_size, input_channels_per_group,
        input_feature_map_size, output_channels_per_group, groups, kernel_size,
        stride, padding, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
        Y_scale, Y_zero_point, use_bias, use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, input_channels_per_group, input_feature_map_size,
            output_channels_per_group, groups, kernel_size, X_scale,
            X_zero_point, W_scale, W_zero_point, use_bias, use_channelwise)

        Y_exp = conv_fn(X, W, b, stride, padding, dilation, groups)
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
        Y_act = qconv_fn(
            X_q, W_q, b, stride, padding, dilation, groups,
            padding_mode="zeros", scale=Y_scale, zero_point=Y_zero_point)

        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)
        # We use decimal = 0 to ignore off-by-1 differences between reference
        # and test. Off-by-1 differences arise due to the order of round and
        # zero_point addition operation, i.e., if addition followed by round is
        # used by reference and round followed by addition is used by test, the
        # results may differ by 1.
        # For example, the result of round(2.5) + 1 is 3 while round(2.5 + 1) is
        # 4 assuming the rounding mode is round-to-nearest, ties-to-even.
        np.testing.assert_array_almost_equal(
            Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)



    @no_deadline
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv2d_api(
        self, batch_size, input_channels_per_group, H, W,
        output_channels_per_group, groups, kernel_h, kernel_w, stride_h,
        stride_w, pad_h, pad_w, dilation, X_scale, X_zero_point, W_scale,
        W_zero_point, Y_scale, Y_zero_point, use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv2d function.

        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        input_feature_map_size = (H, W)
        kernel_size = (kernel_h, kernel_w)
        stride = (stride_h, stride_w)
        padding = (pad_h, pad_w)
        dilation = (dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv2d
            conv_fn = F.conv2d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, input_channels_per_group,
                input_feature_map_size, output_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @no_deadline
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel_d=st.integers(1, 4),
           kernel_h=st.integers(1, 4),
           kernel_w=st.integers(1, 4),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("fbgemm")))
    def test_conv3d_api(
        self, batch_size, input_channels_per_group, D, H, W,
        output_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation, X_scale,
        X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
        use_channelwise, qengine,
    ):
        # Tests the correctness of the conv3d function.
        # Currently conv3d only supports FbGemm engine

        if qengine not in torch.backends.quantized.supported_engines:
            return

        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv3d
            conv_fn = F.conv3d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, input_channels_per_group,
                input_feature_map_size, output_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)


class DynamicModuleAPITest(QuantizationTestCase):
    @no_deadline
    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                         " with instruction set support avx2 or newer.")
    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
        use_default_observer=st.booleans(),
    )
    def test_linear_api(self, batch_size, in_features, out_features, use_bias, use_default_observer):
        """test API functionality for nn.quantized.dynamic.Linear"""
        W = torch.rand(out_features, in_features).float()
        W_scale, W_zp = _calculate_dynamic_qparams(W, torch.qint8)
        W_q = torch.quantize_per_tensor(W, W_scale, W_zp, torch.qint8)
        X = torch.rand(batch_size, in_features).float()
        B = torch.rand(out_features).float() if use_bias else None
        qlinear = nnqd.Linear(in_features, out_features)
        # Run module with default-initialized parameters.
        # This tests that the constructor is correct.
        qlinear.set_weight_bias(W_q, B)
        qlinear(X)

        # Simple round-trip test to ensure weight()/set_weight() API
        self.assertEqual(qlinear.weight(), W_q)
        W_pack = qlinear._packed_params
        Z_dq = qlinear(X)

        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.linear_dynamic(X, W_pack)
        self.assertEqual(Z_ref, Z_dq)

        # Test serialization of dynamic quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        if use_bias:
            self.assertEqual(model_dict['bias'], B)
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qlinear = nnqd.Linear(in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)

        linear_unpack = torch.ops.quantized.linear_unpack
        self.assertEqual(linear_unpack(qlinear._packed_params),
                         linear_unpack(loaded_qlinear._packed_params))
        if use_bias:
            self.assertEqual(qlinear.bias(), loaded_qlinear.bias())
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertTrue(hasattr(qlinear, '_packed_params'))
        self.assertTrue(hasattr(loaded_qlinear, '_packed_params'))
        self.assertTrue(hasattr(qlinear, '_weight_bias'))
        self.assertTrue(hasattr(loaded_qlinear, '_weight_bias'))

        self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
        self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params))
        Z_dq2 = qlinear(X)
        self.assertEqual(Z_dq, Z_dq2)

        # The below check is meant to ensure that `torch.save` and `torch.load`
        # serialization works, however it is currently broken by the following:
        # https://github.com/pytorch/pytorch/issues/24045
        #
        # Instead, we currently check that the proper exception is thrown on save.
        # <start code>
        # b = io.BytesIO()
        # torch.save(qlinear, b)
        # b.seek(0)
        # loaded = torch.load(b)
        # self.assertEqual(qlinear.weight(), loaded.weight())
        # self.assertEqual(qlinear.zero_point, loaded.zero_point)
        # <end code>
        with self.assertRaisesRegex(RuntimeError, r'torch.save\(\) is not currently supported'):
            b = io.BytesIO()
            torch.save(qlinear, b)

        # Test JIT
        self.checkScriptable(qlinear, list(zip([X], [Z_ref])), check_save_load=True)

        # Test from_float
        float_linear = torch.nn.Linear(in_features, out_features).float()
        if use_default_observer:
            float_linear.qconfig = torch.quantization.default_dynamic_qconfig
        prepare_dynamic(float_linear)
        float_linear(X.float())
        quantized_float_linear = nnqd.Linear.from_float(float_linear)

        # Smoke test to make sure the module actually runs
        quantized_float_linear(X)

        # Smoke test extra_repr
        self.assertTrue('QuantizedLinear' in str(quantized_float_linear))


class ModuleAPITest(QuantizationTestCase):
    def test_relu(self):
        relu_module = nnq.ReLU()
        relu6_module = nnq.ReLU6()

        x = torch.arange(-10, 10, dtype=torch.float)
        y_ref = torch.relu(x)
        y6_ref = torch.nn.modules.ReLU6()(x)

        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.qint32)
        qy = relu_module(qx)
        qy6 = relu6_module(qx)

        self.assertEqual(y_ref, qy.dequantize(),
                         message="ReLU module API failed")
        self.assertEqual(y6_ref, qy6.dequantize(),
                         message="ReLU6 module API failed")


    @no_deadline
    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
        use_fused=st.booleans(),
        per_channel=st.booleans(),
        qengine=st.sampled_from(("qnnpack", "fbgemm"))
    )
    def test_linear_api(self, batch_size, in_features, out_features, use_bias, use_fused, per_channel, qengine):
        """test API functionality for nn.quantized.linear and nn.intrinsic.quantized.linear_relu"""
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            per_channel = False
        with override_quantized_engine(qengine):
            W = torch.rand(out_features, in_features).float()
            if per_channel:
                scale_tensor = torch.ones(out_features, dtype=torch.double)
                zero_point_tensor = torch.zeros(out_features, dtype=torch.long)
                for i in range(len(scale_tensor)):
                    scale_tensor[i] = (i + 1.0) / 255.0
                W_q = torch.quantize_per_channel(W, scales=scale_tensor, zero_points=zero_point_tensor, axis=0, dtype=torch.qint8)
            else:
                W_q = torch.quantize_per_tensor(W, 0.1, 4, torch.qint8)

            X = torch.rand(batch_size, in_features).float()
            X_q = torch.quantize_per_tensor(X, 0.2, 10, torch.quint8)
            B = torch.rand(out_features).float() if use_bias else None
            scale = 0.5
            zero_point = 3
            if use_fused:
                qlinear = nnq_fused.LinearReLU(in_features, out_features)
            else:
                qlinear = nnq.Linear(in_features, out_features)

            # Run module with default-initialized parameters.
            # This tests that the constructor is correct.
            qlinear(X_q)

            qlinear.set_weight_bias(W_q, B)
            # Simple round-trip test to ensure weight()/set_weight() API
            self.assertEqual(qlinear.weight(), W_q)
            W_pack = qlinear._packed_params

            qlinear.scale = float(scale)
            qlinear.zero_point = int(zero_point)
            Z_q = qlinear(X_q)
            # Check if the module implementation matches calling the
            # ops directly
            if use_fused:
                Z_ref = torch.ops.quantized.linear_relu(X_q, W_pack, scale, zero_point)

                self.assertTrue('QuantizedLinearReLU' in str(qlinear))
            else:
                Z_ref = torch.ops.quantized.linear(X_q, W_pack, scale, zero_point)

                self.assertTrue('QuantizedLinear' in str(qlinear))
            self.assertEqual(Z_ref, Z_q)

            # Test serialization of quantized Linear Module using state_dict

            model_dict = qlinear.state_dict()
            self.assertEqual(model_dict['weight'], W_q)
            if use_bias:
                self.assertEqual(model_dict['bias'], B)
            b = io.BytesIO()
            torch.save(model_dict, b)
            b.seek(0)
            loaded_dict = torch.load(b)
            for key in model_dict:
                self.assertEqual(model_dict[key], loaded_dict[key])
            if use_fused:
                loaded_qlinear = nnq_fused.LinearReLU(in_features, out_features)
            else:
                loaded_qlinear = nnq.Linear(in_features, out_features)
            loaded_qlinear.load_state_dict(loaded_dict)

            linear_unpack = torch.ops.quantized.linear_unpack
            self.assertEqual(linear_unpack(qlinear._packed_params),
                             linear_unpack(loaded_qlinear._packed_params))
            if use_bias:
                self.assertEqual(qlinear.bias(), loaded_qlinear.bias())
            self.assertEqual(qlinear.scale, loaded_qlinear.scale)
            self.assertEqual(qlinear.zero_point, loaded_qlinear.zero_point)
            self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
            self.assertTrue(hasattr(qlinear, '_packed_params'))
            self.assertTrue(hasattr(loaded_qlinear, '_packed_params'))
            self.assertTrue(hasattr(qlinear, '_weight_bias'))
            self.assertTrue(hasattr(loaded_qlinear, '_weight_bias'))
            self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
            self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params))
            Z_q2 = loaded_qlinear(X_q)
            self.assertEqual(Z_q, Z_q2)

            # The below check is meant to ensure that `torch.save` and `torch.load`
            # serialization works, however it is currently broken by the following:
            # https://github.com/pytorch/pytorch/issues/24045
            #
            # Instead, we currently check that the proper exception is thrown on save.
            # <start code>
            # b = io.BytesIO()
            # torch.save(qlinear, b)
            # b.seek(0)
            # loaded = torch.load(b)
            # self.assertEqual(qlinear.weight(), loaded.weight())
            # self.assertEqual(qlinear.scale, loaded.scale)
            # self.assertEqual(qlinear.zero_point, loaded.zero_point)
            # <end code>
            with self.assertRaisesRegex(RuntimeError, r'torch.save\(\) is not currently supported'):
                b = io.BytesIO()
                torch.save(qlinear, b)

            # Test JIT
            self.checkScriptable(qlinear, list(zip([X_q], [Z_ref])), check_save_load=True)

            # Test from_float.
            float_linear = torch.nn.Linear(in_features, out_features).float()
            float_linear.qconfig = torch.quantization.default_qconfig
            torch.quantization.prepare(float_linear, inplace=True)
            float_linear(X.float())
            # Sequential allows swapping using "convert".
            quantized_float_linear = torch.nn.Sequential(float_linear)
            quantized_float_linear = torch.quantization.convert(quantized_float_linear, inplace=True)

            # Smoke test to make sure the module actually runs
            quantized_float_linear(X_q)

            # Smoke test extra_repr
            self.assertTrue('QuantizedLinear' in str(quantized_float_linear))

    def test_quant_dequant_api(self):
        r = torch.tensor([[1., -1.], [1., -1.]], dtype=torch.float)
        scale, zero_point, dtype = 1.0, 2, torch.qint8
        # testing Quantize API
        qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
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
        per_channel=st.booleans(),
        qengine=st.sampled_from(("qnnpack", "fbgemm"))
    )
    def test_conv_api(self, use_bias, use_fused, per_channel, qengine):
        """Tests the correctness of the conv module.
        The correctness is defined against the functional implementation.
        """
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            per_channel = False
        with override_quantized_engine(qengine):
            N, iC, H, W = 10, 10, 10, 3
            oC, g, kH, kW = 16, 1, 3, 3
            scale, zero_point = 1.0 / 255, 128

            X = torch.randn(N, iC, H, W, dtype=torch.float32)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=128, dtype=torch.quint8)

            w = torch.randn(oC, iC // g, kH, kW, dtype=torch.float32)

            if per_channel:
                scale_tensor = torch.ones(oC, dtype=torch.double)
                zero_point_tensor = torch.zeros(oC, dtype=torch.long)
                for i in range(len(scale_tensor)):
                    scale_tensor[i] = (i + 1.0) / 255.0

                qw = torch.quantize_per_channel(w, scales=scale_tensor, zero_points=zero_point_tensor, axis=0, dtype=torch.qint8)
            else:
                qw = torch.quantize_per_tensor(w, scale=scale, zero_point=0, dtype=torch.qint8)

            b = torch.randn(oC, dtype=torch.float32) if use_bias else None

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
            # Run module with default-initialized parameters.
            # This tests that the constructor is correct.
            conv_under_test.set_weight_bias(qw, b)
            conv_under_test(qX)

            conv_under_test.scale = scale
            conv_under_test.zero_point = zero_point

            # Test members
            self.assertTrue(hasattr(conv_under_test, '_packed_params'))
            self.assertTrue(hasattr(conv_under_test, 'scale'))
            self.assertTrue(hasattr(conv_under_test, 'zero_point'))

            # Test properties
            self.assertEqual(qw, conv_under_test.weight())
            if use_bias:
                self.assertEqual(b, conv_under_test.bias())
            self.assertEqual(scale, conv_under_test.scale)
            self.assertEqual(zero_point, conv_under_test.zero_point)

            # Test forward
            result_under_test = conv_under_test(qX)
            result_reference = qF.conv2d(qX, qw, bias=b,
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
                self.assertEqual(model_dict['bias'], b)
            b = io.BytesIO()
            torch.save(model_dict, b)
            b.seek(0)
            loaded_dict = torch.load(b)
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

                self.assertTrue('QuantizedConvReLU2d' in str(loaded_conv_under_test))
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
                self.assertTrue('QuantizedConv2d' in str(loaded_conv_under_test))
            loaded_conv_under_test.load_state_dict(loaded_dict)
            self.assertEqual(loaded_conv_under_test._weight_bias(), conv_under_test._weight_bias())
            if use_bias:
                self.assertEqual(loaded_conv_under_test.bias(), conv_under_test.bias())
            self.assertEqual(loaded_conv_under_test.scale, conv_under_test.scale)
            self.assertEqual(loaded_conv_under_test.zero_point, conv_under_test.zero_point)
            self.assertTrue(dir(loaded_conv_under_test) == dir(conv_under_test))
            self.assertTrue(hasattr(conv_under_test, '_packed_params'))
            self.assertTrue(hasattr(loaded_conv_under_test, '_packed_params'))
            self.assertTrue(hasattr(conv_under_test, '_weight_bias'))
            self.assertTrue(hasattr(loaded_conv_under_test, '_weight_bias'))
            self.assertEqual(loaded_conv_under_test._weight_bias(), conv_under_test._weight_bias())
            self.assertEqual(loaded_conv_under_test.weight(), qw)
            loaded_result = loaded_conv_under_test(qX)
            self.assertEqual(loaded_result, result_reference)

            # The below check is meant to ensure that `torch.save` and `torch.load`
            # serialization works, however it is currently broken by the following:
            # https://github.com/pytorch/pytorch/issues/24045
            #
            # Instead, we currently check that the proper exception is thrown on save.
            # <start code>
            # b = io.BytesIO()
            # torch.save(conv_under_test, b)
            # b.seek(0)
            # loaded_conv = torch.load(b)
            #
            # self.assertEqual(conv_under_test.bias(), loaded_conv.bias())
            # self.assertEqual(conv_under_test.scale, loaded_conv.scale)
            # self.assertEqual(conv_under_test.zero_point, loaded_conv.zero_point)
            # <end code>
            with self.assertRaisesRegex(RuntimeError, r'torch.save\(\) is not currently supported'):
                b = io.BytesIO()
                torch.save(conv_under_test, b)

            # JIT testing
            self.checkScriptable(conv_under_test, list(zip([qX], [result_reference])), check_save_load=True)

            # Test from_float
            float_conv = torch.nn.Conv2d(in_channels=iC,
                                         out_channels=oC,
                                         kernel_size=(kH, kW),
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         groups=g,
                                         bias=use_bias,
                                         padding_mode='zeros').float()
            float_conv.qconfig = torch.quantization.default_qconfig
            torch.quantization.prepare(float_conv, inplace=True)
            float_conv(X.float())
            quantized_float_conv = torch.nn.Sequential(float_conv)
            torch.quantization.convert(quantized_float_conv, inplace=True)

            # Smoke test to make sure the module actually runs
            quantized_float_conv(qX)
            if use_bias:
                self.assertEqual(quantized_float_conv[0].bias(), float_conv.bias)
            # Smoke test extra_repr
            self.assertTrue('QuantizedConv2d' in str(quantized_float_conv))

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
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)
        qX_expect = torch.nn.functional.max_pool2d(qX, **kwargs)

        pool_under_test = torch.nn.quantized.MaxPool2d(**kwargs)
        qX_hat = pool_under_test(qX)
        self.assertEqual(qX_expect, qX_hat)

        # JIT Testing
        self.checkScriptable(pool_under_test, list(zip([X], [qX_expect])))

if __name__ == '__main__':
    run_tests()
