import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nnq_fused
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.quantized.functional as qF
import torch.quantization

from torch.testing._internal.common_quantization import QuantizationTestCase, prepare_dynamic
from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, override_quantized_engine
from torch.testing._internal.common_utils import run_tests, IS_PPC, TEST_WITH_UBSAN
from hypothesis import assume, given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

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
    batch_size, in_channels_per_group, input_feature_map_size,
    out_channels_per_group, groups, kernel_size, X_scale, X_zero_point, W_scale,
    W_zero_point, use_bias, use_channelwise,
):
    in_channels = in_channels_per_group * groups
    out_channels = out_channels_per_group * groups

    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min, X_value_max,
        (batch_size, in_channels,) + input_feature_map_size)
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)

    W_scale = W_scale * out_channels
    W_zero_point = W_zero_point * out_channels
    # Resize W_scale and W_zero_points arrays equal to out_channels
    W_scale = W_scale[:out_channels]
    W_zero_point = W_zero_point[:out_channels]
    # For testing, we use small values for weights and for activations so that
    # no overflow occurs in vpmaddubsw instruction. If the overflow occurs in
    # qconv implementation and if there is no overflow.
    # In reference we can't exactly match the results with reference.
    # Please see the comment in qconv implementation file
    #   aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
    (W_value_min, W_value_max) = (-5, 5)
    # The operator expects them in the format
    # (out_channels, in_channels/groups,) + kernel_size
    W_init = torch.randint(
        W_value_min, W_value_max,
        (out_channels, in_channels_per_group,) + kernel_size)
    b_init = torch.randint(0, 10, (out_channels,))

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
        self, qconv_fn, conv_fn, batch_size, in_channels_per_group,
        input_feature_map_size, out_channels_per_group, groups, kernel_size,
        stride, padding, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
        Y_scale, Y_zero_point, use_bias, use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale,
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



    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
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
        self, batch_size, in_channels_per_group, H, W, out_channels_per_group,
        groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
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
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
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
           qengine=st.sampled_from(("fbgemm",)))
    def test_conv3d_api(
        self, batch_size, in_channels_per_group, D, H, W,
        out_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
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
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)


class DynamicModuleAPITest(QuantizationTestCase):
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
        W_pack = qlinear._packed_params._packed_params
        Z_dq = qlinear(X)

        # Check if the module implementation matches calling the
        # ops directly
        Z_ref = torch.ops.quantized.linear_dynamic(X, W_pack)
        self.assertEqual(Z_ref, Z_dq)

        # Test serialization of dynamic quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        self.assertEqual(model_dict['_packed_params.weight'], W_q)
        if use_bias:
            self.assertEqual(model_dict['_packed_params.bias'], B)
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qlinear = nnqd.Linear(in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)

        linear_unpack = torch.ops.quantized.linear_unpack
        self.assertEqual(linear_unpack(qlinear._packed_params._packed_params),
                         linear_unpack(loaded_qlinear._packed_params._packed_params))
        if use_bias:
            self.assertEqual(qlinear.bias(), loaded_qlinear.bias())
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertTrue(hasattr(qlinear, '_packed_params'))
        self.assertTrue(hasattr(loaded_qlinear, '_packed_params'))
        self.assertTrue(hasattr(qlinear, '_weight_bias'))
        self.assertTrue(hasattr(loaded_qlinear, '_weight_bias'))

        self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
        self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params._packed_params))
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
            W_pack = qlinear._packed_params._packed_params

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
            self.assertEqual(model_dict['_packed_params.weight'], W_q)
            if use_bias:
                self.assertEqual(model_dict['_packed_params.bias'], B)
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
            self.assertEqual(linear_unpack(qlinear._packed_params._packed_params),
                             linear_unpack(loaded_qlinear._packed_params._packed_params))
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
            self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params._packed_params))
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

    def _test_conv_api_impl(
        self, module_name, qconv_module, conv_module, batch_size,
        in_channels_per_group, input_feature_map_size, out_channels_per_group,
        groups, kernel_size, stride, padding, dilation, X_scale, X_zero_point,
        W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias, use_fused,
        use_channelwise,
    ):
        for i in range(len(kernel_size)):
            assume(input_feature_map_size[i] + 2 * padding[i]
                   >= dilation[i] * (kernel_size[i] - 1) + 1)

        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        (X, X_q, W, W_q, b) = _make_conv_test_input(
            batch_size, in_channels_per_group, input_feature_map_size,
            out_channels_per_group, groups, kernel_size, X_scale, X_zero_point,
            W_scale, W_zero_point, use_bias, use_channelwise)

        qconv_module.set_weight_bias(W_q, b)
        qconv_module.scale = Y_scale
        qconv_module.zero_point = Y_zero_point

        if use_fused:
            conv_module[0].weight.data = W
            if use_bias:
                conv_module[0].bias.data = b
        else:
            conv_module.weight.data = W
            if use_bias:
                conv_module.bias.data = b

        # Test members
        self.assertTrue(module_name in str(qconv_module))
        self.assertTrue(hasattr(qconv_module, '_packed_params'))
        self.assertTrue(hasattr(qconv_module, 'scale'))
        self.assertTrue(hasattr(qconv_module, 'zero_point'))

        # Test properties
        self.assertEqual(W_q, qconv_module.weight())
        if use_bias:
            self.assertEqual(b, qconv_module.bias())
        self.assertEqual(Y_scale, qconv_module.scale)
        self.assertEqual(Y_zero_point, qconv_module.zero_point)

        # Test forward
        Y_exp = conv_module(X)
        Y_exp = torch.quantize_per_tensor(
            Y_exp, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)
        Y_act = qconv_module(X_q)

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

        # Test serialization of quantized Conv Module using state_dict
        model_dict = qconv_module.state_dict()
        self.assertEqual(W_q, model_dict['weight'])
        if use_bias:
            self.assertEqual(b, model_dict['bias'])
        bytes_io = io.BytesIO()
        torch.save(model_dict, bytes_io)
        bytes_io.seek(0)
        loaded_dict = torch.load(bytes_io)
        for key in loaded_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])

        loaded_qconv_module = type(qconv_module)(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, use_bias, padding_mode="zeros")
        loaded_qconv_module.load_state_dict(loaded_dict)

        self.assertTrue(dir(loaded_qconv_module) == dir(qconv_module))
        self.assertTrue(module_name in str(loaded_qconv_module))
        self.assertTrue(hasattr(loaded_qconv_module, '_packed_params'))
        self.assertTrue(hasattr(loaded_qconv_module, '_weight_bias'))

        self.assertEqual(qconv_module.weight(), loaded_qconv_module.weight())
        if use_bias:
            self.assertEqual(qconv_module.bias(), loaded_qconv_module.bias())
        self.assertEqual(qconv_module.scale, loaded_qconv_module.scale)
        self.assertEqual(qconv_module.zero_point,
                         loaded_qconv_module.zero_point)
        Y_loaded = loaded_qconv_module(X_q)
        np.testing.assert_array_almost_equal(
            Y_exp.int_repr().numpy(), Y_loaded.int_repr().numpy(), decimal=0)

        # The below check is meant to ensure that `torch.save` and `torch.load`
        # serialization works, however it is currently broken by the following:
        # https://github.com/pytorch/pytorch/issues/24045
        #
        # Instead, we currently check that the proper exception is thrown on
        # save.
        # <start code>
        # b = io.BytesIO()
        # torch.save(conv_under_test, b)
        # b.seek(0)
        # loaded_conv = torch.load(b)
        #
        # self.assertEqual(loaded_qconv_module.bias(), qconv_module.bias())
        # self.assertEqual(loaded_qconv_module.scale, qconv_module.scale)
        # self.assertEqual(loaded_qconv_module.zero_point,
        #                  qconv_module.zero_point)
        # <end code>
        with self.assertRaisesRegex(
            RuntimeError, r'torch.save\(\) is not currently supported'
        ):
            bytes_io = io.BytesIO()
            torch.save(qconv_module, bytes_io)

        # JIT testing
        self.checkScriptable(
            qconv_module, list(zip([X_q], [Y_exp])),
            check_save_load=True)

        # Test from_float
        conv_module.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(conv_module, inplace=True)
        conv_module(X.float())
        converted_qconv_module = torch.nn.Sequential(conv_module)
        torch.quantization.convert(converted_qconv_module, inplace=True)

        # Smoke test to make sure the module actually runs
        if use_bias:
            if use_fused:
                self.assertEqual(conv_module[0].bias,
                                 converted_qconv_module[0].bias())
            else:
                self.assertEqual(conv_module.bias,
                                 converted_qconv_module[0].bias())
        # Smoke test extra_repr
        self.assertTrue(module_name in str(converted_qconv_module))

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           H=st.integers(4, 16),
           W=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
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
           use_fused=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_conv2d_api(
        self, batch_size, in_channels_per_group, H, W, out_channels_per_group,
        groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_fused, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv2d module.
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        input_feature_map_size = (H, W)
        kernel_size = (kernel_h, kernel_w)
        stride = (stride_h, stride_w)
        padding = (pad_h, pad_w)
        dilation = (dilation, dilation)

        with override_quantized_engine(qengine):
            if use_fused:
                module_name = "QuantizedConvReLU2d"
                qconv_module = nnq_fused.ConvReLU2d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode="zeros")
            else:
                module_name = "QuantizedConv2d"
                qconv_module = nnq.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode="zeros")

            conv_module = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode="zeros")
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU2d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, use_fused, use_channelwise)

    @given(batch_size=st.integers(1, 3),
           in_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           D=st.integers(3, 6),
           H=st.integers(3, 6),
           W=st.integers(3, 6),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           groups=st.integers(1, 4),
           kernel_d=st.integers(1, 3),
           kernel_h=st.integers(1, 3),
           kernel_w=st.integers(1, 3),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 1),
           pad_h=st.integers(0, 1),
           pad_w=st.integers(0, 1),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_fused=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("fbgemm",)))
    def test_conv3d_api(
        self, batch_size, in_channels_per_group, D, H, W,
        out_channels_per_group, groups, kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation, X_scale,
        X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
        use_channelwise, use_fused, qengine,
    ):
        # Tests the correctness of the conv3d module.
        if qengine not in torch.backends.quantized.supported_engines:
            return

        in_channels = in_channels_per_group * groups
        out_channels = out_channels_per_group * groups
        input_feature_map_size = (D, H, W)
        kernel_size = (kernel_d, kernel_h, kernel_w)
        stride = (stride_d, stride_h, stride_w)
        padding = (pad_d, pad_h, pad_w)
        dilation = (dilation, dilation, dilation)

        with override_quantized_engine(qengine):
            if use_fused:
                module_name = "QuantizedConvReLU3d"
                qconv_module = nnq_fused.ConvReLU3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode="zeros")
            else:
                module_name = "QuantizedConv3d"
                qconv_module = nnq.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode="zeros")

            conv_module = nn.Conv3d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode="zeros")
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU3d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, use_fused, use_channelwise)

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

    def test_batch_norm(self):
        """Tests the correctness of the batchnorm2d module.
        The correctness is defined against the functional implementation.
        """
        x = torch.randn((2, 4, 6, 8), dtype=torch.float)
        float_mod = torch.nn.BatchNorm2d(4)
        float_mod.training = False

        y_ref = float_mod(x)
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        quant_mod = nnq.BatchNorm2d(4)
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        qy = quant_mod(qx)

        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         message="BatchNorm2d module API failed")

    def test_batch_norm3d(self):
        """Tests the correctness of the batchnorm3d module.
        The correctness is defined against the functional implementation.
        """
        x = torch.randn((2, 4, 6, 8, 10), dtype=torch.float)
        float_mod = torch.nn.BatchNorm3d(4)
        float_mod.training = False

        y_ref = float_mod(x)
        quant_ref = torch.quantize_per_tensor(y_ref, 1.0, 0, dtype=torch.quint8)

        quant_mod = nnq.BatchNorm3d(4)
        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.quint8)
        qy = quant_mod(qx)

        self.assertEqual(quant_ref.int_repr().numpy(), qy.int_repr().numpy(),
                         message="BatchNorm3d module API failed")

if __name__ == '__main__':
    run_tests()
