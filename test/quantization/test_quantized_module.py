import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.quantized._reference as nniqr
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
import torch.nn.quantized.dynamic as nnqd
import torch.nn.functional as F
import torch.quantization

from torch.quantization import (
    get_default_static_quant_module_mappings,
    default_float_qparams_observer,
    PerChannelMinMaxObserver,
)
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    prepare_dynamic,
    _make_conv_test_input,
    skipIfNoFBGEMM,
    lengths_to_offsets
)
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_qparams,
    override_quantized_engine,
    override_qengines,
)
from hypothesis import assume, given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

import io
import numpy as np
import itertools

"""
Note that tests in this file are just API test, to make sure we wrapped the
quantized operator implementations correctly in the user facing APIs, these are
not correctness test for the underlying quantized operators. For correctness
test please see `test/quantization/test_quantized_op.py`.
"""

class TestStaticQuantizedModule(QuantizationTestCase):
    def test_relu(self):
        relu_module = nn.ReLU()
        relu6_module = nnq.ReLU6()

        x = torch.arange(-10, 10, dtype=torch.float)
        y_ref = torch.relu(x)
        y6_ref = torch.nn.modules.ReLU6()(x)

        qx = torch.quantize_per_tensor(x, 1.0, 0, dtype=torch.qint32)
        qy = relu_module(qx)
        qy6 = relu6_module(qx)

        self.assertEqual(y_ref, qy.dequantize(),
                         msg="ReLU module API failed")
        self.assertEqual(y6_ref, qy6.dequantize(),
                         msg="ReLU6 module API failed")

    @override_qengines
    def test_linear_api(self):
        """test API functionality for nn.quantized.linear and nn.intrinsic.quantized.linear_relu"""
        options = itertools.product(
            [1, 5],
            [16, 32],
            [4, 8],
            [True, False],
            [True, False],
            [True, False],
            [True, False])
        for (batch_size, in_features, out_features, use_bias,
             use_fused, per_channel, is_reference) in options:
            self._test_linear_api_impl(
                batch_size, in_features, out_features, use_bias, use_fused,
                per_channel, is_reference)

    def _test_linear_api_impl(self, batch_size, in_features, out_features, use_bias, use_fused, per_channel, is_reference):
        if torch.backends.quantized.engine == 'qnnpack':
            per_channel = False

        # (use_fused, is_reference) -> quantized class
        class_map = {
            (True, True) : nniqr.LinearReLU,
            (True, False) : nniq.LinearReLU,
            (False, True) : nnqr.Linear,
            (False, False) : nnq.Linear,
        }

        W = torch.rand(out_features, in_features).float()
        if per_channel:
            scale_tensor = torch.ones(out_features, dtype=torch.double)
            zero_point_tensor = torch.zeros(out_features, dtype=torch.long)
            for i in range(len(scale_tensor)):
                scale_tensor[i] = (i + 1.0) / 255.0
            W_q = torch.quantize_per_channel(W, scales=scale_tensor,
                                             zero_points=zero_point_tensor,
                                             axis=0, dtype=torch.qint8)
        else:
            W_q = torch.quantize_per_tensor(W, 0.1, 4, torch.qint8)

        X = torch.rand(batch_size, in_features).float()
        X_q = torch.quantize_per_tensor(X, 0.2, 10, torch.quint8)
        B = torch.rand(out_features).float() if use_bias else None
        scale = 0.5
        zero_point = 3
        qlinear = class_map[(use_fused, is_reference)](in_features, out_features)

        qlinear_copy = qlinear  # deepcopy does not work right now
        # qlinear_copy = copy.deepcopy(qlinear)
        self.checkScriptable(qlinear_copy, [[X_q]], check_save_load=True)
        # Run module with default-initialized parameters.
        # This tests that the constructor is correct.
        qlinear(X_q)

        qlinear.set_weight_bias(W_q, B)
        # Simple round-trip test to ensure weight()/set_weight() API
        self.assertEqual(qlinear.weight(), W_q, atol=1e-5, rtol=0)

        # testing packed param implementation
        qlinear.scale = float(scale)
        qlinear.zero_point = int(zero_point)
        Z_q = qlinear(X_q)

        # Check if the module implementation matches calling the
        # ops directly
        if is_reference:
            weight = qlinear._qweight
            bias = qlinear._bias
            weight_dequant = weight.dequantize()
            X_q_dq = X_q.dequantize()
            Z_ref = F.linear(X_q_dq, weight_dequant, bias)
            if use_fused:
                Z_ref = F.relu(Z_ref, inplace=True)
            Z_ref = torch.quantize_per_tensor(Z_ref, scale, zero_point, torch.quint8)
        else:
            W_pack = qlinear._packed_params._packed_params
            if use_fused:
                Z_ref = torch.ops.quantized.linear_relu(X_q, W_pack, scale, zero_point)
            else:
                Z_ref = torch.ops.quantized.linear(X_q, W_pack, scale, zero_point)

        self.assertEqual(Z_ref, Z_q)
        self.assertTrue(
            ("QuantizedLinearReLU" if use_fused else "QuantizedLinear") in str(qlinear))

        # Test serialization of quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            if isinstance(model_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                w_model, b_model = torch.ops.quantized.linear_unpack(model_dict[key])
                w_loaded, b_loaded = torch.ops.quantized.linear_unpack(loaded_dict[key])
                self.assertEqual(w_model, w_loaded)
                self.assertEqual(b_model, b_loaded)
            else:
                self.assertEqual(model_dict[key], loaded_dict[key])

        loaded_qlinear = class_map[(use_fused, is_reference)](
            in_features, out_features)
        loaded_qlinear.load_state_dict(loaded_dict)
        if is_reference:
            self.assertEqual(qlinear._qweight, loaded_qlinear._qweight)
            self.assertEqual(qlinear._bias, loaded_qlinear._bias)
        else:
            linear_unpack = torch.ops.quantized.linear_unpack
            self.assertEqual(linear_unpack(qlinear._packed_params._packed_params),
                             linear_unpack(loaded_qlinear._packed_params._packed_params))
        self.assertEqual(qlinear.scale, loaded_qlinear.scale)
        self.assertEqual(qlinear.zero_point, loaded_qlinear.zero_point)
        # make sure loaded_qlinear has the same dir as qlinear since
        # scripting the module will add __overloads__ to __dict__
        self.checkScriptable(loaded_qlinear, [[X_q]], check_save_load=True)
        self.assertTrue(dir(qlinear) == dir(loaded_qlinear))
        self.assertEqual(qlinear._weight_bias(), loaded_qlinear._weight_bias())
        if not is_reference:
            self.assertEqual(qlinear._weight_bias(), torch.ops.quantized.linear_unpack(qlinear._packed_params._packed_params))
        Z_q2 = loaded_qlinear(X_q)
        self.assertEqual(Z_q, Z_q2)

        b = io.BytesIO()
        torch.save(qlinear, b)
        b.seek(0)
        loaded = torch.load(b)
        self.assertEqual(qlinear.weight(), loaded.weight())
        self.assertEqual(qlinear.scale, loaded.scale)
        self.assertEqual(qlinear.zero_point, loaded.zero_point)

        # Test JIT
        self.checkScriptable(qlinear, [[X_q]], check_save_load=True)

        # Make sure `from_float` works for all linear variants
        modules_under_test = [torch.nn.Linear, torch.nn.modules.linear._LinearWithBias]

        for mut in modules_under_test:
            # Test from_float.
            float_linear = mut(in_features, out_features).float()
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
        groups, kernel_size, stride, padding, padding_mode, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_fused, use_channelwise, is_reference
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
        self.assertTrue(module_name == qconv_module._get_name(), module_name + " " + qconv_module._get_name())
        if not is_reference:
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
        # skip numerics checking for reference module
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_act.int_repr().numpy(), decimal=0)

        # Test serialization of quantized Conv Module using state_dict
        model_dict = qconv_module.state_dict()
        self.assertEqual(model_dict['weight'], W_q)
        if use_bias:
            self.assertEqual(model_dict['bias'], b)
        bytes_io = io.BytesIO()
        torch.save(model_dict, bytes_io)
        bytes_io.seek(0)
        loaded_dict = torch.load(bytes_io)
        for key in loaded_dict:
            self.assertEqual(model_dict[key], loaded_dict[key])
        loaded_qconv_module = type(qconv_module)(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, use_bias, padding_mode=padding_mode)
        loaded_qconv_module.load_state_dict(loaded_dict)

        self.assertTrue(dir(loaded_qconv_module) == dir(qconv_module))
        self.assertTrue(module_name == loaded_qconv_module._get_name())
        if not is_reference:
            self.assertTrue(hasattr(loaded_qconv_module, '_packed_params'))
        self.assertTrue(hasattr(loaded_qconv_module, '_weight_bias'))

        self.assertEqual(qconv_module.weight(), loaded_qconv_module.weight())
        if use_bias:
            self.assertEqual(qconv_module.bias(), loaded_qconv_module.bias())
        self.assertEqual(qconv_module.scale, loaded_qconv_module.scale)
        self.assertEqual(qconv_module.zero_point,
                         loaded_qconv_module.zero_point)
        Y_loaded = loaded_qconv_module(X_q)
        if not is_reference:
            np.testing.assert_array_almost_equal(
                Y_exp.int_repr().numpy(), Y_loaded.int_repr().numpy(), decimal=0)

        b = io.BytesIO()
        torch.save(qconv_module, b)
        b.seek(0)
        loaded_conv = torch.load(b)

        self.assertEqual(loaded_conv.bias(), qconv_module.bias())
        self.assertEqual(loaded_conv.scale, qconv_module.scale)
        self.assertEqual(loaded_conv.zero_point,
                         qconv_module.zero_point)

        # JIT testing
        self.checkScriptable(
            qconv_module, [[X_q]],
            check_save_load=True)

        # Test from_float
        fused_conv_module = torch.nn.intrinsic._FusedModule(conv_module)
        fused_conv_module.qconfig = torch.quantization.default_qconfig
        torch.quantization.prepare(fused_conv_module, inplace=True)
        fused_conv_module(X.float())
        converted_qconv_module = fused_conv_module
        reference_mapping = get_default_static_quant_module_mappings()
        reference_mapping[type(conv_module)] = type(qconv_module)
        torch.quantization.convert(converted_qconv_module, mapping=reference_mapping, inplace=True)

        # Smoke test to make sure the module actually runs
        if use_bias:
            if use_fused:
                self.assertEqual(conv_module[0].bias,
                                 converted_qconv_module[0].bias())
            else:
                self.assertEqual(conv_module.bias,
                                 converted_qconv_module[0].bias())
        # Smoke test extra_repr
        self.assertTrue(module_name == converted_qconv_module[0]._get_name())

    @override_qengines
    def test_conv1d_api(self):
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for pad_mode, use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            length = 8
            out_channels_per_group = 2
            groups = 3
            kernel = 3
            stride = 2
            pad = 1
            dilation = 1
            # Tests the correctness of the conv2d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (length,)
            kernel_size = (kernel, )
            stride = (stride, )
            pad = (pad, )
            dilation = (dilation, )
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            if torch.backends.quantized.engine == 'qnnpack':
                use_channelwise = False
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU1d, "QuantizedConvReLU1d(Reference)"),
                (True, False): (nniq.ConvReLU1d, "QuantizedConvReLU1d"),
                (False, True): (nnqr.Conv1d, "QuantizedConv1d(Reference)"),
                (False, False): (nnq.Conv1d, "QuantizedConv1d")
            }

            qconv_cls, module_name = class_map[(use_fused, is_reference)]
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode
            )

            conv_module = nn.Conv1d(
                in_channels, out_channels, kernel, stride, pad,
                dilation, groups, use_bias, padding_mode=pad_mode)
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU1d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, pad, pad_mode,
                dilation, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
                Y_zero_point, use_bias, use_fused, use_channelwise, is_reference)

    @override_qengines
    def test_conv2d_api(self):
        options = itertools.product(
            ["zeros", "reflect"],  # pad_mode
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for pad_mode, use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            stride_h = 2
            stride_w = 2
            pad_h = 1
            pad_w = 1
            dilation = 1
            # Tests the correctness of the conv2d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (H, W)
            kernel_size = (kernel_h, kernel_w)
            stride = (stride_h, stride_w)
            padding = (pad_h, pad_w)
            dilation = (dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU2d, "QuantizedConvReLU2d(Reference)"),
                (True, False): (nniq.ConvReLU2d, "QuantizedConvReLU2d"),
                (False, True): (nnqr.Conv2d, "QuantizedConv2d(Reference)"),
                (False, False): (nnq.Conv2d, "QuantizedConv2d")
            }

            qconv_cls, module_name = class_map[(use_fused, is_reference)]
            qconv_module = qconv_cls(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode
            )

            conv_module = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, use_bias, padding_mode=pad_mode)
            if use_fused:
                relu_module = nn.ReLU()
                conv_module = nni.ConvReLU2d(conv_module, relu_module)
            conv_module = conv_module.float()

            self._test_conv_api_impl(
                module_name, qconv_module, conv_module, batch_size,
                in_channels_per_group, input_feature_map_size,
                out_channels_per_group, groups, kernel_size, stride, padding,
                pad_mode, dilation, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_fused, use_channelwise, is_reference)

    @skipIfNoFBGEMM
    def test_conv3d_api(self):
        options = itertools.product(
            [True, False],  # use_bias
            [True, False],  # use_fused
            [True, False],  # use_channelwise
            [True, False]  # is_reference
        )
        for use_bias, use_fused, use_channelwise, is_reference in options:
            if torch.backends.quantized.engine == "qnnpack":
                use_channelwise = False
            batch_size = 2
            in_channels_per_group = 2
            H = 8
            W = 8
            D = 8
            out_channels_per_group = 2
            groups = 3
            kernel_h = 3
            kernel_w = 3
            kernel_d = 3
            stride_h = 2
            stride_w = 2
            stride_d = 2
            pad_mode = "zeros"  # 3d doesn't support reflect padding
            pad_h = 1
            pad_w = 1
            pad_d = 1
            dilation = 1
            # Tests the correctness of the conv3d module.
            in_channels = in_channels_per_group * groups
            out_channels = out_channels_per_group * groups
            input_feature_map_size = (D, H, W)
            kernel_size = (kernel_d, kernel_h, kernel_w)
            stride = (stride_d, stride_h, stride_w)
            padding = (pad_d, pad_h, pad_w)
            dilation = (dilation, dilation, dilation)
            X_scale = 1.3
            X_zero_point = 2
            W_scale = [0.5]
            W_zero_point = [3]
            Y_scale = 5.0
            Y_zero_point = 4
            # (use_fused, is_reference) -> quantized class
            class_map = {
                (True, True): (nniqr.ConvReLU3d, "QuantizedConvReLU3d(Reference)"),
                (True, False): (nniq.ConvReLU3d, "QuantizedConvReLU3d"),
                (False, True): (nnqr.Conv3d, "QuantizedConv3d(Reference)"),
                (False, False): (nnq.Conv3d, "QuantizedConv3d")
            }

            with override_quantized_engine('fbgemm'):
                qconv_cls, module_name = class_map[(use_fused, is_reference)]
                qconv_module = qconv_cls(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode
                )

                conv_module = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, use_bias, padding_mode=pad_mode)
                if use_fused:
                    relu_module = nn.ReLU()
                    conv_module = nni.ConvReLU3d(conv_module, relu_module)
                conv_module = conv_module.float()

                self._test_conv_api_impl(
                    module_name, qconv_module, conv_module, batch_size,
                    in_channels_per_group, input_feature_map_size,
                    out_channels_per_group, groups, kernel_size, stride, padding,
                    pad_mode, dilation, X_scale, X_zero_point, W_scale,
                    W_zero_point, Y_scale, Y_zero_point, use_bias, use_fused,
                    use_channelwise, is_reference)

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
        self.checkScriptable(pool_under_test, [[X]])

    def test_batch_norm2d(self):
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
                         msg="BatchNorm2d module API failed")

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
                         msg="BatchNorm3d module API failed")

    def test_layer_norm(self):
        """Tests the correctness of the layernorm module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = torch.nn.LayerNorm(dqX.size()[1:]).float()
        float_mod.weight = torch.nn.Parameter(torch.rand(*dims[1:]))
        float_mod.bias = torch.nn.Parameter(torch.rand(*dims[1:]))

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = nnq.LayerNorm(
            qX.size()[1:], float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        qY = quant_mod(qX)

        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="LayerNorm module API failed, qY_ref\n{} vs qY\n{}"
                         .format(qY_ref, qY))

    def test_group_norm(self):
        """Tests the correctness of the groupnorm module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = torch.nn.GroupNorm(2, 4).float()
        float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
        float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = nnq.GroupNorm(
            2, 2, float_mod.weight, float_mod.bias, y_scale, y_zero_point)
        qY = quant_mod(qX)

        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="GroupNorm module API failed, qY_ref\n{} vs qY\n{}"
                         .format(qY_ref, qY))

    def test_instance_norm(self):
        """Tests the correctness of the instancenorm{n}d modules.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127

        dims_to_modules = [
            ((1, 4, 8), torch.nn.InstanceNorm1d, nnq.InstanceNorm1d),
            ((1, 4, 8, 1), torch.nn.InstanceNorm2d, nnq.InstanceNorm2d),
            ((1, 4, 8, 1, 1), torch.nn.InstanceNorm3d, nnq.InstanceNorm3d),
        ]

        for dim_to_modules in dims_to_modules:
            dims, float_cls, q_cls = dim_to_modules

            X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
            qX = torch.quantize_per_tensor(
                X, x_scale, x_zero_point, dtype=torch.quint8)
            dqX = qX.dequantize()

            float_mod = float_cls(dims[1]).float()
            float_mod.weight = torch.nn.Parameter(torch.rand(dims[1]))
            float_mod.bias = torch.nn.Parameter(torch.rand(dims[1]))

            dqY_ref = float_mod(dqX)
            qY_ref = torch.quantize_per_tensor(
                dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

            quant_mod = q_cls(
                dims[1], float_mod.weight, float_mod.bias, y_scale,
                y_zero_point)
            qY = quant_mod(qX)

            self.assertEqual(
                qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                msg="InstanceNorm module API failed, qY_ref\n{} vs qY\n{}"
                .format(qY_ref, qY))

    def _test_activation_module_impl(self, name, float_module_class, quantized_module_class, extra_kwargs):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        x_scale = 10.0 / 256
        x_zero_point = 0
        y_scale = 5.0 / 256
        y_zero_point = 127
        alpha = 1.5

        dims = (1, 4, 8)

        X = (torch.randn(dims, dtype=torch.float) - 0.5) * 10
        qX = torch.quantize_per_tensor(X, x_scale, x_zero_point, dtype=torch.quint8)
        dqX = qX.dequantize()

        float_mod = float_module_class(**extra_kwargs).float()

        dqY_ref = float_mod(dqX)
        qY_ref = torch.quantize_per_tensor(
            dqY_ref, y_scale, y_zero_point, dtype=torch.quint8)

        quant_mod = quantized_module_class(y_scale, y_zero_point, **extra_kwargs)
        qY = quant_mod(qX)
        self.assertEqual(qY_ref.int_repr().numpy(), qY.int_repr().numpy(),
                         msg="{} module API failed, qY_ref\n{} vs qY\n{}"
                         .format(name, qY_ref, qY))

    def _test_leaky_relu_serialization(self):
        scale_original = 10.0 / 256
        zero_point_original = 1.0

        quant_mod_original = nnq.LeakyReLU(scale_original, zero_point_original)
        state_dict = quant_mod_original.state_dict()

        scale_new = 5.0 / 256
        zero_point_new = 2.0
        quant_mod_new = nnq.LeakyReLU(scale_new, zero_point_new)
        quant_mod_new.load_state_dict(state_dict)

        self.assertEqual(quant_mod_original.scale, quant_mod_new.scale)
        self.assertEqual(quant_mod_original.zero_point, quant_mod_new.zero_point)

    def test_elu(self):
        """Tests the correctness of the ELU module.
        The correctness is defined against the functional implementation.
        """
        self._test_activation_module_impl("ELU", nn.ELU, nnq.ELU, {"alpha": 1.5})

    def test_leaky_relu(self):
        self._test_activation_module_impl("LeakyReLU", nn.LeakyReLU, nnq.LeakyReLU, {"negative_slope": 0.2})
        self._test_leaky_relu_serialization()

    def test_sigmoid(self):
        self._test_activation_module_impl("Sigmoid", nn.Sigmoid, nnq.Sigmoid, {})

    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        set_qconfig=st.booleans(),
    )
    @skipIfNoFBGEMM
    def test_embedding_api(self, num_embeddings, embedding_dim, set_qconfig):
        num_lengths = np.random.randint(1, 6)
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        obs = default_float_qparams_observer()
        obs(weights)
        qparams = obs.calculate_qparams()
        # Quantize the weights to 8bits
        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)
        qemb = nnq.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        qemb.set_weight(qweight)
        qemb(indices)

        # Ensure the module has the correct weights
        self.assertEqual(qweight, qemb.weight())

        w_packed = qemb._packed_params._packed_weight
        module_out = qemb(indices)

        # Call the qembedding operator directly
        ref = torch.ops.quantized.embedding_byte(w_packed, indices, pruned_weights=False)
        self.assertEqual(module_out, ref)
        self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices, None, set_qconfig=False, is_emb_bag=False)


    @given(
        num_embeddings=st.integers(10, 50),
        embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
        num_offsets=st.integers(1, 20),
        set_qconfig=st.booleans(),
    )
    @skipIfNoFBGEMM
    def test_embedding_bag_api(self, num_embeddings, embedding_dim, num_offsets, set_qconfig):
        r"""Test execution and serialization for dynamic quantized embedding_bag modules on int8
        """

        num_lengths = np.random.randint(1, 6)
        lengths = np.random.randint(0, 21, size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)
        indices = torch.from_numpy(np.random.randint(low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

        offsets = lengths_to_offsets(lengths)
        # include the last offset
        offsets = torch.cat((offsets, torch.tensor([indices.size(0)], dtype=torch.long)), 0)
        weights = torch.from_numpy((np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(np.float32))

        for qdtype in [torch.quint8, torch.quint4x2]:
            obs = PerChannelMinMaxObserver(dtype=qdtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(weights)
            # Get the scale and zero point for the weight tensor
            qparams = obs.calculate_qparams()
            # Quantize the weights to 8bits
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=qdtype)
            qemb = nnq.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                    include_last_offset=True, mode='sum', _weight=qweight, dtype=qdtype)
            qemb(indices, offsets)

            # Ensure the module has the correct weights
            self.assertEqual(qweight, qemb.weight())

            w_packed = qemb._packed_params._packed_weight
            module_out = qemb(indices, offsets)

            # Call the qembedding_bag operator directly
            if qdtype == torch.quint8:
                ref = torch.ops.quantized.embedding_bag_byte(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)
            else:
                ref = torch.ops.quantized.embedding_bag_4bit(w_packed, indices, offsets, mode=0,
                                                             per_sample_weights=None,
                                                             include_last_offset=True)

            self.assertEqual(module_out, ref)
            self.checkEmbeddingSerialization(qemb, num_embeddings, embedding_dim, indices,
                                             offsets, set_qconfig, is_emb_bag=True, dtype=qdtype)

class TestDynamicQuantizedModule(QuantizationTestCase):
    @given(
        batch_size=st.integers(1, 5),
        in_features=st.integers(16, 32),
        out_features=st.integers(4, 8),
        use_bias=st.booleans(),
        use_default_observer=st.booleans(),
    )
    @override_qengines
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
        Z_ref = torch.ops.quantized.linear_dynamic(X, W_pack, reduce_range=True)
        self.assertEqual(Z_ref, Z_dq)

        # Test serialization of dynamic quantized Linear Module using state_dict
        model_dict = qlinear.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        for key in model_dict:
            if isinstance(model_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                w_model, b_model = torch.ops.quantized.linear_unpack(model_dict[key])
                w_loaded, b_loaded = torch.ops.quantized.linear_unpack(loaded_dict[key])
                self.assertEqual(w_model, w_loaded)
                self.assertEqual(b_model, b_loaded)
            else:
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

        b = io.BytesIO()
        torch.save(qlinear, b)
        b.seek(0)
        loaded = torch.load(b)
        self.assertEqual(qlinear.weight(), loaded.weight())
        self.assertEqual(qlinear.zero_point, loaded.zero_point)

        # Test JIT
        self.checkScriptable(qlinear, [[X]], check_save_load=True)

        modules_under_test = [torch.nn.Linear, torch.nn.modules.linear._LinearWithBias]
        for mut in modules_under_test:
            # Test from_float
            float_linear = mut(in_features, out_features).float()
            if use_default_observer:
                float_linear.qconfig = torch.quantization.default_dynamic_qconfig
            prepare_dynamic(float_linear)
            float_linear(X.float())
            quantized_float_linear = nnqd.Linear.from_float(float_linear)

            # Smoke test to make sure the module actually runs
            quantized_float_linear(X)

        # Smoke test extra_repr
        self.assertTrue('QuantizedLinear' in str(quantized_float_linear))

    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),
        bidirectional=st.booleans(),
    )
    @override_qengines
    def test_lstm_api(self, dtype, bidirectional):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes
        seq_len = 4
        batch = 2
        input_size = 3
        hidden_size = 7
        num_layers = 2
        bias = True
        weight_keys = []
        bias_keys = []
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                key_name1 = 'weight_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'weight_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                weight_keys.append(key_name1)
                weight_keys.append(key_name2)
                key_name1 = 'bias_ih_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                key_name2 = 'bias_hh_l{layer_idx}{suffix}'.format(layer_idx=layer, suffix=suffix)
                bias_keys.append(key_name1)
                bias_keys.append(key_name2)

        if not (dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack"):
            # fp16 dynamic quant is not supported for qnnpack
            x = torch.randn(seq_len, batch, input_size)
            h = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            c = torch.randn(num_layers * (bidirectional + 1), batch, hidden_size)
            cell_dq = torch.nn.quantized.dynamic.LSTM(input_size=input_size,
                                                      hidden_size=hidden_size,
                                                      num_layers=num_layers,
                                                      bias=bias,
                                                      batch_first=False,
                                                      dropout=0.0,
                                                      bidirectional=bidirectional,
                                                      dtype=dtype)
            ref_dq = torch.nn.quantized.dynamic.LSTM(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers,
                                                     bias=bias,
                                                     batch_first=False,
                                                     dropout=0.0,
                                                     bidirectional=bidirectional,
                                                     dtype=dtype)

            _all_params = ([m.param for m in cell_dq._all_weight_values])
            result = torch.quantized_lstm(x, (h, c),
                                          _all_params,
                                          cell_dq.bias,
                                          cell_dq.num_layers,
                                          float(cell_dq.dropout),
                                          False,
                                          bidirectional,
                                          False,
                                          dtype=dtype,
                                          use_dynamic=True)


            y, (h, c) = cell_dq(x, (h, c))
            self.assertEqual(result[0], y)
            self.assertEqual(result[1], h)
            self.assertEqual(result[2], c)
            x = torch.randn(10, 20, 3)
            self.check_eager_serialization(cell_dq, ref_dq, [x])
            self.check_weight_bias_api(cell_dq, weight_keys, bias_keys)

    @override_qengines
    def test_gru_api(self):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes

        for dtype in [torch.qint8, torch.float16]:
            if dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack":
                # fp16 dynamic quant is not supported for qnnpack
                continue
                # Test default instantiation
            seq_len = 4
            batch = 2
            input_size = 3
            hidden_size = 7
            num_layers = 2
            bias = True
            bidirectional = False

            x = torch.rand(seq_len, batch, input_size)
            h = torch.rand(num_layers * (bidirectional + 1), batch, hidden_size)


            cell_dq = torch.nn.quantized.dynamic.GRU(input_size=input_size,
                                                     hidden_size=hidden_size,
                                                     num_layers=num_layers,
                                                     bias=bias,
                                                     batch_first=False,
                                                     dropout=0.0,
                                                     bidirectional=bidirectional,
                                                     dtype=dtype)

            _all_params = ([m.param for m in cell_dq._all_weight_values])
            result = torch.quantized_gru(x,
                                         h,
                                         _all_params,
                                         cell_dq.bias,
                                         cell_dq.num_layers,
                                         float(cell_dq.dropout),
                                         False,
                                         bidirectional,
                                         False)


            y, h = cell_dq(x, h)
            self.assertEqual(result[0], y, msg="GRU module API failed")
            self.assertEqual(result[1], h, msg="GRU module API failed")

    @given(
        dtype=st.sampled_from([torch.qint8, torch.float16]),
    )
    @override_qengines
    def test_cell_api(self, dtype):
        r"""Test execution and serialization for dynamic quantized lstm modules on int8 and fp16
        """
        # Check that module matches the numerics of the op and ensure that module can be
        # instantiated for all engines and dtypes
        batch = 7
        input_size = 3
        hidden_size = 7
        bias = True

        x = torch.rand(batch, input_size)
        h = torch.rand(batch, hidden_size)
        cell_dict = {'LSTMCell': torch.nn.quantized.dynamic.LSTMCell,
                     'GRUCell': torch.nn.quantized.dynamic.GRUCell,
                     'RNNTanh': torch.nn.quantized.dynamic.RNNCell,
                     'RNNReLU': torch.nn.quantized.dynamic.RNNCell
                     }
        state = {'LSTMCell': (h, h),
                 'GRUCell': h,
                 'RNNTanh': h,
                 'RNNReLU': h}

        qfn_dict = {'LSTMCell': torch.ops.quantized.quantized_lstm_cell_dynamic,
                    'GRUCell': torch.ops.quantized.quantized_gru_cell_dynamic,
                    'RNNTanh': torch.ops.quantized.quantized_rnn_tanh_cell_dynamic,
                    'RNNReLU': torch.ops.quantized.quantized_rnn_relu_cell_dynamic}

        for rnn_type in cell_dict.keys():
            if not (dtype == torch.float16 and torch.backends.quantized.engine == "qnnpack"):
                # fp16 dynamic quant is not supported for qnnpack
                kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'bias': bias, 'dtype': dtype}
                if rnn_type == 'RNNReLU':
                    kwargs['nonlinearity'] = "relu"
                elif rnn_type == 'RNNTanh':
                    kwargs['nonlinearity'] = "tanh"

                cell_dq = cell_dict[rnn_type](**kwargs)
                result = qfn_dict[rnn_type](x, state[rnn_type],
                                            cell_dq._packed_weight_ih, cell_dq._packed_weight_hh,
                                            cell_dq.bias_ih, cell_dq.bias_hh)
                result_module = cell_dq(x, state[rnn_type])
                self.assertEqual(result[0], result_module[0], msg="RNNCell module API failed")
                self.assertEqual(result[1], result_module[1], msg="RNNCell module API failed")
                weight_keys = ['weight_ih', 'weight_hh']
                bias_keys = ['bias_ih', 'bias_hh']
                self.check_eager_serialization(cell_dq, cell_dict[rnn_type](**kwargs), [x])
                self.check_weight_bias_api(cell_dq, weight_keys, bias_keys)
