# Torch
import torch
import torch.nn.functional as F
import torch.nn.quantized.functional as qF

# Standard library
import numpy as np

# Testing utils
from hypothesis import assume, given
from hypothesis import strategies as st
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    _make_conv_test_input,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import (
    IS_PPC,
    TEST_WITH_UBSAN,
)

class TestQuantizedFunctional(QuantizationTestCase):
    def test_relu_api(self):
        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        qY = torch.relu(qX)
        qY_hat = F.relu(qX)
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
           L=st.integers(4, 16),
           out_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 4),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
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
    def test_conv1d_api(
        self, batch_size, in_channels_per_group, L, out_channels_per_group,
        groups, kernel, stride, pad, dilation,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_channelwise, qengine,
    ):
        # Tests the correctness of the conv1d function.
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        input_feature_map_size = (L, )
        kernel_size = (kernel, )
        stride = (stride, )
        padding = (pad, )
        dilation = (dilation, )

        with override_quantized_engine(qengine):
            qconv_fn = qF.conv1d
            conv_fn = F.conv1d
            self._test_conv_api_impl(
                qconv_fn, conv_fn, batch_size, in_channels_per_group,
                input_feature_map_size, out_channels_per_group, groups,
                kernel_size, stride, padding, dilation, X_scale, X_zero_point,
                W_scale, W_zero_point, Y_scale, Y_zero_point, use_bias,
                use_channelwise)

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
