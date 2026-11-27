# Owner(s): ["oncall: mobile"]

import unittest

from hypothesis import assume, given, strategies as st

import torch
import torch.backends.xnnpack
import torch.testing._internal.hypothesis_utils as hu
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TSAN, TestCase


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN fails with XNNPACK. Does not seem to have a good reason for failures.",
)
class TestXNNPACKOps(TestCase):
    @unittest.skip(
        "Fails on some platforms, see https://github.com/pytorch/pytorch/issues/73488"
    )
    @given(
        batch_size=st.integers(0, 3),
        data_shape=hu.array_shapes(1, 3, 2, 64),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear(self, batch_size, data_shape, weight_output_dim, use_bias):
        data_shape = [batch_size] + list(data_shape)
        input_data = torch.rand(data_shape)
        weight = torch.rand((weight_output_dim, data_shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        ref_result = F.linear(input_data, weight, bias)
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        input_size=st.integers(2, 32),
        weight_output_dim=st.integers(2, 64),
        use_bias=st.booleans(),
    )
    def test_linear_1d_input(self, input_size, weight_output_dim, use_bias):
        input_data = torch.rand(input_size)
        weight = torch.rand((weight_output_dim, input_data.shape[-1]))
        if use_bias:
            bias = torch.rand(weight_output_dim)
        else:
            bias = None
        ref_result = F.linear(input_data, weight, bias)
        packed_weight_bias = torch.ops.prepacked.linear_clamp_prepack(weight, bias)
        output_linearprepacked = torch.ops.prepacked.linear_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result, output_linearprepacked, rtol=1e-2, atol=1e-3
        )

    @given(
        batch_size=st.integers(0, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation,
        use_bias,
        format,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (output_channels, input_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        ref_result = F.conv2d(
            input_data, weight, bias, strides, paddings, dilations, groups
        )
        packed_weight_bias = torch.ops.prepacked.conv2d_clamp_prepack(
            weight, bias, strides, paddings, dilations, groups
        )
        xnnpack_result = torch.ops.prepacked.conv2d_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(ref_result, xnnpack_result, rtol=1e-2, atol=1e-3)

    @given(
        batch_size=st.integers(1, 3),
        input_channels_per_group=st.integers(1, 32),
        height=st.integers(5, 64),
        width=st.integers(5, 64),
        output_channels_per_group=st.integers(1, 32),
        groups=st.integers(1, 16),
        kernel_h=st.integers(1, 7),
        kernel_w=st.integers(1, 7),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.integers(0, 2),
        pad_w=st.integers(0, 2),
        output_pad_h=st.integers(0, 2),
        output_pad_w=st.integers(0, 2),
        dilation=st.integers(1, 2),
        use_bias=st.booleans(),
        format=st.sampled_from(
            [None, torch.preserve_format, torch.contiguous_format, torch.channels_last]
        ),
    )
    def test_conv2d_transpose(
        self,
        batch_size,
        input_channels_per_group,
        height,
        width,
        output_channels_per_group,
        groups,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        output_pad_h,
        output_pad_w,
        dilation,
        use_bias,
        format,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        output_paddings = (output_pad_h, output_pad_w)
        dilations = (dilation, dilation)
        assume(height + 2 * paddings[0] >= dilations[0] * (kernels[0] - 1) + 1)
        assume(width + 2 * paddings[1] >= dilations[1] * (kernels[1] - 1) + 1)
        assume((output_pad_h < stride_h) and (output_pad_h < dilation))
        assume((output_pad_w < stride_w) and (output_pad_w < dilation))

        input_data = torch.rand((batch_size, input_channels, height, width))
        if format is not None:
            input_data = input_data.contiguous(memory_format=format)
        weight = torch.rand(
            (input_channels, output_channels_per_group, kernel_h, kernel_w)
        )
        bias = None
        if use_bias:
            bias = torch.rand(output_channels)

        # Note that groups/dilation is in reverse order from conv2d
        ref_result = F.conv_transpose2d(
            input_data,
            weight,
            bias,
            strides,
            paddings,
            output_paddings,
            groups,
            dilation,
        )
        packed_weight_bias = torch.ops.prepacked.conv2d_transpose_clamp_prepack(
            weight, bias, strides, paddings, output_paddings, dilations, groups
        )
        xnnpack_result = torch.ops.prepacked.conv2d_transpose_clamp_run(
            input_data, packed_weight_bias
        )
        torch.testing.assert_close(
            ref_result.contiguous(), xnnpack_result.contiguous(), rtol=1e-2, atol=1e-3
        )


if __name__ == "__main__":
    run_tests()
