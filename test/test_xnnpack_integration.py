# Owner(s): ["oncall: mobile"]

import itertools
import unittest

from hypothesis import assume, given, strategies as st

import torch
import torch.backends.xnnpack
import torch.testing._internal.hypothesis_utils as hu
from torch.nn import functional as F
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    slowTest,
    TEST_WITH_TSAN,
    TestCase,
)
from torch.utils.mobile_optimizer import optimize_for_mobile


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


@unittest.skipUnless(
    torch.backends.xnnpack.enabled,
    " XNNPACK must be enabled for these tests. Please build with USE_XNNPACK=1.",
)
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class TestXNNPACKMobileOptimizer(TestCase):
    @staticmethod
    def _check_mobile_optimizer(module, pattern_counts, data_shape):
        input_data = torch.normal(1, 20, size=data_shape)
        module.eval()
        expected = module(input_data)
        for fixture_kind in ("script", "trace"):
            if fixture_kind == "script":
                fixture = torch.jit._script.script(module)
            else:
                fixture = torch.jit._trace.trace(module, input_data)
            optimized = optimize_for_mobile(fixture)
            for pattern, count in pattern_counts.items():
                if count == 0:
                    FileCheck().check(pattern).run(optimized.graph)
                elif count == -1:
                    FileCheck().check_not(pattern).run(optimized.graph)
                else:
                    FileCheck().check_count(pattern, count, exactly=True).run(
                        optimized.graph
                    )
            torch.testing.assert_close(
                expected, optimized(input_data), rtol=1e-2, atol=1e-3
            )

    @unittest.skipIf(IS_FBCODE, "T137513244")
    def test_conv1d_basic(self):
        batch_size_list = range(1, 3)
        input_channels_per_group_list = range(10, 12)
        width_list = range(10, 12)
        output_channels_per_group_list = range(10, 12)
        groups_list = range(1, 3)
        kernel_list = range(1, 4)
        stride_list = range(1, 3)
        padding_list = range(3)
        dilation_list = range(1, 3)
        for hparams in itertools.product(
            batch_size_list,
            input_channels_per_group_list,
            width_list,
            output_channels_per_group_list,
            groups_list,
            kernel_list,
            stride_list,
            padding_list,
            dilation_list,
        ):
            (
                batch_size,
                input_channels_per_group,
                width,
                output_channels_per_group,
                groups,
                kernel,
                stride,
                padding,
                dilation,
            ) = hparams
            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            conv_weight_shape = (output_channels, input_channels_per_group, kernel)
            conv_bias_shape = output_channels

            class Conv1D(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.rand(conv_weight_shape), requires_grad=False
                    )
                    self.bias = torch.nn.Parameter(
                        torch.rand(conv_bias_shape), requires_grad=False
                    )
                    self.stride = stride
                    self.padding = padding
                    self.dilation = dilation
                    self.groups = groups

                def forward(self, x):
                    return F.conv1d(
                        x,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )

            data_shape = (batch_size, input_channels, width)
            pattern_count_optimized_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": -1,
                "prepacked::conv2d_clamp_prepack": -1,
                "prepacked::conv2d_clamp_run": 1,
            }
            TestXNNPACKMobileOptimizer._check_mobile_optimizer(
                Conv1D(), pattern_count_optimized_map, data_shape
            )

    @slowTest
    def test_conv1d_with_relu_fc(self):
        batch_size_list = range(1, 3)
        input_channels_per_group_list = range(10, 12)
        width_list = range(10, 12)
        output_channels_per_group_list = range(10, 12)
        groups_list = range(1, 3)
        kernel_list = range(1, 4)
        stride_list = range(1, 3)
        padding_list = range(3)
        dilation_list = range(1, 3)
        output_features_list = range(1, 3)
        for hparams in itertools.product(
            batch_size_list,
            input_channels_per_group_list,
            width_list,
            output_channels_per_group_list,
            groups_list,
            kernel_list,
            stride_list,
            padding_list,
            dilation_list,
            output_features_list,
        ):
            (
                batch_size,
                input_channels_per_group,
                width,
                output_channels_per_group,
                groups,
                kernel,
                stride,
                padding,
                dilation,
                output_features,
            ) = hparams
            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            conv_weight_shape = (output_channels, input_channels_per_group, kernel)
            conv_bias_shape = output_channels
            conv_output_width = (
                int((width + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
            )
            fc_weight_shape = (output_features, output_channels * conv_output_width)
            fc_bias_shape = output_features

            class Net(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv_weight = torch.nn.Parameter(
                        torch.rand(conv_weight_shape), requires_grad=False
                    )
                    self.conv_bias = torch.nn.Parameter(
                        torch.rand(conv_bias_shape), requires_grad=False
                    )
                    self.stride = stride
                    self.padding = padding
                    self.dilation = dilation
                    self.groups = groups
                    self.fc_weight = torch.nn.Parameter(
                        torch.rand(fc_weight_shape), requires_grad=False
                    )
                    self.fc_bias = torch.nn.Parameter(
                        torch.rand(fc_bias_shape), requires_grad=False
                    )

                def forward(self, x):
                    x = F.conv1d(
                        x,
                        self.conv_weight,
                        self.conv_bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                    x = F.relu(x)
                    x = x.view(x.size(0), -1)
                    x = F.linear(x, self.fc_weight, self.fc_bias)
                    return x

            data_shape = (batch_size, input_channels, width)
            pattern_count_optimized_map = {
                "Tensor = aten::conv1d": -1,
                "Tensor = aten::conv2d": -1,
                "prepacked::conv2d_clamp_prepack": -1,
                "prepacked::conv2d_clamp_run": 1,
            }
            TestXNNPACKMobileOptimizer._check_mobile_optimizer(
                Net(), pattern_count_optimized_map, data_shape
            )


if __name__ == "__main__":
    run_tests()
