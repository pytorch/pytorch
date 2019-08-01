from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torch.nn._intrinsic.qat import ConvBn2d, ConvBnReLU2d
from torch.quantization.QConfig import default_qat_qconfig
from torch.nn import Parameter
from common_utils import TestCase, run_tests
from hypothesis import given
from hypothesis import strategies as st
from functools import reduce


class IntrinsicQATModuleTest(TestCase):

    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 3),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 1),
           padding_mode=st.sampled_from(['zeros', 'circular']),
           use_relu=st.booleans(),
           eps=st.sampled_from([1e-5, 1e-4, 1e-3, 0.01, 0.1]),
           momentum=st.sampled_from([0.1, 0.2, 0.3]),
           freeze_bn=st.booleans())
    def test_conv_bn_relu(
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
            padding_mode,
            use_relu,
            eps,
            momentum,
            freeze_bn
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        dilation_h = dilation_w = dilation

        conv_op = Conv2d(
            input_channels,
            output_channels,
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (pad_h, pad_w),
            (dilation_h, dilation_w),
            groups,
            False,  # No bias
            padding_mode
        ).to(dtype=torch.float)
        bn_op = BatchNorm2d(output_channels, eps, momentum).to(dtype=torch.float)
        relu_op = ReLU()

        cls = ConvBnReLU2d if use_relu else ConvBn2d
        qat_op = cls(
            input_channels,
            output_channels,
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (pad_h, pad_w),
            (dilation_h, dilation_w),
            groups,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            default_qat_qconfig.activation,
            default_qat_qconfig.weight
        ).to(dtype=torch.float).disable_fake_quant()

        # align inputs and internal parameters
        input = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)
        input.requires_grad_()
        conv_op.weight = Parameter(qat_op.weight)
        bn_op.running_mean = qat_op.running_mean
        bn_op.running_var = qat_op.running_var
        bn_op.weight = qat_op.gamma
        bn_op.bias = qat_op.beta

        def compose(functions):
            # functions are reversed for natural reading order
            return reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)

        if not use_relu:
            def relu_op(x):
                return x

        if freeze_bn:
            def ref_op(x):
                x = conv_op(x)
                x = (x - bn_op.running_mean.reshape([1, -1, 1, 1])) * \
                    (bn_op.weight / torch.sqrt(bn_op.running_var + bn_op.eps)) \
                    .reshape([1, -1, 1, 1]) + bn_op.bias.reshape([1, -1, 1, 1])
                x = relu_op(x)
                return x
        else:
            ref_op = compose([conv_op, bn_op, relu_op])

        result_ref = ref_op(input)
        result_actual = qat_op(input)
        self.assertEqual(result_ref, result_actual)

        # backward
        dout = torch.randn(result_ref.size(), dtype=torch.float)
        result_actual.backward(dout, retain_graph=True)
        grad_ref = input.grad.cpu()
        result_actual.backward(dout)
        grad_actual = input.grad.cpu()
        self.assertEqual(grad_ref, grad_actual)

if __name__ == '__main__':
    run_tests()
