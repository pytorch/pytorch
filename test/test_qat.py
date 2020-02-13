from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torch.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
from torch.quantization.qconfig import default_qat_qconfig
import torch.backends.mkldnn
from torch.testing._internal.common_utils import TestCase, run_tests
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from functools import reduce


class IntrinsicQATModuleTest(TestCase):

    @given(batch_size=st.integers(2, 4),
           input_channels_per_group=st.sampled_from([2, 3, 4]),
           height=st.integers(5, 10),
           width=st.integers(5, 10),
           output_channels_per_group=st.sampled_from([2, 3]),
           groups=st.integers(1, 3),
           kernel_h=st.integers(1, 3),
           kernel_w=st.integers(1, 3),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 1),
           padding_mode=st.sampled_from(['zeros', 'circular']),
           use_relu=st.booleans(),
           eps=st.sampled_from([1e-5, 1e-4, 1e-3]),
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
        # **** WARNING: This is used to temporarily disable MKL-DNN convolution due
        # to a bug: https://github.com/pytorch/pytorch/issues/23825
        # Once this bug is fixed, this context manager as well as its callsites
        # should be removed!
        with torch.backends.mkldnn.flags(enabled=False):
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
            ).to(dtype=torch.double)
            bn_op = BatchNorm2d(output_channels, eps, momentum).to(dtype=torch.double)
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
                freeze_bn=True,
                qconfig=default_qat_qconfig
            ).to(dtype=torch.double)
            qat_op.apply(torch.quantization.disable_fake_quant)
            if freeze_bn:
                qat_op.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            else:
                qat_op.apply(torch.nn.intrinsic.qat.update_bn_stats)

            # align inputs and internal parameters
            input = torch.randn(batch_size, input_channels, height, width, dtype=torch.double, requires_grad=True)
            conv_op.weight = torch.nn.Parameter(qat_op.weight.detach())
            bn_op.running_mean = qat_op.running_mean.clone()
            bn_op.running_var = qat_op.running_var.clone()
            bn_op.weight = torch.nn.Parameter(qat_op.gamma.detach())
            bn_op.bias = torch.nn.Parameter(qat_op.beta.detach())

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

            input_clone = input.clone().detach().requires_grad_()
            for i in range(2):
                result_ref = ref_op(input)
                result_actual = qat_op(input_clone)
                self.assertEqual(result_ref, result_actual)

                # backward
                dout = torch.randn(result_ref.size(), dtype=torch.double)
                loss = (result_ref - dout).sum()
                loss.backward()
                input_grad_ref = input.grad.cpu()
                weight_grad_ref = conv_op.weight.grad.cpu()
                gamma_grad_ref = bn_op.weight.grad.cpu()
                beta_grad_ref = bn_op.bias.grad.cpu()
                running_mean_ref = bn_op.running_mean
                running_var_ref = bn_op.running_var
                num_batches_tracked_ref = bn_op.num_batches_tracked
                loss = (result_actual - dout).sum()
                loss.backward()
                input_grad_actual = input_clone.grad.cpu()
                weight_grad_actual = qat_op.weight.grad.cpu()
                gamma_grad_actual = qat_op.gamma.grad.cpu()
                beta_grad_actual = qat_op.beta.grad.cpu()
                running_mean_actual = qat_op.running_mean
                running_var_actual = qat_op.running_var
                num_batches_tracked_actual = qat_op.num_batches_tracked
                precision = 1e-10
                self.assertEqual(input_grad_ref, input_grad_actual, prec=precision)
                self.assertEqual(weight_grad_ref, weight_grad_actual, prec=precision)
                self.assertEqual(gamma_grad_ref, gamma_grad_actual, prec=precision)
                self.assertEqual(beta_grad_ref, beta_grad_actual, prec=precision)
                self.assertEqual(num_batches_tracked_ref, num_batches_tracked_actual, prec=precision)
                self.assertEqual(running_mean_ref, running_mean_actual, prec=precision)
                self.assertEqual(running_var_ref, running_var_actual, prec=precision)


if __name__ == '__main__':
    run_tests()
