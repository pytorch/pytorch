from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import math
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, init
from torch.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
from torch.nn.modules.utils import _pair
from torch.quantization.qconfig import default_qat_qconfig
import torch.backends.mkldnn
from torch.testing._internal.common_utils import TestCase
from hypothesis import given, seed
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from functools import reduce


class _ReferenceConvBnNd(torch.nn.Conv2d, torch.nn.modules.conv._ConvNd):
    """
    Conv-BN fusion implemented with explicit folding. Useful
    to verify numerical equivalency with non-folded version.
    """
    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups,
                 bias,
                 padding_mode,
                 # BatchNormNd args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.eps = eps
        self.momentum = momentum
        self.freeze_bn = freeze_bn if self.training else True
        self.num_features = out_channels
        self.gamma = nn.Parameter(torch.Tensor(out_channels))
        self.beta = nn.Parameter(torch.Tensor(out_channels))
        self.affine = True
        self.track_running_stats = True
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.activation_post_process = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super(_ReferenceConvBnNd, self).reset_parameters()
        # A hack to avoid resetting on undefined parameters
        if hasattr(self, 'gamma'):
            self.reset_bn_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        return self

    def _forward(self, input):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and not self.freeze_bn and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # we use running statistics from the previous batch, so this is an
        # approximation of the approach mentioned in the whitepaper, but we only
        # need to do one convolution in this case instead of two
        running_std = torch.sqrt(self.running_var + self.eps)
        scale_factor = self.gamma / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        conv = self._conv_forward(input, self.weight_fake_quant(scaled_weight))

        if self.training and not self.freeze_bn:
            # recovering original conv to get original batch_mean and batch_var
            if self.bias is not None:
                conv_orig = conv / scale_factor.reshape([1, -1, 1, 1]) + self.bias.reshape([1, -1, 1, 1])
            else:
                conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
            batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)
            n = float(conv_orig.numel() / conv_orig.size()[1])
            unbiased_batch_var = batch_var * (n / (n - 1))
            batch_rstd = torch.ones_like(batch_var, memory_format=torch.contiguous_format) / torch.sqrt(batch_var + self.eps)

            conv = (self.gamma * batch_rstd).reshape([1, -1, 1, 1]) * conv_orig + \
                (self.beta - self.gamma * batch_rstd * batch_mean).reshape([1, -1, 1, 1])
            self.running_mean = exponential_average_factor * batch_mean.detach() + \
                (1 - exponential_average_factor) * self.running_mean
            self.running_var = exponential_average_factor * unbiased_batch_var.detach() + \
                (1 - exponential_average_factor) * self.running_var
        else:
            if self.bias is None:
                conv = conv + (self.beta - self.gamma * self.running_mean /
                               running_std).reshape([1, -1, 1, 1])
            else:
                conv = conv + (self.gamma * (self.bias - self.running_mean) / running_std + self.beta).reshape([1, -1, 1, 1])
        return conv

    def extra_repr(self):
        # TODO(jerryzh): extend
        return super(_ReferenceConvBnNd, self).extra_repr()

    def forward(self, input):
        return self.activation_post_process(self._forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict
            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.gamma = bn.weight
        qat_convbn.beta = bn.bias
        qat_convbn.running_mean = bn.running_mean
        qat_convbn.running_var = bn.running_var
        qat_convbn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

class _ReferenceConvBn2d(_ReferenceConvBnNd, nn.Conv2d):
    _FLOAT_MODULE = torch.nn.intrinsic.ConvBn2d

    def __init__(self,
                 # ConvNd args
                 in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None,
                 padding_mode='zeros',
                 # BatchNorm2d args
                 # num_features: out_channels
                 eps=1e-05, momentum=0.1,
                 # affine: True
                 # track_running_stats: True
                 # Args for this module
                 freeze_bn=False,
                 qconfig=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ReferenceConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride,
                                    padding, dilation, False, _pair(0), groups, bias, padding_mode,
                                    eps, momentum, freeze_bn, qconfig)

class TestQATModule(TestCase):

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
                copy.deepcopy(bn_op),
                input_channels,
                output_channels,
                (kernel_h, kernel_w),
                (stride_h, stride_w),
                (pad_h, pad_w),
                (dilation_h, dilation_w),
                groups,
                None,  # bias
                padding_mode,
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
            bn_op.running_mean = qat_op.bn.running_mean.clone()
            bn_op.running_var = qat_op.bn.running_var.clone()
            bn_op.weight = torch.nn.Parameter(qat_op.bn.weight.detach())
            bn_op.bias = torch.nn.Parameter(qat_op.bn.bias.detach())

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
                gamma_grad_actual = qat_op.bn.weight.grad.cpu()
                beta_grad_actual = qat_op.bn.bias.grad.cpu()
                running_mean_actual = qat_op.bn.running_mean
                running_var_actual = qat_op.bn.running_var
                num_batches_tracked_actual = qat_op.bn.num_batches_tracked
                precision = 1e-10
                self.assertEqual(input_grad_ref, input_grad_actual, atol=precision)
                self.assertEqual(weight_grad_ref, weight_grad_actual, atol=precision)
                self.assertEqual(gamma_grad_ref, gamma_grad_actual, atol=precision)
                self.assertEqual(beta_grad_ref, beta_grad_actual, atol=precision)
                self.assertEqual(num_batches_tracked_ref, num_batches_tracked_actual, atol=precision)
                self.assertEqual(running_mean_ref, running_mean_actual, atol=precision)
                self.assertEqual(running_var_ref, running_var_actual, atol=precision)

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
           eps=st.sampled_from([1e-5, 1e-4, 1e-3]),
           momentum=st.sampled_from([0.1, 0.2, 0.3]),
           freeze_bn=st.booleans())
    def test_conv_bn_folded_vs_unfolded(
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
            eps,
            momentum,
            freeze_bn
    ):
        # TODO: remove before land
        if False:
            batch_size = 2
            input_channels_per_group = 2
            height = 5
            width = 5
            output_channels_per_group = 2
            groups = 1
            kernel_h = 1
            kernel_w = 1
            stride_h = 1
            stride_w = 1
            pad_h = 0
            pad_w = 0
            dilation = 1
            padding_mode = 'zeros'
            eps = 1e-5
            momentum = 0.1
            freeze_bn = False
            freeze_bn = True
        # **** WARNING: This is used to temporarily disable MKL-DNN convolution due
        # to a bug: https://github.com/pytorch/pytorch/issues/23825
        # Once this bug is fixed, this context manager as well as its callsites
        # should be removed!
        with torch.backends.mkldnn.flags(enabled=False):
            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            dilation_h = dilation_w = dilation

            bn_op = BatchNorm2d(output_channels, eps, momentum).to(dtype=torch.double)

            qat_op = ConvBn2d(
                copy.deepcopy(bn_op),
                input_channels,
                output_channels,
                (kernel_h, kernel_w),
                (stride_h, stride_w),
                (pad_h, pad_w),
                (dilation_h, dilation_w),
                groups,
                None,  # bias
                padding_mode,
                freeze_bn=freeze_bn,
                qconfig=default_qat_qconfig
            ).to(dtype=torch.double)

            qat_ref_op = _ReferenceConvBn2d(
                input_channels,
                output_channels,
                (kernel_h, kernel_w),
                (stride_h, stride_w),
                (pad_h, pad_w),
                (dilation_h, dilation_w),
                groups,
                None,  # bias
                padding_mode,
                eps,
                momentum,
                freeze_bn=freeze_bn,
                qconfig=default_qat_qconfig
            ).to(dtype=torch.double)

            qat_op.apply(torch.quantization.disable_fake_quant)
            qat_ref_op.apply(torch.quantization.disable_fake_quant)

            # align inputs and internal parameters
            qat_ref_op.weight = torch.nn.Parameter(qat_op.weight.detach().clone())
            qat_ref_op.running_mean = qat_op.bn.running_mean.clone()
            qat_ref_op.running_var = qat_op.bn.running_var.clone()
            qat_ref_op.gamma = torch.nn.Parameter(qat_op.bn.weight.detach().clone())
            qat_ref_op.beta = torch.nn.Parameter(qat_op.bn.bias.detach().clone())
            # if qat_op.bias != None:
                # qat_ref_op.bias = torch.nn.Parameter(qat_op.bias.detach().clone())

            lr = 0.01
            qat_op_optim = torch.optim.SGD(qat_op.parameters(), lr=lr)
            qat_ref_op_optim = torch.optim.SGD(qat_ref_op.parameters(), lr=lr)

            for i in range(5):
                qat_op_optim.zero_grad()
                qat_ref_op_optim.zero_grad()

                input = torch.randn(batch_size, input_channels, height, width, dtype=torch.double, requires_grad=True)
                input_clone = input.clone().detach().requires_grad_()

                if i > 2:
                    qat_op.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                    qat_ref_op.freeze_bn_stats()

                if i > 3:
                    qat_op.apply(torch.quantization.disable_observer)
                    qat_ref_op.apply(torch.quantization.disable_observer)

                result_ref = qat_ref_op(input)
                result_actual = qat_op(input_clone)
                self.assertEqual(result_ref, result_actual)

                # backward
                dout = torch.randn(result_ref.size(), dtype=torch.double) + 10.0

                loss = (result_ref - dout).sum()
                loss.backward()
                input_grad_ref = input.grad.cpu()
                weight_grad_ref = qat_ref_op.weight.grad.cpu()
                gamma_grad_ref = qat_ref_op.gamma.grad.cpu()
                beta_grad_ref = qat_ref_op.beta.grad.cpu()
                running_mean_ref = qat_ref_op.running_mean
                running_var_ref = qat_ref_op.running_var
                num_batches_tracked_ref = qat_ref_op.num_batches_tracked

                loss = (result_actual - dout).sum()
                loss.backward()
                input_grad_actual = input_clone.grad.cpu()
                weight_grad_actual = qat_op.weight.grad.cpu()
                gamma_grad_actual = qat_op.bn.weight.grad.cpu()
                beta_grad_actual = qat_op.bn.bias.grad.cpu()
                running_mean_actual = qat_op.bn.running_mean
                running_var_actual = qat_op.bn.running_var
                num_batches_tracked_actual = qat_op.bn.num_batches_tracked

                precision = 1e-5
                self.assertEqual(input_grad_ref, input_grad_actual, atol=precision)
                self.assertEqual(weight_grad_ref, weight_grad_actual, atol=precision)
                self.assertEqual(gamma_grad_ref, gamma_grad_actual, atol=precision)
                self.assertEqual(beta_grad_ref, beta_grad_actual, atol=precision)
                self.assertEqual(num_batches_tracked_ref, num_batches_tracked_actual, atol=precision)
                self.assertEqual(running_mean_ref, running_mean_actual, atol=precision)
                self.assertEqual(running_var_ref, running_var_actual, atol=precision)

                qat_op_optim.step()
                qat_ref_op_optim.step()
