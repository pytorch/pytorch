from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import unittest

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.jit
import torch.backends.mkldnn
from torch.utils import mkldnn as mkldnn_utils
from torch.testing._internal.common_utils import TestCase, run_tests, \
    TemporaryFileName, TEST_NUMPY

from torch.autograd.gradcheck import gradgradcheck, gradcheck
from hypothesis import given, settings
from hypothesis import strategies as st

if TEST_NUMPY:
    import numpy as np

dtype2prec = {torch.float: 1e-5, torch.bfloat16: 1e-2}

# Comment the line below to find out the CI machines having MKL-DNN build disabled
@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnn(TestCase):
    @given(stype=st.sampled_from((torch.float, torch.bfloat16)),
           otype=st.sampled_from((torch.float, torch.bfloat16)))
    def test_conversion(self, stype, otype):
        for cpu_tensor in [torch.randn((1, 2, 3, 4),
                                       dtype=torch.float, device=torch.device('cpu')),
                           torch.randn((1, 2, 3, 4, 5),
                                       dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
            cpu_tensor.requires_grad_()
            mkldnn_tensor = cpu_tensor.to_mkldnn(stype)
            cpu_tensor_1 = mkldnn_tensor.to_dense(otype)
            self.assertEqual(cpu_tensor, cpu_tensor_1,
                dtype2prec[torch.bfloat16 if torch.bfloat16 in [stype, otype] else torch.float])
            self.assertEqual(mkldnn_tensor.dtype, stype)
            self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
            self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
            self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
            self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.to(stype).element_size())
            self.assertRaisesRegex(RuntimeError,
                                   "Cannot access data pointer of Tensor that doesn't have storage",
                                   lambda: mkldnn_tensor.data_ptr() != 0)

    def test_unsupported(self):
        # unsupported types and unsupported types with gpu
        for dtype in [torch.double, torch.half, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_mkldnn()
            if torch.cuda.is_available():
                with self.assertRaises(RuntimeError) as context:
                    torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cuda')).to_mkldnn()
        # supported type with gpu
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cuda')).to_mkldnn()
        # some factory functions
        for creator in [torch.ones, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError) as context:
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._mkldnn)

    def test_autograd_to_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        def func(root):
            return root.to_mkldnn().to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')
        self.assertWarnsRegex(lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')

    def test_autograd_from_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        def func(root):
            return root.to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')

    def test_detach(self):
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        detach = root.detach()
        self.assertEqual((4, 5), detach.size())
        self.assertFalse(detach.requires_grad)
        self.assertTrue(root.requires_grad)

        detach_ = root.detach_()
        self.assertEqual((4, 5), detach_.size())
        self.assertFalse(detach_.requires_grad)
        self.assertFalse(root.requires_grad)

    def test_repr(self):
        self.assertTrue("layout=torch._mkldnn" in str(torch.randn((1, 2, 3, 4),
                                                                  dtype=torch.float, device=torch.device('cpu')).to_mkldnn()))

    def _test_conv_base(self, func):
        if func is torch.nn.Conv2d:
            ndims = 2
            transpose = False
        elif func is torch.nn.Conv3d:
            ndims = 3
            transpose = False
        elif func is torch.nn.ConvTranspose2d:
            ndims = 2
            transpose = True
        elif func is torch.nn.ConvTranspose3d:
            ndims = 3
            transpose = True
        else:
            raise RuntimeError(str(func) + ' is not supported')

        @settings(deadline=None)
        @given(
            batch=st.integers(1, 10),
            height=st.integers(8, 32),
            width=st.integers(8, 32),
            depth=st.integers(8, 32),
            in_channels_per_group=st.sampled_from([2, 4, 7, 8, 16, 32]),
            out_channels_per_group=st.sampled_from([2, 4, 7, 8, 16, 32]),
            groups=st.integers(1, 4),
            kernel=st.integers(1, 5),
            stride=st.integers(2, 3),
            pad=st.integers(0, 2),
            opad=st.integers(0, 1),
            dilation=st.integers(1, 2),
            use_bias=st.booleans())
        def _test_conv(
            batch,
            height,
            width,
            depth,
            in_channels_per_group,
            out_channels_per_group,
            groups,
            kernel,
            stride,
            pad,
            opad,
            dilation,
            use_bias
        ):
            ic = in_channels_per_group * groups
            oc = out_channels_per_group * groups
            if ndims == 2:
                x = torch.randn(batch, ic, height, width)
            elif ndims == 3:
                x = torch.randn(batch, ic, depth, height, width)

            kwargs = {
                'in_channels': ic,
                'out_channels': oc,
                'kernel_size': kernel,
                'stride': stride,
                'padding': pad,
                'bias': use_bias,
                'dilation': dilation,
                'groups': groups,
            }
            if transpose:
                kwargs['output_padding'] = opad

            module = func(**kwargs).float()
            module_mkldnn = copy.deepcopy(module)

            x_aten = x.clone().requires_grad_()
            x_mkldnn = x.clone().to_mkldnn().requires_grad_()

            with torch.backends.mkldnn.flags(enabled=False):
                y_aten = module(x_aten)
                y_aten.sum().backward()

            y_mkldnn = module_mkldnn(x_mkldnn).to_dense()
            y_mkldnn.sum().backward()

            np.testing.assert_allclose(
                y_aten.detach(), y_mkldnn.detach(), rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(
                module.weight.grad, module_mkldnn.weight.grad, rtol=1e-3, atol=1e-3)
            self.assertEqual(x_aten.grad, x_mkldnn.grad.to_dense())
            if use_bias:
                self.assertEqual(module.bias.grad, module_mkldnn.bias.grad)

        _test_conv()

    def test_conv2d(self):
        self._test_conv_base(torch.nn.Conv2d)

    def test_conv3d(self):
        self._test_conv_base(torch.nn.Conv3d)

    def test_deconv2d(self):
        self._test_conv_base(torch.nn.ConvTranspose2d)

    def test_deconv3d(self):
        self._test_conv_base(torch.nn.ConvTranspose3d)

    def test_conv2d_jit(self):
        for groups in [1, 4]:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 224, 224, dtype=torch.float32)
            for bias in [True, False]:
                conv2d = torch.nn.Conv2d(in_channels=C,
                                         out_channels=M,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=bias,
                                         groups=groups).float()
                mkldnn_conv2d = mkldnn_utils.to_mkldnn(copy.deepcopy(conv2d))
                self._test_serialization(mkldnn_conv2d, (x.to_mkldnn(),))
                self._test_tracing(mkldnn_conv2d, (x.to_mkldnn(),))

    def test_conv3d_jit(self):
        for groups in [1, 4]:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 112, 112, 112, dtype=torch.float32)
            for bias in [True, False]:
                conv3d = torch.nn.Conv3d(in_channels=C,
                                         out_channels=M,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=bias,
                                         groups=groups).float()
                mkldnn_conv3d = mkldnn_utils.to_mkldnn(copy.deepcopy(conv3d))
                self._test_serialization(mkldnn_conv3d, (x.to_mkldnn(),))
                self._test_tracing(mkldnn_conv3d, (x.to_mkldnn(),))
    
    def test_conv2d_bf16(self):
        for groups in [1, 4]:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 224, 224, dtype=torch.float32)
            x_bf16 = x.bfloat16()
            for bias in [True, False]:
                conv2d = torch.nn.Conv2d(in_channels=C,
                                         out_channels=M,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=bias,
                                         groups=groups)
                mkldnn_conv2d = mkldnn_utils.to_mkldnn(copy.deepcopy(conv2d))
                mkldnn_conv2d_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(conv2d), torch.bfloat16)
                y = mkldnn_conv2d(x.to_mkldnn()).to_dense()
                y_bf16 = mkldnn_conv2d_bf16(x_bf16.to_mkldnn()).to_dense()
                self.assertEqual(y, y_bf16, prec=0.1)

    def test_conv3d_bf16(self):
        for groups in [1, 4]:
            N = torch.randint(3, 10, (1,)).item()
            C = torch.randint(1, 3, (1,)).item() * groups
            M = torch.randint(1, 3, (1,)).item() * groups
            x = torch.randn(N, C, 112, 112, 112, dtype=torch.float32)
            x_bf16 = x.bfloat16()
            for bias in [True, False]:
                conv3d = torch.nn.Conv3d(in_channels=C,
                                         out_channels=M,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         bias=bias,
                                         groups=groups)
                mkldnn_conv3d = mkldnn_utils.to_mkldnn(copy.deepcopy(conv3d))
                mkldnn_conv3d_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(conv3d), torch.bfloat16)
                y = mkldnn_conv3d(x.to_mkldnn()).to_dense()
                y_bf16 = mkldnn_conv3d_bf16(x_bf16.to_mkldnn()).to_dense()
                self.assertEqual(y, y_bf16, prec=0.1)

    def test_conv2d_legacy_jit_model(self):
        g = 4
        conv2d = torch.nn.Conv2d(16, 16, 3, groups=g)
        conv2d_mkldnn = torch.utils.mkldnn.to_mkldnn(conv2d)

        # contrive legacy conv2d module with a 5-d weight
        o, i, h, w = conv2d.weight.shape
        weight_5d = conv2d.weight.reshape((g, o // g, i, h, w))
        conv2d_mkldnn.weight = weight_5d.to_mkldnn()

        x = torch.randn(1, 16, 8, 8)

        with TemporaryFileName() as fname:
            torch.jit.save(conv2d_mkldnn, fname)
            conv2d_loaded = torch.jit.load(fname)

            self.assertEqual(conv2d_mkldnn.weight.ndimension(), 5)
            self.assertEqual(conv2d_loaded.weight.ndimension(), 4)
            self.assertEqual(
                conv2d(x),
                conv2d_loaded(x.to_mkldnn()).to_dense())

    def test_relu(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        self.assertEqual(torch.relu(x), torch.relu(x.to_mkldnn()).to_dense())

    def test_relu_(self):
        x1 = torch.randn((4, 5), dtype=torch.float32) * 10
        x2 = x1.clone().to_mkldnn()
        self.assertEqual(torch.relu_(x1), torch.relu_(x2).to_dense())

    def test_relu_backward(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu(x1).sum()
        y2 = torch.relu(x2).to_dense().sum()
        y1.backward()
        y2.backward()
        self.assertEqual(x1.grad, x2.grad.to_dense())
        # inplace
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu_(x1.clone()).sum()
        y2 = torch.relu_(x2.clone()).to_dense().sum()
        y1.backward()
        y2.backward()
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_max_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
                x = torch.randn(N, C, H, W, dtype=torch.float32) * 10

                for ceil_mode in [False, True]:
                    max_pool2d = torch.nn.MaxPool2d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(
                        max_pool2d(x),
                        max_pool2d(x.to_mkldnn()).to_dense())

    def test_max_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for stride in [1, 2, 3]:
            for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
                x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10

                for ceil_mode in [False, True]:
                    max_pool3d = torch.nn.MaxPool3d(
                        kernel_size=3 if not ceil_mode else 7,
                        stride=stride,
                        padding=1,
                        ceil_mode=ceil_mode)

                    self.assertEqual(
                        max_pool3d(x),
                        max_pool3d(x.to_mkldnn()).to_dense())

    def test_max_pool2d_backward(self):
        x = torch.randn(10, 3, 64, 64, dtype=torch.float32) * 10
        for ceil_mode in [False, True]:
            max_pool2d = torch.nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                ceil_mode=ceil_mode)

            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_mkldnn().requires_grad_()

            y1 = max_pool2d(x1).sum()
            y2 = max_pool2d(x2).to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(
                avg_pool2d(x),
                avg_pool2d(x.to_mkldnn()).to_dense())

    def test_avg_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            avg_pool3d = torch.nn.AvgPool3d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(
                avg_pool3d(x),
                avg_pool3d(x.to_mkldnn()).to_dense())

    def test_avg_pool2d_backward(self):
        x = torch.randn(10, 3, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_mkldnn().requires_grad_()
            avg_pool2d = torch.nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            y1 = avg_pool2d(x1).sum()
            y2 = avg_pool2d(x2).to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_adaptive_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        self.assertEqual(
            adaptive_avg_pool2d(x),
            adaptive_avg_pool2d(x.to_mkldnn()).to_dense())

    def test_adaptive_avg_pool2d_backward(self):
        x = torch.randn(10, 3, 224, 224, dtype=torch.float32) * 100

        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        y1 = adaptive_avg_pool2d(x1).sum()
        y2 = adaptive_avg_pool2d(x2).to_dense().sum()
        y1.backward()
        y2.backward()
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_batch_norm2d(self):
        x = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10

        for train in [True, False]:
            # TODO: support none affine
            for affine in [True]:
                for track_running_stats in [True, False]:
                    bn = torch.nn.BatchNorm2d(
                        3,
                        affine=affine,
                        track_running_stats=track_running_stats).float().train(train)
                    if (train or not track_running_stats):
                        mkldnn_bn = copy.deepcopy(bn)
                    else:
                        mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
                    self.assertEqual(
                        bn(x),
                        mkldnn_bn(x.to_mkldnn()).to_dense(), prec=1e-4)
                    if train and track_running_stats:
                        self.assertEqual(
                            bn.running_mean,
                            mkldnn_bn.running_mean)
                        self.assertEqual(
                            bn.running_var,
                            mkldnn_bn.running_var, prec=1e-3)
                    if (not train and track_running_stats):
                        self._test_serialization(mkldnn_bn, (x.to_mkldnn(),))
                        self._test_tracing(mkldnn_bn, (x.to_mkldnn(),))

    def test_batch_norm3d(self):
        x = torch.randn(4, 3, 30, 30, 30, dtype=torch.float32) * 10

        for train in [True, False]:
            # TODO: support none affine
            for affine in [True]:
                for track_running_stats in [True, False]:
                    bn = torch.nn.BatchNorm3d(
                        3,
                        affine=affine,
                        track_running_stats=track_running_stats).float().train(train)
                    if (train or not track_running_stats):
                        mkldnn_bn = copy.deepcopy(bn)
                    else:
                        mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
                    self.assertEqual(
                        bn(x),
                        mkldnn_bn(x.to_mkldnn()).to_dense(), prec=1e-4)
                    if train and track_running_stats:
                        self.assertEqual(
                            bn.running_mean,
                            mkldnn_bn.running_mean)
                        self.assertEqual(
                            bn.running_var,
                            mkldnn_bn.running_var, prec=1e-3)
                    if (not train and track_running_stats):
                        self._test_serialization(mkldnn_bn, (x.to_mkldnn(),))
                        self._test_tracing(mkldnn_bn, (x.to_mkldnn(),))

    def test_batch_norm2d_backward(self):
        x = torch.randn(64, 3, 35, 45, dtype=torch.float32) * 10

        # TODO: support none affine
        for affine in [True]:
            for track_running_stats in [True, False]:
                x1 = x.clone().requires_grad_()
                x2 = x.clone().to_mkldnn().requires_grad_()
                bn = torch.nn.BatchNorm2d(
                    3,
                    affine=affine,
                    track_running_stats=track_running_stats).float().train(True)
                mkldnn_bn = copy.deepcopy(bn)
                y1 = bn(x1).sum()
                y2 = mkldnn_bn(x2).to_dense().sum()
                y1.backward()
                y2.backward()
                self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_batch_norm2d_relu(self):
        input = torch.randn(2, 10, 35, 35, dtype=torch.float32)
        input_mkldnn=input.to_mkldnn()
        weight = torch.randn(10, dtype=torch.float32)
        bias = torch.randn(10, dtype=torch.float32)
        running_mean = torch.randn(10, dtype=torch.float32)
        running_var = torch.randn(10, dtype=torch.float32)
        momentum = 1.0
        eps = 1.0
        cudnn_enabled = False

        for is_training in [True, False]:
            out = torch.batch_norm(input, weight, bias, \
			    running_mean, running_var, is_training, \
			    momentum, eps, cudnn_enabled)
            out = torch.relu(out)

            out_mkldnn = torch._C._nn.batch_norm_relu(input_mkldnn, \
			    weight, bias, running_mean, running_var, \
			    is_training, momentum, eps)
            self.assertEqual(out, out_mkldnn[0].to_dense())

    def test_batch_norm2d_relu_backward(self):
        input = torch.randn(2, 10, 35, 35, dtype=torch.float32)
        input_grad = input.clone().requires_grad_()
        input_mkldnn=input.clone().to_mkldnn().requires_grad_()
        input_perf=input.clone().to_mkldnn().requires_grad_()
        weight = torch.randn(10, dtype=torch.float32)
        bias = torch.randn(10, dtype=torch.float32)
        running_mean = torch.randn(10, dtype=torch.float32)
        running_var = torch.randn(10, dtype=torch.float32)
        is_training = True
        momentum = 1.0
        eps = 1.0
        cudnn_enabled = False

        out = torch.batch_norm(input_grad, weight, bias,
              running_mean, running_var, is_training, momentum, eps, cudnn_enabled)
        out = torch.relu(out)
        y1 = out.sum()
        y1.backward()

        out_mkldnn = torch._C._nn.batch_norm_relu(input_mkldnn,
              weight, bias, running_mean, running_var, is_training, momentum, eps)
        y2 = out_mkldnn[0].to_dense().sum()
        y2.backward()
        self.assertEqual(input_grad.grad, input_mkldnn.grad.to_dense())

    def test_add(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        alpha = torch.randn(1, dtype=torch.float32).item()

        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        mx = x.to_mkldnn()
        my = y.to_mkldnn()

        # add
        self.assertEqual(
            x + y,
            (mx + my).to_dense())

        self.assertEqual(
            torch.add(x, y, alpha=alpha),
            torch.add(mx, my, alpha=alpha).to_dense())

        # add_
        x += y
        mx += my
        self.assertEqual(x, mx.to_dense())

        # add_out
        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.add(x, y, alpha=alpha, out=out)
        torch.add(mx, my, alpha=alpha, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())

    def test_mul(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        value = torch.randn(1, dtype=torch.float32).item()

        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        y = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        mx = x.to_mkldnn()
        my = y.to_mkldnn()

        # mul
        self.assertEqual(
            x * y,
            (mx * my).to_dense())

        self.assertEqual(
            x * value,
            (mx * value).to_dense())

        self.assertEqual(
            torch.mul(x, y),
            torch.mul(mx, my).to_dense())

        self.assertEqual(
            torch.mul(x, value),
            torch.mul(mx, value).to_dense())

        # mul_
        x *= y
        mx *= my
        self.assertEqual(x, mx.to_dense())

        x *= value
        mx *= value
        self.assertEqual(x, mx.to_dense())

        # mul_out
        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.mul(x, y, out=out)
        torch.mul(mx, my, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())

        out = x.clone()
        mkldnn_out = out.to_mkldnn()
        torch.mul(x, value, out=out)
        torch.mul(mx, value, out=mkldnn_out)
        self.assertEqual(out, mkldnn_out.to_dense())

    def test_view(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32).to_mkldnn()
        self.assertRaisesRegex(RuntimeError,
                               "Change to use reshape",
                               lambda: x.view(x.size(0), -1))

    def test_reshape(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        size = (x.size(0), -1)

        self.assertEqual(
            x.reshape(size),
            x.to_mkldnn().reshape(size).to_dense(),
        )
        # test whether share same memory for plain format tensor
        y = x.to_mkldnn()
        z = y.reshape(size).add_(y.reshape(size))
        self.assertEqual(
            y.reshape(size).to_dense(),
            z.to_dense(),
        )

    def test_reshape_backward(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        size = (x.size(0), -1)

        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()

        in_features = 20
        out_features = out_features = torch.randint(3, 100, (1,)).item()
        linear = torch.nn.Linear(in_features, out_features).float()

        y1 = linear(x1.reshape(size)).sum()
        y2 = linear(x2.reshape(size).to_dense()).sum()
        y1.backward()
        y2.backward()

        self.assertEqual(
            x1.grad,
            x2.grad.to_dense())

    def test_clone(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        self.assertEqual(
            x.clone(),
            x.to_mkldnn().clone().to_dense(),
        )
        # test whether share same memory
        y = x.to_mkldnn()
        z = y.clone().add_(y)
        self.assertNotEqual(
            y.to_dense(),
            z.to_dense(),
        )

    def test_transpose(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        for dim1 in range(x.ndim):
            for dim2 in range(x.ndim):
                self.assertEqual(
                    x.transpose(dim1, dim2),
                    x.to_mkldnn().transpose(dim1, dim2).to_dense(),
                )

    def test_linear(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
            mkldnn_linear = mkldnn_utils.to_mkldnn(copy.deepcopy(linear))
            self.assertEqual(
                linear(x),
                mkldnn_linear(x.to_mkldnn()).to_dense())

            self._test_serialization(mkldnn_linear, (x.to_mkldnn(),))
            self._test_tracing(mkldnn_linear, (x.to_mkldnn(),))

    # we should first expose aten::linear, depend on https://github.com/pytorch/pytorch/pull/20039
    def test_linear_backward(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        
        x1 = x.clone().requires_grad_()
        linear = torch.nn.Linear(in_features, out_features)
        y1 = linear(x1).sum()
        y1.backward()
        for dtype in [torch.bfloat16, torch.float]:
            for bias in [True, False]:
                x2 = x.clone().to(dtype).to_mkldnn().requires_grad_()
                mkldnn_linear = copy.deepcopy(linear)
                y2 = mkldnn_linear(x2).to_dense().sum()
                y2.backward()

                self.assertEqual(x2.dtype, dtype)
                self.assertEqual(x1.grad, x2.grad.float().to_dense(), prec=5e-2)
                # for current bf16 design, the weight/weight.grad are in fp32 datatype  
                self.assertEqual(mkldnn_linear.weight.grad.dtype, torch.float)
                self.assertEqual(linear.weight.grad, mkldnn_linear.weight.grad, prec=6e-2)
                if bias:
                    self.assertEqual(mkldnn_linear.bias.grad.dtype, torch.float)
                    self.assertEqual(linear.bias.grad, mkldnn_linear.bias.grad, prec=5e-2)

    def test_linear_bf16(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias)
            mkldnn_linear = mkldnn_utils.to_mkldnn(copy.deepcopy(linear))
            mkldnn_linear_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(linear), torch.bfloat16)
            y = mkldnn_linear(x.to_mkldnn()).to_dense()
            y_bf16 = mkldnn_linear_bf16(x_bf16.to_mkldnn()).to_dense()
            self.assertEqual(y, y_bf16, prec=0.1)

    def test_mm(self):
        M, N, O = 23, 8, 12
        b1 = torch.randn(M, N, dtype=torch.float32)
        b2 = torch.randn(N, O, dtype=torch.float32)
        mm = torch.mm(b1, b2)
        
        b1_in = b1.clone()
        b1_in.mm(b2)
        for dtype in [torch.bfloat16, torch.float]:
            b1_ = b1.clone().to_mkldnn().to(dtype)
            b2_ = b2.clone().to_mkldnn().to(dtype)
            mkldnn_mm = torch.mm(b1_, b2_)
            self.assertEqual(mkldnn_mm.dtype, dtype)
            self.assertEqual(mm, mkldnn_mm.float().to_dense(), prec=5e-02)

            y = torch.randn(M, O, dtype=torch.float32)
            mkldnn_y = y.clone().to_mkldnn().to(dtype)
            torch.mm(b1_, b2_, out=mkldnn_y)
            self.assertEqual(mkldnn_y.dtype, dtype)
            self.assertEqual(mm, mkldnn_y.float().to_dense(), prec=5e-02)

            b1_in_ = b1.clone().to_mkldnn().to(dtype)
            b1_in_.mm(b2_)
            self.assertEqual(b1_in_.dtype, dtype)
            self.assertEqual(b1_in, b1_in_.float().to_dense(), prec=5e-02)

    def test_mm_backward(self):
        M, N, O = 23, 8, 12
        b1 = torch.randn(M, N, dtype=torch.float32)
        b2 = torch.randn(N, O, dtype=torch.float32)

        b1_ = b1.clone().requires_grad_()
        b2_ = b2.clone().requires_grad_()
        mm = torch.mm(b1_, b2_)
        y1 = mm.sum()
        y1.backward()
        for dtype in [torch.bfloat16, torch.float]:
            b1_m = b1.clone().to_mkldnn().to(dtype).requires_grad_()
            b2_m = b2.clone().to_mkldnn().to(dtype).requires_grad_()
            mkldnn_mm = torch.mm(b1_m, b2_m)
            y2 = mkldnn_mm.to_dense().sum()
            y2.backward()
                    
            self.assertEqual(b1_.grad, b1_m.grad.to(dtype).to_dense(), prec=5e-02)
            self.assertEqual(b2_.grad, b2_m.grad.to(dtype).to_dense(), prec=5e-02)

    def test_bmm(self):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
        b2 = torch.randn(num_batches, N, O, dtype=torch.float32)
        bmm = torch.bmm(b1, b2)
        
        b1_in = b1.clone()
        b1_in.bmm(b2)
        for dtype in [torch.bfloat16, torch.float]:
            b1_ = b1.clone().to_mkldnn().to(dtype)
            b2_ = b2.clone().to_mkldnn().to(dtype)
            mkldnn_bmm = torch.bmm(b1_, b2_)
            self.assertEqual(mkldnn_bmm.dtype, dtype)
            self.assertEqual(bmm, mkldnn_bmm.float().to_dense(), prec=6e-02)

            y = torch.randn(num_batches, M, O, dtype=torch.float32)
            mkldnn_y = y.clone().to_mkldnn().to(dtype)
            torch.bmm(b1_, b2_, out=mkldnn_y)
            self.assertEqual(mkldnn_y.dtype, dtype)
            self.assertEqual(bmm, mkldnn_y.float().to_dense(), prec=6e-02)
            
            b1_in_ = b1.clone().to_mkldnn().to(dtype)
            b1_in_.bmm(b2_)
            self.assertEqual(b1_in_.dtype, dtype)
            self.assertEqual(b1_in, b1_in_.float().to_dense(), prec=5e-02)
     
    def test_bmm_backward(self):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
        b2 = torch.randn(num_batches, N, O, dtype=torch.float32)

        b1_ = b1.clone().requires_grad_()
        b2_ = b2.clone().requires_grad_()
        bmm = torch.bmm(b1_, b2_)
        y1 = bmm.sum()
        y1.backward()
        for dtype in [torch.bfloat16, torch.float]:
            b1_m = b1.clone().to_mkldnn().to(dtype).requires_grad_()
            b2_m = b2.clone().to_mkldnn().to(dtype).requires_grad_()
            mkldnn_bmm = torch.bmm(b1_m, b2_m)
            y2 = mkldnn_bmm.to_dense().sum()
            y2.backward()
            self.assertEqual(mkldnn_bmm.dtype, dtype)
            self.assertEqual(b1_.grad, b1_m.grad.float().to_dense(), prec=5e-02)
            self.assertEqual(b2_.grad, b2_m.grad.float().to_dense(), prec=5e-02)
    
    def test_addmm(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                M, N, O = 23, 8, 12
                b1 = torch.randn(M, N, dtype=torch.float32)
                b2 = torch.randn(N, O, dtype=torch.float32)
                res = torch.randn(M, O, dtype=torch.float32)

                addmm = torch.addmm(input=res, mat1=b1, mat2=b2, alpha=alpha, beta=beta)
                res1 = res.clone()
                res1.addmm_(b1, b2, alpha=alpha, beta=beta)
                for dtype in [torch.bfloat16, torch.float]:
                    b1_ = b1.clone().to_mkldnn().to(dtype)
                    b2_ = b2.clone().to_mkldnn().to(dtype)
                    res_ = res.clone().to_mkldnn().to(dtype)
                    mkldnn_addmm = torch.addmm(input=res_, mat1=b1_, mat2=b2_, \
                                   alpha=alpha, beta=beta)
                    self.assertEqual(mkldnn_addmm.dtype, dtype)
                    self.assertEqual(addmm, mkldnn_addmm.float().to_dense(), prec=5e-02)

                    y = torch.randn(M, O, dtype=torch.float32)
                    mkldnn_y = y.clone().to_mkldnn().to(dtype)
                    torch.addmm(input=res_, mat1=b1_, mat2=b2_, alpha=alpha, beta=beta, \
                                out=mkldnn_y),
                    self.assertEqual(mkldnn_y.dtype, dtype)
                    self.assertEqual(addmm, mkldnn_y.float().to_dense(), prec=5e-02)
                    
                    res1_ = res.clone().to_mkldnn().to(dtype)
                    res1_.addmm_(b1_, b2_, alpha=alpha, beta=beta)
                    self.assertEqual(res1, res1_.float().to_dense(), prec=5e-02)
    
    def test_addmm_backward(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                M, N, O = 23, 8, 12
                b1 = torch.randn(M, N, dtype=torch.float32)
                b2 = torch.randn(N, O, dtype=torch.float32)
                res = torch.randn(M, O, dtype=torch.float32)

                b1_ = b1.clone().requires_grad_()
                b2_ = b2.clone().requires_grad_()
                res_ = res.clone().requires_grad_()
                addmm = torch.addmm(input=res_, mat1=b1_, mat2=b2_, alpha=alpha, beta=beta)
                y1 = addmm.sum()
                y1.backward()
                for dtype in [torch.bfloat16, torch.float]:
                    b1_m = b1.clone().to_mkldnn().to(dtype).requires_grad_()
                    b2_m = b2.clone().to_mkldnn().to(dtype).requires_grad_()
                    res_m = res.clone().to_mkldnn().to(dtype).requires_grad_()
                    mkldnn_addmm = torch.addmm(input=res_m, mat1=b1_m, mat2=b2_m, \
                                   alpha=alpha, beta=beta)
                    y2 = mkldnn_addmm.to_dense().sum()
                    y2.backward()
                    
                    self.assertEqual(b1_.grad, b1_m.grad.to(dtype).to_dense(), prec=5e-02)
                    self.assertEqual(b2_.grad, b2_m.grad.to(dtype).to_dense(), prec=5e-02)
                    self.assertEqual(res_.grad, res_m.grad.to(dtype).to_dense(), prec=5e-02)

    def test_addbmm(self):
        dtype2prec = {torch.float: 2e-5, torch.bfloat16: 5e-1}
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                num_batches = 10
                M, N, O = 23, 8, 12
                b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2 = torch.randn(num_batches, N, O, dtype=torch.float32)
                res = torch.randn(M, O, dtype=torch.float32)

                addbmm = torch.addbmm(res, b1, b2, beta=beta, alpha=alpha)
                res1 = res.clone()
                res1.addbmm_(b1, b2, alpha=alpha, beta=beta)
                for dtype in [torch.bfloat16, torch.float]:
                    b1_ = b1.clone().to_mkldnn().to(dtype)
                    b2_ = b2.clone().to_mkldnn().to(dtype)
                    res_ = res.clone().to_mkldnn().to(dtype)
                    mkldnn_addbmm = torch.addbmm(res_, b1_, b2_, beta=beta, alpha=alpha)
                    self.assertEqual(mkldnn_addbmm.dtype, dtype)
                    self.assertEqual(addbmm, mkldnn_addbmm.float().to_dense(), prec=dtype2prec[dtype])

                    y = torch.randn(M, O, dtype=torch.float32)
                    mkldnn_y = y.clone().to_mkldnn().to(dtype)
                    torch.addbmm(res_, b1_, b2_, beta=beta, alpha=alpha, out=mkldnn_y)
                    self.assertEqual(mkldnn_y.dtype, dtype)
                    self.assertEqual(addbmm, mkldnn_y.float().to_dense(), prec=dtype2prec[dtype])
    
                    res1_ = res.clone().to_mkldnn().to(dtype)
                    res1_.addbmm_(b1_, b2_, alpha=alpha, beta=beta)
                    self.assertEqual(res1, res1_.float().to_dense(), prec=dtype2prec[dtype])
    
    def test_addbmm_backward(self):
        dtype2prec = {torch.float: 5e-2, torch.bfloat16: 7e-2}
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10 
                num_batches = 10 
                M, N, O = 23, 8, 12
                b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2 = torch.randn(num_batches, N, O, dtype=torch.float32)
                res = torch.randn(M, O, dtype=torch.float32)

                b1_ = b1.clone().requires_grad_()
                b2_ = b2.clone().requires_grad_()
                res_ = res.clone().requires_grad_()
                addbmm = torch.addbmm(res_, b1_, b2_, beta=beta, alpha=alpha)
                y1 = addbmm.sum()
                y1.backward()
                for dtype in [torch.bfloat16, torch.float]:
                    b1_m = b1.clone().to_mkldnn().to(dtype).requires_grad_()
                    b2_m = b2.clone().to_mkldnn().to(dtype).requires_grad_()
                    res_m = res.clone().to_mkldnn().to(dtype).requires_grad_()
                    mkldnn_addbmm = torch.addbmm(res_m, b1_m, b2_m, beta=beta, alpha=alpha)
                    y2 = mkldnn_addbmm.to_dense().sum()
                    y2.backward()
                    self.assertEqual(b1_.grad, b1_m.grad.to(dtype).to_dense(), prec=dtype2prec[dtype])
                    self.assertEqual(b2_.grad, b2_m.grad.to(dtype).to_dense(), prec=dtype2prec[dtype])
                    self.assertEqual(res_.grad, res_m.grad.to(dtype).to_dense(), prec=dtype2prec[dtype])

    # mkldnn baddbmm now doesn't support broadcast
    # self tensor should be in 3 dims
    def test_baddbmm(self):
        dtype2prec = {torch.float: 2e-5, torch.bfloat16: 5e-1}
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10
                num_batches = 10
                M, N, O = 23, 8, 12
                b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2 = torch.randn(num_batches, N, O, dtype=torch.float32)
                res = torch.randn(num_batches, M, O, dtype=torch.float32)

                baddbmm = torch.baddbmm(res, b1, b2, alpha=alpha, beta=beta)
                res1 = res.clone()
                res1.baddbmm_(b1, b2, alpha=alpha, beta=beta)
                for dtype in [torch.bfloat16, torch.float]:
                    b1_ = b1.clone().to_mkldnn().to(dtype)
                    b2_ = b2.clone().to_mkldnn().to(dtype)
                    res_ = res.clone().to_mkldnn().to(dtype)
                    mkldnn_baddbmm = torch.baddbmm(res_, b1_, b2_, alpha=alpha, beta=beta)
                    self.assertEqual(mkldnn_baddbmm.dtype, dtype)
                    self.assertEqual(baddbmm, mkldnn_baddbmm.float().to_dense(), prec=dtype2prec[dtype])

                    y = torch.randn(num_batches, M, O, dtype=torch.float32)
                    mkldnn_y = y.clone().to_mkldnn().to(dtype)
                    torch.baddbmm(res_, b1_, b2_, alpha=alpha, beta=beta, out=mkldnn_y),
                    self.assertEqual(mkldnn_y.dtype, dtype)
                    self.assertEqual(baddbmm, mkldnn_y.float().to_dense(), prec=dtype2prec[dtype])
                    
                    res1_ = res.clone().to_mkldnn().to(dtype)
                    res1_.baddbmm_(b1_, b2_, alpha=alpha, beta=beta)
                    self.assertEqual(res1, res1_.float().to_dense(), prec=dtype2prec[dtype])

    def test_baddbmm_backward(self):
        for i in range(8, 14, 2):
            for j in range(8, 14, 2):
                alpha = i / 10
                beta = j / 10 
                M, N, O = 23, 8, 12
                num_batches = 10 
                b1 = torch.randn(num_batches, M, N, dtype=torch.float32)
                b2 = torch.randn(num_batches, N, O, dtype=torch.float32)
                res = torch.randn(num_batches, M, O, dtype=torch.float32)

                b1_ = b1.clone().requires_grad_()
                b2_ = b2.clone().requires_grad_()
                res_ = res.clone().requires_grad_()
                baddbmm = torch.baddbmm(res_, b1_, b2_, alpha=alpha, beta=beta)
                y1 = baddbmm.sum()
                y1.backward()
                
                for dtype in [torch.bfloat16, torch.float]:
                    b1_m = b1.clone().to_mkldnn().to(dtype).requires_grad_()
                    b2_m = b2.clone().to_mkldnn().to(dtype).requires_grad_()
                    res_m = res.clone().to_mkldnn().to(dtype).requires_grad_()
                    mkldnn_baddbmm = torch.baddbmm(res_m, b1_m, b2_m, alpha=alpha, beta=beta)
                    y2 = mkldnn_baddbmm.to_dense().sum()
                    y2.backward()
                     
                    self.assertEqual(b1_.grad, b1_m.grad.to(dtype).to_dense(), prec=5e-02)
                    self.assertEqual(b2_.grad, b2_m.grad.to(dtype).to_dense(), prec=5e-02)
                    self.assertEqual(res_.grad, res_m.grad.to(dtype).to_dense(), prec=5e-02)

    def test_softmax(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        for dim in range(x.ndim):
            softmax = torch.nn.Softmax(dim=dim)
            self.assertEqual(
                softmax(x),
                softmax(x.to_mkldnn()).to_dense())

    def test_softmax_backward(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        for dim in range(x.ndim):
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_mkldnn().requires_grad_()
            softmax = torch.nn.Softmax(dim=dim)
            y1 = softmax(x1).sum()
            y2 = softmax(x2).to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_sigmoid(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        mkldnn_x = x.to_mkldnn()
        self.assertEqual(
            torch.sigmoid(x),
            torch.sigmoid(mkldnn_x).to_dense(),
        )
        # inplace
        torch.sigmoid_(x)
        torch.sigmoid_(mkldnn_x)
        self.assertEqual(x, mkldnn_x.to_dense())

    def test_sigmoid_backward(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        dense_x = x.clone().requires_grad_()
        mkldnn_x1 = x.clone().to_mkldnn().requires_grad_()
        dense_y = torch.sigmoid(dense_x)
        dense_y1 = torch.sigmoid(mkldnn_x1).to_dense()
        self.assertEqual(dense_y, dense_y1)

        dense_y.sum().backward()
        dense_y1.sum().backward()
        #self.assertEqual(dense_x.grad, mkldnn_x1.grad.to_dense())

    def _test_serialization(self, module, inputs):
        with TemporaryFileName() as fname:
            torch.jit.save(module, fname)
            loaded = torch.jit.load(fname)
            self.assertEqual(
                module(*inputs).to_dense(),
                loaded(*inputs).to_dense())

    def _test_tracing(self, module, inputs):
        traced = torch.jit.trace(module, inputs, check_trace=False)
        self.assertEqual(
            module(*inputs).to_dense(),
            traced(*inputs).to_dense())

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while MKL-DNN tensor has impl
        # of type `OpaqueTensorImpl<IDeepTensorWrapperPtr>`.
        x = torch.randn((1, 2), dtype=torch.float, device=torch.device('cpu'))
        x_mkldnn = x.to_mkldnn()
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_mkldnn

    def test_empty(self):
        x1 = torch.empty(4, 5, 2, 3, dtype=torch.float32)
        x2 = torch.empty(4, 5, 2, 3, dtype=torch.float32, layout=torch._mkldnn)
        self.assertEqual(x1.size(), x2.to_dense().size())
        self.assertEqual(x1.dtype, x2.to_dense().dtype)

    def test_zero_(self):
        x1 = torch.randn(4, 5, dtype=torch.float32) * 10
        x2 = x1.clone().to_mkldnn()
        self.assertEqual(
            x1.zero_(),
            x2.zero_().to_dense(),
        )

    def test_is_mkldnn(self):
        x = torch.randn(1, dtype=torch.float32)
        self.assertFalse(x.is_mkldnn)
        self.assertTrue(x.to_mkldnn().is_mkldnn)

    # legacy constructor/new doesn't support mkldnn tensors
    def test_legacy_new_failure(self):
        x = torch.randn(1, dtype=torch.float32)
        x_mkldnn = x.to_mkldnn()
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(device='cpu'))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(x.storage()))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(x))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new(torch.Size([2, 3])))
        self.assertRaises(RuntimeError, lambda: x_mkldnn.new([6]))

    def test_is_mkldnn_jit(self):
        class EnsureMkldnn(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if not x.is_mkldnn:
                    x = x.to_mkldnn()
                return x

        m = EnsureMkldnn()
        x = torch.randn(1, dtype=torch.float32)
        self.assertTrue(m(x).is_mkldnn)
        self.assertTrue(m(x.to_mkldnn()).is_mkldnn)

    def _test_imagenet_model(self, model):
        model = model.train(False).float()
        mkldnn_model = mkldnn_utils.to_mkldnn(copy.deepcopy(model))
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            self.assertEqual(
                model(x),
                mkldnn_model(x.to_mkldnn()).to_dense(),
            )

    @skipIfNoTorchVision
    def test_resnet18(self):
        model = torchvision.models.resnet.resnet18(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        self._test_imagenet_model(model)

    @skipIfNoTorchVision
    def test_resnext50_32x4d_bf16(self):
        model = torchvision.models.resnet.resnext50_32x4d(pretrained=False)
        model = model.train(False)
        mkldnn_model = mkldnn_utils.to_mkldnn(copy.deepcopy(model))
        mkldnn_model_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(model), torch.bfloat16)
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        x_bf16 = x.bfloat16()
        y = mkldnn_model(x.to_mkldnn()).to_dense()
        y_bf16 = mkldnn_model_bf16(x_bf16.to_mkldnn()).to_dense()
        with torch.no_grad():
            self.assertEqual(y, y_bf16, prec=1e-3)

    def test_dropout(self):
        p = 0.2
        input = torch.randn(1000, dtype=torch.float32)
        input = input.fill_(1 - p)
        module = torch.nn.Dropout(p)
        input_var = input.clone().to_mkldnn().requires_grad_()
        output = module(input_var)
        self.assertLess(abs(output.to_dense().data.mean() - (1 - p)), 0.05)
        output.backward(input_var)
        self.assertLess(abs(input_var.grad.to_dense().data.mean() - (1 - p)), 0.05)

        # check eval mode doesn't change anything
        for inplace in [True, False]:
            module = torch.nn.Dropout(p, inplace).eval()
            self.assertEqual(input_var.to_dense(), module(input_var).to_dense())

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_cat(self):
        x = torch.randn(4, 5, dtype=torch.float32) * 10
        mkldnn_x = x.to_mkldnn()
        for dim in [0, 1]:
            self.assertEqual(
                torch.cat((x, x, x), dim=dim),
                torch.cat((mkldnn_x, mkldnn_x, mkldnn_x), dim=dim).to_dense(),
            )
        #cat_out
        y = torch.randn(12, 5, dtype=torch.float32)*10
        mkldnn_y = y.to_mkldnn()
        torch.cat((x, x, x), dim=0, out=y),
        torch.cat((mkldnn_x, mkldnn_x, mkldnn_x), dim=0, out=mkldnn_y)
        self.assertEqual(y, mkldnn_y.to_dense())
        y = torch.randn(4, 15, dtype=torch.float32)*10
        mkldnn_y = y.to_mkldnn()
        torch.cat((x, x, x), dim=1, out=y),
        torch.cat((mkldnn_x, mkldnn_x, mkldnn_x), dim=1, out=mkldnn_y)
        self.assertEqual(y, mkldnn_y.to_dense())

    def test_cat_backward(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.cat((x1, x1, x1)).sum()
        y2 = torch.cat((x2, x2, x2)).to_dense().sum()
        y1.backward()
        y2.backward()
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_split(self):
        x = torch.randn(5, 5, dtype=torch.float32) * 10
        mkldnn_x = x.to_mkldnn()
        for dim in [0, 1]:
            self.assertEqual(
                torch.split(x, (2,3), dim=dim)[0],
                torch.split(mkldnn_x, (2,3), dim=dim)[0].to_dense(),
            )
            self.assertEqual(
                torch.split(x, (2,3), dim=dim)[1],
                torch.split(mkldnn_x, (2,3), dim=dim)[1].to_dense(),
            )
            self.assertEqual(
                torch.split(x, 3, dim=dim)[0],
                torch.split(mkldnn_x, 3, dim=dim)[0].to_dense(),
            )
            self.assertEqual(
                torch.split(x, 3, dim=dim)[1],
                torch.split(mkldnn_x, 3, dim=dim)[1].to_dense(),
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[0],
                torch.split(mkldnn_x, 2, dim=dim)[0].to_dense(),
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[1],
                torch.split(mkldnn_x, 2, dim=dim)[1].to_dense(),
            )
            self.assertEqual(
                torch.split(x, 2, dim=dim)[2],
                torch.split(mkldnn_x, 2, dim=dim)[2].to_dense(),
            )

    def test_split_backward(self):
        x = torch.randn(5, 5, dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        for dim in [0, 1]:
            y1 = torch.split(x1, (2,3), dim=dim)[0].sum() \
                    + torch.split(x1, (2,3), dim=dim)[1].sum()
            y2 = torch.split(x2, (2,3), dim=dim)[0].to_dense().sum() \
                    + torch.split(x2, (2,3), dim=dim)[1].to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())
            y1 = torch.split(x1, 3, dim=dim)[0].sum() \
                    + torch.split(x1, 3, dim=dim)[1].sum()
            y2 = torch.split(x2, 3, dim=dim)[0].to_dense().sum() \
                    + torch.split(x2, 3, dim=dim)[1].to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())
            y1 = torch.split(x1, 2, dim=dim)[0].sum() \
                    + torch.split(x1, 2, dim=dim)[1].sum() \
                    + torch.split(x1, 2, dim=dim)[2].sum()
            y2 = torch.split(x2, 2, dim=dim)[0].to_dense().sum() \
                    + torch.split(x2, 2, dim=dim)[1].to_dense().sum() \
                    + torch.split(x2, 2, dim=dim)[2].to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())

if __name__ == '__main__':
    run_tests()
