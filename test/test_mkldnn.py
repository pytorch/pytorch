import copy
import itertools
import functools
import unittest

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

import torch
import torch.nn.functional as F
import torch.jit
import torch.backends.mkldnn
from torch.utils import mkldnn as mkldnn_utils
from torch.testing._internal.common_utils import TestCase, \
    run_tests, TemporaryFileName, gradcheck, gradgradcheck, IS_WINDOWS

# batched grad doesn't support mkldnn
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
gradgradcheck = functools.partial(gradgradcheck, check_batched_grad=False)


# For OneDNN bf16 path, OneDNN requires the cpu has intel avx512 with avx512bw,
# avx512vl, and avx512dq at least. So we will skip the test case if one processor
# is not meet the requirement.
def has_bf16_support():
    import subprocess
    try:
        cmd = "grep avx512bw /proc/cpuinfo | grep avx512vl | grep avx512dq"
        subprocess.check_output(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

types = [torch.float, torch.bfloat16]

# Comment the line below to find out the CI machines having MKL-DNN build disabled
@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnn(TestCase):
    def test_conversion(self):
        for cpu_tensor in [torch.randn((1, 2, 3, 4),
                                       dtype=torch.float, device=torch.device('cpu')),
                           torch.randn((1, 2, 3, 4, 5),
                                       dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
            cpu_tensor.requires_grad_()
            # float cpu tensor to mkldnn float tensor or bfloat tensor.
            for dtype1 in types:
                mkldnn_tensor = cpu_tensor.to_mkldnn(dtype1)
                self.assertEqual(mkldnn_tensor.dtype, dtype1)
                cpu_tensor_1 = mkldnn_tensor.to_dense()
                # not given dtype for to_dense, mkldnn tensor has same dtype with cpu tensor
                self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)
                # mkldnn float/bfloat tensor to cpu float or bfloat tensor
                for dtype2 in types:
                    cpu_tensor_2 = mkldnn_tensor.to_dense(dtype2)
                    self.assertEqual(cpu_tensor_2.dtype, dtype2)
                    atol = 1e-5 if dtype1 == torch.float and dtype2 == torch.float else 1e-2
                    self.assertEqual(cpu_tensor, cpu_tensor_2.float(), atol=atol, rtol=0)

                self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
                self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
                self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
                if dtype1 == torch.float:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size())
                else:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size() / 2)
                self.assertRaisesRegex(RuntimeError,
                                       "Cannot access data pointer of Tensor that doesn't have storage",
                                       lambda: mkldnn_tensor.data_ptr() != 0)

            # bfloat cpu tensor to mkldnn float tensor or bfloat tensor.
            cpu_tensor_bf16 = cpu_tensor.bfloat16()
            for dtype1 in types:
                mkldnn_tensor = cpu_tensor_bf16.to_mkldnn(dtype1)
                self.assertEqual(mkldnn_tensor.dtype, dtype1)
                cpu_tensor_1 = mkldnn_tensor.to_dense()
                # not given dtype for to_dense, mkldnn tensor has same dtype with cpu tensor
                self.assertEqual(mkldnn_tensor.dtype, cpu_tensor_1.dtype)
                # mkldnn float/bfloat tensor to cpu float or bfloat tensor
                for dtype2 in types:
                    cpu_tensor_2 = mkldnn_tensor.to_dense(dtype2)
                    self.assertEqual(cpu_tensor_2.dtype, dtype2)
                    self.assertEqual(cpu_tensor_bf16, cpu_tensor_2.bfloat16(), atol=1e-5, rtol=0)

                self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
                self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
                self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
                if dtype1 == torch.bfloat16:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor_bf16.element_size())
                else:
                    self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor_bf16.element_size() * 2)
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
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_autograd_from_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        def func(root):
            return root.to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(UserWarning,
                              'double precision floating point',
                              lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2))

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

    def _test_conv_base(self, dim):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [True, False], [1, 2], [1, 4])
        for train, bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)
            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).float()
            x1 = x.clone()
            x2 = x.clone().to_mkldnn()
            if not train:
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
            elif train and dim != 1:
                # TODO: enable conv1d training.
                x1.requires_grad_()
                x2.requires_grad_()
                mkldnn_conv = copy.deepcopy(conv)
            with torch.backends.mkldnn.flags(enabled=False):
                y_aten = conv(x1)
                if train and dim != 1:
                    loss1 = y_aten.sum()
                    loss1.backward()
            if not train or (train and dim != 1):
                y_mkldnn = mkldnn_conv(x2).to_dense()
                self.assertEqual(y_aten, y_mkldnn)
            if not train:
                self._test_serialization(mkldnn_conv, (x.to_mkldnn(),))
                self._test_tracing(mkldnn_conv, (x.to_mkldnn(),))
            elif dim != 1:
                loss2 = y_mkldnn.sum()
                loss2.backward()
                self.assertTrue(x2.grad.is_mkldnn)
                self.assertEqual(x1.grad, x2.grad.to_dense())
                self.assertEqual(conv.weight.grad,
                                 mkldnn_conv.weight.grad,
                                 atol=1e-3,
                                 rtol=1e-3)
                if bias:
                    self.assertEqual(conv.bias.grad, mkldnn_conv.bias.grad)

    def test_conv1d(self):
        self._test_conv_base(dim=1)

    def test_conv2d(self):
        self._test_conv_base(dim=2)

    def test_conv3d(self):
        self._test_conv_base(dim=3)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_conv_bf16_base(self, dim):
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}
        input_shapes = {1: (224,), 2: (224, 224), 3: (55, 55, 55)}
        options = itertools.product([True, False], [1, 2], [1, 4])
        for bias, dilation, groups in options:
            N = torch.randint(3, 10, (1,)).item()
            M = torch.randint(1, 3, (1,)).item() * groups
            C = torch.randint(1, 3, (1,)).item() * groups
            x_shape = (N, C) + input_shapes[dim]
            x = torch.randn(x_shape, dtype=torch.float32)

            conv = conv_module[dim](in_channels=C,
                                    out_channels=M,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias,
                                    groups=groups).float()
            x_bf16 = x.bfloat16()
            if has_bf16_support():
                mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
                mkldnn_conv_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), torch.bfloat16)
                y = mkldnn_conv(x.to_mkldnn()).to_dense()
                y_bf16 = mkldnn_conv_bf16(x_bf16.to_mkldnn()).to_dense(torch.float32)
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                with self.assertRaisesRegex(RuntimeError, msg):
                    mkldnn_conv_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), torch.bfloat16)
                    y_bf16 = mkldnn_conv_bf16(x_bf16.to_mkldnn()).to_dense(torch.float32)

    def test_conv1d_bf16(self):
        self._test_conv_bf16_base(dim=1)

    def test_conv2d_bf16(self):
        self._test_conv_bf16_base(dim=2)

    def test_conv3d_bf16(self):
        self._test_conv_bf16_base(dim=3)

    def test_conv2d_legacy_jit_model(self):
        """
        MKLDNN integration used to serialize models with 5d weight for grouped
        convolutions, we'd like to preserve this behavior
        """
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
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu(x1)
        y2 = torch.relu(x2).to_dense()
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_relu_(self):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = torch.relu_(x1.clone())
        y2 = torch.relu_(x2.clone()).to_dense()
        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()
        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_relu_bf16_base(self, name):
        x = torch.randn((4, 5), dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()
        fn = getattr(torch, name)
        if has_bf16_support():
            y = fn(x.to_mkldnn()).to_dense()
            y_bf16 = fn(x_bf16.to_mkldnn()).to_dense(torch.float32)
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            msg = r"bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: fn(x_bf16.to_mkldnn()))

    def test_relu_bf16(self):
        self._test_relu_bf16_base("relu")

    def test_relu_inplace_bf16(self):
        self._test_relu_bf16_base("relu_")

    def _test_max_pool_base(self, dim, input):
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                x1 = input.clone().requires_grad_()
                x2 = input.clone().to_mkldnn().requires_grad_()
                y1 = max_pool(x1)
                y2 = max_pool(x2).to_dense()
                loss1 = y1.sum()
                loss2 = y2.sum()
                loss1.backward()
                loss2.backward()
                self.assertEqual(y1, y2)
                self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_max_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            self._test_max_pool_base(dim=2, input=x)

    def test_max_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            self._test_max_pool_base(dim=3, input=x)


    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_max_pool_bf16_base(self, dim, input):
        pool_module = {2: torch.nn.MaxPool2d, 3: torch.nn.MaxPool3d}
        x_bf16 = input.bfloat16()
        for stride in [1, 2, 3]:
            for ceil_mode in [False, True]:
                max_pool = pool_module[dim](
                    kernel_size=3 if not ceil_mode else 7,
                    stride=stride,
                    padding=1,
                    ceil_mode=ceil_mode)

                if has_bf16_support():
                    y = max_pool(input.to_mkldnn()).to_dense()
                    y_bf16 = max_pool(x_bf16.to_mkldnn()).to_dense(torch.float32)
                    self.assertEqual(y, y_bf16, atol=0.1, rtol=1e-3)
                else:
                    msg = "mkldnn_max_pool%dd: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq" % dim
                    self.assertRaisesRegex(RuntimeError,
                                           msg,
                                           lambda: max_pool(x_bf16.to_mkldnn()))

    def test_max_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            self._test_max_pool_bf16_base(dim=2, input=x)

    def test_max_pool3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        for D, H, W in [(64, 64, 64), (35, 39, 35), (16, 19, 20), [7, 8, 9]]:
            x = torch.randn(N, C, D, H, W, dtype=torch.float32) * 10
            self._test_max_pool_bf16_base(dim=3, input=x)

    def test_max_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        for H, W in [(64, 64), (35, 39), (16, 19), [7, 8]]:
            x = torch.randn(N, C, H, W, dtype=torch.float32) * 10
            for ceil_mode in [False, True]:
                y1 = F.max_pool2d(
                    x,
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                y2 = F.max_pool2d(
                    x.to_mkldnn(),
                    kernel_size=3 if not ceil_mode else 7,
                    stride=None,
                    padding=1,
                    ceil_mode=ceil_mode)

                self.assertEqual(y1, y2.to_dense())

    def test_max_pool_unsupported(self):
        # OneDNN not support dilation max_pooling, will be avilabled in v2.0.
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()

        # 2d dilation case
        x = torch.randn(N, C, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool2d = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool2d does not support dilation case',
                               lambda: max_pool2d(x))

        # 3d dilation case
        x = torch.randn(N, C, 7, 7, 7, dtype=torch.float32).to_mkldnn()
        max_pool3d = torch.nn.MaxPool3d(
            kernel_size=3,
            stride=3,
            padding=1,
            dilation=2)
        self.assertRaisesRegex(RuntimeError,
                               'mkldnn_max_pool3d does not support dilation case',
                               lambda: max_pool3d(x))

    def _test_avg_pool_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)

            x1 = input.clone().requires_grad_()
            x2 = input.clone().to_mkldnn().requires_grad_()
            y1 = avg_pool(x1)
            y2 = avg_pool(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2)
            self.assertEqual(x1.grad, x2.grad.to_dense())

    def test_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=2, input=x)

    def test_avg_pool3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_base(dim=3, input=x)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_avg_pool_bf16_base(self, dim, input):
        avg_module = {2: torch.nn.AvgPool2d, 3: torch.nn.AvgPool3d}
        x_bf16 = input.bfloat16()
        for count_include_pad in [True, False]:
            avg_pool = avg_module[dim](
                kernel_size=3,
                stride=2,
                padding=1,
                count_include_pad=count_include_pad)
            if has_bf16_support():
                y = avg_pool(input.to_mkldnn()).to_dense()
                y_bf16 = avg_pool(x_bf16.to_mkldnn()).to_dense(torch.float)
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = "mkldnn_avg_pool%dd: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq" % dim
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: avg_pool(x_bf16.to_mkldnn()))

    def test_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_bf16_base(dim=2, input=x)

    def test_avg_pool3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, 64, dtype=torch.float32) * 10
        self._test_avg_pool_bf16_base(dim=3, input=x)

    def test_avg_pool2d_stride_none(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 64, 64, dtype=torch.float32) * 10

        for count_include_pad in [True, False]:
            y1 = F.avg_pool2d(
                x,
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)
            y2 = F.avg_pool2d(
                x.to_mkldnn(),
                kernel_size=3,
                stride=None,
                padding=1,
                count_include_pad=count_include_pad)

            self.assertEqual(y1, y2.to_dense())

    def test_adaptive_avg_pool2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)
        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        y1 = adaptive_avg_pool2d(x1)
        y2 = adaptive_avg_pool2d(x2).to_dense()

        loss1 = y1.sum()
        loss2 = y2.sum()
        loss1.backward()
        loss2.backward()

        self.assertEqual(y1, y2)
        self.assertEqual(x1.grad, x2.grad.to_dense())

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_adaptive_avg_pool2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 10, (1,)).item()
        x = torch.randn(N, C, 224, 224, dtype=torch.float32) * 100

        x_bf16 = x.bfloat16()
        adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)

        if has_bf16_support():
            y = adaptive_avg_pool2d(x.to_mkldnn()).to_dense()
            y_bf16 = adaptive_avg_pool2d(x.to_mkldnn()).to_dense(torch.float32)
            self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
        else:
            msg = "mkldnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
            self.assertRaisesRegex(RuntimeError,
                                   msg,
                                   lambda: adaptive_avg_pool2d(x_bf16.to_mkldnn()))

    def _test_batch_norm_base(self, dim, channels, input):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        bn = bn_module[dim](channels).float().train(False)
        mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
        self.assertEqual(
            bn(input),
            mkldnn_bn(input.to_mkldnn()).to_dense())

        self._test_serialization(mkldnn_bn, (input.to_mkldnn(),))
        self._test_tracing(mkldnn_bn, (input.to_mkldnn(),))

    def _test_batch_norm_train_base(self, dim, channels, input):
        # TODO: support 3d batchnorm training.
        bn_module = {2 : torch.nn.BatchNorm2d}
        # TODO: support none affine.
        options = itertools.product([True], [True, False])
        for affine, track_running_stats in options:
            bn = bn_module[dim](
                num_features=channels,
                affine=affine,
                track_running_stats=track_running_stats).float().train(True)
            mkldnn_bn = copy.deepcopy(bn)
            x1 = input.clone().requires_grad_()
            x2 = input.clone().to_mkldnn().requires_grad_()
            y1 = bn(x1)
            y2 = mkldnn_bn(x2).to_dense()
            loss1 = y1.sum()
            loss2 = y2.sum()
            loss1.backward()
            loss2.backward()
            self.assertEqual(y1, y2)
            self.assertEqual(x1.grad, x2.grad.to_dense())
            self.assertEqual(bn.weight.grad, mkldnn_bn.weight.grad, rtol=1e-3, atol=1e-3)
            if track_running_stats:
                self.assertEqual(bn.running_mean, mkldnn_bn.running_mean)
                self.assertEqual(bn.running_var, mkldnn_bn.running_var, rtol=1e-5, atol=1e-5)

    def test_batch_norm_2d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=2, channels=C, input=x)
        self._test_batch_norm_train_base(dim=2, channels=C, input=x)

    def test_batch_norm_3d(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        self._test_batch_norm_base(dim=3, channels=C, input=x)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def _test_batch_norm_bf16_base(self, dim, channels, input):
        bn_module = {2 : torch.nn.BatchNorm2d, 3 : torch.nn.BatchNorm3d}
        x_bf16 = input.bfloat16()
        # TODO: support training
        for train in [False]:
            bn = bn_module[dim](channels).float().train(train)
            mkldnn_bn = mkldnn_utils.to_mkldnn(copy.deepcopy(bn))
            if has_bf16_support():
                y = bn(input.to_mkldnn().to_dense())
                y_bf16 = bn(input.to_mkldnn().to_dense(torch.float))
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: bn(x_bf16.to_mkldnn()))

    def test_batch_norm_2d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 35, 45, dtype=torch.float32) * 10
        self._test_batch_norm_bf16_base(dim=2, channels=C, input=x)

    def test_batch_norm_3d_bf16(self):
        N = torch.randint(3, 10, (1,)).item()
        C = torch.randint(3, 100, (1,)).item()
        x = torch.randn(N, C, 30, 30, 30, dtype=torch.float32) * 10
        self._test_batch_norm_bf16_base(dim=3, channels=C, input=x)

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

        # add_out inplace case: first input
        torch.add(x, y, alpha=alpha, out=x)
        torch.add(mx, my, alpha=alpha, out=mx)
        self.assertEqual(x, mx.to_dense())

        # add_out inplace case: second input
        torch.add(x, y, alpha=alpha, out=y)
        torch.add(mx, my, alpha=alpha, out=my)
        self.assertEqual(y, my.to_dense())

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

    def test_0_dimension_tensor(self):
        x = torch.rand([20, 20, 1, 1], dtype=torch.float)
        y = torch.rand([20, 20, 0, 1], dtype=torch.float)

        # unary ops work without modification
        out_relu = torch.relu(y)
        out_relu_mkldnn = torch.relu(y.to_mkldnn()).to_dense()
        self.assertEqual(out_relu, out_relu_mkldnn)

        out_mul = x * y
        out_mul_mkldnn = (x.to_mkldnn() * y.to_mkldnn()).to_dense()
        self.assertEqual(out_mul, out_mul_mkldnn)

        out_add = x + y
        out_add_mkldnn = (x.to_mkldnn() + y.to_mkldnn()).to_dense()
        self.assertEqual(out_add, out_add_mkldnn)

        x.requires_grad_(True)
        y.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "0-dimension Tensor in training"):
            x.to_mkldnn() + y.to_mkldnn()

        with self.assertRaisesRegex(RuntimeError, "must match"):
            torch.rand([5]).to_mkldnn() + torch.rand([0]).to_mkldnn()

        C = 7
        m = torch.nn.Conv2d(C, C, 3)
        x = torch.randn(0, C, C, 8, dtype=torch.float)
        out_eager = m(x)
        out_mkldnn = mkldnn_utils.to_mkldnn(m)(x)
        self.assertEqual(out_eager, out_mkldnn)

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

    def test_reshape_blocked_format(self):
        # construct an mkldnn blocked tensor with mkldnn conv2d
        C = 7
        m = mkldnn_utils.to_mkldnn(torch.nn.Conv2d(C, C, 3))
        x = torch.randn(1, C, 8, 8).to_mkldnn()

        # mkldnn tensor w/ blocked format
        y_block = m(x)
        # aten tensor w/ plain format
        y_plain = y_block.to_dense()

        y_block_reshape = y_block.reshape(C, -1)
        y_plain_reshape = y_plain.reshape(C, -1)

        self.assertEqual(y_plain_reshape, y_block_reshape.to_dense())

    def test_reshape_backward(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        size = (x.size(0), -1)

        x1 = x.clone().requires_grad_()
        x2 = x.clone().to_mkldnn().requires_grad_()
        in_features = 20
        out_features = torch.randint(3, 100, (1,)).item()
        linear = torch.nn.Linear(in_features, out_features).float()

        y1 = linear(x1.reshape(size)).sum()
        y2 = linear(x2.reshape(size).to_dense()).sum()
        y1.backward()
        y2.backward()
        self.assertEqual(x1.grad, x2.grad.to_dense())

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

    def test_linear_non_contiguous_weight(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        w = torch.randn(in_features, out_features, dtype=torch.float32)
        for bias in [True, False]:
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_mkldnn().requires_grad_()
            linear = torch.nn.Linear(in_features, out_features).float()
            linear.weight = torch.nn.Parameter(w.t())
            mkldnn_linear = copy.deepcopy(linear)
            y1 = linear(x1).sum()
            y2 = mkldnn_linear(x2).to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())
            self.assertEqual(linear.weight.grad, mkldnn_linear.weight.grad)
            if bias:
                self.assertEqual(linear.bias.grad, mkldnn_linear.bias.grad)

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

    def test_linear_backward(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        for bias in [True, False]:
            x1 = x.clone().requires_grad_()
            x2 = x.clone().to_mkldnn().requires_grad_()
            linear = torch.nn.Linear(in_features, out_features).float()
            mkldnn_linear = copy.deepcopy(linear)
            y1 = linear(x1).sum()
            y2 = mkldnn_linear(x2).to_dense().sum()
            y1.backward()
            y2.backward()
            self.assertEqual(x1.grad, x2.grad.to_dense())
            self.assertEqual(linear.weight.grad, mkldnn_linear.weight.grad)
            if bias:
                self.assertEqual(linear.bias.grad, mkldnn_linear.bias.grad)

    @unittest.skipIf(IS_WINDOWS, "Limit support for bf16 path")
    def test_linear_bf16(self):
        in_features = torch.randint(3, 10, (1,)).item()
        out_features = torch.randint(3, 100, (1,)).item()
        x = torch.randn(3, in_features, dtype=torch.float32) * 10
        x_bf16 = x.bfloat16()

        for bias in [True, False]:
            linear = torch.nn.Linear(in_features, out_features, bias=bias).float()
            mkldnn_linear = mkldnn_utils.to_mkldnn(copy.deepcopy(linear))
            mkldnn_linear_bf16 = mkldnn_utils.to_mkldnn(copy.deepcopy(linear), torch.bfloat16)
            if has_bf16_support():
                y = mkldnn_linear(x.to_mkldnn()).to_dense()
                y_bf16 = mkldnn_linear_bf16(x_bf16.to_mkldnn()).to_dense(torch.float32)
                self.assertEqual(y, y_bf16, atol=1e-1, rtol=1e-3)
            else:
                msg = "mkldnn_linear: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq"
                self.assertRaisesRegex(RuntimeError,
                                       msg,
                                       lambda: mkldnn_linear_bf16(x_bf16.to_mkldnn()))

    def test_softmax(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32) * 10
        for dim in range(x.ndim):
            softmax = torch.nn.Softmax(dim=dim)
            self.assertEqual(
                softmax(x),
                softmax(x.to_mkldnn()).to_dense())

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


if __name__ == '__main__':
    run_tests()
