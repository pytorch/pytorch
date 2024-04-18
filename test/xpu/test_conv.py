# Owner(s): ["module: intel"]

import itertools
import math
import unittest
from itertools import product

import torch

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from torch.testing import make_tensor
from torch.testing._internal.common_cuda import tf32_is_not_fp32
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_nn import _test_module_empty_input, NNTestCase
from torch.testing._internal.common_utils import (
    dtype2prec_DONTUSE,
    gradcheck,
    gradgradcheck,
    parametrize as parametrize_test,
    run_tests,
    set_default_dtype,
    skipIfRocmVersionLessThan,
    TEST_SCIPY,
    TEST_WITH_ROCM,
)

AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

if TEST_SCIPY:
    import scipy.ndimage
    import scipy.signal


class TestConvolutionNNDeviceType(NNTestCase):
    def run_conv_double_back_test(
        self,
        kern,
        stride,
        padding,
        chan_in,
        chan_out,
        batch_size,
        inp_size,
        dilation,
        no_weight,
        groups=1,
        use_xpu=False,
        use_bias=True,
        dtype=torch.double,
    ):
        device = torch.device("xpu" if use_xpu else "cpu")

        x = torch.randn(
            batch_size,
            chan_in,
            inp_size,
            inp_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        weight = torch.randn(
            chan_out,
            chan_in // groups,
            kern,
            kern,
            device=device,
            dtype=dtype,
            requires_grad=not no_weight,
        )
        if use_bias:
            bias = torch.randn(chan_out, device=device, dtype=dtype, requires_grad=True)
        else:
            bias = None

        def func(*inputs):
            if use_bias:
                lx, lweight, lbias = inputs
            else:
                lx, lweight = inputs
                lbias = None
            # We disable cudnn during forward to avoid finite difference imprecision issues
            # with cudnn.flags(enabled=False):
            out = F.conv2d(lx, lweight, lbias, stride, padding, dilation, groups)
            return out

        if use_bias:
            inputs = x, weight, bias
        else:
            inputs = x, weight

        dummy_out = func(*inputs)
        grad_y = torch.randn_like(
            dummy_out, device=device, dtype=dtype, requires_grad=True
        )

        # Issue #15353: test mkldnn double backward, don't run gradgradcheck due
        # to imprecision issues
        if dtype == torch.float:
            (g,) = torch.autograd.grad(dummy_out.sum(), x, create_graph=True)
            return g.requires_grad

        return gradgradcheck(func, inputs, (grad_y,))

    @unittest.skipIf(True, "deterministic not supported for oneDNN version<3.4")
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_Conv2d_deterministic_cudnn(self, device, dtype):
        inputs = torch.randn(2, 3, 5, 5, device=device, dtype=dtype, requires_grad=True)
        with cudnn.flags(enabled=True, benchmark=True, deterministic=True):
            conv1 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2 = torch.nn.Conv2d(3, 3, 3).to(device, dtype)
            conv2.bias.data.copy_(conv1.bias.data)
            conv2.weight.data.copy_(conv1.weight.data)
            out1 = conv1(inputs)
            out2 = conv2(inputs)
            self.assertEqual(out1, out2, atol=0.0, rtol=0)
            y = torch.randn(out1.size(), device=device, dtype=dtype)
            out1.backward(y)
            out2.backward(y)
            self.assertEqual(
                conv1.bias.grad.data, conv2.bias.grad.data, atol=0.0, rtol=0
            )
            self.assertEqual(
                conv1.weight.grad.data, conv2.weight.grad.data, atol=0.0, rtol=0
            )

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_Conv2d_large_workspace(self, device, dtype):
        # These sizes require huge cuDNN workspaces. Make sure we choose a
        # reasonable algorithm that does not run out of memory
        sizes = [
            (1, 256, 109, 175),
            (1, 256, 80, 128),
            (1, 256, 120, 192),
        ]

        def run_test(benchmark):
            with torch.backends.cudnn.flags(enabled=True, benchmark=benchmark):
                conv = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1).to(
                    device, dtype
                )
                for size in sizes:
                    x = torch.randn(size, device=device, dtype=dtype)
                    out = conv(x.detach().clone().requires_grad_())
                    out.backward(torch.ones_like(out))

        run_test(benchmark=False)
        run_test(benchmark=True)

    @dtypes(torch.half, torch.float)
    def test_ConvTranspose2d_large_output_padding(self, device, dtype):
        net1 = torch.nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net2 = torch.nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        net3 = torch.nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        ).to(device=device, dtype=dtype)
        x = torch.rand(1, 128, 6, 6, device=device, dtype=dtype, requires_grad=True)
        x = net1(x)
        x = net2(x)
        x = net3(x)
        x.backward(torch.randn_like(x))

    @dtypes(torch.float, torch.double, torch.half)
    def test_Conv2d_depthwise_naive_groups(self, device, dtype):
        if dtype == torch.half and "xpu" in device:
            self.skipTest(
                "The accuracy issue of dtype fp16 would be fixed in oneDNN v3.4"
            )
        for depth_multiplier in [1, 2]:
            m = nn.Conv2d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(2, 2 * depth_multiplier, 4, 4, device=device, dtype=dtype)
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv2d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())

            self.assertEqual(
                output,
                torch.cat([output1, output2], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )

    @dtypes(torch.float, torch.double, torch.half)
    def test_Conv3d_depthwise_naive_groups(self, device, dtype):
        for depth_multiplier in [1, 2]:
            m = nn.Conv3d(2, 2 * depth_multiplier, kernel_size=3, groups=2).to(
                device, dtype
            )
            i = (
                torch.randn(2, 2, 6, 6, 6, device=device, dtype=dtype)
                .div_(2)
                .requires_grad_()
            )
            output = m(i)
            grad_output = (
                torch.randn(
                    2, 2 * depth_multiplier, 4, 4, 4, device=device, dtype=dtype
                )
                / 2
            )
            output.backward(grad_output)

            offset = 1 * depth_multiplier

            m1 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m1.weight.data = m.weight.data[:offset].clone()
            m1.bias.data = m.bias.data[:offset].clone()
            i1 = i.detach()[:, :1].clone().requires_grad_()
            output1 = m1(i1)
            output1.backward(grad_output[:, :offset].contiguous())

            m2 = nn.Conv3d(1, 1 * depth_multiplier, kernel_size=3).to(device, dtype)
            m2.weight.data.copy_(m.weight.data[offset:])
            m2.bias.data.copy_(m.bias.data[offset:])
            i2 = i.detach()[:, 1:].clone().requires_grad_()
            output2 = m2(i2)
            output2.backward(grad_output[:, offset:].contiguous())
            is_cuda_sm86 = device.startswith(
                "cuda"
            ) and torch.cuda.get_device_capability(0) == (8, 6)
            atol, rtol = (
                (3e-4, 3e-2)
                if dtype == torch.float32 and is_cuda_sm86
                else (dtype2prec_DONTUSE[dtype], 0)
            )

            self.assertEqual(
                output, torch.cat([output1, output2], 1), atol=atol, rtol=rtol
            )
            self.assertEqual(
                i.grad.data,
                torch.cat([i1.grad.data, i2.grad.data], 1),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.bias.grad.data,
                torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
                atol=dtype2prec_DONTUSE[dtype],
                rtol=0,
            )
            self.assertEqual(
                m.weight.grad.data,
                torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
                atol=atol,
                rtol=rtol,
            )

    @dtypes(torch.float, torch.double, torch.half)
    def test_noncontig_conv_grad(self, device, dtype):
        # FIXME: remove after adding non-contiguous grad tests for all modules
        module = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device, dtype)
        input = torch.randn(
            2, 3, 10, 10, dtype=dtype, device=device, requires_grad=True
        )
        output = module(input)

        grad = torch.randn(2, 2, 5, 10, 10, dtype=dtype, device=device)[:, 1]
        assert not grad.is_contiguous()
        output.backward(grad, retain_graph=True)
        self.assertIsNotNone(input.grad)
        result = input.grad.data.clone()
        input.grad.data.zero_()

        output.backward(grad.contiguous())
        self.assertEqual(
            result, input.grad.data, atol=dtype2prec_DONTUSE[dtype], rtol=0
        )

    @dtypes(torch.double)
    def test_conv_double_backward(self, device, dtype):
        with torch.backends.cudnn.flags(enabled=True, deterministic=True):
            # Double backward only runs with DoubleTensor due to precision reason
            batch_size = 1
            for kern, inp_size, dilations in [(3, 5, [1, 2]), (4, 9, [1])]:
                for stride, padding, chan_in, chan_out, dilation in product(
                    [1], [2], [2], [3], dilations
                ):
                    no_weight = stride == 2
                    result = self.run_conv_double_back_test(
                        kern,
                        stride,
                        padding,
                        chan_in,
                        chan_out,
                        batch_size,
                        inp_size,
                        dilation,
                        no_weight,
                        use_xpu=True,
                        dtype=dtype,
                    )
                    self.assertTrue(
                        result,
                        "Conv double backward test failed with parameters:"
                        + "\nkern: "
                        + str(kern)
                        + "\nstride: "
                        + str(stride)
                        + "\npadding: "
                        + str(padding)
                        + "\nchan_in: "
                        + str(chan_in)
                        + "\nchan_out: "
                        + str(chan_out)
                        + "\nbatch_size: "
                        + str(batch_size)
                        + "\ninp_size: "
                        + str(inp_size)
                        + "\ndilation: "
                        + str(dilation),
                    )

    def test_conv_double_backward_no_bias(self):
        kern = 3
        stride = 2
        chan_in, chan_out = 2, 4
        batch_size = 2
        inp_size = 5
        padding = 1
        dilation = 1
        no_weight = False
        use_bias = True
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in,
            chan_out,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            use_bias=use_bias,
        )
        self.assertTrue(
            result,
            "Conv double backward test failed with parameters:"
            + "\nkern: "
            + str(kern)
            + "\nstride: "
            + str(stride)
            + "\npadding: "
            + str(padding)
            + "\nchan_in: "
            + str(chan_in)
            + "\nchan_out: "
            + str(chan_out)
            + "\nbatch_size: "
            + str(batch_size)
            + "\ninp_size: "
            + str(inp_size)
            + "\ndilation: "
            + str(dilation),
        )

    def test_conv_double_backward_groups(self):
        kern = 3
        stride = 1
        padding = 2
        chan_in, chan_out = 2, 4
        batch_size = 2
        inp_size = 6
        dilation = 1
        no_weight = False
        groups = 2
        result = self.run_conv_double_back_test(
            kern,
            stride,
            padding,
            chan_in * groups,
            chan_out * groups,
            batch_size,
            inp_size,
            dilation,
            no_weight,
            groups=groups,
        )
        self.assertTrue(
            result,
            "Conv double backward test failed with parameters:"
            + "\nkern: "
            + str(kern)
            + "\nstride: "
            + str(stride)
            + "\npadding: "
            + str(padding)
            + "\nchan_in: "
            + str(chan_in)
            + "\nchan_out: "
            + str(chan_out)
            + "\nbatch_size: "
            + str(batch_size)
            + "\ninp_size: "
            + str(inp_size)
            + "\ndilation: "
            + str(dilation)
            + "\ngroups: "
            + str(groups),
        )

    def test_conv_double_backward_stride(self):
        batch_size = 2

        # Cannot provide ggW when stride is > 1
        for kern, inp_size, dilations in [(3, 5, [1, 2]), (3, 7, [1])]:
            for stride, padding, chan_in, chan_out, dilation in product(
                [2], [0, 1], [1], [2], dilations
            ):
                no_weight = False
                self.run_conv_double_back_test(
                    kern,
                    stride,
                    padding,
                    chan_in,
                    chan_out,
                    batch_size,
                    inp_size,
                    dilation,
                    no_weight,
                )

    @dtypes(torch.float)
    def test_conv1d_same_padding(self, device, dtype):
        # Test padding='same' outputs the correct shape
        test_args = [
            # in_size
            range(50, 55),
            # kernel_size
            [1, 2, 3, 8],
            # dilation
            range(1, 4),
            # stride
            [1],
        ]
        for in_size, k_size, dilation, stride in itertools.product(*test_args):
            x = torch.rand(1, 1, in_size, device=device, dtype=dtype)
            y = torch.rand(1, 1, k_size, device=device, dtype=dtype)
            z = F.conv1d(x, y, padding="same", dilation=dilation, stride=stride)
            self.assertEqual(z.size(2), int(math.ceil(in_size / stride)))

        # Compare F.conv1d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 3, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=1)
        actual = F.conv1d(x, y, padding="same")
        self.assertEqual(expect, actual)

        # With dilation
        x = torch.rand(1, 1, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y, padding=3, dilation=2)
        actual = F.conv1d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual)

        # Dilation with asymmetric padding
        expect = F.conv1d(x, y, padding=5, dilation=3)[..., 1:]
        actual = F.conv1d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv3d_same_padding(self, device, dtype):
        if dtype is torch.cfloat:
            rtol, atol = 2e-6, 2e-6
        else:
            rtol, atol = None, None
        # Compare F.conv3d padding='same' output against manual padding
        # Without strides/dilation
        x = torch.rand(1, 1, 10, 11, 12, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 2, 5, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=(0, 1, 2))[..., :, 1:, :]
        actual = F.conv3d(x, y, padding="same")
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # With dilation
        expect = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        actual = F.conv3d(x, y, padding="same", dilation=2)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

        # Dilation with asymmetric padding
        y = torch.rand(1, 1, 4, 4, 4, device=device, dtype=dtype)
        expect = F.conv3d(x, y, padding=5, dilation=3)[..., 1:, 1:, 1:]
        actual = F.conv3d(x, y, padding="same", dilation=3)
        self.assertEqual(expect, actual, rtol=rtol, atol=atol)

    @dtypes(torch.float)
    def test_conv1d_valid_padding(self, device, dtype):
        # Test F.conv1d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 10, device=device, dtype=dtype)
        y = torch.rand(1, 1, 4, device=device, dtype=dtype)
        expect = F.conv1d(x, y)
        actual = F.conv1d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv2d_valid_padding(self, device, dtype):
        # Test F.conv2d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype)
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype)
        expect = F.conv2d(x, y)
        actual = F.conv2d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv3d_valid_padding(self, device, dtype):
        # Test F.conv3d padding='valid' is the same as no padding
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device)
        expect = F.conv3d(x, y)
        actual = F.conv3d(x, y, padding="valid")
        self.assertEqual(expect, actual)

    @dtypes(torch.float)
    def test_conv1d_same_padding_backward(self, device, dtype):
        # Test F.conv1d gradients work with padding='same'
        x = torch.rand(1, 1, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv1d(x, y, padding=3, dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # Asymmetric padding
        z = F.conv1d(x, y, padding=2)[..., 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv1d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    @dtypes(torch.float)
    def test_conv2d_same_padding_backward(self, device, dtype):
        # Test F.conv2d gradients work with padding='same'
        x = torch.rand(1, 1, 10, 11, device=device, dtype=dtype, requires_grad=True)
        y = torch.rand(1, 1, 4, 5, device=device, dtype=dtype, requires_grad=True)

        # Symmetric padding
        z = F.conv2d(x, y, padding=(3, 4), dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv2d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        # Asymmetric padding
        y = torch.rand(1, 1, 4, 4, device=device, dtype=dtype, requires_grad=True)
        z = F.conv2d(x, y, padding=2)[..., 1:, 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv2d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

    @dtypes(torch.double)
    def test_conv3d_same_padding_backward(self, device, dtype):
        check_forward_ad = torch.device(device).type != "xla"

        # Test F.conv3d gradients work with padding='same'
        x = torch.rand(1, 1, 1, 11, 12, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 2, 5, dtype=dtype, device=device, requires_grad=True)

        # Symmetric padding
        z = F.conv3d(x, y, padding=(0, 1, 4), dilation=2)
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding="same", dilation=2)
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)
        x.grad, y.grad = None, None

        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
            (x, y),
            check_forward_ad=check_forward_ad,
            nondet_tol=1e-5,
        )
        if torch.device(device).type != "cuda":
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(
                lambda x, y: F.conv3d(x, y, padding="same", dilation=2),
                (x, y),
                check_fwd_over_rev=True,
            )

        # Asymmetric padding
        y = torch.rand(1, 1, 1, 4, 4, dtype=dtype, device=device, requires_grad=True)
        z = F.conv3d(x, y, padding=2)[..., 1:, 1:]
        z.sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        z = F.conv3d(x, y, padding="same")
        z.sum().abs().backward()
        self.assertEqual(gx_expect, x.grad)
        self.assertEqual(gy_expect, y.grad)

        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="same"),
            (x, y),
            check_forward_ad=check_forward_ad,
            nondet_tol=1e-5,
        )
        if torch.device(device).type != "cuda":
            # https://github.com/pytorch/pytorch/issues/70702
            gradgradcheck(
                lambda x, y: F.conv3d(x, y, padding="same"),
                (x, y),
                check_fwd_over_rev=True,
            )

    @dtypes(torch.float)
    def test_conv1d_valid_padding_backward(self, device, dtype):
        # Test F.conv1d gradients work with padding='valid'
        x = torch.rand(1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 4, dtype=dtype, device=device, requires_grad=True)
        F.conv1d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv1d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv1d_vs_scipy(self, device, dtype, mode):
        t = make_tensor((1, 10), device=device, dtype=dtype)
        feat_dim = t.shape[1]
        weight_even = make_tensor((1, 1, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy expects two 1-D inputs.
            t_a = t.view(-1).cpu().numpy()
            w_a = weight.view(-1).cpu().numpy()
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                # `same` padding in PyTorch conv1d is different
                # from SciPy
                p = weight.shape[2] // 2
                t = torch.nn.functional.pad(t, (p, p))
                # We have already taken care of padding
                kwargs.pop("padding")

            # second input is flipped in SciPy's convolve
            weight_flipped = torch.flip(weight, (2,))
            actual = torch.nn.functional.conv1d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:feat_dim]

            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-5)

        # Global dtype for this test suite is torch.double
        # This leads to change in type-promotion
        # and conv1d outputs `complex128` for `complex64` input.
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv2d_vs_scipy(self, device, dtype, mode):
        t = make_tensor((1, 5, 10), device=device, dtype=dtype)
        weight_even = make_tensor((1, 1, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy expects two 2-D inputs.
            t_a = t.squeeze(0).cpu().numpy()
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            expected = scipy.signal.convolve2d(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                # `same` padding in PyTorch conv2d is different
                # from SciPy
                left_right_pad = weight.shape[3] // 2
                top_bottom_pad = weight.shape[2] // 2
                p = (left_right_pad, left_right_pad, top_bottom_pad, top_bottom_pad)
                t = torch.nn.functional.pad(t, p)
                # We have already taken care of padding
                kwargs.pop("padding")

            # second input is flipped in SciPy's convolve2d
            weight_flipped = torch.flip(weight, (2, 3))
            actual = torch.nn.functional.conv2d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:5, :10]

            self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # Global dtype for this test suite is torch.double
        # This leads to change in type-promotion
        # and conv1d outputs `complex128` for `complex64` input.
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @unittest.skipIf(not TEST_SCIPY, "Scipy required for the test.")
    @dtypes(torch.float)
    @parametrize_test("mode", ("valid", "same"))
    def test_conv3d_vs_scipy(self, device, dtype, mode):
        t = make_tensor((1, 5, 5, 10), device=device, dtype=dtype)
        weight_even = make_tensor((1, 1, 2, 2, 4), device=device, dtype=dtype)
        weight_odd = make_tensor((1, 1, 2, 3, 5), device=device, dtype=dtype)

        def _test(t, weight, mode):
            # SciPy expects two 3-D inputs.
            t_a = t.squeeze(0).cpu().numpy()
            w_a = weight.squeeze(0).squeeze(0).cpu().numpy()
            expected = scipy.signal.convolve(t_a, w_a, mode=mode)

            kwargs = {"padding": mode}
            if mode == "same":
                # `same` padding in PyTorch conv3d is different
                # from SciPy
                left_right_pad = weight.shape[4] // 2
                top_bottom_pad = weight.shape[3] // 2
                front_back_pad = weight.shape[2] // 2
                p = (
                    left_right_pad,
                    left_right_pad,
                    top_bottom_pad,
                    top_bottom_pad,
                    front_back_pad,
                    front_back_pad,
                )
                t = torch.nn.functional.pad(t, p)
                # We have already taken care of padding
                kwargs.pop("padding")

            # second input is flipped in SciPy's convolve
            weight_flipped = torch.flip(weight, (2, 3, 4))
            actual = torch.nn.functional.conv3d(t, weight_flipped, **kwargs).squeeze(0)
            if mode == "same":
                actual = actual[:5, :5, :10]

            if tf32_is_not_fp32() and (
                dtype == torch.float or dtype == torch.complex64
            ):
                self.assertEqual(actual, expected, atol=0.05, rtol=0.05)
            else:
                self.assertEqual(actual, expected, rtol=2e-5, atol=5e-6)

        # Global dtype for this test suite is torch.double
        # This leads to change in type-promotion
        # and conv1d outputs `complex128` for `complex64` input.
        with set_default_dtype(torch.float):
            _test(t, weight_even, mode)
            _test(t, weight_odd, mode)

    @dtypes(torch.float)
    def test_conv2d_valid_padding_backward(self, device, dtype):
        # Test F.conv2d gradients work with padding='valid'
        x = torch.rand(1, 1, 1, 10, device=device, dtype=dtype, requires_grad=True)
        y = torch.rand(1, 1, 1, 4, device=device, dtype=dtype, requires_grad=True)
        F.conv2d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv2d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

    @dtypes(torch.double)
    def test_conv3d_valid_padding_backward(self, device, dtype):
        check_forward_ad = torch.device(device).type != "xla"

        # Test F.conv3d gradients work with padding='valid'
        x = torch.rand(1, 1, 1, 1, 10, dtype=dtype, device=device, requires_grad=True)
        y = torch.rand(1, 1, 1, 1, 4, dtype=dtype, device=device, requires_grad=True)
        F.conv3d(x, y, padding=0).sum().abs().backward()
        gx_expect, gy_expect = x.grad, y.grad
        x.grad, y.grad = None, None

        F.conv3d(x, y, padding="valid").sum().abs().backward()
        gx_actual, gy_actual = x.grad, y.grad
        self.assertEqual(gx_expect, gx_actual)
        self.assertEqual(gy_expect, gy_actual)

        gradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_forward_ad=check_forward_ad,
        )
        gradgradcheck(
            lambda x, y: F.conv3d(x, y, padding="valid"),
            (x, y),
            check_fwd_over_rev=check_forward_ad,
        )

    @parametrize_test("N", range(2, 4), name_fn=lambda N: f"ConvTranspose{N}d")
    def test_conv_transpose_with_output_size_and_no_batch_dim(self, device, N):
        # For inputs with no batch dim, verify output is the correct shape when output_size is set.
        # See https://github.com/pytorch/pytorch/issues/75889
        inp = torch.randn((1, 15, 13) if N == 2 else (1, 15, 13, 13), device=device)
        output_size = (1, 240, 200) if N == 2 else (1, 240, 200, 200)
        ConvTransposeNd = getattr(nn, f"ConvTranspose{N}d")
        m = ConvTransposeNd(
            1, 1, kernel_size=16, stride=16, padding=7, bias=False, device=device
        )
        output = m(inp, output_size=output_size)
        self.assertEqual(output.shape, output_size)

    @dtypes(torch.float)
    def test_conv_empty_channel(self, device, dtype):
        in_channels = 0
        mod = torch.nn.Conv1d(in_channels, 8, 2, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 15, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 0, device=device, dtype=dtype)
            mod(inp)

        mod = torch.nn.Conv2d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 50, 100, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 40, 0, device=device, dtype=dtype)
            mod(inp)

        mod = torch.nn.Conv3d(in_channels, 33, 3, stride=2, dtype=dtype).to(device)
        inp = torch.randn(2, 0, 50, 20, 40, device=device, dtype=dtype)
        _test_module_empty_input(self, mod, inp, check_size=False)

        with self.assertRaisesRegex(RuntimeError, "Given groups=1, weight"):
            inp = torch.randn(2, 1, 50, 0, 40, device=device, dtype=dtype)
            mod(inp)

    def test_group_conv_empty(self, device):
        mod = torch.nn.Conv2d(4, 4, stride=2, kernel_size=3, padding=1, groups=4).to(
            device
        )
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    def test_group_convTranspose_empty(self, device):
        mod = torch.nn.ConvTranspose2d(
            4, 4, stride=2, kernel_size=3, padding=1, groups=4
        ).to(device)
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    def test_convTranspose_empty(self, device):
        mod = torch.nn.ConvTranspose2d(4, 4, stride=2, kernel_size=3, padding=1).to(
            device
        )
        inp = torch.randn(0, 4, 4, 4, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        if self.device_type == "cuda" and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                _test_module_empty_input(self, mod, inp, check_size=False)

    def test_conv_large_nosplit(self, device):
        # Here we just test the convolution correctly route to the fallback implementation
        # that is, it does not crash. The correctness of fallback implementation should be
        # covered in other tests
        dtype = torch.half
        conv1 = nn.Conv2d(2, 2, 8, 8).to(device).to(dtype)
        input_large = torch.randn(1, 2, 1024, 1024 * 1024, dtype=dtype, device=device)
        conv1(input_large)
        conv2 = torch.nn.Conv2d(1, 1024, 1, 1).to(device).to(dtype)
        input_large = torch.randn(1, 1, 2048, 1024, dtype=dtype, device=device)
        conv2(input_large)

    def test_conv_noncontig_weights(self, device):
        for dim in (1, 2, 3):
            for grouped in (False, True):
                nc = 3
                groups = 3 if grouped else 1
                w = torch.randn([3] * dim, device=device)
                w = w.expand([nc, int(nc / groups)] + list(w.shape))
                w = w.detach().requires_grad_()
                x = torch.randn(
                    [1, nc] + ([5] * dim), device=device, requires_grad=True
                )
                y = getattr(F, f"conv{dim}d")(x, w, groups=groups)
                y.sum().backward()
                y = getattr(F, f"conv_transpose{dim}d")(x, w, groups=groups)
                y.sum().backward()

    def test_conv_noncontig_weights_and_bias(self, device):
        # need floats to exercise https://github.com/pytorch/pytorch/issues/16018
        for bias in [True, False]:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=bias).to(
                device, torch.float
            )

            input_nc = torch.randn(
                (1, 3, 224, 224, 2), device=device, dtype=torch.float
            )[:, :, :, :, 1]
            input_c = input_nc.contiguous()

            weight_nc = torch.randn((64, 3, 7, 7, 2), device=device, dtype=torch.float)[
                :, :, :, :, 1
            ]
            conv1.weight = nn.Parameter(weight_nc)
            weight_c = conv1.weight.contiguous()

            if bias:
                bias_nc = torch.randn((64, 2), device=device, dtype=torch.float)[:, 1]
                conv1.bias = nn.Parameter(bias_nc)
                bias_c = conv1.bias.contiguous()

            out1 = conv1(input_nc)
            conv1.weight = nn.Parameter(weight_c)
            if bias:
                conv1.bias = nn.Parameter(bias_c)
            out2 = conv1(input_c)
            self.assertEqual(out1, out2)

    @largeTensorTest("12GB")
    @skipIfRocmVersionLessThan((6, 0))
    def test_conv_transposed_large(self, device):
        dtype = torch.half if self.device_type == "cuda" else torch.float
        conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False).to(device).to(dtype)
        input_large = torch.randn(4096, 1, 512, 1024, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        maxdiff0 = (
            (ret.narrow(0, 0, 1024) - conv(input_large.narrow(0, 0, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff1 = (
            (ret.narrow(0, 1024, 1024) - conv(input_large.narrow(0, 1024, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff2 = (
            (ret.narrow(0, 2048, 1024) - conv(input_large.narrow(0, 2048, 1024)))
            .abs_()
            .max()
            .item()
        )
        maxdiff3 = (
            (ret.narrow(0, 3072, 1024) - conv(input_large.narrow(0, 3072, 1024)))
            .abs_()
            .max()
            .item()
        )
        if self.device_type == "cuda":
            # cuDNN may use algorithms such as FFT that don't guarantee a diff of 0
            self.assertEqual(maxdiff0, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff1, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff2, 0, atol=2e-3, rtol=1e-5)
            self.assertEqual(maxdiff3, 0, atol=2e-3, rtol=1e-5)
        else:
            self.assertEqual(maxdiff0, 0)
            self.assertEqual(maxdiff1, 0)
            self.assertEqual(maxdiff2, 0)
            self.assertEqual(maxdiff3, 0)

    def test_conv_large(self, device):
        dtype = torch.half if self.device_type == "cuda" else torch.float
        conv = nn.Conv2d(2, 2, 8, 8, bias=False).to(device).to(dtype)
        input_large = torch.randn(4097, 2, 512, 512, dtype=dtype, device=device)
        # forward
        ret = conv(input_large)
        self.assertEqual(ret[:2048], conv(input_large[:2048]))
        self.assertEqual(ret[2048:4096], conv(input_large[2048:4096]))
        self.assertEqual(ret[4096:], conv(input_large[4096:]))

        # backward
        conv.zero_grad()
        # When computing the backward, we are using the `max(dim=1)`` to create
        # some sparsity. Without this sparsity, the rounding error would be
        # too large (as large as 1e-5) to satisfy the creterion (1e-6) of `assertEqual`
        ret.view(4097, -1).max(dim=1).values.sum().backward()
        del ret
        grad1 = conv.weight.grad.detach().clone()
        conv.zero_grad()
        conv(input_large[:2048]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[2048:4096]).view(2048, -1).max(dim=1).values.sum().backward()
        conv(input_large[4096:]).view(1, -1).max(dim=1).values.sum().backward()
        grad2 = conv.weight.grad.detach().clone()
        # gradients are at the order of hundreds, we need to scale it to
        # the order of one so that we can compare
        scale = 1 / grad2.abs().mean()
        grad1 = grad1 * scale
        grad2 = grad2 * scale
        self.assertEqual(grad1, grad2, atol=5e-2, rtol=5e-3)

    def test_Conv2d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    def test_ConvTranspose2d_size_1_kernel(self, device):
        x_cpu = torch.randn(2, 3, 5, 5)
        conv_cpu = torch.nn.ConvTranspose2d(3, 3, kernel_size=1)
        y_cpu = conv_cpu(x_cpu)
        y = torch.rand_like(y_cpu)
        y_cpu.backward(y)

        with cudnn.flags(enabled=False):
            conv_cuda = torch.nn.ConvTranspose2d(3, 3, kernel_size=1).to(device)
            conv_cuda.bias.data.copy_(conv_cpu.bias.data)
            conv_cuda.weight.data.copy_(conv_cpu.weight.data)
            y_cuda = conv_cuda(x_cpu.to(device))
            y_cuda.backward(y.to(device))

        self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
        self.assertEqual(
            conv_cpu.bias.grad.data,
            conv_cuda.bias.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )
        self.assertEqual(
            conv_cpu.weight.grad.data,
            conv_cuda.weight.grad.data,
            atol=1e-5,
            rtol=0,
            exact_device=False,
        )

    def test_ConvTranspose3d_size_1_kernel(self, device):
        with set_default_dtype(torch.double):
            x_cpu = torch.randn(2, 3, 3, 5, 5)
            conv_cpu = torch.nn.ConvTranspose3d(3, 3, kernel_size=1)
            y_cpu = conv_cpu(x_cpu)
            y = torch.rand_like(y_cpu)
            y_cpu.backward(y)

            with cudnn.flags(enabled=False):
                conv_cuda = torch.nn.ConvTranspose3d(3, 3, kernel_size=1).to(device)
                conv_cuda.bias.data.copy_(conv_cpu.bias.data)
                conv_cuda.weight.data.copy_(conv_cpu.weight.data)
                y_cuda = conv_cuda(x_cpu.to(device))
                y_cuda.backward(y.to(device))

            self.assertEqual(y_cpu, y_cuda, atol=1e-5, rtol=0, exact_device=False)
            self.assertEqual(
                conv_cpu.bias.grad.data,
                conv_cuda.bias.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )
            self.assertEqual(
                conv_cpu.weight.grad.data,
                conv_cuda.weight.grad.data,
                atol=1e-5,
                rtol=0,
                exact_device=False,
            )

    @dtypesIfCUDA(
        *floating_types_and(torch.half, *[torch.bfloat16] if AMPERE_OR_ROCM else [])
    )
    @dtypes(torch.float)
    def test_Conv2d_naive_groups(self, device, dtype):
        # Check that grouped convolutions matches two half convolutions
        m = nn.Conv2d(4, 4, kernel_size=3, groups=2).to(device, dtype)
        i = torch.randn(2, 4, 6, 6, device=device, dtype=dtype, requires_grad=True)
        output = m(i)
        grad_output = torch.randn(2, 4, 4, 4, device=device, dtype=dtype)
        output.backward(grad_output)

        m1 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m1.weight.data.copy_(m.weight.data[:2])
        m1.bias.data.copy_(m.bias.data[:2])
        i1 = i.data[:, :2].contiguous().requires_grad_(True)
        output1 = m1(i1)
        output1.backward(grad_output[:, :2].contiguous())

        m2 = nn.Conv2d(2, 2, kernel_size=3).to(device, dtype)
        m2.weight.data.copy_(m.weight.data[2:])
        m2.bias.data.copy_(m.bias.data[2:])
        i2 = i.data[:, 2:].contiguous().requires_grad_(True)
        output2 = m2(i2)
        output2.backward(grad_output[:, 2:].contiguous())

        self.assertEqual(output, torch.cat([output1, output2], 1))
        self.assertEqual(
            i.grad.data,
            torch.cat([i1.grad.data, i2.grad.data], 1),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        self.assertEqual(
            m.bias.grad.data,
            torch.cat([m1.bias.grad.data, m2.bias.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )
        self.assertEqual(
            m.weight.grad.data,
            torch.cat([m1.weight.grad.data, m2.weight.grad.data], 0),
            atol=dtype2prec_DONTUSE[dtype],
            rtol=0,
        )

    @dtypes(torch.double)
    def test_Conv2d_backward_depthwise(self, device, dtype):
        x = torch.randn(2, 2, 4, 20, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(2, 1, 3, 5, device=device, dtype=dtype, requires_grad=True)

        def conv2d_depthwise(x, weight):
            return torch.nn.functional.conv2d(
                x, weight, bias=None, stride=(1, 10), groups=2
            )

        for cudnn_enabled in [False, True]:
            with torch.backends.cudnn.flags(enabled=cudnn_enabled):
                torch.autograd.gradcheck(conv2d_depthwise, (x, weight))

    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_nhwc(self, device, dtype):
        def helper(n, c, h, w, out_channels, kernel_size, groups):
            input = torch.randint(-3, 3, (n, c, h, w), dtype=dtype, device=device).to(
                memory_format=torch.channels_last
            )
            input.requires_grad_()
            conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -3, 3)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv2d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -3, 3)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(input.grad.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 56, 56, out_channels=16, kernel_size=3, groups=16)

    @dtypes(torch.half, torch.float)
    def test_conv_cudnn_ndhwc(self, device, dtype):
        def helper(n, c, d, h, w, out_channels, kernel_size, groups):
            input = torch.randint(
                -2, 2, (n, c, d, h, w), dtype=dtype, device=device
            ).to(memory_format=torch.channels_last_3d)
            input.requires_grad_()
            conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups).to(
                device=device, dtype=dtype, memory_format=torch.channels_last_3d
            )
            for p in conv.parameters():
                p.data = torch.randint_like(p, -2, 2)

            # use FP64 channels-first conv as reference
            ref_input = input.detach().clone().contiguous().double().requires_grad_()
            ref_conv = nn.Conv3d(c, out_channels, kernel_size, groups=groups)
            # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
            ref_conv.load_state_dict(conv.state_dict())
            ref_conv = ref_conv.to(
                device=device, dtype=torch.double, memory_format=torch.contiguous_format
            )

            out = conv(input)
            ref_out = ref_conv(ref_input)

            grad = torch.randint_like(out, -2, 2)
            ref_grad = grad.detach().clone().double().contiguous()

            out.backward(grad)
            ref_out.backward(ref_grad)

            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(
                input.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )
            self.assertTrue(
                conv.weight.grad.is_contiguous(memory_format=torch.channels_last_3d)
            )

            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ref_input.grad.is_contiguous())
            self.assertTrue(ref_conv.weight.grad.is_contiguous())

            self.assertEqual(out, ref_out, exact_dtype=False)
            self.assertEqual(conv.weight.grad, ref_conv.weight.grad, exact_dtype=False)
            self.assertEqual(conv.bias.grad, ref_conv.bias.grad, exact_dtype=False)
            self.assertEqual(input.grad, ref_input.grad, exact_dtype=False)

        helper(2, 8, 4, 4, 4, out_channels=4, kernel_size=3, groups=1)
        helper(2, 8, 4, 4, 4, out_channels=8, kernel_size=3, groups=8)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=1)
        helper(1, 16, 18, 18, 18, out_channels=16, kernel_size=3, groups=16)

    def _run_conv(
        self,
        layer,
        device,
        inp,
        grad,
        ref_conv,
        ref_input,
        ref_out,
        input_format,
        weight_format,
        grad_format,
        output_format,
    ):
        conv = (
            layer(inp.size(1), grad.size(1), ref_conv.weight.size(2)).float().to(device)
        )
        # load_state_dict will restore the stride & memory_layout on ref_conv.weight.
        conv.load_state_dict(ref_conv.state_dict())
        weight_data = (
            conv.weight.detach().clone().contiguous(memory_format=weight_format)
        )
        conv.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=weight_format
        )
        input = inp.clone().contiguous(memory_format=input_format)
        input.resize_(input.size(), memory_format=input_format)
        input = input.requires_grad_()
        grad = grad.contiguous(memory_format=grad_format)
        grad.resize_(grad.size(), memory_format=grad_format)
        out = conv(input)
        out.backward(grad)
        self.assertTrue(out.is_contiguous(memory_format=output_format))
        self.assertEqual(out, ref_out)
        self.assertEqual(conv.weight.grad, ref_conv.weight.grad)
        self.assertEqual(conv.bias.grad, ref_conv.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    def _test_conv_cudnn_nhwc_nchw(self, layer, n, c, h, w, k, filter_size, device):
        data = torch.randint(1, 10, (n, c, h, w), dtype=torch.float32, device=device)
        ref_input = data.clone().contiguous().requires_grad_(True)
        ref_conv = layer(c, k, filter_size).float().to(device)
        ref_out = ref_conv(ref_input)
        grad = torch.randint(1, 10, ref_out.size(), dtype=torch.float32, device=device)
        ref_out.backward(grad)

        for w_f in [torch.contiguous_format, torch.channels_last]:
            for g_f in [torch.contiguous_format, torch.channels_last]:
                for input_format in [torch.contiguous_format, torch.channels_last]:
                    output_format = torch.contiguous_format
                    # # Older versions of CudNN have Channels Last support disabled
                    # if torch.backends.cudnn.version() >= 7603:
                    if input_format == torch.channels_last:
                        output_format = torch.channels_last
                    # This is because we have N111 weight that cannot handle
                    # the ambiguous memory_format
                    if w_f == torch.channels_last:
                        # if layer == nn.Conv2d and filter_size * c != 1:
                        output_format = torch.channels_last
                        # if layer == nn.ConvTranspose2d and filter_size * k != 1:
                        # output_format = torch.channels_last
                    self._run_conv(
                        layer,
                        device,
                        data,
                        grad,
                        ref_conv,
                        ref_input,
                        ref_out,
                        input_format,
                        w_f,
                        g_f,
                        output_format,
                    )

    # torch.half is erroring out on Windows with CUDA 10.1 + cuDNN 7.6.4
    # returning CUDNN_STATUS_BAD_PARAM
    # Disabling that specific test for now [see issue # 33918]
    @dtypes(torch.float, torch.double)
    def test_conv_cudnn_nhwc_support(self, device, dtype):
        input = torch.randn(
            (1, 16, 1, 1), dtype=dtype, device=device, requires_grad=True
        )
        weight = torch.randn(
            (8, 16, 3, 3), dtype=dtype, device=device, requires_grad=True
        )
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        self.assertTrue(o.is_contiguous(memory_format=torch.channels_last))
        o.sum().backward()

    # Test that faster algorithms used for inference produce the same results
    # Validates depthwise3x3 bug reported in https://github.com/pytorch/pytorch/issues/60176
    @dtypes(torch.float)
    def test_conv2d_no_grad(self, device, dtype):
        for batch in [1, 2, 3]:
            for groups in [1, 2, 4]:
                input = torch.rand(batch, groups, 8, 8, dtype=dtype, device=device)
                m = nn.Conv2d(
                    groups,
                    8,
                    kernel_size=(3, 3),
                    groups=groups,
                    dtype=dtype,
                    device=device,
                )
                with torch.no_grad():
                    output_ng = m(input)
                output = m(input)
                self.assertEqual(output, output_ng, rtol=1e-2, atol=1e-5)

    def test_convert_conv2d_weight_memory_format(self, device):
        input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, device=device)
        model = nn.Sequential(nn.Conv2d(8, 4, 3), nn.BatchNorm2d(4)).to(device).float()
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

        model = (
            nn.Sequential(nn.ConvTranspose2d(8, 4, 3), nn.BatchNorm2d(4))
            .to(device)
            .float()
        )
        for memory_format in [torch.channels_last, torch.contiguous_format]:
            model = nn.utils.convert_conv2d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

    def test_convert_conv3d_weight_memory_format(self, device):
        input = torch.randint(
            1, 10, (2, 8, 4, 4, 4), dtype=torch.float32, device=device
        )
        model = (
            nn.Sequential(nn.ConvTranspose3d(8, 4, 3), nn.BatchNorm3d(4))
            .to(device)
            .float()
        )
        for memory_format in [torch.channels_last_3d, torch.contiguous_format]:
            model = nn.utils.convert_conv3d_weight_memory_format(model, memory_format)
            out = model(input)
            self.assertTrue(out.is_contiguous(memory_format=memory_format))

    def test_conv_double_backward_strided_with_3D_input_and_weight(self, device):
        # Test that _convolution_double_backward() outputs the correct grad shapes
        # for 3D input / weight when stride > 1. This is an ad-hoc regression test for a
        # specific case that was uncovered during the convolution consolidation effort.
        # The test can be safely deleted if _convolution_double_backward() is removed.

        input = torch.randn(2, 3, 6, device=device)
        weight = torch.randn(3, 3, 3, device=device)
        bias = torch.randn(3, device=device)
        stride = (2,)
        padding = (1,)
        dilation = (1,)
        transposed = False
        output_padding = (0,)
        groups = 1
        output = torch.ops.aten.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )

        ggI = torch.randn(input.shape, device=device)
        ggW = torch.randn(weight.shape, device=device)
        ggB = torch.randn(bias.shape, device=device)
        gO = torch.randn(output.shape, device=device)
        output_mask = [True, True, True]
        (
            grad_grad_output,
            grad_input,
            grad_weight,
        ) = torch.ops.aten._convolution_double_backward(
            ggI,
            ggW,
            ggB,
            gO,
            weight,
            input,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

        # Make sure the correct shapes are computed.
        self.assertEqual(grad_grad_output.shape, gO.shape)
        self.assertEqual(grad_input.shape, input.shape)
        self.assertEqual(grad_weight.shape, weight.shape)


instantiate_device_type_tests(TestConvolutionNNDeviceType, globals(), only_for="xpu")

if __name__ == "__main__":
    run_tests()
