# Owner(s): ["module: rocm"]

import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import TEST_HIPDNN
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not TEST_HIPDNN, "hipDNN not available")
class TestHipdnnConvolution(TestCase):

    def _compare_conv(
        self,
        x_shape,
        w_shape,
        bias,
        stride,
        padding,
        dilation,
        groups,
        dtype,
        transposed=False,
        output_padding=0,
    ):
        atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (5e-2, 5e-2)

        x_gpu = torch.randn(*x_shape, dtype=dtype, device="cuda")
        w_gpu = torch.randn(*w_shape, dtype=dtype, device="cuda")
        b_gpu = torch.randn(w_shape[1] if transposed else w_shape[0],
                            dtype=dtype, device="cuda") if bias else None

        x_cpu = x_gpu.float().cpu().requires_grad_(True)
        w_cpu = w_gpu.float().cpu().requires_grad_(True)
        b_cpu = b_gpu.float().cpu().requires_grad_(True) if bias else None

        conv_fn = F.conv_transpose2d if transposed else F.conv2d
        kwargs = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
        if transposed:
            kwargs["output_padding"] = output_padding

        out_cpu = conv_fn(x_cpu, w_cpu, b_cpu, **kwargs)
        out_cpu.sum().backward()

        x_gpu = x_gpu.detach().requires_grad_(True)
        w_gpu = w_gpu.detach().requires_grad_(True)
        b_gpu = b_gpu.detach().requires_grad_(True) if bias else None

        with torch.backends.hipdnn.flags(enabled=True):
            out_gpu = conv_fn(x_gpu, w_gpu, b_gpu, **kwargs)
            out_gpu.sum().backward()

        self.assertEqual(out_cpu.float(), out_gpu.float().cpu(), atol=atol, rtol=rtol)
        self.assertEqual(x_cpu.grad.float(), x_gpu.grad.float().cpu(), atol=atol, rtol=rtol)
        self.assertEqual(w_cpu.grad.float(), w_gpu.grad.float().cpu(), atol=atol, rtol=rtol)
        if bias:
            self.assertEqual(b_cpu.grad.float(), b_gpu.grad.float().cpu(), atol=atol, rtol=rtol)

    # -----------------------------------------------------------------------
    # Passing: basic configs (stride=1, groups=1)
    # -----------------------------------------------------------------------
    def test_conv2d_fp32(self):
        self._compare_conv(
            (2, 64, 32, 32), (128, 64, 3, 3),
            bias=False, stride=1, padding=1, dilation=1, groups=1,
            dtype=torch.float32,
        )

    def test_conv2d_fp16(self):
        self._compare_conv(
            (2, 8, 32, 32), (16, 8, 3, 3),
            bias=False, stride=1, padding=1, dilation=1, groups=1,
            dtype=torch.float16,
        )

    def test_conv2d_dilation(self):
        self._compare_conv(
            (2, 64, 32, 32), (128, 64, 3, 3),
            bias=False, stride=1, padding=2, dilation=2, groups=1,
            dtype=torch.float32,
        )

    def test_backend_selection(self):
        x = torch.randn(2, 64, 32, 32, device="cuda")
        w = torch.randn(128, 64, 3, 3, device="cuda")

        with torch.backends.hipdnn.flags(enabled=True):
            out_hipdnn = F.conv2d(x, w, padding=1)

        with torch.backends.hipdnn.flags(enabled=False):
            out_miopen = F.conv2d(x, w, padding=1)

        self.assertEqual(out_hipdnn, out_miopen, atol=1e-4, rtol=1e-4)

    def test_conv2d_stride2(self):
        self._compare_conv(
            (2, 64, 32, 32), (128, 64, 3, 3),
            bias=False, stride=2, padding=1, dilation=1, groups=1,
            dtype=torch.float32,
        )

    def test_conv2d_grouped(self):
        self._compare_conv(
            (2, 128, 32, 32), (128, 32, 3, 3),
            bias=False, stride=1, padding=1, dilation=1, groups=4,
            dtype=torch.float32,
        )

    def test_conv2d_resnet_first_layer(self):
        self._compare_conv(
            (1, 3, 224, 224), (64, 3, 7, 7),
            bias=False, stride=2, padding=3, dilation=1, groups=1,
            dtype=torch.float32,
        )

    def test_conv2d_bf16(self):
        self._compare_conv(
            (2, 8, 32, 32), (16, 8, 3, 3),
            bias=False, stride=1, padding=1, dilation=1, groups=1,
            dtype=torch.bfloat16,
        )

    def test_conv2d_bias_fp32(self):
        self._compare_conv(
            (2, 64, 32, 32), (128, 64, 3, 3),
            bias=True, stride=1, padding=1, dilation=1, groups=1,
            dtype=torch.float32,
        )

    def test_conv2d_bias_fp16(self):
        self._compare_conv(
            (2, 8, 32, 32), (16, 8, 3, 3),
            bias=True, stride=1, padding=1, dilation=1, groups=1,
            dtype=torch.float16,
        )

    def test_conv_transpose2d_fp32(self):
        self._compare_conv(
            (2, 128, 16, 16), (128, 64, 3, 3),
            bias=False, stride=2, padding=1, dilation=1, groups=1,
            dtype=torch.float32, transposed=True, output_padding=1,
        )

    def test_conv_transpose2d_no_output_padding(self):
        self._compare_conv(
            (2, 128, 16, 16), (128, 64, 3, 3),
            bias=False, stride=2, padding=1, dilation=1, groups=1,
            dtype=torch.float32, transposed=True, output_padding=0,
        )

    def test_conv_transpose2d_bias(self):
        self._compare_conv(
            (2, 128, 16, 16), (128, 64, 3, 3),
            bias=True, stride=2, padding=1, dilation=1, groups=1,
            dtype=torch.float32, transposed=True, output_padding=1,
        )

    def test_conv2d_depthwise(self):
        self._compare_conv(
            (2, 128, 32, 32), (128, 1, 3, 3),
            bias=False, stride=1, padding=1, dilation=1, groups=128,
            dtype=torch.float32,
        )


if __name__ == "__main__":
    run_tests()
