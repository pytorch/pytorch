# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo
import torch.utils.flop_counter
import torch.nn.functional as F
import unittest

try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

HAS_CUDA = torch.cuda.is_available()

def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)

def get_total_flops(mode):
    return str(sum([v for _, v in mode.flop_counts["Global"].items()]))

def T(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)

@skipIfTorchDynamo
class TestFlopCounter(TestCase):
    def test_flop_counter_variety(self):
        mode = FlopCounterMode()
        mod = torch.nn.Linear(9, 10)
        with mode:
            torch.mm(T(4, 5), T(5, 6))
            torch.addmm(T(4, 6), T(4, 5), T(5, 6), beta=0.5, alpha=0.5)
            torch.matmul(T(5, 6), T(6, 7))
            torch.einsum("ab,bc->ac", T(6, 7), T(7, 8))
            mod(T(8, 9))

        self.assertExpectedInline(get_total_flops(mode), """3012""")

    def test_op(self):
        mode = FlopCounterMode()
        with mode:
            torch.mm(T(4, 5), T(5, 6))
        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """240""")

        with mode:
            torch.bmm(T(3, 4, 5), T(3, 5, 6))
        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.addmm(T(4, 6), T(4, 5), T(5, 6))
            torch.addmm(T(4, 1), T(4, 5), T(5, 6))
            torch.addmm(T(6), T(4, 5), T(5, 6))

        # 4 * 6 * 2 * 5 = 240
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.baddbmm(T(3, 4, 6), T(3, 4, 5), T(3, 5, 6))

        # 3 * 4 * 6 * 2 * 5 = 720
        self.assertExpectedInline(get_total_flops(mode), """720""")

        with mode:
            torch.conv2d(T(2, 3, 6, 6), T(6, 3, 4, 4), padding=1)

        # out_image_size = 2 * 5 * 5
        # kernel_size = 4 * 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """28800""")

        with mode:
            torch.conv1d(T(2, 3, 6), T(6, 3, 4), padding=1)

        # out_image_size = 2 * 5
        # kernel_size = 4
        # c_out = 6
        # c_in = 3
        # out_image_size * kernel_size * c_out * 2 * c_in

        # NB: I don't think this properly accounts for padding?
        self.assertExpectedInline(get_total_flops(mode), """1440""")

    def test_backward(self):
        mode = FlopCounterMode()
        with mode:
            a = T(4, 5, requires_grad=True)
            a = torch.mm(a, T(5, 6))
            a = a.unsqueeze(0).expand(7, 4, 6)
            a = torch.bmm(a, T(7, 6, 7))
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """5184""")

    def test_torchscript(self):
        def foo(x):
            return torch.mm(x, x)
        mode = FlopCounterMode()
        with mode:
            foo(T(5, 5))
        unscripted_flops = get_total_flops(mode)
        ts_foo = torch.jit.script(foo)
        with mode:
            ts_foo(T(5, 5))
        self.assertEqual(unscripted_flops, get_total_flops(mode))

    def test_autograd_op(self):
        class _CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input: torch.Tensor) -> torch.Tensor:
                return torch.mm(input, input)

            @staticmethod
            def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
                return torch.mm(grad_output, grad_output) + torch.mm(grad_output, grad_output)

        a = T(5, 5, requires_grad=True)
        mode = FlopCounterMode()
        with mode:
            a = _CustomOp.apply(a)
            a.sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """750""")



    @skipIfNoTorchVision
    def test_module(self):
        resnet18 = torchvision_models.resnet18()
        mode = FlopCounterMode(resnet18)
        with mode:
            a = T(1, 3, 224, 224, requires_grad=True)
            resnet18(a).sum().backward()

        self.assertExpectedInline(get_total_flops(mode), """10884440064""")
        layer1_conv_flops = mode.flop_counts['ResNet.layer1'][torch.ops.aten.convolution]
        layer1_conv_back_flops = mode.flop_counts['ResNet.layer1'][torch.ops.aten.convolution_backward]
        self.assertExpectedInline(str(layer1_conv_flops), """924844032""")
        self.assertExpectedInline(str(layer1_conv_back_flops), """1849688064""")

    def test_custom(self):
        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: lambda *args, out: 5})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """5""")

    def test_noop(self):
        mode = FlopCounterMode()
        with mode:
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    def test_sdpa(self):
        batch_size = 4
        n_heads = 8
        seq_len_q = 128
        seq_len_k = 128
        head_dim = 64
        dtype = torch.float16

        backends = [
            torch.backends.cuda.enable_mem_efficient_sdp,
            torch.backends.cuda.enable_flash_sdp,
            torch.backends.cuda.enable_math_sdp
        ]

        def enable_backend(backend):
            for cur in backends:
                if backend == cur:
                    backend(True)
                else:
                    backend(False)

        torch.manual_seed(0)
        qkv = torch.randn(batch_size, n_heads, seq_len_q, 3 * head_dim, device='cuda', dtype=dtype, requires_grad=True)
        query, key, value = qkv.split(head_dim, dim=-1)

        def f_forward(query, key, value):
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0, is_causal=True)

        def f_forward_backward(query, key, value):
            return F.scaled_dot_product_attention(query, key, value, dropout_p=0, is_causal=True).sum().backward()

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            mode = FlopCounterMode()
            with mode:
                f_forward(query, key, value)
            flops_fw = get_total_flops(mode)
            self.assertExpectedInline(flops_fw, """134217728""")
            with mode:
                f_forward_backward(query, key, value)

            # Note: The "math" backend does *not* do recomputation, which is why this value is lower
            flops_bw = get_total_flops(mode)
            self.assertEqual(int(flops_bw), int(flops_fw) * 3)
            self.assertExpectedInline(flops_bw, """402653184""")

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            mode = FlopCounterMode()
            with mode:
                f_forward(query, key, value)
            flops_fw = get_total_flops(mode)
            self.assertExpectedInline(flops_fw, """134217728""")
            with mode:
                f_forward_backward(query, key, value)

            flops_bw = get_total_flops(mode)
            self.assertEqual(int(flops_bw), int(flops_fw) * 7 // 2)
            self.assertExpectedInline(flops_bw, """469762048""")

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            mode = FlopCounterMode()
            with mode:
                f_forward(query, key, value)
            flops_fw = get_total_flops(mode)
            self.assertExpectedInline(flops_fw, """134217728""")
            with mode:
                f_forward_backward(query, key, value)

            flops_bw = get_total_flops(mode)
            self.assertEqual(int(flops_bw), int(flops_fw) * 7 // 2)
            self.assertExpectedInline(flops_bw, """469762048""")




if __name__ == '__main__':
    run_tests()
