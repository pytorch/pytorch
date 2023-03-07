# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
import torch.utils.flop_counter
import unittest

try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)

def get_total_flops(mode):
    return str(sum([v for _, v in mode.flop_counts["Global"].items()]))

def T(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)

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
            a = T(1, 3, 224, 224)
            resnet18(a)

        self.assertExpectedInline(get_total_flops(mode), """3628146688""")
        layer1_conv_flops = mode.flop_counts['layer1'][torch.ops.aten.convolution]
        layer1_conv_back_flops = mode.flop_counts['layer1'][torch.ops.aten.convolution_backward]
        self.assertExpectedInline(str(layer1_conv_flops), """0""")
        self.assertExpectedInline(str(layer1_conv_back_flops), """0""")

    def test_custom(self):
        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: lambda *args: 5})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """5""")

    def test_noop(self):
        mode = FlopCounterMode()
        with mode:
            T(4, 5).cos()




if __name__ == '__main__':
    run_tests()
