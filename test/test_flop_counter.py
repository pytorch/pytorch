# Owner(s): ["module: unknown"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_TORCHDYNAMO
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION, PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
import torch.utils.flop_counter
import torch.nn.functional as F
import unittest
import functools

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

@unittest.skipIf(TEST_WITH_TORCHDYNAMO, "torchdynamo doesn't work with __torch_dispatch__ right now")
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

    def test_conv_backwards_as_decomposition(self):
        # [conv backwards decomposition as conv forwards]

        class onlyConvs(torch.autograd.Function):
            @staticmethod
            def forward(inp, weight, transposed):
                if not transposed:
                    return F.conv1d(inp, weight)
                else:
                    return F.conv_transpose1d(inp, weight)

            @staticmethod
            def setup_context(ctx, inputs, output):
                inp, weight, transposed = inputs
                ctx.save_for_backward(inp, weight)
                ctx.transposed = transposed

            @staticmethod
            def backward(ctx, grad_out):
                inp, weight = ctx.saved_tensors
                if not ctx.transposed:
                    grad_inp = F.conv_transpose1d(grad_out, weight)
                    grad_weight = F.conv1d(inp, grad_out)
                    return grad_inp, grad_weight, None
                else:
                    grad_inp = F.conv1d(grad_out, weight)
                    grad_weight = F.conv1d(grad_out.transpose(1, 0), inp.transpose(1, 0))
                    return grad_inp, grad_weight.transpose(1, 0), None


        from torch.func import grad
        x = torch.randn(2, 3, 16, dtype=torch.float64)
        weight = torch.randn(3, 4, 4, dtype=torch.float64)

        def boring_conv(x, weight, transposed):
            if not transposed:
                return F.conv1d(x, weight).pow(2).sum()
            else:
                return F.conv_transpose1d(x, weight).pow(2).sum()

        def only_convs(x, weight, transposed):
            return onlyConvs.apply(x, weight, transposed).pow(2).sum()

        boring_grads = grad(boring_conv, argnums=(0, 1))(x, weight, True)
        fun_grads = grad(only_convs, argnums=(0, 1))(x, weight, True)

        self.assertEqual(boring_grads, fun_grads)


    def test_convs(self):
        def assert_equivalence(f, expected_forward=None):
            mode = FlopCounterMode()
            with mode:
                f()
            conv_forward_flops = mode.get_flop_counts()['Global'][torch.ops.aten.convolution]
            conv_backward_flops = mode.get_flop_counts()['Global'][torch.ops.aten.convolution_backward]

            self.assertEqual(conv_forward_flops * 2, conv_backward_flops)
            if expected_forward is not None:
                self.assertEqual(conv_forward_flops, expected_forward)

        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 2, 2, requires_grad=True)
        assert_equivalence(lambda: F.conv_transpose2d(x, weight).sum().backward(), 32)

        x = torch.rand(1, 1, 2, 2, requires_grad=True)
        weight = torch.randn(1, 1, 1, 1, requires_grad=True)
        assert_equivalence(lambda: F.conv2d(x, weight).sum().backward(), 8)

        for in_channels, out_channels, groups in [
            (1, 1, 1),
            (1, 3, 1),
            (3, 1, 1),
            (3, 7, 1),
            (2, 4, 2),
            (4, 2, 2),
        ]:
            x = torch.rand(1, in_channels, 4, 4, requires_grad=True)
            weight = torch.randn(out_channels, in_channels, 2, 2, requires_grad=True)
            assert_equivalence(lambda: F.conv2d(x, weight).sum().backward())
            transposed_weight = torch.randn(in_channels, out_channels, 2, 2, requires_grad=True)
            assert_equivalence(lambda: F.conv_transpose2d(x, transposed_weight).sum().backward())

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


    def test_conv_transpose_loop(self):
        x = torch.rand(1, 4, 30, 2)
        model = torch.nn.ConvTranspose2d(4, 8, (2, 2), stride=2)

        mode = FlopCounterMode(model)
        with mode:
            for i in range(50):
                out = model(x)
                out.sum().backward()
        self.assertExpectedInline(str(mode.get_total_flops()), """1536000""")

    def test_custom(self):
        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: lambda *args, out_shape: 5})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """5""")

        def count(*args, out):
            return out.numel()
        count._get_raw = True

        mode = FlopCounterMode(custom_mapping={torch.ops.aten.add: count})
        with mode:
            a = T(4, 5)
            a + a

        self.assertExpectedInline(get_total_flops(mode), """20""")

    def test_noop(self):
        mode = FlopCounterMode()
        with mode:
            T(4, 5).cos()

    @unittest.skipIf(not HAS_CUDA, "CUDA not available")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION or not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
                     "Does not support all SDPA backends (pre-SM80 hardware on CUDA)")
    def test_sdpa(self):
        batch_size = 4
        n_heads = 8
        seq_len_q = 128
        seq_len_k = 256
        head_dim = 64
        head_dim_v = 64
        dtype = torch.float16

        torch.manual_seed(0)

        def get_flops(batch_size, n_heads, seq_len_q, seq_len_k, head_dim, head_dim_v, dtype, backend, with_backward=False):
            query = torch.randn(batch_size, n_heads, seq_len_q, head_dim, device='cuda', dtype=dtype, requires_grad=True)
            key = torch.randn(batch_size, n_heads, seq_len_k, head_dim, device='cuda', dtype=dtype, requires_grad=True)
            value = torch.randn(batch_size, n_heads, seq_len_k, head_dim_v, device='cuda', dtype=dtype, requires_grad=True)

            if backend == "math":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            elif backend == "flash":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            elif backend == "mem_efficient":
                backend = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

            mode = FlopCounterMode()
            with backend, mode:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=0, is_causal=True)
                if with_backward:
                    out.sum().backward()
            return int(get_total_flops(mode))

        # Sets seq_len_q == seq_len_k and dim_q == dim_v
        run_uniform_flops = functools.partial(get_flops, batch_size, n_heads, seq_len_q, seq_len_q, head_dim, head_dim, dtype)

        flops = [run_uniform_flops(backend, with_backward=False) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_math, flops_fw_flash, flops_fw_efficient = flops
        self.assertEqual(flops_fw_math, flops_fw_flash)
        self.assertEqual(flops_fw_math, flops_fw_efficient)

        self.assertExpectedInline(str(flops_fw_math), """134217728""")

        flops = [run_uniform_flops(backend, with_backward=True) for backend in ["math", "flash", "mem_efficient"]]
        flops_fw_bw_math, flops_fw_bw_flash, flops_fw_bw_efficient = flops
        self.assertEqual(flops_fw_math * 3, flops_fw_bw_math)
        self.assertEqual(flops_fw_math * 7 // 2, flops_fw_bw_flash)
        self.assertEqual(flops_fw_bw_flash, flops_fw_bw_efficient)


        run_nonuniform_flops = functools.partial(get_flops, batch_size, n_heads, seq_len_q, seq_len_k, head_dim, head_dim_v, dtype)
        # Flash does not support non-uniform attention, i.e. seq_len_q != seq_len_k or dim_q != dim_v"
        non_uniform_backends = ["math", "mem_efficient"]
        flops = [run_nonuniform_flops(backend, with_backward=False) for backend in non_uniform_backends]
        flops_fw_math, flops_fw_efficient = flops
        self.assertEqual(flops_fw_math, flops_fw_efficient)

        self.assertExpectedInline(str(flops_fw_math), """268435456""")

        flops = [run_nonuniform_flops(backend, with_backward=True) for backend in non_uniform_backends]
        flops_fw_bw_math, flops_fw_bw_efficient = flops
        self.assertExpectedInline(str(flops_fw_bw_math), """805306368""")
        self.assertExpectedInline(str(flops_fw_bw_efficient), """939524096""")

    def test_hook_registration(self):
        model = torch.nn.Linear(100, 100)
        x = torch.randn(3, 100)

        flop_counter = FlopCounterMode(model)
        with flop_counter:
            self.assertEqual(len(model._forward_pre_hooks), 1)
            self.assertEqual(len(model._forward_hooks), 1)
            model(x).sum().backward()

        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_pytrees(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                x = x['a'].relu_()
                return {'a': torch.mm(x, x)}

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Foo()
                self.b = Foo()

            def forward(self, x):
                return self.b(self.a(x))

        mod = Mod()
        mode = FlopCounterMode(mod)
        with mode:
            mod({'a': torch.randn(10, 10, requires_grad=True).clone()})['a'].sum().backward()
        self.assertExpectedInline((mode.flop_counts['Mod'][torch.ops.aten.mm]), """12000""")

        class Mod2(torch.nn.Module):
            def forward(self, x):
                return (torch.mm(x, x),)

        mod = Mod2()
        mode = FlopCounterMode(mod)
        with mode:
            mod(torch.randn(10, 10, requires_grad=True))[0].sum().backward()
        self.assertExpectedInline((mode.flop_counts['Mod2'][torch.ops.aten.mm]), """6000""")





if __name__ == '__main__':
    run_tests()
