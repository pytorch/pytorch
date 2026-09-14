# Owner(s): ["module: inductor"]
import contextlib
import re
import unittest
from unittest.mock import patch

import functorch
import torch
import torch._inductor.config as config
import torch.autograd
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor import metrics
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

########################
# Explanation of Tests #
########################
# These tests are all testing *memory accesses* of TorchInductor.
# They are intended to be deterministic performance tests.
# The expect tests are all measuring the number of memory bytes read/written by
# the code that Inductor has generated
#
# If the test is failing because the number became smaller, feel free to lower it.
# On the other hand, if the test is failing because the number became larger,
# that means that your change is leading to *more* memory accesses on this test.
#
# That may still be aceeptable, but be aware that you are likely lowering
# performance for that setting.
#
# Defines all the kernels for tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    skipIfXpu,
    TEST_WITH_ROCM,
)
from torch.utils._triton import has_triton


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

if has_triton():
    import triton  # @manual
    import triton.language as tl  # @manual

    from torch.testing._internal.triton_utils import add_kernel

aten = torch.ops.aten


def compile_but_use_eager(gm, example_inputs):
    def inner_compile(gm, *args, **kwargs):
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def count_numel(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch.compile(f, backend=compile_but_use_eager)(*args)
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


def count_numel_train(f, *args):
    """
    Assumes all inputs are fp32
    """
    metrics.reset()

    f = torch.compile(f, backend=compile_but_use_eager)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


def T(*size, dtype=torch.float32, device="cpu", grad=False):
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


def TI(*size, mx=10, dtype=torch.int32, device="cpu"):
    return torch.randint(0, mx, size, dtype=dtype, device=device)


class TestCase(InductorTestCase):
    hw_classification = HardwareClassification.ACCELERATOR


class NumBytesMetricTests(TestCase):
    """
    Primarily used for sanity testing that the num_bytes_accessed metrics is correct.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def test_pointwise(self, device):
        def f(x):
            return x.cos()

        inp = (T(10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x, y):
            return x + y

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        def f(x, y):
            return x + y

        inp = (T(10, 10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

        def f(x):
            return x + x

        inp = (T(10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x):
            return x + x.t()

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b, c):
            return a.cos(), b.sin() + c.sin()

        inp = (T(10, device=device), T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    def test_reduction(self, device):
        def f(x):
            return x.sum(dim=1)

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

        def f(x):
            return x.sum(dim=0)

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_extern(self, device):
        def f(x):
            return torch.mm(x, x)

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b):
            return torch.mm(a, b)

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(x):
            x = x.cos()
            x = torch.mm(x, x)
            x = x.cos()
            return x

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        def f(x):
            a = x.cos()
            b = x.sin()
            x = torch.mm(a, b)
            return x

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

    def test_cat(self, device):
        def f(a, b):
            return torch.cat([a.sin(), b.sin()])

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            return torch.cat([a, b])

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            return torch.cat([a.cos(), b])

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a):
            return torch.cat([a.cos(), a.sin()])

        inp = (T(10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        def f(a, b):
            return torch.cat([torch.mm(a, a), b.sin()])

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b, c):
            return torch.cat((a + 1, b + 2, c + 3)) + 10

        inp = (
            T(10, 10, device=device),
            T(10, 10, device=device),
            T(10, 10, device=device),
        )
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        def f(a, b, c, d, e):
            return torch.cat((a + 1, b + 2, c + 3, d + 4, e + 5)) + 10

        inp = [T(10, 10, device=device) for _ in range(5)]
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

        def f(a, b):
            return torch.cat([a.sum(dim=0), b.sum(dim=0)]) + 10

        inp = [T(10, 10, 10, device=device), T(10, 10, 10, device=device)]
        self.assertExpectedInline(count_numel(f, *inp), """2600""")

    @config.patch("fx_graph_cache", False)
    def test_cat_pointwise(self, device):
        def f(a, b):
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)])

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)]).cos()

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """680""")

        # Should turn into pointwise even if only some of inputs are pointwise.
        def f(a, b):
            out = torch.cat([a.cos(), torch.mm(b, b)])
            return out.cos()

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        # Should not turn into pointwise if all inputs are not pointwise
        def f(a, b):
            out = torch.cat([torch.mm(a, a), torch.mm(b, b)])
            return out.cos()

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """800""")

        def f(a, b):
            out = torch.cat([a, b])
            return out.cos()

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            b = b.cos()
            return torch.cat([a, b])

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            a = a @ a
            return torch.constant_pad_nd(torch.cat([a, b]), [2, 2], 0.5)

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """680""")

    @patch.object(config, "split_cat_fx_passes", False)
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})
    def test_cat_pointwise_many_complex_inputs(self, device):
        def f(*inputs):
            input = [torch.nn.functional.gelu(val) for val in inputs]
            return torch.cat(input) + 10

        inp = (T(10, 10, device=device) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """6400""")

    @patch.object(config, "split_cat_fx_passes", False)
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})
    def test_cat_pointwise_many_simple_inputs(self, device):
        def f(*inputs):
            input = [torch.nn.functional.relu(val) for val in inputs]
            return torch.cat(input) + 10

        inp = (T(10, 10, device=device) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """9600""")

    @patch.object(config, "max_pointwise_cat_inputs", 0)
    def test_cat_pointwise_config_option(self, device):
        def f(a, b):
            return torch.cat([a + 1, b + 2]) + 3

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

    def test_index(self, device):
        def f(a, b):
            return a[b]

        inp = (T(10, device=device), TI(10, mx=10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """30""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_delay_realize_cheap_outputs_shared_mask(self, device):
        # Shared tril mask across multiple users gets eagerly materialized
        # as an output buffer, inflating downstream read counts. With
        # delay_realize_cheap_outputs, the mask stays inlined as index
        # arithmetic, reducing memory traffic.
        def f(x1, x2, x3):
            mask = torch.tril(torch.ones(32, 32, device=x1.device))
            return x1 + mask, x2 + mask, x3 + mask

        inp = (
            T(32, 32, grad=True, device=device),
            T(32, 32, grad=True, device=device),
            T(32, 32, grad=True, device=device),
        )

        # Without deferred realization: mask gets materialized as output buffer
        with patch.object(config, "delay_realize_cheap_outputs", False):
            metrics.reset()
            torch.compile(f, backend=compile_but_use_eager)(*inp)
            eager_bytes = metrics.num_bytes_accessed

        # Default (deferred realization on): mask stays inlined
        metrics.reset()
        torch.compile(f, backend=compile_but_use_eager)(*inp)
        deferred_bytes = metrics.num_bytes_accessed

        self.assertLessEqual(deferred_bytes, eager_bytes)


class FusionTests(TestCase):
    """
    Tests that things can be fused into a single kernel
    """

    hw_classification = HardwareClassification.ACCELERATOR

    def test_horizontal_reduction_pointwise(self, device):
        def f(a):
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_horizontal_reduction_reduction(self, device):
        def f(a):
            b = a.sum(dim=1)
            c = a.amax(dim=1)
            return b, c

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    @config.patch({"force_disable_caches": True, "triton.multi_kernel": 0})
    def test_aot_autograd_cse_preserves_reduction_fusion(self, device):
        def fn(x):
            return x.abs().max(), x.abs().mean(), x.square().mean()

        def count_kernels(requires_grad):
            torch._dynamo.reset()
            metrics.reset()
            x = torch.rand((1024, 32768), device=device, requires_grad=requires_grad)
            result = torch.compile(fn, fullgraph=True, dynamic=False)(x)
            expected = fn(x)
            self.assertEqual(result, expected)
            get_interface_for_device(device).synchronize()
            forward_kernel_count = metrics.generated_kernel_count
            if requires_grad:
                result_sum = result[0] + result[1] + result[2]
                expected_sum = expected[0] + expected[1] + expected[2]
                (result_grad,) = torch.autograd.grad(result_sum, (x,))
                (expected_grad,) = torch.autograd.grad(expected_sum, (x,))
                self.assertEqual(result_grad, expected_grad)
            return forward_kernel_count

        inference_kernel_count = count_kernels(requires_grad=False)
        autograd_kernel_count = count_kernels(requires_grad=True)

        self.assertEqual(autograd_kernel_count, inference_kernel_count)

    def test_horizontal_reduction_pointwise2(self, device):
        def f(a, b):
            c = a.sum(dim=1)
            b = b.cos()
            return b + c

        inp = (T(10, 10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_reduction_outer_pointwise(self, device):
        def f(a, b):
            c = a.sum(dim=0)
            b = b.cos()
            return b + c

        inp = (T(10, 10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_sum_pw_broadcast(self, device):
        def f(a, b):
            a = a.sum(dim=1, keepdim=True)
            b = b.cos()
            return a * b

        inp = (T(10, 10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_vertical_sum_pw(self, device):
        def f(a):
            a = a.cos()
            a = a.sum(dim=1)
            return a.cos()

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_norm_chain(self, device):
        def f(a):
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            return a

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_inner(self, device):
        def f(a):
            return torch.softmax(a, dim=1)

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_layer_norm(self, device):
        # TODO: Suboptimal! We shouldn't need to save normalization stats.
        mod = torch.nn.LayerNorm(10, device=device)

        def f(x):
            return mod(x)

        inp = (T(10, 10, device=device),)
        with torch.no_grad():
            self.assertExpectedInline(count_numel(f, *inp), """220""")

    def test_double_softmax(self, device):
        def f(x):
            x = torch.softmax(x, dim=1)
            x = torch.softmax(x, dim=1)
            return x

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_backward(self, device):
        def f(grad_out, out):
            return aten._softmax_backward_data(grad_out, out, 1, torch.float32)

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_neighbor(self, device):
        def f(a, b):
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 4, device=device), T(1, 10, 4, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """90""")

    def test_factory_reduction(self, device):
        def f():
            a = torch.ones(10, device=device)
            b = torch.ones(10, 10, device=device)
            return (a + b).sum(dim=-1)

        inp = ()
        self.assertExpectedInline(count_numel(f, *inp), """10""")

    def test_index_pointwise(self, device):
        def f(a, b):
            return a[b].cos()

        inp = (T(10, 10, device=device), TI(20, mx=10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """320""")

    def test_index_reduction(self, device):
        def f(a, b):
            return a[b].cos().sum(dim=1)

        inp = (T(10, 10, device=device), TI(20, mx=10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """140""")

    def test_mutation_fusion(self, device):
        def f(a, b, c):
            a0 = a.add(c)
            b0 = b.add(a0)
            b.copy_(b0)
            a.copy_(a0)

        inp = (
            T(10, 10, device=device),
            T(10, 10, device=device),
            T(10, 10, device=device),
        )
        self.assertExpectedInline(count_numel(f, *inp), """500""")

    @skipIfXpu(msg="copy_(cat()) fusion not supported on XPU")
    @unittest.skipIf(TEST_WITH_ROCM, "copy_(cat()) fusion not supported on ROCm")
    # TODO(ivankobzarev): enable copy_(cat()) fusion for CUDA 13+
    @unittest.skipIf(
        torch.version.cuda
        and tuple(int(x) for x in torch.version.cuda.split(".")) >= (13, 0),
        "copy_(cat()) fusion not supported on CUDA 13+",
    )
    def test_copy_cat_fusion(self, device):
        """copy_(cat(...)) should fuse: no intermediate allocation for cat."""

        def f(dst, a, b):
            dst.copy_(torch.cat([a, b]))

        dst = T(20, device=device)
        inp = (dst, T(10, device=device), T(10, device=device))
        # The cat is fully fused: the generated kernel reads a and b and writes
        # them straight into dst's slices, with no intermediate allocation --
        # real traffic is 10 (read a) + 10 (read b) + 20 (write dst) = 40.
        # count_numel reports 60 here rather than 40 because its static
        # num_bytes_accessed estimate over-counts the ConcatKernel slices once
        # canonicalization reorders the placeholders (dst is no longer input 0);
        # the emitted kernel is unchanged, so this is an estimate artifact, not
        # a real regression. Without fusion cat would allocate intermediate: 80.
        self.assertExpectedInline(count_numel(f, *inp), """60""")

    def test_reduction_pointwise_multi_level_reduction(self, device):
        hidden_size = 4096
        layer_norm = torch.nn.LayerNorm(hidden_size).to(device).float()

        @torch.inference_mode()
        def f(x, scale, amax_keep_dim):
            x = layer_norm(x.to(dtype=torch.float))
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)

        inp = (
            T(4, 2048, hidden_size, dtype=torch.float, device=device),
            T(1, dtype=torch.float, device=device),
        )

        # 2 kernels:
        # kernel 1: (input = X, scale, LN scale, LN bias, output = LN_pointwise(X), first-level amax (split-reduction))
        # kernel 2: (input = first-level amax, output = final amax)
        # scale (1) + X (4*2048*hidden_size) * 2 + LN scale (hidden_size) + LN bias (hidden_size) + amax (4 * 2048 * 2 + 1)
        expected_numel = (
            1 + hidden_size * 2 + 4 * 2048 * hidden_size * 2 + 4 * 2048 * 2 + 1
        )
        if (
            config.triton.cooperative_reductions
            or config.triton.force_cooperative_reductions
        ):
            expected_numel = 134225922

        # Allow ~0.05% tolerance: canonicalization may reorder nodes, causing
        # slightly different fusion decisions and a small numel change.
        actual_keep = int(count_numel(f, *inp, True))
        actual_no_keep = int(count_numel(f, *inp, False))
        self.assertAlmostEqual(actual_keep, expected_numel, delta=expected_numel * 5e-4)
        self.assertAlmostEqual(
            actual_no_keep, expected_numel, delta=expected_numel * 5e-4
        )

    def test_pointwise_multi_level_reduction(self, device):
        # TODO: this can be optimized by having the first pointwise kernel leveraging block sizes
        # of the first-level reduction kernel.
        hidden_size = 4096

        def f(x, scale, amax_keep_dim):
            x = x * 1.1
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)

        inp = (
            T(4, 2048, hidden_size, dtype=torch.float, device=device),
            T(1, dtype=torch.float, device=device),
        )

        compiled_f = torch.compile(f)
        compiled_f(*inp, True)

        # 3 kernels:
        # kernel 1: (input = X, scale, output = pointwise(X))
        # kernel 2: (input = X, output = first-level amax)
        # kernel 3: (input = first-level amax, output = final amax)
        # scale (1) + X (4*2048*hidden_size) * 3 + amax (num_splits * 2 + 1)
        # num_splits depends on SM architectures.
        expected_numel = 1 + 4 * 2048 * hidden_size * 3 + 1
        actual_numel_amax_keep_dim = count_numel(f, *inp, True)
        actual_numel_amax_no_keep_dim = count_numel(f, *inp, False)
        self.assertEqual(actual_numel_amax_keep_dim, actual_numel_amax_no_keep_dim)
        self.assertGreaterAlmostEqual(actual_numel_amax_keep_dim, str(expected_numel))

    def test_create_block_mask(self, device):
        def mk_3d_flex_natten_mask(dims, kernel_size):
            T, H, W = dims
            K_T, K_H, K_W = kernel_size
            spatial = H * W

            def get_x_y_t(idx: int) -> tuple[int, int, int]:
                t = idx // spatial
                s = idx % spatial
                x = s // W
                y = s % W
                return x, y, t

            def get_mask(b, h, q_idx, kv_idx):
                q_x, q_y, q_t = get_x_y_t(q_idx)
                kv_x, kv_y, kv_t = get_x_y_t(kv_idx)
                kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
                kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
                kernel_t = q_t.clamp(K_T // 2, (T - 1) - K_T // 2)
                hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
                vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
                temp_mask = (kernel_t - kv_t).abs() <= K_T // 2
                return hori_mask & vert_mask & temp_mask

            return get_mask

        T = 4
        H = 16
        W = 16
        t = 5
        h = 5
        w = 5
        data_size = (T, H, W)
        kernel_size = (t, h, w)
        S = T * H * W
        from torch.nn.attention.flex_attention import create_block_mask

        mask_mod = mk_3d_flex_natten_mask(data_size, kernel_size)

        torch.compile(create_block_mask)(mask_mod, None, None, S, S)
        numel = int(count_numel(create_block_mask, mask_mod, None, None, S, S))

        # We should be writing way less than a quadratic amount of bytes here
        # With fusion, we should only be writing a linear number of bytes
        self.assertLess(numel * 5, S * S)


class SchedulerFusionTests(TestCase):
    """
    Testing the fusion group creation heuristic (i.e. cases where we can't fuse
    everything into a single kernel)
    Disables inductor rematerialization for easier reasoning of tests.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "realize_opcount_threshold", 0))

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice1(self, device):
        # Doesn't matter where we break fusion group here
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c.cos()
            return d + e

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """700""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice2(self, device):
        # We should materialize e (it's smaller!)
        # [c, e]: 210, [f]: 210, [d]: 200
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c.sum(dim=1)
            f = d + e
            return f

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """620""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice3(self, device):
        # We should materialize e.
        # [c, e]: 300, [f]: 300, [d]: 200
        def f(a):
            c = a.cos()
            d = torch.mm(c, c)
            e = c + a
            f = d + e
            return f, e

        inp = (T(10, 10, device=device),)
        self.assertExpectedInline(count_numel(f, *inp), """800""")


class SchedulerFusionTestsOnlyCPU(TestCase):
    """
    CPU-specific fusion group creation heuristic tests.
    Disables inductor rematerialization for easier reasoning of tests.
    """

    hw_classification = HardwareClassification.CPU

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "realize_opcount_threshold", 0))

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice4_cpu(self, device):
        # Fuse nodes with same number of elements and compatible original var ranges
        # [buf0: {d0: 60, d1: 11}, buf1: {d0: 660}] -> buf0_buf1
        def f(x, w):
            o1 = x * w
            output = o1 + 1.0
            return output

        inp = (T(2, 3, 10, 11, device=device), T(11, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """1331""")

        # [buf0_buf1: {d0: 60, d1: 11}, buf2: {d0: 660}] -> buf0_buf1_buf2
        def f(x, w1, w2):
            o1 = x * w1
            o2 = x * w2
            output = o1 + o2
            return output

        inp = (
            T(2, 3, 10, 11, device=device),
            T(11, device=device),
            T(11, device=device),
        )
        self.assertExpectedInline(count_numel(f, *inp), """1342""")


class TilingTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_tiling_simple(self, device):
        def f(a, b):
            return a + b.t()

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(a, b):
            return a.t() + b

        inp = (T(10, 10, device=device), T(10, 10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_tiling_three(self, device):
        def f(a, b, c):
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)

        inp = (
            T(10, 10, 10, device=device),
            T(10, 10, 10, device=device),
            T(10, 10, 10, device=device),
        )
        self.assertExpectedInline(count_numel(f, *inp), """4000""")


class MinCutPartitioningTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_partitioning_full_remat(self, device):
        def f(x):
            return x.cos().cos().cos()

        inp = (T(10, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """50""")

    def test_partitioning_partial_remat(self, device):
        def f(a, b, c, d):
            x = a + b + c + d
            return x.cos().cos()

        inp = (
            T(10, grad=True, device=device),
            T(10, grad=True, device=device),
            T(10, grad=True, device=device),
            T(10, grad=True, device=device),
        )
        self.assertExpectedInline(count_numel_train(f, *inp), """90""")

    def test_partitioning_dtype(self, device):
        def f(x):
            return (x < 0) * x

        inp = (T(100, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """450""")

    @config.patch("fx_graph_cache", False)
    @config.patch("autotune_local_cache", False)
    @patch.object(functorch.compile.config, "max_dist_from_bw", 1000)
    def test_partitioning_unremat_bw(self, device):
        def f(x):
            return torch.mm(x, x.new_ones(x.shape)).tanh().tanh()

        inp = (T(10, 10, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """1300""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_unremat_bw2(self, device):
        def f(a):
            a = torch.mm(a, a)
            a = a + 1
            b = a + 2
            c = torch.mm(a, b)
            return c

        inp = (T(10, 10, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """2600""")

    def test_partitioning_keops(self, device):
        def f(a, b):
            return (a * b).cos().sum(dim=1)

        inp = (T(20, 1, grad=True, device=device), T(1, 20, grad=True, device=device))
        self.assertExpectedInline(count_numel_train(f, *inp), """220""")

    def test_partitioning_cat(self, device):
        def f(a, b):
            a = torch.tanh(a)
            return torch.cat([a, b])

        inp = (T(10, grad=True, device=device), T(10, grad=True, device=device))
        self.assertExpectedInline(count_numel_train(f, *inp), """70""")

    def test_partitioning_relu(self, device):
        def f(x):
            return torch.relu(x)

        inp = (T(16, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """72""")

    def test_partitioning_with_view(self, device):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                x = x.cos()
                x = x.view(10, 10)
                ctx.save_for_backward(x, y)
                x = x.cos()
                return x

            @staticmethod
            def backward(ctx, gradOut):
                x, y = ctx.saved_tensors
                return torch.mm(gradOut, x).view(100) * y

        def f(a):
            return Foo.apply(a)

        inp = (T(100, grad=True, device=device),)
        # We do not want to recompute the x.cos().view() chain, as it's
        # materialized in backwards
        self.assertExpectedInline(count_numel_train(f, *inp), """900""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_long_chain_add(self, device):
        def f(x):
            orig = x
            for _ in range(2):
                x = x * x
                x = torch.mm(x, x)
                x = x * 2
                x = orig + x
                orig = x
            return x

        inp = (T(10, 10, grad=True, device=device),)
        self.assertExpectedInline(count_numel_train(f, *inp), """3900""")


def unfusible(x):
    # For the purpose of noop tests, we want inductor to fall back to
    # eager mode, so, below we must use a aten operator that does not
    # have decomposition nor lowering:
    return aten._lazy_clone(x)


class NoopTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_noop_clones(self, device):
        def f(a):
            b = a.clone()
            b = unfusible(b)
            return b

        inp = T(10, device=device)
        self.assertExpectedInline(count_numel(f, inp), """20""")

        def f(a):
            b = a.clone()
            c = unfusible(b)
            return b, c

        self.assertExpectedInline(count_numel(f, inp), """40""")

    def test_noop_slice_scatter(self, device):
        def f(a):
            b = aten.slice_scatter(a, a)
            c = unfusible(b)
            return c

        inp = T(10, device=device)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_dtype_conversion(self, device):
        def f(a):
            b = torch.ops.prims.convert_element_type(a, torch.float32)
            c = unfusible(b)
            return c

        inp = T(10, device=device)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_device_conversion(self, device):
        def f(a):
            b = torch.ops.prims.device_put(a, device)
            c = unfusible(b)
            return c

        inp = T(10, device=device)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_int_ops(self, device):
        def f1(a):
            b = torch.ceil(a)
            c = unfusible(b)
            return c

        def f2(a):
            d = torch.floor(a)
            e = unfusible(d)
            return e

        def f3(a):
            f = torch.round(a)
            g = unfusible(f)
            return g

        def f4(a):
            f = torch.pow(a, 1)
            g = unfusible(f)
            return g

        inp = TI(10, device=device)
        self.assertExpectedInline(count_numel(f1, inp), """20""")
        self.assertExpectedInline(count_numel(f2, inp), """20""")
        self.assertExpectedInline(count_numel(f3, inp), """20""")
        self.assertExpectedInline(count_numel(f4, inp), """20""")

    def test_noop_cat(self, device):
        def f1(a):
            b = torch.cat([a])
            return unfusible(b)

        inp = T(10, device=device)
        self.assertExpectedInline(count_numel(f1, inp), """20""")

        def f2(a):
            b = torch.cat([a])
            c = torch.cat([b])
            return c

        self.assertExpectedInline(count_numel(f2, inp), """20""")


class InplacingTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_inplace_scatter(self, device):
        def f(a, b):
            a = a.cos()
            a[b] = 1
            return a

        inp = (T(10, device=device), TI(2, mx=5, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """26""")

        def f(a, b):
            out = aten.index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        inp = (T(10, device=device), TI(2, mx=5, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """6""")

        def f(a, b):
            out = aten._unsafe_index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        inp = (T(10, device=device), TI(2, mx=5, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """6""")

    def test_inplace_scatter_noop_view(self, device):
        def f(a, b):
            a[:, b] = 1
            return a

        inp = (T(10, 10, device=device), TI(2, mx=5, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """42""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_training(self, device):
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            n_elements = x.numel()
            sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

        factory_op = torch.empty_like

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = factory_op(x)
                sin_triton(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin_triton(saved, out)
                return out

        def f(x):
            return MySin.apply(x)

        x = T(3, grad=True, device=device)
        self.assertExpectedInline(count_numel_train(f, x), """18""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_triton_kernel_not_fusable_with_users(self, device):
        @triton.jit
        def _sin_kernel(
            in_ptr0,
            out_ptr,
            out2_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)
            tl.store(out2_ptr + offsets, output, mask=mask)

        from torch._library import capture_triton, triton_op

        @triton_op("mylib::sin_kernel", mutates_args={})
        def sin_kernel(x: torch.Tensor) -> list[torch.Tensor]:
            n_elements = x.numel()
            out = torch.empty_like(x)
            out2 = torch.empty_like(x)
            capture_triton(_sin_kernel)[(n_elements,)](
                x, out, out2, n_elements, BLOCK_SIZE=4
            )
            return [out, out2]

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out, saved = tuple(torch.ops.mylib.sin_kernel(x))
                ctx.save_for_backward(x, saved)
                return out

            @staticmethod
            def backward(ctx, grad):
                (x, saved) = ctx.saved_tensors
                return grad * saved.sigmoid() * x

        def f(x):
            return MySin.apply(x)

        x = T(3, grad=True, device=device)
        # Important bit: saved.sigmoid() can be fused into its consumer (mul),
        # but not its producer (user triton kernel).
        # So we should not compute it in the fw and save it for backward
        # (it will cost an extra kernel)
        self.assertExpectedInline(count_numel_train(f, x), """27""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_custom_op_training_two_mutated_inputs(self, device):
        @torch.library.custom_op(
            "_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"}
        )
        def sin_cos(
            x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor
        ) -> None:
            out_sin.copy_(x.sin())
            out_cos.copy_(x.cos())

        def f(x):
            out0 = torch.empty_like(x)
            out1 = torch.empty_like(x)
            sin_cos(x, out0, out1)
            return x.clone(), out0, out1

        x = T(3, grad=True, device=device)
        self.assertExpectedInline(count_numel(f, x), """21""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_custom_op_training(self, device):
        @torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
        def sin(x: torch.Tensor, result: torch.Tensor) -> None:
            result.copy_(x.sin())

        factory_op = torch.empty_like

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                out = factory_op(x)
                sin(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin(saved, out)
                return out

        def f(x):
            return MySin.apply(x)

        x = T(3, grad=True, device=device)
        self.assertExpectedInline(count_numel_train(f, x), """18""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_custom_op(self, device):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3, device=device)
            out = T(3, device=device)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 0)

            self.assertExpectedInline(count_numel(f, x, out), """21""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_custom_op_intermediate(self, device):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                out = torch.empty_like(x)
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3, device=device)
            out = T(3, device=device)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 1)

            self.assertExpectedInline(count_numel(f, x, out), """21""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_custom_op_two_mutated_inputs(self, device):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache) -> Tensor")

            def foo(q, k_cache, v_cache):
                k_cache.add_(1)
                v_cache.add_(1)
                return q + 1

            m.impl("foo", foo, "CompositeExplicitAutograd")

            q = T(3, device=device)
            k_cache = T(3, device=device)
            v_cache = torch.rand_like(k_cache)

            def f():
                x = 0
                for _ in range(2):
                    x = x + torch.ops.mylib.foo(q, k_cache, v_cache)
                return x

            _, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True),
            )

            # Check that we are not allocate intermediate buffers
            # which can be reused.
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 0)
            self.assertEqual("in_out" in code, True)

            self.assertExpectedInline(count_numel(f), """45""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v1(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v2(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            tmp = torch.add(x, 1)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output, tmp

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """70""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v3(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x.add_(1)
            return output

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """80""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v4(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            output2 = x_view.mul(2)
            return output, output2

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """70""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v5(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x_view.mul_(2)
            return output

        inp = (T(10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(f, *inp), """80""")

    @unittest.skipUnless(has_triton(), "Triton not available")
    def test_inplace_triton_kernel_v6(self, device):
        def f(x: torch.Tensor, y: torch.Tensor):
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        t = T(10, device=device)
        inp = (t, t.view(-1))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    def test_inplace_randperm_scatter(self, device):
        def scaled_index_add(x, y, scale_y):
            index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
            out = x.index_add_(dim=0, source=y * scale_y, index=index)
            return out

        inp = (T(10, 10, device=device), T(5, 10, device=device), T(10, device=device))
        self.assertExpectedInline(count_numel(scaled_index_add, *inp), """250""")


# Test cases where we don't do the right thing yet.
class WouldBeNiceIfItWorked:
    def test_horizontal(self):
        def f(a):
            b = a.sum(dim=0)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # TODO: We aren't fusing outer dim softmaxes
    def test_softmax_outer(self):
        def f(a):
            return torch.softmax(a, dim=0)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # TODO: The greedy fusion strategy results in suboptimal grouping
    @patch.object(config, "realize_opcount_threshold", 0)
    def test_fusion_choice4(self):
        def f(a, b, b2):
            c = a + b
            d = torch.mm(c, c)
            e = c + b + b2
            f = d + e + b2
            return f, e

        inp = (T(10, 10), T(10, 10, dtype=torch.float16), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

    # TODO: We materialize the intermediate if we don't unroll the reduction
    def test_neighbor(self):
        def f(a, b):
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 8), T(1, 10, 8))
        self.assertExpectedInline(count_numel(f, *inp), """170""")


instantiate_device_type_tests(
    NumBytesMetricTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(FusionTests, globals(), except_for="cpu", allow_xpu=True)
instantiate_device_type_tests(
    SchedulerFusionTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(SchedulerFusionTestsOnlyCPU, globals(), only_for="cpu")
instantiate_device_type_tests(TilingTests, globals(), except_for="cpu", allow_xpu=True)
instantiate_device_type_tests(
    MinCutPartitioningTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(NoopTests, globals(), except_for="cpu", allow_xpu=True)
instantiate_device_type_tests(
    InplacingTests, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if has_triton():
        run_tests(needs="filelock")
