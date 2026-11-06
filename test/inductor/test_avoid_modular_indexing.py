# Owner(s): ["module: inductor"]
import contextlib
import math

import sympy

import torch
import torch.nn.functional as F
from torch._inductor import config, ir, metrics
from torch._inductor.exc import InductorError, NonAffineIndexingError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_affine_indexing
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU
from torch.utils._sympy.functions import FloorDiv, Identity, Min, ModularIndexing, Where
from torch.utils._sympy.symbol import make_symbol, SymT


_p0, _p1 = make_symbol(SymT.INDEX, 0), make_symbol(SymT.INDEX, 1)
_s0, _s1 = make_symbol(SymT.SIZE, 0), make_symbol(SymT.SIZE, 1)


class AffineIndexingPredicateTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "expr,affine",
        [
            subtest((sympy.Integer(7), True), name="constant"),
            subtest((_p0 + 3 * _p1, True), name="affine"),
            subtest((_s0 * _s1 * _p0 + _s1 * _p1, True), name="symbolic_strides"),
            subtest(
                (FloorDiv(_s0, 2) * _p0 + ModularIndexing(_s1, 1, 4), True),
                name="floordiv_of_sizes",
            ),
            subtest((Identity(_p0 - 3) + 5, True), name="identity_wrapper"),
            subtest((ModularIndexing(_p0, 1, 4), False), name="modular"),
            subtest((FloorDiv(_p0, 2) + _p1, False), name="floordiv"),
            subtest((Min(_p0, 3), False), name="min"),
            subtest((sympy.Abs(_p0 - 1), False), name="abs"),
            subtest((Where(_p0 < 0, _p0 + 8, _p0), False), name="where"),
            subtest((_p0 * _p1, False), name="product"),
        ],
    )
    def test_is_affine_indexing(self, expr, affine):
        self.assertEqual(is_affine_indexing(expr), affine)


instantiate_parametrized_tests(AffineIndexingPredicateTest)


class AvoidModularIndexingTest(TestCase):
    """
    With avoid_modular_indexing every load/store index stays affine in the loop
    and indirect variables; raise_on_modular_indexing turns any remaining
    ModularIndexing, FloorDiv, Min/Max/Abs or product of loop variables into an
    error.
    """

    hw_classification = HardwareClassification.ACCELERATOR

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._config_stack = contextlib.ExitStack()
        cls._config_stack.enter_context(
            config.patch(
                {
                    "avoid_modular_indexing": True,
                    "test_configs.raise_on_modular_indexing": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._config_stack.close()
        super().tearDownClass()

    def check(self, fn, *args, kernel_count=None, modular_without_option=False):
        """
        Compile fn with the option on and compare with eager. kernel_count is
        checked on GPUs (the CPU backend splits work into kernels differently).
        modular_without_option asserts that the same program hits the
        detector with the option off, so the test cannot pass vacuously once
        a lowering stops producing modular indexing on its own.
        """
        metrics.reset()
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        if kernel_count is not None and args[0].device.type != "cpu":
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
        if modular_without_option:
            self.assert_modular_without_option(fn, *args)

    def assert_modular_without_option(self, fn, *args):
        torch._dynamo.reset()
        with config.patch(avoid_modular_indexing=False):
            with self.assertRaises(InductorError) as cm:
                torch.compile(fn)(*args)
        self.assertIsInstance(cm.exception.inner_exception, NonAffineIndexingError)
        self.assertIn("non-affine indexing in", str(cm.exception.inner_exception))
        torch._dynamo.reset()

    def test_detection(self, device):
        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        self.assert_modular_without_option(fn, a, b)
        # the detector names the kernel and the FX origins of the offending index
        with config.patch(avoid_modular_indexing=False):
            with self.assertRaises(InductorError) as cm:
                torch.compile(fn)(a, b)
        self.assertIn("origins:", str(cm.exception.inner_exception))

    def test_reshape_of_permuted_pointwise(self, device):
        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        # the realized copy fuses with its consumer
        self.check(fn, a, b, kernel_count=1, modular_without_option=True)

    def test_contiguous_merge_stays_fused(self, device):
        def fn(a, b):
            return (a + b).view(-1, a.shape[-1]) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 8, 16, device=device)
        self.check(fn, a, b, kernel_count=1)

    def test_split_view_is_inlined(self, device):
        def fn(a):
            return (a * 3).view(4, 8, 16).sum(dim=1)

        self.check(fn, torch.randn(32, 16, device=device), kernel_count=1)

    def test_repeat(self, device):
        def fn(x):
            return x.repeat(2, 3) + 1

        x = torch.randn(5, 7, device=device)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_repeat_unrealized_input(self, device):
        def fn(x, y):
            return (x + y).repeat(3, 1, 2).sum(dim=0)

        x = torch.randn(2, 5, 7, device=device)
        y = torch.randn(5, 7, device=device)
        self.check(fn, x, y, modular_without_option=True)

    def test_repeat_dynamic_shapes(self, device):
        def fn(x):
            return x.repeat(2, 3) + 1

        x = torch.randn(5, 7, device=device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_cat_with_merged_view(self, device):
        # flatten(2).transpose(1, 2) of a conv-bias add followed by cat with a
        # cls token: the cat shifts the index, so the merge must be realized.
        def fn(x, cls_tok, w, b):
            y = (x + b.view(1, -1, 1, 1)).flatten(2).transpose(1, 2)
            y = torch.cat((cls_tok.expand(x.shape[0], -1, -1), y), dim=1)
            return torch.nn.functional.layer_norm(y, (y.shape[-1],), w, None)

        x = torch.randn(2, 8, 4, 4, device=device)
        cls_tok = torch.randn(1, 1, 8, device=device)
        w = torch.randn(8, device=device)
        b = torch.randn(8, device=device)
        # the CPU backend copies each input into the output, and that copy's
        # loop splits at the merge; only the pointwise cat keeps the modular
        # index
        self.check(
            fn, x, cls_tok, w, b, kernel_count=2, modular_without_option=device != "cpu"
        )

    def test_reshape_of_strided_buffer(self, device):
        # a realized channels-last buffer reshaped across non-contiguous dims
        def fn(x):
            y = torch.nn.functional.relu(x).to(memory_format=torch.channels_last)
            return y.reshape(x.shape[0], -1) + 1

        self.check(fn, torch.randn(2, 3, 4, 5, device=device))

    def check_split_reduction(
        self, fn, *args, kernel_count=2, modular_without_option=True
    ):
        """
        The CPU backend never splits reductions, so these tests only compare
        numbers there; on GPUs the kernel count proves the split happened and
        the detector run proves the flat split is what the option replaces.
        """
        if args[0].device.type == "cpu":
            self.check(fn, *args)
        else:
            self.check(
                fn,
                *args,
                kernel_count=kernel_count,
                modular_without_option=modular_without_option,
            )

    def test_split_reduction(self, device):
        # the split of a multi-dim reduction must fall on a dim boundary so the
        # per-batch operand is indexed linearly
        def fn(x, y):
            return (x.float() * y).sum(dim=(0, 2, 3), keepdim=True)

        x = torch.randn(128, 256, 7, 7, device=device, dtype=torch.float16)
        x = x.to(memory_format=torch.channels_last)
        y = torch.randn(128, 1, 1, 1, device=device)
        self.check_split_reduction(fn, x, y)

    def test_split_reduction_to_scalar(self, device):
        def fn(x, y):
            return (x * y).sum()

        x = torch.randn(1000, 3, 65, device=device)
        y = torch.randn(1000, 1, 1, device=device)
        self.check_split_reduction(fn, x, y)

    def test_split_reduction_channels_last_dense_dim(self, device):
        # a full reduction over a channels-last buffer reduces the channel dim
        # innermost, like the flat split does, so the inner loop stays
        # contiguous after the split
        def fn(x):
            return (x * 2).mean()

        x = torch.randn(1, 64, 128, 128, device=device)
        # the flat split's block is a multiple of the inner dims here, so loop
        # splitting keeps even the flat split affine
        self.check_split_reduction(
            fn, x.to(memory_format=torch.channels_last), modular_without_option=False
        )

    def test_roll(self, device):
        # roll decomposes to two slices instead of a (iota + shift) % size gather
        def fn(x):
            return torch.roll(x, shifts=(-3, 2), dims=(1, 2)) * 2

        x = torch.randn(2, 7, 9, 4, device=device)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_roll_flattened(self, device):
        def fn(x):
            return torch.roll(x, 5) + 1

        x = torch.randn(6, 5, device=device)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_roll_dynamic_shapes(self, device):
        # the shift is reduced modulo a symbolic size and the no-op check must
        # not guard on it
        def fn(x):
            return torch.roll(x, shifts=(-3, 2), dims=(0, 1)) * 2

        x = torch.randn(7, 9, 4, device=device)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_roll_argument_checks(self, device):
        x = torch.randn(3, 4, device=device)
        for shifts, dims in [((1, 2), ()), ((), (0,)), ((1,), (0, 1))]:
            with self.assertRaisesRegex(Exception, "shifts"):
                torch.compile(torch.roll)(x, shifts, dims)
            torch._dynamo.reset()

    def test_split_reduction_uneven_dim(self, device):
        # a prime leading dim has no even split, so the boundary dim is split
        # into masked chunks instead of falling back to a flat split
        def fn(x, y):
            return (x * y).sum()

        x = torch.randn(100003, 2, device=device)
        y = torch.randn(100003, 1, device=device)
        self.check_split_reduction(fn, x, y)

    def test_var_mean_split_reduction(self, device):
        def fn(x, y):
            return torch.var_mean(x * y, dim=(0, 2))

        x = torch.randn(1000, 4, 65, device=device)
        y = torch.randn(1000, 1, 1, device=device)
        self.check_split_reduction(fn, x, y)

    def test_var_mean_split_reduction_uneven_dim(self, device):
        # the masked welford_combine path: the chunks of the prime dim
        # overshoot it, so the tail must not count towards the weight
        def fn(x, y):
            return torch.var_mean(x * y, dim=(0, 2))

        x = torch.randn(100003, 4, 3, device=device)
        y = torch.randn(100003, 1, 1, device=device)
        self.check_split_reduction(fn, x, y)

    def test_online_softmax_split_reduction(self, device):
        # a long softmax row is split; the chunks of the prime row length
        # overshoot it, so the tail is masked out of the max and the weight
        def fn(x):
            return torch.softmax(x, -1)

        x = torch.randn(4, 100003, device=device)
        # the flat split of a one-dim reduction is affine too
        self.check_split_reduction(fn, x, kernel_count=3, modular_without_option=False)

    def test_fusion_does_not_reindex_loops(self, device):
        # RMS norm over a reshape: fusion would reindex the pointwise's
        # [256, 8192] loops to the reduction's [16384, 128], which turns the
        # per-row bias read into FloorDiv(t0, 64); the detector also runs
        # after every loop transformation, so this fails if the reindex
        # happens
        def fn(x, b, w):
            h = (x + b[:, None]).view(-1, 128)
            var = (h * h).mean(-1, keepdim=True)
            return h * torch.rsqrt(var + 1e-6) * w

        x = torch.randn(256, 8192, device=device)
        b = torch.randn(256, device=device)
        w = torch.randn(128, device=device)
        self.check(fn, x, b, w)

    def test_roll_noop_does_not_alias(self, device):
        def fn(x):
            y = torch.roll(x, 7, 1)
            x.add_(1)
            return y

        x = torch.zeros(3, 7, device=device)
        self.assertEqual(torch.compile(fn)(x.clone()), fn(x.clone()))

    def test_max_pool2d_backward_with_padding(self, device):
        # the realized copy of the pooling indices is fused back into the
        # scatter kernel; the bounds assert must keep the kernel mask
        def fn(x):
            return torch.nn.functional.max_pool2d(x, 3, 2, 1).square().sum()

        x = torch.randn(2, 3, 17, 19, device=device, requires_grad=True)
        expected = torch.autograd.grad(fn(x), x)
        actual = torch.autograd.grad(torch.compile(fn)(x), x)
        self.assertEqual(actual, expected)

    @parametrize("start,end,step", [(3, 11, 3), (4, 12, 3), (1, 12, 2)])
    def test_strided_slice_scatter(self, device, start, end, step):
        # start // step and start % step are both nonzero in the second case
        def fn(x, src):
            return torch.slice_scatter(x, src, 1, start, end, step) + 1

        x = torch.randn(4, 12, device=device)
        src = torch.randn(4, len(range(start, end, step)), device=device)
        self.check(fn, x, src, modular_without_option=True)

    def test_strided_slice_backward(self, device):
        # gpt-j rotate_every_two: the backward is a slice_scatter with step 2
        def fn(x, w):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return (torch.stack((-x2, x1), dim=-1).flatten(-2) * w).sum()

        x = torch.randn(2, 4, 64, device=device, requires_grad=True)
        w = torch.randn(64, device=device)
        expected = torch.autograd.grad(fn(x, w), x)
        actual = torch.autograd.grad(torch.compile(fn)(x, w), x)
        self.assertEqual(actual, expected)

    def test_dynamic_shapes(self, device):
        def fn(a, b):
            return (a.transpose(1, 2) * b).reshape(a.shape[0], -1).cumsum(-1)

        a = torch.randn(3, 8, 6, device=device)
        b = torch.randn(3, 6, 8, device=device)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        # the CPU backend realizes the scan input, and that copy's loop
        # splits at the merge
        self.check(fn, a, b, modular_without_option=device != "cpu")

    def test_dynamic_merged_dims(self, device):
        # the merged dims themselves are symbolic
        def fn(a, b):
            return (a.transpose(1, 2) * b).reshape(a.shape[0], -1) * 2

        a = torch.randn(3, 8, 6, device=device)
        b = torch.randn(3, 6, 8, device=device)
        for t in (a, b):
            torch._dynamo.mark_dynamic(t, 1)
            torch._dynamo.mark_dynamic(t, 2)
        self.check(fn, a, b, kernel_count=1, modular_without_option=True)

    def test_index_arithmetic_stays_indirect(self, device):
        # %, // and negative wrapping of an index tensor are kept as value
        # computations; the loads go through an indirect variable instead of
        # a ModularIndexing/FloorDiv/Where address
        def fn(x):
            i = torch.arange(16, device=x.device)
            return x[i % 4] + x[i // 2] + x[i - 8]

        x = torch.randn(8, 3, device=device)
        self.check(fn, x, kernel_count=1, modular_without_option=True)

    def test_reflection_and_replication_pad(self, device):
        # the decompositions index with abs() and clamp() of an iota
        def fn(x):
            return F.pad(x, (1, 2, 2, 1), mode="reflect") + F.pad(
                x, (1, 2, 2, 1), mode="replicate"
            )

        x = torch.randn(2, 3, 6, 7, device=device)
        self.check(fn, x, modular_without_option=True)

    def test_reflection_pad_backward(self, device):
        def fn(x):
            return F.pad(x, (1, 2, 2, 1), mode="reflect").square().sum()

        x = torch.randn(2, 3, 6, 7, device=device, requires_grad=True)
        expected = torch.autograd.grad(fn(x), x)
        actual = torch.autograd.grad(torch.compile(fn)(x), x)
        self.assertEqual(actual, expected)

    def test_avg_pool2d_backward_with_padding(self, device):
        # the window bounds are FloorDiv/Min of the output index and reach the
        # grad load through indirect_indexing
        def fn(x):
            return F.avg_pool2d(x, 3, 2, 1, count_include_pad=False).square().sum()

        x = torch.randn(2, 3, 17, 19, device=device, requires_grad=True)
        expected = torch.autograd.grad(fn(x), x)
        actual = torch.autograd.grad(torch.compile(fn)(x), x)
        self.assertEqual(actual, expected)


instantiate_device_type_tests(AvoidModularIndexingTest, globals())


class SplitReductionRangesTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "reduction_ranges,split",
        [
            subtest(((1000, 4), 900), name="uneven_candidate_divides"),
            subtest(((10007, 2), 1181), name="prime_dim"),
            subtest(((8, 384, 512), 1181), name="chunk_inner_dim"),
            subtest(((32, 7, 7), 13), name="small_dims"),
            subtest(((64, 196, 512), 16), name="even_leading_dim"),
        ],
    )
    @config.patch(avoid_modular_indexing=True)
    def test_chunks_cover_boundary_dim(self, reduction_ranges, split):
        aligned = ir.Reduction._split_reduction_ranges(reduction_ranges, split)
        outer, inner = aligned.outer_ranges, aligned.inner_ranges
        k = len(outer) - 1
        self.assertEqual([int(s) for s in outer[:-1]], list(reduction_ranges[:k]))
        self.assertEqual([int(s) for s in inner[1:]], list(reduction_ranges[k + 1 :]))
        self.assertEqual(int(aligned.boundary_size), reduction_ranges[k])
        self.assertIsNone(aligned.perm)
        chunks, chunk = int(outer[-1]), int(inner[0])
        # the chunks cover the boundary dim and none of them is empty; the
        # aligned loader masks the tail exactly when chunks * chunk != size
        self.assertGreaterEqual(chunks * chunk, reduction_ranges[k])
        self.assertLess((chunks - 1) * chunk, reduction_ranges[k])
        self.assertEqual(aligned.need_mask, chunks * chunk != reduction_ranges[k])
        self.assertEqual(int(aligned.split), math.prod(reduction_ranges[:k]) * chunks)

    def test_disabled_without_option(self):
        with config.patch(avoid_modular_indexing=False):
            self.assertIsNone(ir.Reduction._split_reduction_ranges((1000, 4), 900))


instantiate_parametrized_tests(SplitReductionRangesTest)

if __name__ == "__main__":
    if HAS_CPU or HAS_GPU:
        run_tests()
