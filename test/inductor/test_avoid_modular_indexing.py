# Owner(s): ["module: inductor"]
import contextlib

import sympy

import torch
import torch.nn.functional as F
from torch._inductor import config, ir, metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.exc import NonLinearIndexingError
from torch._inductor.ir import is_triton
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_linear_indexing
from torch._inductor.virtualized import V
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


class LinearIndexingPredicateTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "expr,linear",
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
    def test_is_linear_indexing(self, expr, linear):
        self.assertEqual(is_linear_indexing(expr), linear)


instantiate_parametrized_tests(LinearIndexingPredicateTest)


class ModularIndexingDecisionTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_config_overrides_backend(self):
        with config.patch(avoid_modular_indexing=True):
            self.assertFalse(V.choices.use_modular_indexing("cpu", 8, "view"))
        with config.patch(avoid_modular_indexing=False):
            self.assertTrue(V.choices.use_modular_indexing("mtia", 8, "view"))

    @parametrize("kind", ["view", "reduction", "slice_scatter", "repeat"])
    @config.patch(avoid_modular_indexing=None)
    def test_cpp_backend_inlines(self, kind):
        self.assertTrue(V.choices.use_modular_indexing("cpu", 2**30, kind))

    @config.patch(avoid_modular_indexing=None)
    def test_affine_backends(self):
        # backends declaring PREFER_AFFINE_INDEXING keep load/store indices
        # affine; a third-party scheduling that declares nothing inlines
        self.assertFalse(V.choices.use_modular_indexing("mtia", 8, "view"))
        self.assertFalse(V.choices.use_modular_indexing("mtia", 8, "view_linear"))
        self.assertFalse(V.choices.use_modular_indexing("mtia", 8, "index"))
        with config.patch(cpu_backend="pallas"):
            self.assertFalse(V.choices.use_modular_indexing("cpu", 8, "reduction"))
            # index arithmetic can only move behind an indirect variable where
            # the backend supports indirect indexing
            self.assertTrue(V.choices.use_modular_indexing("cpu", 8, "index"))

    @parametrize("kind", ["reduction", "slice_scatter"])
    @config.patch(avoid_modular_indexing=None)
    def test_triton_gpu_avoids(self, kind):
        self.assertFalse(V.choices.use_modular_indexing("cuda", 8, kind))

    @parametrize("kind", ["repeat", "roll", "index", "view_linear", "view_copy"])
    @config.patch(avoid_modular_indexing=None)
    def test_triton_gpu_inlines(self, kind):
        self.assertTrue(V.choices.use_modular_indexing("cuda", 2**30, kind))

    @config.patch(avoid_modular_indexing=None, realize_modular_view_min_numel=1024)
    def test_triton_gpu_view_threshold(self):
        self.assertTrue(V.choices.use_modular_indexing("cuda", 1023, "view"))
        self.assertFalse(V.choices.use_modular_indexing("cuda", 1024, "view"))
        self.assertTrue(V.choices.use_modular_indexing("cuda", None, "view"))
        self.assertFalse(V.choices.use_modular_indexing("cuda", 1024, lambda: "view"))

    @config.patch(avoid_modular_indexing=None)
    def test_unknown_device_inlines(self):
        # no Inductor backend or no device: inline as before
        self.assertTrue(V.choices.use_modular_indexing("meta", 8, "reduction"))
        self.assertTrue(V.choices.use_modular_indexing(None, 8, "reduction"))

    def test_view_classification_is_lazy(self):
        def classify():
            raise AssertionError("classified where the answer does not need it")

        with config.patch(avoid_modular_indexing=None):
            self.assertTrue(V.choices.use_modular_indexing("cpu", 2**30, classify))
            self.assertFalse(V.choices.use_modular_indexing("mtia", 2**30, classify))
            # below the realization threshold every view kind inlines
            with config.patch(realize_modular_view_min_numel=1024):
                self.assertTrue(V.choices.use_modular_indexing("cuda", 1023, classify))
        with config.patch(avoid_modular_indexing=True):
            self.assertFalse(V.choices.use_modular_indexing("cuda", 2**30, classify))
        with config.patch(avoid_modular_indexing=False):
            self.assertTrue(V.choices.use_modular_indexing("cuda", 2**30, classify))

    def test_choices_override(self):
        # a backend can replace the whole decision, config check included
        class AlwaysAffine(InductorChoices):
            def use_modular_indexing(self, device, numel=None, kind="view"):
                return False

        with (
            V.set_choices_handler(AlwaysAffine()),
            config.patch(avoid_modular_indexing=False),
        ):
            self.assertFalse(V.choices.use_modular_indexing("cpu", 8, "view"))


instantiate_parametrized_tests(ModularIndexingDecisionTest)


class AvoidModularIndexingTest(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR
    """
    With avoid_modular_indexing every load/store index stays affine in the loop
    and indirect variables; raise_on_modular_indexing turns any remaining
    ModularIndexing, FloorDiv, Min/Max/Abs or product of loop variables into an
    error.
    """

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

    def check(self, fn, *args, kernel_count=None):
        metrics.reset()
        expected = fn(*args)
        actual = torch.compile(fn)(*args)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        # the CPU backend splits work into kernels differently
        if kernel_count is not None and args[0].device.type != "cpu":
            self.assertEqual(metrics.generated_kernel_count, kernel_count)

    def test_detection(self, device):
        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        with config.patch(avoid_modular_indexing=False):
            with self.assertRaisesRegex(Exception, "NonLinearIndexingError"):
                torch.compile(fn)(a, b)
        self.assertTrue(issubclass(NonLinearIndexingError, RuntimeError))

    def test_reshape_of_permuted_pointwise(self, device):
        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        # the realized copy fuses with its consumer
        self.check(fn, a, b, kernel_count=1)

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

        self.check(fn, torch.randn(5, 7, device=device), kernel_count=1)

    def test_repeat_unrealized_input(self, device):
        def fn(x, y):
            return (x + y).repeat(3, 1, 2).sum(dim=0)

        x = torch.randn(2, 5, 7, device=device)
        y = torch.randn(5, 7, device=device)
        self.check(fn, x, y)

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
        self.check(fn, x, cls_tok, w, b, kernel_count=2)

    def test_reshape_of_strided_buffer(self, device):
        # a realized channels-last buffer reshaped across non-contiguous dims
        def fn(x):
            y = torch.nn.functional.relu(x).to(memory_format=torch.channels_last)
            return y.reshape(x.shape[0], -1) + 1

        self.check(fn, torch.randn(2, 3, 4, 5, device=device))

    def test_split_reduction(self, device):
        # the split of a multi-dim reduction must fall on a dim boundary so the
        # per-batch operand is indexed linearly
        def fn(x, y):
            return (x.float() * y).sum(dim=(0, 2, 3), keepdim=True)

        x = torch.randn(128, 256, 7, 7, device=device, dtype=torch.float16)
        x = x.to(memory_format=torch.channels_last)
        y = torch.randn(128, 1, 1, 1, device=device)
        self.check(fn, x, y)

    def test_split_reduction_to_scalar(self, device):
        def fn(x, y):
            return (x * y).sum()

        x = torch.randn(1000, 3, 65, device=device)
        y = torch.randn(1000, 1, 1, device=device)
        self.check(fn, x, y)

    def test_roll(self, device):
        # roll decomposes to two slices instead of a (iota + shift) % size gather
        def fn(x):
            return torch.roll(x, shifts=(-3, 2), dims=(1, 2)) * 2

        self.check(fn, torch.randn(2, 7, 9, 4, device=device), kernel_count=1)

    def test_roll_flattened(self, device):
        def fn(x):
            return torch.roll(x, 5) + 1

        self.check(fn, torch.randn(6, 5, device=device), kernel_count=1)

    def test_split_reduction_uneven_dim(self, device):
        # a prime leading dim has no even split, so the boundary dim is split
        # into masked chunks instead of falling back to a flat split
        def fn(x, y):
            return (x * y).sum()

        x = torch.randn(10007, 2, device=device)
        y = torch.randn(10007, 1, device=device)
        self.check(fn, x, y)

    def test_var_mean_split_reduction(self, device):
        def fn(x, y):
            return torch.var_mean(x * y, dim=(0, 2))

        x = torch.randn(1000, 4, 65, device=device)
        y = torch.randn(1000, 1, 1, device=device)
        self.check(fn, x, y)

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

    def test_strided_slice_scatter(self, device):
        def fn(x, src):
            return torch.slice_scatter(x, src, 1, 3, 11, 3) + 1

        x = torch.randn(4, 12, device=device)
        src = torch.randn(4, 3, device=device)
        self.check(fn, x, src)
        # Triton GPU backends take the split-space lowering by default too,
        # and the realized split result keeps the consumer's index affine
        if is_triton(torch.device(device)):
            torch._dynamo.reset()
            with config.patch(avoid_modular_indexing=None):
                self.check(fn, x, src)

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
        self.check(fn, a, b)

    def test_index_arithmetic_stays_indirect(self, device):
        # %, // and negative wrapping of an index tensor are kept as value
        # computations; the loads go through an indirect variable instead of
        # a ModularIndexing/FloorDiv/Where address
        def fn(x):
            i = torch.arange(16, device=x.device)
            return x[i % 4] + x[i // 2] + x[i - 8]

        self.check(fn, torch.randn(8, 3, device=device), kernel_count=1)

    def test_reflection_and_replication_pad(self, device):
        # the decompositions index with abs() and clamp() of an iota
        def fn(x):
            return F.pad(x, (1, 2, 2, 1), mode="reflect") + F.pad(
                x, (1, 2, 2, 1), mode="replicate"
            )

        self.check(fn, torch.randn(2, 3, 6, 7, device=device))

    def test_reflection_pad_backward(self, device):
        def fn(x):
            return F.pad(x, (1, 2, 2, 1), mode="reflect").square().sum()

        x = torch.randn(2, 3, 6, 7, device=device, requires_grad=True)
        expected = torch.autograd.grad(fn(x), x)
        actual = torch.autograd.grad(torch.compile(fn)(x), x)
        self.assertEqual(actual, expected)

    def test_auto_mode_inlines_small_merge(self, device):
        # with the default (per-backend) decision, CPU and GPU inline a small
        # merge into the cat instead of realizing it, so there is no extra
        # kernel
        def fn(x, cls_tok, w, b):
            y = (x + b.view(1, -1, 1, 1)).flatten(2).transpose(1, 2)
            y = torch.cat((cls_tok.expand(x.shape[0], -1, -1), y), dim=1)
            return torch.nn.functional.layer_norm(y, (y.shape[-1],), w, None)

        x = torch.randn(2, 8, 4, 4, device=device)
        cls_tok = torch.randn(1, 1, 8, device=device)
        w = torch.randn(8, device=device)
        b = torch.randn(8, device=device)
        with config.patch(
            {
                "avoid_modular_indexing": None,
                "test_configs.raise_on_modular_indexing": False,
            }
        ):
            self.check(fn, x, cls_tok, w, b, kernel_count=1)

    def test_auto_mode_view_threshold(self, device):
        # a merge of fresh compute with modular loads is realized on Triton
        # GPU backends once it reaches realize_modular_view_min_numel; below
        # that (and on other backends) it is inlined
        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        with config.patch(
            {
                "avoid_modular_indexing": None,
                "test_configs.raise_on_modular_indexing": True,
            }
        ):
            with config.patch(realize_modular_view_min_numel=1024):
                with self.assertRaisesRegex(Exception, "NonLinearIndexingError"):
                    torch.compile(fn)(a, b)
            torch._dynamo.reset()
            with config.patch(realize_modular_view_min_numel=512):
                if is_triton(torch.device(device)):
                    self.check(fn, a, b, kernel_count=1)
                else:
                    with self.assertRaisesRegex(Exception, "NonLinearIndexingError"):
                        torch.compile(fn)(a, b)

    def test_auto_mode_keeps_cast_merge_inlined(self, device):
        # the HF causal-LM loss: logits.float().view(-1, V) is a row-major
        # merge of a cast, so inlining needs no ModularIndexing and realizing
        # it would write a float32 copy of the logits
        def fn(logits):
            return torch.log_softmax(logits.float().view(-1, logits.shape[-1]), -1)

        logits = torch.randn(2, 8, 64, device=device, dtype=torch.bfloat16)
        with config.patch(
            {
                "avoid_modular_indexing": None,
                "test_configs.raise_on_modular_indexing": True,
                "realize_modular_view_min_numel": 1,
            }
        ):
            metrics.reset()
            torch.compile(fn)(logits)
            kernels = metrics.generated_kernel_count
            nodes = metrics.ir_nodes_pre_fusion
        torch._dynamo.reset()
        with config.patch(avoid_modular_indexing=False):
            metrics.reset()
            torch.compile(fn)(logits)
            self.assertEqual(kernels, metrics.generated_kernel_count)
            self.assertEqual(nodes, metrics.ir_nodes_pre_fusion)

    def test_choices_override_realizes_merge(self, device):
        class AlwaysAffine(InductorChoices):
            def use_modular_indexing(self, device, numel=None, kind="view"):
                return False

        def fn(a, b):
            return (a.permute(0, 2, 1) + b).reshape(a.shape[0], -1) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 16, 8, device=device)
        with (
            V.set_choices_handler(AlwaysAffine()),
            config.patch(
                {
                    "avoid_modular_indexing": False,
                    "test_configs.raise_on_modular_indexing": True,
                }
            ),
        ):
            self.check(fn, a, b, kernel_count=1)

    def test_auto_mode_keeps_broadcast_merge_inlined(self, device):
        # the only modular load is the table broadcast over the batch (the
        # labels in the causal-LM loss backward, position embeddings): its
        # index arithmetic is negligible next to the affine traffic, so the
        # merge is inlined even above the size threshold
        def fn(a, p):
            return (a + p[None]).view(-1, a.shape[-1]) * 2

        a = torch.randn(8, 8, 16, device=device)
        p = torch.randn(8, 16, device=device)
        with config.patch(
            {
                "avoid_modular_indexing": None,
                "test_configs.raise_on_modular_indexing": False,
                "realize_modular_view_min_numel": 1,
            }
        ):
            self.check(fn, a, p, kernel_count=1)
            self.assertEqual(metrics.ir_nodes_pre_fusion, 1)

    def test_auto_mode_keeps_row_major_merge_inlined(self, device):
        # a row-major merge of fresh compute: the loads are affine once
        # inlined, so there is nothing to gain from realizing it even above
        # the size threshold
        def fn(a, b):
            return torch.nn.functional.gelu(a + b).view(-1, a.shape[-1]) * 2

        a = torch.randn(4, 8, 16, device=device)
        b = torch.randn(4, 8, 16, device=device)
        with config.patch(
            {
                "avoid_modular_indexing": None,
                "test_configs.raise_on_modular_indexing": True,
                "realize_modular_view_min_numel": 1,
            }
        ):
            self.check(fn, a, b, kernel_count=1)
            self.assertEqual(metrics.ir_nodes_pre_fusion, 1)

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
    def test_chunks_cover_boundary_dim(self, reduction_ranges, split):
        outer, inner = ir.Reduction._split_reduction_ranges(reduction_ranges, split)
        k = len(outer) - 1
        self.assertEqual([int(s) for s in outer[:-1]], list(reduction_ranges[:k]))
        self.assertEqual([int(s) for s in inner[1:]], list(reduction_ranges[k + 1 :]))
        chunks, chunk = int(outer[-1]), int(inner[0])
        # the chunks cover the boundary dim and none of them is empty; the
        # aligned loader masks the tail exactly when chunks * chunk != size
        self.assertGreaterEqual(chunks * chunk, reduction_ranges[k])
        self.assertLess((chunks - 1) * chunk, reduction_ranges[k])


instantiate_parametrized_tests(SplitReductionRangesTest)

if __name__ == "__main__":
    if HAS_CPU or HAS_GPU:
        run_tests()
