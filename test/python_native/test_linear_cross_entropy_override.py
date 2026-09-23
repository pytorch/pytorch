# Owner(s): ["module: dsl-native-ops"]
#
# Routing tests for the CuTeDSL overrides of the chunked ``linear_cross_entropy``
# ops. Both tests read ``cutedsl_impl._OVERRIDES``, so adding an override needs
# no change here.

import math
import unittest
import unittest.mock

import torch
import torch._native.registry as registry_module
from torch._native import cutedsl_utils as cu, variants
from torch._native.ops.linear_cross_entropy import cutedsl_impl
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions
from torch.testing._internal.common_cuda import (
    has_device_side_assert,
    TEST_CUDA,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


_OP_SYMBOLS = [op_symbol for op_symbol, _, _ in cutedsl_impl._OVERRIDES]
_SCALAR_OP = "_linear_cross_entropy_batch_chunked"
# Self-discovered, so promoting or adding a variant needs no change here.
_KERNEL_VARIANTS = sorted(
    name for name in cutedsl_impl._VARIANTS[_SCALAR_OP] if name != variants.PASSTHROUGH
)


# Repeated on every test that needs a live kernel rather than a fallback.
_needs_kernel = unittest.skipIf(
    not TEST_CUDA or not cutedsl_impl._arch_supported(),
    "no kernel variant is eligible on this device, so there is no kernel "
    "path to test -- the call would fall back to eager",
)


def _compact_options(**overrides):
    """The one options shape the kernel's gate accepts; callers set what varies."""
    return LinearCrossEntropyOptions(
        **{
            "acc_policy": "compact",
            "acc_dtype": torch.float32,
            "chunking_method": None,
            **overrides,
        }
    )


@unittest.skipIf(not TEST_CUDA, "the overrides are registered on CUDA")
class TestLinearCrossEntropyOverride(TestCase):
    def setUp(self):
        super().setUp()
        # The version check belongs here too: `cu.register_op_override` drops
        # every registration when the installed CuTeDSL is not known-good, so
        # these tests would look for overrides that were never installed and
        # fail where they should skip.
        if (
            not cu.runtime_available()
            or cu.check_native_jit_disabled()
            or not cu._version_is_ok()
        ):
            self.skipTest(
                "CuTeDSL runtime unavailable, disabled, or at an unsupported version"
            )

    def _cutedsl_nodes(self, op_symbol):
        key = ("torch_nn", op_symbol, "CUDA")
        nodes = registry_module._graphs.get(key, [])
        # A key can hold several overrides, and deregistered ones stay listed
        # with active=False.
        return key, [n for n in nodes if n.dsl_name == "cutedsl" and n.active]

    def test_every_override_is_installed(self):
        """`import torch` must leave each override registered with a live
        router -- which it can only do if the lazily-imported module defining
        the ops was pulled in before registration ran."""
        self.assertTrue(_OP_SYMBOLS, "the impl module registers nothing")
        for op_symbol in _OP_SYMBOLS:
            key, live = self._cutedsl_nodes(op_symbol)
            self.assertTrue(live, f"no live cutedsl override for {op_symbol}")
            self.assertIn(key, registry_module._override_libs)

    @_needs_kernel
    def test_empty_batch_returns_a_zeroed_bias_gradient(self):
        """An empty batch has no loop to write `grad_linear_bias`, so the early
        return must zero it. `fill_uninitialized_memory` makes an unzeroed
        allocation visible, since `torch.empty` often returns zeroed pages."""
        in_features, num_classes = 32, 64
        input = torch.zeros(
            0, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        linear_weight = torch.randn(
            num_classes, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        linear_bias = torch.randn(
            num_classes, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        target = torch.zeros(0, device="cuda", dtype=torch.int64)
        options = _compact_options(batch_chunk_size=8)
        with DeterministicGuard(True, fill_uninitialized_memory=True):
            loss = torch.nn.functional.linear_cross_entropy(
                input, linear_weight, target, linear_bias=linear_bias, options=options
            )
            loss.backward()
        self.assertEqual(
            linear_bias.grad,
            torch.zeros_like(linear_bias.grad),
            msg="the empty-batch path returned an unwritten bias gradient",
        )

    @unittest.skipIf(not TEST_CUDA, "the overrides register at a CUDA key")
    @parametrize("reduction", ["mean", "none"])
    def test_cutedsl_path_is_used(self, reduction):
        """A call must route to one of the registered overrides: exporting and
        applying the registry's decomposition table puts the override's own
        ``_native::<node_id>`` op in the graph exactly when the call is routed
        to it. ``mean`` reaches the scalar-reduction override, ``none`` the
        no_reduction one. Pinned to ``passthrough``, so this tests the router's
        installation independently of which inputs a kernel variant accepts on
        this GPU.
        """
        expected = {
            f"_native::{node.node_id}"
            for op_symbol in _OP_SYMBOLS
            for node in self._cutedsl_nodes(op_symbol)[1]
        }

        num_batches, in_features, num_classes = 8, 4, 16
        module = torch.nn.LinearCrossEntropyLoss(
            in_features,
            num_classes,
            reduction=reduction,
            device="cuda",
            dtype=torch.float16,
            options=LinearCrossEntropyOptions(
                batch_chunk_size=4, allow_retain_graph=True
            ),
        )
        args = (
            torch.randn(num_batches, in_features, device="cuda", dtype=torch.float16),
            torch.randint(0, num_classes, (num_batches,), device="cuda"),
        )
        with torch.backends.python_native.override_variant(
            f"torch_nn::{_SCALAR_OP}", variants.PASSTHROUGH
        ):
            exported = torch.export.export(module, args)
            decomposed = exported.run_decompositions(
                registry_module.native_decomp_table()
            )
        # Match on namespace and exact node id: node_id embeds the op symbol,
        # and one op symbol is a prefix of the other.
        routed = [
            node.target.name()
            for node in decomposed.graph.nodes
            if node.op == "call_function"
            and getattr(node.target, "namespace", None) == "_native"
        ]
        # One call, and the `_native` node is opaque, so the aten ops inside it
        # are never exposed -- overrides of those (sum, scatter_add, ...) cannot
        # add nodes here.
        self.assertEqual(len(routed), 1, msg=f"expected one _native node, got {routed}")
        self.assertIn(routed[0], expected)

    @parametrize(
        "reduction, requires_grad",
        [("mean", False), ("mean", True), ("none", False), ("none", True)],
    )
    def test_sample_filter_matches_the_gate(self, reduction, requires_grad):
        """The OpInfo variants select samples with ``F.linear_cross_entropy``'s
        own predicate for "a chunked op is reached". This checks, over each
        generator's whole sample space and in both grad modes, that the filter
        passes the right arguments and that the predicate agrees with the call:
        the shared accumulator runs if and only if a chunked op does. The
        cutedsl overrides are disabled, since an eligible kernel variant
        replaces the accumulator and would hide a reached op. The predicate is
        evaluated under the call's grad mode, since one clause reads
        ``torch.is_grad_enabled()``.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module
        from torch.testing._internal import common_methods_invocations as cmi

        # Keyed on the reduction that has its OWN op, so adding a scalar
        # reduction to the parametrization picks the scalar generator rather
        # than silently falling through to the no_reduction one.
        generator = (
            cmi.sample_inputs_linear_cross_entropy_chunked_none
            if reduction == "none"
            else cmi.sample_inputs_linear_cross_entropy_chunked
        )
        with (
            torch.backends.python_native.cutedsl.disabled(),
            unittest.mock.patch.object(
                lce_module,
                "_linear_cross_entropy_batch_chunked_accumulator",
                wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
            ) as accumulator,
        ):
            for i, sample in enumerate(
                generator(None, "cuda", torch.float16, requires_grad), start=1
            ):
                predicted = cmi._cutedsl_linear_cross_entropy_eligible(sample)
                before = accumulator.call_count
                torch.nn.functional.linear_cross_entropy(
                    sample.input, *sample.args, **sample.kwargs
                )
                entered = accumulator.call_count > before
                if entered != predicted:
                    consequence = (
                        "would cover a sample the override never sees"
                        if predicted
                        else "would drop a sample the override handles"
                    )
                    self.fail(
                        f"{generator.__name__} sample {i}: filter predicted "
                        f"chunked={predicted}, dispatch entered={entered}, so the cutedsl "
                        f"OpInfo variants {consequence}. {sample.summary()}"
                    )
            self.assertGreater(
                accumulator.call_count,
                0,
                f"{generator.__name__} admits none of its samples",
            )

    @_needs_kernel
    def test_noncontiguous_target_reads_the_right_classes(self):
        """The kernel is compiled for a stride-1 target, and `_corrected_target`
        returns the caller's tensor when `ignore_index` is a valid class, so a
        strided target must be made contiguous before the launch. Compared
        exactly against the same call with a contiguous target; the accumulator
        witness ensures both legs ran the kernel, since two eager legs would
        also agree exactly.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        torch.manual_seed(0)
        num_batches, in_features, num_classes = 64, 64, 512
        input = torch.randn(
            num_batches, in_features, device="cuda", dtype=torch.bfloat16
        )
        linear_weight = (
            torch.randn(num_classes, in_features, device="cuda", dtype=torch.bfloat16)
            / in_features**0.5
        )
        # One column of a 2-column tensor: 1-D, correct values, stride 2.
        pairs = torch.randint(0, num_classes, (num_batches, 2), device="cuda")
        strided = pairs[:, 0]
        self.assertFalse(strided.is_contiguous())
        options = _compact_options(batch_chunk_size=16)

        def run(target):
            leaves = [
                t.detach().clone().requires_grad_() for t in (input, linear_weight)
            ]
            with unittest.mock.patch.object(
                lce_module,
                "_linear_cross_entropy_batch_chunked_accumulator",
                wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
            ) as accumulator:
                loss = torch.nn.functional.linear_cross_entropy(
                    leaves[0],
                    leaves[1],
                    target,
                    # In range, so `_corrected_target` passes `target` through
                    # with its stride intact.
                    ignore_index=0,
                    options=options,
                )
                self.assertEqual(
                    accumulator.call_count, 0, "the call fell back to the accumulator"
                )
            loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        got = run(strided)
        want = run(strided.contiguous())
        for name, a, b in zip(("loss", "grad_input", "grad_linear_weight"), got, want):
            self.assertEqual(a, b, atol=0, rtol=0, msg=f"{name} depends on the layout")

    @_needs_kernel
    @parametrize("logit", [1.0, 2.0**12, 2.0**24, -(2.0**24)])
    def test_a_large_common_logit_keeps_the_loss(self, logit):
        """Equal logits make the loss log(C) whatever their common value, which
        holds in fp32 only if both statistics are shifted by the row max: at
        2**24, `m + log(l)` rounds back to `m`. The kernel tests cannot see this,
        since their reference is built the same way.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        num_batches, num_classes = 4, 8
        # One feature, so every logit is exactly `root * root`: bf16 holds
        # each root exactly and the product is exact in the fp32 buffer.
        root = math.sqrt(abs(logit))
        input = torch.full((num_batches, 1), root, device="cuda", dtype=torch.bfloat16)
        sign = 1.0 if logit > 0 else -1.0
        linear_weight = torch.full(
            (num_classes, 1), sign * root, device="cuda", dtype=torch.bfloat16
        )
        target = torch.randint(0, num_classes, (num_batches,), device="cuda")
        options = _compact_options(batch_chunk_size=2)

        leaves = [t.detach().clone().requires_grad_() for t in (input, linear_weight)]
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            loss = torch.nn.functional.linear_cross_entropy(
                leaves[0], leaves[1], target, options=options
            )
            self.assertEqual(
                accumulator.call_count, 0, "the call fell back to the accumulator"
            )
        loss.backward()

        # bf16 carries ~3 decimal digits, and the reference is the same number
        # through an independent path.
        want = math.log(num_classes)
        self.assertEqual(loss.item(), want, atol=1e-2, rtol=0)
        eager = torch.nn.functional.linear_cross_entropy(input, linear_weight, target)
        self.assertEqual(loss.item(), eager.item(), atol=1e-2, rtol=0)
        self.assertTrue(torch.isfinite(leaves[0].grad).all())

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the gate declines this device first, so the class count never decides",
    )
    def test_kernel_declines_a_class_count_it_cannot_index(self):
        """Both ends of the class count, which the gate has to decline rather
        than truncate or read past.

        Zero classes is the end that matters: every target is out of range and
        the kernel traps, taking the CUDA context with it -- where eager raises
        on the empty reduction. The int32 end is unreachable, since the weight
        alone would be terabytes, so the predicate is asked directly under
        `FakeTensorMode`, which is also how the router asks it while tracing.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        def eligible(num_classes):
            with FakeTensorMode():
                return cutedsl_impl._kernel_eligible(
                    input=torch.empty(8, 16, device="cuda", dtype=torch.bfloat16),
                    linear_weight=torch.empty(
                        num_classes, 16, device="cuda", dtype=torch.bfloat16
                    ),
                    target=torch.empty(8, device="cuda", dtype=torch.int64),
                    linear_bias=None,
                    weight=None,
                    reduction="mean",
                    ignore_index=-100,
                    label_smoothing=0.0,
                    batch_chunk_size=8,
                    acc_policy="compact",
                    acc_dtype=torch.float32,
                    allow_retain_graph=False,
                    compute_input_grad=True,
                    compute_linear_weight_grad=True,
                    compute_linear_bias_grad=False,
                )

        self.assertFalse(eligible(0), "zero classes: every target is out of range")
        self.assertTrue(eligible(1), "one class is indexable")
        # Where the gate switches; the kernel's exact ceiling is lower (see the gate).
        self.assertTrue(eligible(2**31 - 1), "the largest count int32 can hold")
        self.assertFalse(eligible(2**31), "one class past int32")

    @_needs_kernel
    def test_kernel_declines_a_tensor_on_another_device(self):
        """Every tensor must be on the input's device; the gate declines
        otherwise, so eager raises its own device-mismatch error. Asked of the
        predicate directly, with small real tensors.
        """

        def eligible(**on_cpu):
            def make(name, *shape, dtype=torch.bfloat16):
                dev = "cpu" if on_cpu.get(name) else "cuda"
                if dtype is torch.int64:
                    return torch.zeros(*shape, device=dev, dtype=dtype)
                return torch.empty(*shape, device=dev, dtype=dtype)

            return cutedsl_impl._kernel_eligible(
                input=make("input", 8, 16),
                linear_weight=make("linear_weight", 32, 16),
                target=make("target", 8, dtype=torch.int64),
                linear_bias=make("linear_bias", 32),
                weight=make("weight", 32, dtype=torch.float32),
                reduction="mean",
                ignore_index=-100,
                label_smoothing=0.0,
                batch_chunk_size=8,
                acc_policy="compact",
                acc_dtype=torch.float32,
                allow_retain_graph=False,
                compute_input_grad=True,
                compute_linear_weight_grad=True,
                compute_linear_bias_grad=True,
            )

        self.assertTrue(eligible(), "all on one device")
        for name in ("linear_weight", "target", "linear_bias", "weight"):
            self.assertFalse(eligible(**{name: True}), f"{name} on another device")

    @_needs_kernel
    @unittest.skipIf(not TEST_MULTIGPU, "requires at least 2 visible CUDA devices")
    def test_tensors_on_a_non_current_device(self):
        def grads():
            loss = torch.nn.functional.linear_cross_entropy(
                input, weight, target, options=_compact_options(batch_chunk_size=1)
            )
            return (loss, *torch.autograd.grad(loss, (input, weight)))

        with torch.cuda.device(0):
            input = torch.randn(
                2, 8, device="cuda:1", dtype=torch.bfloat16, requires_grad=True
            )
            weight = torch.randn(
                4, 8, device="cuda:1", dtype=torch.bfloat16, requires_grad=True
            )
            target = torch.tensor([1, 3], device="cuda:1")
            fused = grads()
            with torch.backends.python_native.cutedsl.disabled():
                plain = grads()
        self.assertEqual(fused, plain)

    @_needs_kernel
    def test_an_out_of_range_target_traps(self):
        """Through the public op, in a child process since the trap takes the
        CUDA context with it. The fallback is disabled there, so eager's own
        index assert cannot pass for the kernel's."""
        stdout, stderr = self.run_process_no_exception("""
import torch
import torch.nn.modules.linear_cross_entropy as lce_module
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions
lce_module._linear_cross_entropy_batch_chunked_accumulator = None
input = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(512, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
target = torch.randint(0, 512, (64,), device="cuda")
target[7] = 517
options = LinearCrossEntropyOptions(
    acc_policy="compact", acc_dtype=torch.float32, chunking_method=None,
    batch_chunk_size=16,
)
try:
    torch.nn.functional.linear_cross_entropy(input, weight, target, options=options)
except RuntimeError:
    pass  # raised by whichever later op notices first, e.g. a cuBLAS call
torch.cuda.synchronize()
""")
        self.assertTrue(has_device_side_assert(stderr.decode()))
        self.assertIn("linear_cross_entropy: target >= num_classes", stdout.decode())

    @_needs_kernel
    @parametrize("reduction", ["mean", "sum"])
    @parametrize("weighted", [False, True])
    def test_the_parameter_axes_the_kernel_accepts_match_eager(
        self, reduction, weighted
    ):
        """`_neg_weight_target` branches on `reduction` (`sum` keeps the class
        weight, `mean` divides by its total) and on whether a class weight is
        given, and the kernel multiplies its result into every row, so all four
        branches are compared against eager. `ignore_index` is set on a quarter
        of the rows.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        num_batches, in_features, num_classes = 64, 32, 256
        ignore_index = 3
        gen = torch.Generator(device="cuda").manual_seed(0)
        input = torch.randn(
            num_batches, in_features, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        linear_weight = (
            torch.randn(
                num_classes,
                in_features,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            / in_features**0.5
        )
        target = torch.randint(
            0, num_classes, (num_batches,), device="cuda", generator=gen
        )
        target[::4] = ignore_index
        weight = None
        if weighted:
            weight = (
                torch.rand(num_classes, device="cuda", generator=gen) + 0.5
            ).float()
        options = _compact_options(batch_chunk_size=16)

        def once():
            leaves = [
                t.detach().clone().requires_grad_() for t in (input, linear_weight)
            ]
            loss = torch.nn.functional.linear_cross_entropy(
                leaves[0],
                leaves[1],
                target,
                weight=weight,
                reduction=reduction,
                ignore_index=ignore_index,
                options=options,
            )
            loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            fused = once()
            self.assertEqual(
                accumulator.call_count, 0, "the call fell back to the accumulator"
            )
        with torch.backends.python_native.cutedsl.disabled():
            plain = once()

        # `sum` leaves out the division by the row count that `mean` applies,
        # so every gradient is larger by roughly that factor and an absolute
        # bound has to scale with it. The relative disagreement is one bf16 ULP
        # either way.
        atol = 1e-3 if reduction == "mean" else 1e-3 * num_batches
        for name, a, b in zip(
            ("grad_input", "grad_linear_weight"), fused[1:], plain[1:]
        ):
            self.assertEqual(a, b, atol=atol, rtol=0, msg=f"{name} disagrees")
        self.assertEqual(
            fused[0], plain[0], rtol=4 * torch.finfo(torch.bfloat16).eps, atol=0
        )

    @_needs_kernel
    def test_label_smoothing_never_reaches_the_chunked_path(self):
        """Neither the kernel nor the chunked path implements label smoothing,
        so `options` are refused one level above the override, with a warning,
        and the call takes the reference path. The gate's own
        `label_smoothing == 0.0` clause is unreachable through the public API.
        """
        num_batches, in_features, num_classes = 64, 32, 256
        gen = torch.Generator(device="cuda").manual_seed(0)
        input = torch.randn(
            num_batches, in_features, device="cuda", dtype=torch.bfloat16, generator=gen
        )
        linear_weight = (
            torch.randn(
                num_classes,
                in_features,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            / in_features**0.5
        )
        target = torch.randint(
            0, num_classes, (num_batches,), device="cuda", generator=gen
        )
        options = _compact_options(batch_chunk_size=16)
        with self.assertWarnsRegex(UserWarning, r"label_smoothing == 0"):
            asked = torch.nn.functional.linear_cross_entropy(
                input, linear_weight, target, label_smoothing=0.1, options=options
            )
        reference = torch.nn.functional.linear_cross_entropy(
            input, linear_weight, target, label_smoothing=0.1
        )
        # Not merely close: `options` were ignored, so this IS the reference
        # path and the two are the same computation.
        self.assertEqual(asked, reference, atol=0, rtol=0)

    @_needs_kernel
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("requires_grad", [True, False])
    def test_forward_only_loss_matches_the_reference(self, dtype, requires_grad):
        """With no gradient requested the override does not call the kernel: it
        forms the loss with eager ops, a second loss implementation reached by
        any `no_grad` call the gate accepts. Both `requires_grad` settings run,
        since the wrapper decides through `torch.is_grad_enabled()` as well as
        the leaves.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        num_batches, in_features, num_classes = 64, 32, 512
        gen = torch.Generator(device="cuda").manual_seed(0)
        input = torch.randn(
            num_batches, in_features, device="cuda", dtype=dtype, generator=gen
        )
        linear_weight = (
            torch.randn(
                num_classes, in_features, device="cuda", dtype=dtype, generator=gen
            )
            / in_features**0.5
        )
        linear_bias = torch.randn(
            num_classes, device="cuda", dtype=dtype, generator=gen
        )
        target = torch.randint(
            0, num_classes, (num_batches,), device="cuda", generator=gen
        )
        if requires_grad:
            for leaf in (input, linear_weight, linear_bias):
                leaf.requires_grad_()
        options = _compact_options(batch_chunk_size=16)
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            with torch.no_grad():
                fused = torch.nn.functional.linear_cross_entropy(
                    input,
                    linear_weight,
                    target,
                    linear_bias=linear_bias,
                    options=options,
                )
            self.assertEqual(
                accumulator.call_count, 0, "the call fell back to the accumulator"
            )
        with torch.no_grad():
            reference = torch.nn.functional.linear_cross_entropy(
                input, linear_weight, target, linear_bias=linear_bias
            )
        # A scalar of order log(C), so a relative bound in ULP of its dtype --
        # the same reasoning as the bias-dtype test below.
        self.assertEqual(fused, reference, rtol=4 * torch.finfo(dtype).eps, atol=0)

    @_needs_kernel
    @parametrize(
        "dtype, bias_dtype",
        [
            (torch.float16, torch.float16),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ],
    )
    def test_every_bias_dtype_the_op_takes_reaches_the_kernel(self, dtype, bias_dtype):
        """Every bias dtype the op takes must reach the kernel. `addmm` takes
        `self` only in `out_dtype` or `mat1`'s dtype, so with fp16 inputs an
        fp32 bias needs a cast; each pair is compared against the same call with
        the kernel disabled.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        torch.manual_seed(0)
        num_batches, in_features, num_classes = 64, 64, 512
        input = torch.randn(num_batches, in_features, device="cuda", dtype=dtype)
        linear_weight = (
            torch.randn(num_classes, in_features, device="cuda", dtype=dtype)
            / in_features**0.5
        )
        linear_bias = torch.randn(num_classes, device="cuda", dtype=bias_dtype)
        target = torch.randint(0, num_classes, (num_batches,), device="cuda")
        options = _compact_options(batch_chunk_size=16)

        def once():
            leaves = [
                t.detach().clone().requires_grad_()
                for t in (input, linear_weight, linear_bias)
            ]
            loss = torch.nn.functional.linear_cross_entropy(
                leaves[0], leaves[1], target, linear_bias=leaves[2], options=options
            )
            loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        # Any entry into the accumulator means the call fell back, and a
        # fallback would hide exactly the failure this test is about.
        # Under `fill_uninitialized_memory`, any gradient element the first chunk
        # fails to write comes back NaN, which pins write-not-accumulate.
        with (
            unittest.mock.patch.object(
                lce_module,
                "_linear_cross_entropy_batch_chunked_accumulator",
                wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
            ) as accumulator,
            DeterministicGuard(True, fill_uninitialized_memory=True),
        ):
            fused = once()
            self.assertEqual(
                accumulator.call_count,
                0,
                f"a {bias_dtype} bias on {dtype} inputs fell back to the "
                "accumulator instead of reaching the kernel",
            )
        with torch.backends.python_native.cutedsl.disabled():
            plain = once()

        names = ("grad_input", "grad_linear_weight", "grad_linear_bias")
        for name, a, b in zip(names, fused[1:], plain[1:]):
            # Absolute only: these gradients hold near-zero elements, where a
            # relative bound is dominated by division rather than by error.
            self.assertEqual(a, b, atol=1e-3, rtol=0, msg=f"{name} disagrees")
        # The loss is a scalar of order log(C) returned in `dtype`, where one ULP
        # exceeds the gradients' 1e-3, so it gets a few ULP, relative.
        self.assertEqual(
            fused[0],
            plain[0],
            rtol=4 * torch.finfo(dtype).eps,
            atol=0,
            msg="loss disagrees",
        )

    @_needs_kernel
    def test_empty_batch_returns_a_zeroed_weight_gradient(self):
        """An empty batch has no first chunk to write `grad_linear_weight`, so
        the early return must zero it. `fill_uninitialized_memory` makes an
        unzeroed allocation visible, since `torch.empty` often returns zeroed
        pages."""
        import torch.nn.modules.linear_cross_entropy as lce_module

        in_features, num_classes = 64, 512
        input = torch.zeros(
            0, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        linear_weight = torch.randn(
            num_classes, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        target = torch.zeros(0, device="cuda", dtype=torch.int64)
        options = _compact_options(batch_chunk_size=8)

        with (
            unittest.mock.patch.object(
                lce_module,
                "_linear_cross_entropy_batch_chunked_accumulator",
                wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
            ) as accumulator,
            DeterministicGuard(True, fill_uninitialized_memory=True),
        ):
            loss = torch.nn.functional.linear_cross_entropy(
                input, linear_weight, target, options=options
            )
            self.assertEqual(
                accumulator.call_count,
                0,
                "the call fell back to the accumulator, so this asserted the "
                "eager path's allocation rather than the override's",
            )
            loss.backward()

        self.assertTrue(torch.isnan(loss), "mean over an empty batch is nan")
        self.assertEqual(
            linear_weight.grad,
            torch.zeros_like(linear_weight.grad),
            atol=0,
            rtol=0,
            msg="the weight gradient is not zeroed on the path that has no "
            "chunk to write it",
        )

    @_needs_kernel
    @parametrize("variant", _KERNEL_VARIANTS)
    def test_kernel_path_is_deterministic(self, variant):
        """Two identical calls give bit-identical gradients: a GEMM and a
        fixed-order column sum replace eager's atomic `index_add_`. The absence
        of the scatter is asserted separately, by spying on `index_add_`,
        because `torch.use_deterministic_algorithms(True)` lets `index_add` run
        deterministically instead of raising, so it would not catch one.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        torch.manual_seed(0)
        num_batches, in_features, num_classes = 96, 64, 512
        input = torch.randn(
            num_batches, in_features, device="cuda", dtype=torch.bfloat16
        )
        linear_weight = (
            torch.randn(num_classes, in_features, device="cuda", dtype=torch.bfloat16)
            / in_features**0.5
        )
        linear_bias = torch.randn(num_classes, device="cuda", dtype=torch.bfloat16)
        target = torch.randint(0, num_classes, (num_batches,), device="cuda")
        options = _compact_options(batch_chunk_size=32)

        def once():
            leaves = [
                t.detach().clone().requires_grad_()
                for t in (input, linear_weight, linear_bias)
            ]
            with torch.backends.python_native.override_variant(
                f"torch_nn::{_SCALAR_OP}", variant
            ):
                loss = torch.nn.functional.linear_cross_entropy(
                    leaves[0],
                    leaves[1],
                    target,
                    linear_bias=leaves[2],
                    options=options,
                )
                loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        scatters = []
        unpatched_index_add_ = torch.Tensor.index_add_

        def counting_index_add_(self, *args, **kwargs):
            scatters.append(1)
            return unpatched_index_add_(self, *args, **kwargs)

        # The kernel variants replace the accumulator, so any entry into it
        # means the call fell back and this test never saw the kernel path.
        with (
            unittest.mock.patch.object(
                lce_module,
                "_linear_cross_entropy_batch_chunked_accumulator",
                wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
            ) as accumulator,
            unittest.mock.patch.object(torch.Tensor, "index_add_", counting_index_add_),
        ):
            first, second = once(), once()
            self.assertEqual(scatters, [], "the backward scattered")
            self.assertEqual(
                accumulator.call_count,
                0,
                f"variant {variant!r} fell back to the accumulator, so this "
                "asserted determinism of the eager path, not the kernel's",
            )
        names = ("loss", "grad_input", "grad_linear_weight", "grad_linear_bias")
        for name, a, b in zip(names, first, second):
            if not torch.equal(a, b):
                self.fail(
                    f"variant {variant!r}: {name} differs between two identical "
                    f"runs (max abs diff {(a - b).abs().max().item():.3e}), so "
                    "the kernel path is not deterministic"
                )


instantiate_parametrized_tests(TestLinearCrossEntropyOverride)

if __name__ == "__main__":
    run_tests()
