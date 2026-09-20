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
from torch.testing._internal.common_cuda import TEST_CUDA
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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    def test_empty_batch_returns_a_zeroed_bias_gradient(self):
        """`grad_linear_bias` is left uninitialized because the `copy_` after
        the loop writes all of it. An empty batch has no loop, so the early
        return is the one path where that allocation has to be zeroed, and the
        only place where skipping the fill would be visible as a wrong result.

        `fill_uninitialized_memory` is what gives this teeth: `torch.empty`
        otherwise tends to hand back zeroed pages, and the assertion would pass
        whether or not the allocation is guarded.
        """
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=8,
        )
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
        """A call must route to one of the registered overrides.

        Checked by exporting and applying the registry's decomposition table:
        the override's own ``_native::<node_id>`` op appears in the graph
        exactly when the call is routed to it. The router emits nothing per
        call, so the graph is the only place the choice is observable.

        ``mean`` reaches the scalar-reduction override, ``none`` the
        no_reduction one.

        Pinned to ``passthrough`` because the question is whether the ROUTER
        installed the override, not whether a kernel accepts these inputs: a
        kernel variant's eligibility is dtype- and architecture-specific, so
        letting the default variant answer would make this test pass or fail on
        the GPU it runs on.
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
        """The OpInfo variants select their samples by asking
        ``F.linear_cross_entropy``'s own predicate whether a chunked op -- and
        so an override -- is reached. Sharing the predicate means the answer
        cannot drift from the dispatch, but the filter still has to hand it the
        right arguments out of a ``SampleInput``, and the predicate still has
        to agree with what the call actually does. Both are checked here across
        each generator's whole sample space, in both grad modes the variants run
        in.

        The shared accumulator is called if and only if a chunked op runs (the
        reference path uses linear + cross_entropy instead), so counting entries
        into it is the same question the predicate answers -- but only while no
        override REPLACES that call. A kernel variant eligible for the sample
        would run instead of the accumulator and make the witness read False for
        a sample that did reach the op, so the whole loop runs with the cutedsl
        overrides disabled. That is also the honest scope: the predicate
        describes the functional's gate, which is about whether a chunked op is
        reached, not about which implementation answers.

        The predicate is evaluated in the same grad mode as the call it
        describes: one of its clauses reads ``torch.is_grad_enabled()``, so
        evaluating it under a different mode would make the two disagree by
        construction.
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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "no kernel variant is eligible on this device, so there is no kernel "
        "path to test -- the call would fall back to eager",
    )
    def test_noncontiguous_target_reads_the_right_classes(self):
        """The kernel is compiled for a stride-1 target, and `_corrected_target`
        returns the caller's tensor untouched when `ignore_index` is itself a
        valid class -- so without a contiguity fix a strided target reaches the
        kernel and is read at the wrong offsets.

        Checked against the SAME call with a contiguous target rather than
        against eager: identical values through an identical path, so the
        comparison is exact and the only difference under test is the layout.
        Eager would need tolerances for the formulation difference in
        `grad_linear_weight`, which is the noise this test has to see through.

        Comparing the override against itself is what makes the accumulator
        witness necessary rather than decorative: on a fallback both legs would
        be eager, agree exactly, and pass while testing nothing.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )

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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "no kernel variant is eligible on this device, so there is no kernel "
        "path to test -- the call would fall back to eager",
    )
    @parametrize("logit", [1.0, 2.0**12, 2.0**24, -(2.0**24)])
    def test_a_large_common_logit_keeps_the_loss(self, logit):
        """Equal logits make the loss log(C) whatever their common value is:
        the row max cancels out of it exactly.

        It cancels numerically only if the two statistics the loss is built
        from are themselves shifted by that max. From the unshifted pair,
        `m + log(l)` rounds back to `m` in fp32 once `m` is large -- at 2**24
        the next float is two away and log(2) does not reach it -- so the
        difference comes out zero and the loss with it. A kernel-level test
        cannot see this: it compares against a reference built the same way.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=2,
        )

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

        Zero classes is the end that matters: every target is out of range, the
        kernel clamps the read to column 0, and on a `(Bc, 0)` buffer that is an
        illegal access that poisons the CUDA context -- where eager raises on
        the empty reduction. The int32 end is unreachable, since the weight
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

        self.assertFalse(eligible(0), "no column for the clamped read to land in")
        self.assertTrue(eligible(1), "one class is indexable")
        # Not "the largest safe count": the kernel strides past V, so the real
        # ceiling is lower by a knob-dependent margin. This pins where the gate
        # switches, which is what the gate promises.
        self.assertTrue(eligible(2**31 - 1), "the largest count int32 can hold")
        self.assertFalse(eligible(2**31), "one class past int32")

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "no kernel variant is eligible on this device, so there is no kernel "
        "path to test -- the call would fall back to eager",
    )
    def test_kernel_declines_a_tensor_on_another_device(self):
        """The gate is the only place that can require one device. Past it the
        launch takes each tensor through the FFI, where a stray CPU one comes
        back as an argument-type error naming an FFI parameter -- where eager
        said "expected all tensors to be on the same device". Declining sends
        the call back through the router to eager, which says that again.

        Asked directly, with real tensors: they are small, and the point is the
        predicate rather than the launch.
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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    def test_an_out_of_range_target_poisons_both_parameter_gradients(self):
        """What the caller sees when a target is out of range, which is not
        what the kernel does.

        `test_out_of_range_target_poisons_its_row` pins the kernel's behaviour
        on `g`: the offending row is nan and the others are finite. That
        row-locality does not survive the loop. Both parameter gradients are
        reductions OVER rows -- `g.t() @ input_chunk` and `g.sum(dim=0)` -- so
        one bad target in one chunk makes the whole `(C, F)` weight gradient,
        the whole `(C,)` bias gradient and the scalar loss nan. `grad_input` is
        the only output that stays row-local.

        The public note on `linear_cross_entropy` states exactly this, so it is
        pinned here rather than inferred from the kernel test, whose assertion
        is true and yet does not imply it.

        The class-`weight` path raises instead of poisoning, which the note
        also states. That is not asserted here: the raise is a CUDA device-side
        assert, and it would poison the context for every test after it.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        num_tokens, in_features, num_classes = 64, 32, 512
        bad_row = 7
        gen = torch.Generator(device="cuda").manual_seed(0)
        input = torch.randn(
            num_tokens, in_features, device="cuda", dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        linear_weight = torch.randn(
            num_classes, in_features, device="cuda", dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        linear_bias = torch.randn(
            num_classes, device="cuda", dtype=torch.bfloat16, generator=gen
        ).requires_grad_()
        target = torch.randint(
            0, num_classes, (num_tokens,), device="cuda", generator=gen
        )
        target[bad_row] = num_classes + 5
        # Several chunks, with the bad row in the first: the accumulators live
        # across the whole loop, so which chunk it lands in must not matter.
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            loss = torch.nn.functional.linear_cross_entropy(
                input, linear_weight, target, linear_bias=linear_bias, options=options
            )
            self.assertEqual(
                accumulator.call_count, 0, "the call fell back to the accumulator"
            )
        loss.backward()

        self.assertTrue(torch.isnan(loss), "the loss carries the poison")
        self.assertTrue(
            torch.isnan(linear_weight.grad).all(),
            "grad_linear_weight is a reduction over rows, so all of it is nan",
        )
        self.assertTrue(
            torch.isnan(linear_bias.grad).all(), "grad_linear_bias likewise"
        )
        nan_rows = torch.isnan(input.grad).any(dim=1).nonzero().flatten().tolist()
        self.assertEqual(
            nan_rows, [bad_row], "grad_input is the one output that stays row-local"
        )

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    @parametrize("reduction", ["mean", "sum"])
    @parametrize("weighted", [False, True])
    def test_the_parameter_axes_the_kernel_accepts_match_eager(
        self, reduction, weighted
    ):
        """The row scale is where the op's parameters meet the kernel.

        `_neg_weight_target` branches on both of these -- `sum` keeps the class
        weight where `mean` divides by its total, and a `None` weight takes a
        different branch from a real one -- and the kernel multiplies whatever
        comes out into every row. The tests around this one all use
        `reduction='mean'` with no class weight, so three of those four
        branches reach the kernel nowhere.

        `ignore_index` is set on a quarter of the rows by construction. Picking
        a valid class and hoping, as a strided-target test elsewhere does, puts
        the expected number of ignored rows below one.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )

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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    def test_label_smoothing_never_reaches_the_chunked_path(self):
        """The kernel implements no label smoothing, and neither does the
        chunked path it overrides -- so `options` are refused one level above
        the override, with a warning, and the call goes to the reference path.

        Worth pinning because it is the layer the refusal happens at that keeps
        the kernel safe here. The gate's own `label_smoothing == 0.0` clause is
        belt-and-braces: nothing reaching it through the public API can carry a
        non-zero one, so that clause cannot be exercised end to end and a test
        that tried would be asserting against the wrong layer.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )
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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("requires_grad", [True, False])
    def test_forward_only_loss_matches_the_reference(self, dtype, requires_grad):
        """The override has a second, separate loss path.

        With no gradient requested there is no `g` to write, so the override
        does not call the kernel at all: it shifts, exponentiates and reduces
        with eager ops, reimplementing the loss a second time. That code is
        reached by any `no_grad` call the gate accepts -- ordinary inference --
        and a wrong loss there is silent, since nothing downstream of it is
        compared against anything.

        `test_nn`'s `test_linear_cross_entropy_loss_no_grad` does not cover it:
        it builds fp32 tensors, which the gate declines, so it measures eager
        against eager. Both `requires_grad` settings are run because the
        wrapper decides through `torch.is_grad_enabled()` as well as the leaves.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )
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

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
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
        """The override has to accept every bias dtype the op accepts.

        It forms the logits with `addmm`, which takes `self` only in `out_dtype`
        or in `mat1`'s dtype. With fp16 inputs the logits buffer is fp16, so an
        fp32 bias -- which the chunked path takes -- matches neither and the
        matmul rejects it unless the override casts first. The other three pairs
        need no cast and are here to pin that they do not get one: each is
        compared against the same call with the kernel disabled, which is the
        behaviour the override has to reproduce.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=16,
        )

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
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
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
        # The loss is a single scalar of order log(C), nowhere near zero, and
        # it comes back in `dtype` -- where one ULP at that magnitude is
        # 3.1e-2 in bf16 and 3.9e-3 in fp16. The gradients' absolute 1e-3 is
        # below both, so applying it here would be a bitwise assertion dressed
        # as a tolerance, holding only while the two paths agree exactly. A
        # few ULP, relative, is what this comparison can actually promise.
        self.assertEqual(
            fused[0],
            plain[0],
            rtol=4 * torch.finfo(dtype).eps,
            atol=0,
            msg="loss disagrees",
        )

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "no kernel variant is eligible on this device, so there is no kernel "
        "path to test -- the call would fall back to eager",
    )
    def test_empty_batch_returns_a_zeroed_weight_gradient(self):
        """`grad_linear_weight` is left uninitialized because the first chunk
        writes it outright. An empty batch has no first chunk, so the early
        return is the one path where that allocation has to be zeroed, and the
        only place where skipping the fill would be visible as a wrong result.

        `fill_uninitialized_memory` is what gives this teeth: `torch.empty`
        otherwise tends to hand back zeroed pages, and the assertion would pass
        whether or not the allocation is guarded. Nothing else in the mode
        bears on this path: an empty batch returns before the chunk loop, so
        the allocation under test is the only one the fill can reach.
        """
        import torch.nn.modules.linear_cross_entropy as lce_module

        in_features, num_classes = 64, 512
        input = torch.zeros(
            0, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        linear_weight = torch.randn(
            num_classes, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_()
        target = torch.zeros(0, device="cuda", dtype=torch.int64)
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=8,
        )

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
        self.assertTrue(
            torch.equal(linear_weight.grad, torch.zeros_like(linear_weight.grad)),
            "the weight gradient is not zeroed on the path that has no chunk "
            "to write it",
        )

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "no kernel variant is eligible on this device, so there is no kernel "
        "path to test -- the call would fall back to eager",
    )
    @parametrize("variant", _KERNEL_VARIANTS)
    def test_kernel_path_is_deterministic(self, variant):
        """Two identical calls must give bit-identical gradients.

        The dense grad-logits formulation replaced eager's `index_add_` --
        atomic on CUDA, and used there for both `grad_linear_weight` and
        `grad_linear_bias` -- with a GEMM and a fixed-order column sum, so the
        whole backward is deterministic. That is a user-visible property, and
        the reason a bias-grad epilogue built on atomics was rejected, so it is
        pinned here rather than left aspirational.

        Two assertions, because the claim has two halves. Bit-identical repeats
        are the property a caller sees, and atomics would break them. The
        absence of the scatter is asserted separately, by spying on
        `index_add_`: `torch.use_deterministic_algorithms(True)` would NOT pin
        it, since `index_add` on CUDA is listed as acting deterministically
        under that flag rather than raising, so a reintroduced scatter would
        pass a deterministic-mode check and merely run slower.
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
        options = LinearCrossEntropyOptions(
            acc_policy="compact",
            acc_dtype=torch.float32,
            chunking_method=None,
            batch_chunk_size=32,
        )

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
