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
from torch._native import cutedsl_utils as cu
from torch._native.ops.linear_cross_entropy import cutedsl_impl
from torch.nn.modules.linear_cross_entropy_options import LinearCrossEntropyOptions
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


_OP_SYMBOLS = [op_symbol for op_symbol, _, _ in cutedsl_impl._OVERRIDES]


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
        "the scalar op's kernel declines this device, so `mean` would not route",
    )
    @parametrize("reduction", ["mean", "none"])
    def test_cutedsl_path_is_used(self, reduction):
        """A call must route to one of the registered overrides.

        Checked by exporting and applying the registry's decomposition table:
        the override's own ``_native::<node_id>`` op appears in the graph
        exactly when the call is routed to it. The router emits nothing per
        call, so the graph is the only place the choice is observable.

        ``mean`` reaches the scalar-reduction override, ``none`` the
        no_reduction one.
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
        exported = torch.export.export(module, args)
        decomposed = exported.run_decompositions(registry_module.native_decomp_table())
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
        override REPLACES that call. The kernel, where it is eligible for the
        sample, runs instead of the accumulator and makes the witness read
        False for a sample that did reach the op, so the whole loop runs with
        the cutedsl overrides disabled. That is also the honest scope: the predicate
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
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
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
        """
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
            loss = torch.nn.functional.linear_cross_entropy(
                leaves[0],
                leaves[1],
                target,
                # In range, so `_corrected_target` passes `target` through with
                # its stride intact.
                ignore_index=0,
                options=options,
            )
            loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        got = run(strided)
        want = run(strided.contiguous())
        for name, a, b in zip(("loss", "grad_input", "grad_linear_weight"), got, want):
            self.assertEqual(a, b, atol=0, rtol=0, msg=f"{name} depends on the layout")

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
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
    def test_kernel_declines_a_class_count_beyond_int32(self):
        """The class count crosses the FFI as int32 and the kernel indexes
        columns in int32, so the gate has to decline rather than truncate.

        No allocation reaches that size -- the weight alone would be terabytes
        -- so the predicate is asked directly, under `FakeTensorMode`, which is
        also how the router asks it while tracing.
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

        self.assertTrue(eligible(2**31 - 1), "the largest count int32 holds")
        self.assertFalse(eligible(2**31), "one class past int32")

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

        names = ("loss", "grad_input", "grad_linear_weight", "grad_linear_bias")
        for name, a, b in zip(names, fused, plain):
            # Absolute only: these gradients hold near-zero elements, where a
            # relative bound is dominated by division rather than by error.
            self.assertEqual(a, b, atol=1e-3, rtol=0, msg=f"{name} disagrees")

    @unittest.skipIf(
        not TEST_CUDA or not cutedsl_impl._arch_supported(),
        "the kernel declines this device, so there is no kernel path to test "
        "-- the call would fall back to eager",
    )
    def test_kernel_path_is_deterministic(self):
        """Two identical calls must give bit-identical gradients.

        The dense grad-logits formulation replaced eager's `index_add_` --
        atomic on CUDA, and used there for both `grad_linear_weight` and
        `grad_linear_bias` -- with a GEMM and a fixed-order column sum, so the
        whole backward is deterministic. That is a user-visible property, and
        the reason a bias-grad epilogue built on atomics was rejected, so it is
        pinned here rather than left aspirational.
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
            loss = torch.nn.functional.linear_cross_entropy(
                leaves[0], leaves[1], target, linear_bias=leaves[2], options=options
            )
            loss.backward()
            return (loss.detach(), *(t.grad for t in leaves))

        # The kernel replaces the accumulator, so any entry into it means the
        # call fell back and this test never saw the kernel path.
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            first, second = once(), once()
            self.assertEqual(
                accumulator.call_count,
                0,
                "the call fell back to the accumulator, so this asserted "
                "determinism of the eager path, not the kernel's",
            )
        names = ("loss", "grad_input", "grad_linear_weight", "grad_linear_bias")
        for name, a, b in zip(names, first, second):
            if not torch.equal(a, b):
                self.fail(
                    f"{name} differs between two identical runs (max abs diff "
                    f"{(a - b).abs().max().item():.3e}), so the kernel path is "
                    "not deterministic"
                )


instantiate_parametrized_tests(TestLinearCrossEntropyOverride)

if __name__ == "__main__":
    run_tests()
