# Owner(s): ["module: dsl-native-ops"]
#
# Routing tests for the CuTeDSL overrides of the chunked ``linear_cross_entropy``
# ops. Both tests read ``cutedsl_impl._OVERRIDES``, so adding an override needs
# no change here.

import unittest
import unittest.mock

import torch
import torch._native.registry as registry_module
from torch._native import cutedsl_utils as cu, variants
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
_SCALAR_OP = "_linear_cross_entropy_batch_chunked"
# Self-discovered, so promoting or adding a variant needs no change here.
_KERNEL_VARIANTS = sorted(
    name for name in cutedsl_impl._VARIANTS[_SCALAR_OP] if name != variants.PASSTHROUGH
)


@unittest.skipIf(not TEST_CUDA, "the overrides are registered on CUDA")
class TestLinearCrossEntropyOverride(TestCase):
    def setUp(self):
        super().setUp()
        if not cu.runtime_available() or cu.check_native_jit_disabled():
            self.skipTest("CuTeDSL runtime unavailable or native DSL disabled")

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
    @parametrize("variant", _KERNEL_VARIANTS)
    def test_kernel_path_is_deterministic(self, variant):
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

        # The kernel variants replace the accumulator, so any entry into it
        # means the call fell back and this test never saw the kernel path.
        with unittest.mock.patch.object(
            lce_module,
            "_linear_cross_entropy_batch_chunked_accumulator",
            wraps=lce_module._linear_cross_entropy_batch_chunked_accumulator,
        ) as accumulator:
            first, second = once(), once()
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
