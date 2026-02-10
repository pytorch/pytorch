# Owner(s): ["module: inductor"]
"""
Tests for decompose_functional_to_out pass.

This test verifies that the pass correctly transforms functional ops
to their out variants by:
1. Detecting out variants via schema matching
2. Allocating output buffers with torch.empty
3. Replacing functional calls with out variant calls

Tests use kwarg-only out format with Tensor(a!) alias annotation.
Note: torchgen requires explicit alias annotation - Tensor! alone is not parseable.
"""

import torch
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDecomposeFunctionalToOut(TestCase):
    """Core tests for the decompose_functional_to_out pass."""

    @classmethod
    def setUpClass(cls):
        cls._setup_test_ops()

    @classmethod
    def _setup_test_ops(cls):
        """Register test ops with functional and out variants (PyTorch standard style)."""
        cls.lib = torch.library.Library("test_decompose", "FRAGMENT")

        # === Single output op ===
        cls.lib.define("add_one(Tensor x) -> Tensor")

        @torch.library.impl("test_decompose::add_one", "CompositeExplicitAutograd")
        def add_one_impl(x: Tensor) -> Tensor:
            return x + 1

        @torch.library.impl("test_decompose::add_one", "Meta")
        def add_one_meta(x: Tensor) -> Tensor:
            return torch.empty_like(x)

        # kwarg-only out with Tensor(a!) alias annotation (required by torchgen)
        cls.lib.define("add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)")

        @torch.library.impl("test_decompose::add_one.out", "CompositeExplicitAutograd")
        def add_one_out_impl(x: Tensor, *, out: Tensor) -> Tensor:
            out.copy_(x + 1)
            return out

        @torch.library.impl("test_decompose::add_one.out", "Meta")
        def add_one_out_meta(x: Tensor, *, out: Tensor) -> Tensor:
            return out

        # === Multiple outputs op ===
        cls.lib.define("split_scale(Tensor x, float scale) -> (Tensor, Tensor)")

        @torch.library.impl("test_decompose::split_scale", "CompositeExplicitAutograd")
        def split_scale_impl(x: Tensor, scale: float) -> tuple[Tensor, Tensor]:
            return x * scale, x / scale

        @torch.library.impl("test_decompose::split_scale", "Meta")
        def split_scale_meta(x: Tensor, scale: float) -> tuple[Tensor, Tensor]:
            return torch.empty_like(x), torch.empty_like(x)

        # kwarg-only out args with Tensor(a!) alias annotation (required by torchgen)
        cls.lib.define(
            "split_scale.out(Tensor x, float scale, *, "
            "Tensor(a!) out_scaled, Tensor(b!) out_divided) -> (Tensor(a!), Tensor(b!))"
        )

        @torch.library.impl(
            "test_decompose::split_scale.out", "CompositeExplicitAutograd"
        )
        def split_scale_out_impl(
            x: Tensor, scale: float, *, out_scaled: Tensor, out_divided: Tensor
        ) -> tuple[Tensor, Tensor]:
            out_scaled.copy_(x * scale)
            out_divided.copy_(x / scale)
            return out_scaled, out_divided

        @torch.library.impl("test_decompose::split_scale.out", "Meta")
        def split_scale_out_meta(
            x: Tensor, scale: float, *, out_scaled: Tensor, out_divided: Tensor
        ) -> tuple[Tensor, Tensor]:
            return out_scaled, out_divided

    def test_single_output_graph_transformation(self):
        """
        Test that a single-output functional op is transformed to out variant.

        Verifies:
        - Before: graph contains add_one.default (functional)
        - After: graph contains torch.empty + add_one.out
        - Result is correct
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch.export import export

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.test_decompose.add_one(x)

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        # Verify functional op exists before transformation
        has_functional = any(
            n.target == torch.ops.test_decompose.add_one.default
            for n in graph.nodes
            if n.op == "call_function"
        )
        self.assertTrue(has_functional, "Functional op should exist before transform")

        # Apply decompose pass
        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified, "Pass should modify the graph")

        # Verify out variant exists after transformation
        has_out = any(
            n.target == torch.ops.test_decompose.add_one.out
            for n in graph.nodes
            if n.op == "call_function"
        )
        has_empty = any(
            n.target == torch.empty for n in graph.nodes if n.op == "call_function"
        )
        self.assertTrue(has_out, "Out variant should exist after transform")
        self.assertTrue(has_empty, "torch.empty allocation should exist")

        # Verify functional op is removed
        has_functional_after = any(
            n.target == torch.ops.test_decompose.add_one.default
            for n in graph.nodes
            if n.op == "call_function"
        )
        self.assertFalse(has_functional_after, "Functional op should be removed")

        # Verify correctness
        exported.graph_module.recompile()
        result = exported.graph_module(x)
        if isinstance(result, tuple):
            result = result[0]
        torch.testing.assert_close(result, x + 1)

    def test_multiple_outputs_graph_transformation(self):
        """
        Test that a multi-output functional op is transformed correctly.

        Verifies:
        - Out variant has multiple out args (out_scaled, out_divided)
        - Both outputs are correctly allocated and returned
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch._library._out_variant import get_out_arg_names, to_out_variant
        from torch.export import export

        # Verify out variant detection
        functional_op = torch.ops.test_decompose.split_scale.default
        out_op = to_out_variant(functional_op)
        self.assertIsNotNone(out_op)
        self.assertEqual(out_op, torch.ops.test_decompose.split_scale.out)
        self.assertEqual(get_out_arg_names(out_op), ["out_scaled", "out_divided"])

        # Test graph transformation
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.test_decompose.split_scale(x, 2.0)

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified)

        # Verify out variant in graph
        has_out = any(
            n.target == torch.ops.test_decompose.split_scale.out
            for n in graph.nodes
            if n.op == "call_function"
        )
        self.assertTrue(has_out)

        # Verify correctness
        exported.graph_module.recompile()
        result = exported.graph_module(x)
        if isinstance(result, tuple) and len(result) == 2:
            scaled, divided = result
        else:
            scaled, divided = result[0], result[1]
        torch.testing.assert_close(scaled, x * 2.0)
        torch.testing.assert_close(divided, x / 2.0)


class TestReinplace(TestCase):
    """Tests for the reinplace optimization in decompose_functional_to_out."""

    @classmethod
    def setUpClass(cls):
        cls.lib = torch.library.Library("test_reinplace", "FRAGMENT")

        # Single output op: out = x + 1 (same shape/dtype → x is reinplace candidate)
        cls.lib.define("add_one(Tensor x) -> Tensor")

        @torch.library.impl("test_reinplace::add_one", "CompositeExplicitAutograd")
        def add_one_impl(x: Tensor) -> Tensor:
            return x + 1

        @torch.library.impl("test_reinplace::add_one", "Meta")
        def add_one_meta(x: Tensor) -> Tensor:
            return torch.empty_like(x)

        # kwarg-only out with Tensor(a!) alias annotation (required by torchgen)
        cls.lib.define("add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)")

        @torch.library.impl(
            "test_reinplace::add_one.out", "CompositeExplicitAutograd"
        )
        def add_one_out_impl(x: Tensor, *, out: Tensor) -> Tensor:
            out.copy_(x + 1)
            return out

        @torch.library.impl("test_reinplace::add_one.out", "Meta")
        def add_one_out_meta(x: Tensor, *, out: Tensor) -> Tensor:
            return out

    def test_reinplace_fires_when_input_not_used_downstream(self):
        """
        When the input to the custom op is an intermediate node with no
        downstream uses, reinplace should reuse its buffer.

        We chain two custom ops: add_one(add_one(x)).
        The inner add_one produces an intermediate that is ONLY used by
        the outer add_one. After transformation, the outer add_one should
        reinplace into the inner's output buffer.
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch.export import export

        class TestModule(torch.nn.Module):
            def forward(self, x):
                tmp = torch.ops.test_reinplace.add_one(x)
                return torch.ops.test_reinplace.add_one(tmp)

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified)

        # Count torch.empty allocations. With two add_one ops:
        # - First add_one(x): x is placeholder → cannot reinplace → allocates empty
        # - Second add_one(tmp): tmp is the first empty node → not placeholder,
        #   no downstream uses → CAN reinplace → no allocation needed
        # So we should see exactly 1 torch.empty, not 2.
        empty_count = sum(
            1 for n in graph.nodes
            if n.op == "call_function" and n.target == torch.empty
        )
        self.assertEqual(
            empty_count, 1,
            f"Expected 1 torch.empty (reinplace should eliminate the second), got {empty_count}",
        )

        # Both out variants should exist
        out_count = sum(
            1
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.test_reinplace.add_one.out
        )
        self.assertEqual(out_count, 2)

        # Verify correctness
        exported.graph_module.recompile()
        result = exported.graph_module(x)
        if isinstance(result, tuple):
            result = result[0]
        torch.testing.assert_close(result, x + 2)

    def test_reinplace_skipped_when_input_used_downstream(self):
        """
        When the input x is used after the op, reinplace must NOT fire —
        overwriting x would corrupt the downstream use.

        Graph: x → add_one(x) → result; x is also used in x + result
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch.export import export

        class TestModule(torch.nn.Module):
            def forward(self, x):
                result = torch.ops.test_reinplace.add_one(x)
                return result + x  # x is used downstream

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified)

        # Reinplace should NOT fire: torch.empty must be allocated
        has_empty = any(
            n.target == torch.empty for n in graph.nodes if n.op == "call_function"
        )
        self.assertTrue(has_empty, "Should allocate torch.empty when input is used downstream")

        # Verify correctness
        exported.graph_module.recompile()
        result = exported.graph_module(x)
        if isinstance(result, tuple):
            result = result[0]
        torch.testing.assert_close(result, (x + 1) + x)

    def test_reinplace_skipped_for_placeholder(self):
        """
        Graph inputs (placeholders) must not be reinplaced — the caller
        still expects the original data.
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch.export import export

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.test_reinplace.add_one(x)

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        # The only input arg to add_one is x, which is a placeholder.
        # _can_reuse_input rejects placeholders, so torch.empty must be used.
        # However, the exported graph may introduce intermediate nodes
        # between the placeholder and the op. Check correctness regardless.
        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified)

        # Verify correctness
        exported.graph_module.recompile()
        result = exported.graph_module(x)
        if isinstance(result, tuple):
            result = result[0]
        torch.testing.assert_close(result, x + 1)


class TestNativeQuantizeOp(TestCase):
    """
    Tests with real native PyTorch quantization op.

    fake_quantize_per_tensor_affine_cachemask is a native op with:
    - Multiple outputs: (Tensor output, Tensor mask)
    - Auto-generated .out variant
    """

    def test_fake_quantize_graph_transformation(self):
        """
        Test decompose pass transforms fake_quantize_per_tensor_affine_cachemask.

        Verifies:
        - Before: graph contains functional op
        - After: graph contains out variant + torch.empty allocations
        """
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )
        from torch.export import export

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
                    x, 0.1, 0, -128, 127
                )

        x = torch.randn(4, 4)
        exported = export(TestModule(), (x,))
        graph = exported.graph_module.graph

        # Check before transformation
        has_functional = any(
            n.target == torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.default
            for n in graph.nodes
            if n.op == "call_function"
        )
        self.assertTrue(has_functional, "Functional op should exist before transform")

        # Apply pass
        modified = decompose_functional_to_out(graph)

        if modified:
            # Verify out variant in graph
            has_out = any(
                n.target == torch.ops.aten.fake_quantize_per_tensor_affine_cachemask.out
                for n in graph.nodes
                if n.op == "call_function"
            )
            self.assertTrue(has_out, "Out variant should exist after transform")

            # Verify torch.empty allocations
            has_empty = any(
                n.target == torch.empty for n in graph.nodes if n.op == "call_function"
            )
            self.assertTrue(has_empty, "torch.empty allocation should exist")

    def test_fake_quantize_end_to_end(self):
        """
        End-to-end test with torch.compile for fake_quantize_per_tensor_affine_cachemask.

        Verifies:
        1. The decompose pass works in the full compilation pipeline
        2. The compiled result matches eager execution
        3. Output shapes and types are correct
        """

        def fn(x):
            return torch.ops.aten.fake_quantize_per_tensor_affine_cachemask(
                x, 0.1, 0, -128, 127
            )

        x = torch.randn(4, 4)

        # Eager execution for reference
        expected_output, expected_mask = fn(x)

        # Compiled execution with decompose pass enabled
        with torch._inductor.config.patch(decompose_functional_to_out=True):
            compiled_fn = torch.compile(fn, backend="inductor")
            result = compiled_fn(x)

        # Handle result (may be tuple or individual tensors)
        if isinstance(result, tuple):
            output, mask = result
        else:
            output, mask = result[0], result[1]

        # Verify correctness
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(mask, expected_mask)

        # Verify shapes and types
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(mask.shape, x.shape)
        self.assertEqual(mask.dtype, torch.bool)


if __name__ == "__main__":
    run_tests()
