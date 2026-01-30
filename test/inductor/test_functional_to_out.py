"""
Tests for functional → out variant decomposition.

This module tests:
1. The registration API (register_functional_to_out)
2. The Inductor decompose pass
3. End-to-end compilation with custom ops
4. CUDAGraph compatibility (when CUDA available)
"""

import operator
import unittest
from typing import Any
from unittest.mock import patch

import torch
import torch.fx as fx
from torch import Tensor


# Import the modules we're testing
from torch._library.functional_to_out import (
    register_functional_to_out,
    unregister_functional_to_out,
    get_out_variant,
    has_out_variant,
    clear_registry,
    TensorSpec,
    FunctionalToOutMapping,
)
from torch._inductor.fx_passes.decompose_functional_to_out import (
    decompose_functional_to_out,
    _decompose_node,
    _get_output_specs_from_node,
)


class TestRegistryAPI(unittest.TestCase):
    """Tests for the functional→out registration API."""

    def setUp(self):
        """Clear the registry before each test."""
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        """Clean up after each test."""
        clear_registry()
        self._cleanup_test_ops()

    def _setup_test_ops(self):
        """Define test custom ops."""
        # Functional variant
        @torch.library.custom_op("test_f2o::add_scale", mutates_args=())
        def add_scale_functional(x: Tensor, y: Tensor, scale: float) -> Tensor:
            return (x + y) * scale

        @add_scale_functional.register_fake
        def _(x: Tensor, y: Tensor, scale: float) -> Tensor:
            return torch.empty_like(x)

        # Out variant
        @torch.library.custom_op("test_f2o::add_scale_out", mutates_args=("out",))
        def add_scale_out(out: Tensor, x: Tensor, y: Tensor, scale: float) -> None:
            out.copy_((x + y) * scale)

        @add_scale_out.register_fake
        def _(out: Tensor, x: Tensor, y: Tensor, scale: float) -> None:
            pass

        # Multi-output functional variant
        @torch.library.custom_op("test_f2o::split_quant", mutates_args=())
        def split_quant_functional(x: Tensor) -> tuple[Tensor, Tensor]:
            quant = (x * 127).to(torch.int8)
            scale = x.abs().max(dim=-1, keepdim=True).values
            return quant, scale

        @split_quant_functional.register_fake
        def _(x: Tensor) -> tuple[Tensor, Tensor]:
            quant = torch.empty_like(x, dtype=torch.int8)
            scale = torch.empty(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)
            return quant, scale

        # Multi-output out variant
        @torch.library.custom_op(
            "test_f2o::split_quant_out", mutates_args=("out_quant", "out_scale")
        )
        def split_quant_out(
            out_quant: Tensor, out_scale: Tensor, x: Tensor
        ) -> None:
            out_quant.copy_((x * 127).to(torch.int8))
            out_scale.copy_(x.abs().max(dim=-1, keepdim=True).values)

        @split_quant_out.register_fake
        def _(out_quant: Tensor, out_scale: Tensor, x: Tensor) -> None:
            pass

    def _cleanup_test_ops(self):
        """Clean up test ops (if needed)."""
        # torch.library ops are automatically cleaned up
        pass

    def test_register_basic(self):
        """Test basic registration."""
        register_functional_to_out(
            functional_op=torch.ops.test_f2o.add_scale,
            out_op=torch.ops.test_f2o.add_scale_out,
            out_arg_positions=(0,),
        )

        self.assertTrue(has_out_variant(torch.ops.test_f2o.add_scale))
        mapping = get_out_variant(torch.ops.test_f2o.add_scale)
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.out_op, torch.ops.test_f2o.add_scale_out)
        self.assertEqual(mapping.out_arg_positions, (0,))

    def test_register_multi_output(self):
        """Test registration with multiple outputs."""
        register_functional_to_out(
            functional_op=torch.ops.test_f2o.split_quant,
            out_op=torch.ops.test_f2o.split_quant_out,
            out_arg_positions=(0, 1),
        )

        mapping = get_out_variant(torch.ops.test_f2o.split_quant)
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.num_outputs, 2)

    def test_register_duplicate_raises(self):
        """Test that duplicate registration raises ValueError."""
        register_functional_to_out(
            functional_op=torch.ops.test_f2o.add_scale,
            out_op=torch.ops.test_f2o.add_scale_out,
            out_arg_positions=(0,),
        )

        with self.assertRaises(ValueError):
            register_functional_to_out(
                functional_op=torch.ops.test_f2o.add_scale,
                out_op=torch.ops.test_f2o.add_scale_out,
                out_arg_positions=(0,),
            )

    def test_unregister(self):
        """Test unregistration."""
        register_functional_to_out(
            functional_op=torch.ops.test_f2o.add_scale,
            out_op=torch.ops.test_f2o.add_scale_out,
            out_arg_positions=(0,),
        )

        self.assertTrue(has_out_variant(torch.ops.test_f2o.add_scale))

        result = unregister_functional_to_out(torch.ops.test_f2o.add_scale)
        self.assertTrue(result)
        self.assertFalse(has_out_variant(torch.ops.test_f2o.add_scale))

        # Unregistering again should return False
        result = unregister_functional_to_out(torch.ops.test_f2o.add_scale)
        self.assertFalse(result)

    def test_get_nonexistent(self):
        """Test getting mapping for unregistered op."""
        mapping = get_out_variant(torch.ops.test_f2o.add_scale)
        self.assertIsNone(mapping)

    def test_has_nonexistent(self):
        """Test has_out_variant for unregistered op."""
        self.assertFalse(has_out_variant(torch.ops.test_f2o.add_scale))


class TestTensorSpec(unittest.TestCase):
    """Tests for TensorSpec."""

    def test_allocate(self):
        """Test tensor allocation from spec."""
        spec = TensorSpec(
            shape=(4, 8),
            dtype=torch.float32,
            device="cpu",
        )

        tensor = spec.allocate()
        self.assertEqual(tensor.shape, (4, 8))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, "cpu")

    def test_allocate_with_requires_grad(self):
        """Test allocation with requires_grad."""
        spec = TensorSpec(
            shape=(2, 3),
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )

        tensor = spec.allocate()
        self.assertTrue(tensor.requires_grad)


class TestOutputSpecsInference(unittest.TestCase):
    """Tests for output specs inference from fake tensors."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Set up test ops."""
        @torch.library.custom_op("test_f2o_infer::simple_op", mutates_args=())
        def simple_op(x: Tensor) -> Tensor:
            return x * 2

        @simple_op.register_fake
        def _(x: Tensor) -> Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o_infer::simple_op_out", mutates_args=("out",))
        def simple_op_out(out: Tensor, x: Tensor) -> None:
            out.copy_(x * 2)

        @simple_op_out.register_fake
        def _(out: Tensor, x: Tensor) -> None:
            pass

    def test_infer_single_output(self):
        """Test inferring specs for single output op."""
        mapping = FunctionalToOutMapping(
            functional_op=torch.ops.test_f2o_infer.simple_op,
            out_op=torch.ops.test_f2o_infer.simple_op_out,
            out_arg_positions=(0,),
        )

        x = torch.randn(4, 8)
        specs = mapping.get_output_specs((x,), {})

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape, (4, 8))
        self.assertEqual(specs[0].dtype, torch.float32)


class TestDecomposePass(unittest.TestCase):
    """Tests for the Inductor decompose pass."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Set up test ops for pass testing."""
        @torch.library.custom_op("test_f2o_pass::mul2", mutates_args=())
        def mul2_functional(x: Tensor) -> Tensor:
            return x * 2

        @mul2_functional.register_fake
        def _(x: Tensor) -> Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o_pass::mul2_out", mutates_args=("out",))
        def mul2_out(out: Tensor, x: Tensor) -> None:
            out.copy_(x * 2)

        @mul2_out.register_fake
        def _(out: Tensor, x: Tensor) -> None:
            pass

        # Register mapping
        register_functional_to_out(
            functional_op=torch.ops.test_f2o_pass.mul2,
            out_op=torch.ops.test_f2o_pass.mul2_out,
            out_arg_positions=(0,),
        )

    def test_decompose_basic(self):
        """Test basic decomposition in an FX graph."""

        def fn(x):
            y = torch.ops.test_f2o_pass.mul2(x)
            return y + 1

        # Trace to FX graph
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as fake_mode:
            fake_x = fake_mode.from_tensor(torch.randn(4, 8))

            # Create a simple graph manually for testing
            graph = fx.Graph()
            x_node = graph.placeholder("x")
            x_node.meta["val"] = fake_x

            # Add the functional op call
            mul_node = graph.call_function(
                torch.ops.test_f2o_pass.mul2,
                args=(x_node,),
            )
            mul_node.meta["val"] = torch.empty_like(fake_x)

            # Add a user of the result
            add_node = graph.call_function(
                torch.add,
                args=(mul_node, 1),
            )
            add_node.meta["val"] = torch.empty_like(fake_x)

            graph.output(add_node)

        # Run the decompose pass
        modified = decompose_functional_to_out(graph)
        self.assertTrue(modified)

        # Verify the graph structure
        nodes = list(graph.nodes)
        node_targets = [n.target for n in nodes if n.op == "call_function"]

        # Should have: torch.empty, mul2_out, torch.add
        self.assertIn(torch.empty, node_targets)
        self.assertIn(torch.ops.test_f2o_pass.mul2_out, node_targets)
        self.assertNotIn(torch.ops.test_f2o_pass.mul2, node_targets)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests with torch.compile."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Set up realistic test ops."""
        # Functional variant that mimics a quantization op
        @torch.library.custom_op("test_f2o_e2e::quant", mutates_args=())
        def quant_functional(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
            quant = (x * scale).clamp(-128, 127).to(torch.int8)
            quant_scale = x.abs().amax(dim=-1, keepdim=True)
            return quant, quant_scale

        @quant_functional.register_fake
        def _(x: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
            quant = torch.empty_like(x, dtype=torch.int8)
            quant_scale = torch.empty(
                x.shape[:-1] + (1,), dtype=x.dtype, device=x.device
            )
            return quant, quant_scale

        # Out variant
        @torch.library.custom_op(
            "test_f2o_e2e::quant_out", mutates_args=("out_quant", "out_scale")
        )
        def quant_out(
            out_quant: Tensor, out_scale: Tensor, x: Tensor, scale: Tensor
        ) -> None:
            out_quant.copy_((x * scale).clamp(-128, 127).to(torch.int8))
            out_scale.copy_(x.abs().amax(dim=-1, keepdim=True))

        @quant_out.register_fake
        def _(out_quant: Tensor, out_scale: Tensor, x: Tensor, scale: Tensor) -> None:
            pass

        # Register the mapping
        register_functional_to_out(
            functional_op=torch.ops.test_f2o_e2e.quant,
            out_op=torch.ops.test_f2o_e2e.quant_out,
            out_arg_positions=(0, 1),
        )

    def test_compile_with_custom_op(self):
        """Test that compilation works with registered functional→out mapping."""

        def fn(x, scale):
            quant, quant_scale = torch.ops.test_f2o_e2e.quant(x, scale)
            # Do something with the results
            return quant.float() * quant_scale

        x = torch.randn(4, 16)
        scale = torch.tensor([100.0])

        # Run without compilation for reference
        ref_result = fn(x, scale)

        # Compile with Inductor
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(x, scale)

        # Results should match
        torch.testing.assert_close(compiled_result, ref_result, rtol=1e-4, atol=1e-4)

    def test_multiple_custom_ops_in_sequence(self):
        """Test multiple custom ops in sequence."""

        def fn(x, scale):
            # First quant
            q1, s1 = torch.ops.test_f2o_e2e.quant(x, scale)
            # Some computation
            intermediate = q1.float() * s1
            # Second quant
            q2, s2 = torch.ops.test_f2o_e2e.quant(intermediate, scale)
            return q2, s2

        x = torch.randn(4, 16)
        scale = torch.tensor([100.0])

        ref_q, ref_s = fn(x, scale)

        compiled_fn = torch.compile(fn, backend="inductor")
        comp_q, comp_s = compiled_fn(x, scale)

        torch.testing.assert_close(comp_q, ref_q)
        torch.testing.assert_close(comp_s, ref_s)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestCUDAGraphCompatibility(unittest.TestCase):
    """Tests for CUDAGraph compatibility."""

    def setUp(self):
        clear_registry()
        self._setup_test_ops()

    def tearDown(self):
        clear_registry()

    def _setup_test_ops(self):
        """Set up CUDA test ops."""

        @torch.library.custom_op("test_f2o_cuda::scale", mutates_args=())
        def scale_functional(x: Tensor, s: Tensor) -> Tensor:
            return x * s

        @scale_functional.register_fake
        def _(x: Tensor, s: Tensor) -> Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o_cuda::scale_out", mutates_args=("out",))
        def scale_out(out: Tensor, x: Tensor, s: Tensor) -> None:
            out.copy_(x * s)

        @scale_out.register_fake
        def _(out: Tensor, x: Tensor, s: Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o_cuda.scale,
            out_op=torch.ops.test_f2o_cuda.scale_out,
            out_arg_positions=(0,),
        )

    def test_cudagraph_capture(self):
        """Test that decomposed ops work with CUDAGraph."""

        def fn(x, s):
            return torch.ops.test_f2o_cuda.scale(x, s)

        x = torch.randn(4, 8, device="cuda")
        s = torch.tensor([2.0], device="cuda")

        # Compile with reduce-overhead (enables CUDAGraph)
        compiled_fn = torch.compile(fn, backend="inductor", mode="reduce-overhead")

        # Warm up (CUDAGraph capture happens here)
        for _ in range(3):
            _ = compiled_fn(x, s)

        # Run several times - should use captured graph
        results = []
        for _ in range(5):
            result = compiled_fn(x, s)
            results.append(result.clone())

        # Verify all results are consistent
        for r in results[1:]:
            torch.testing.assert_close(r, results[0])


# =============================================================================
# Demo/Example Usage
# =============================================================================


def demo_basic_usage():
    """Demonstrate basic usage of the functional→out API."""
    print("=" * 60)
    print("Demo: Basic Functional→Out Usage")
    print("=" * 60)

    clear_registry()

    # Step 1: Define functional variant
    @torch.library.custom_op("demo::my_quant", mutates_args=())
    def my_quant(x: Tensor, scale: float) -> Tensor:
        return (x * scale).to(torch.int8)

    @my_quant.register_fake
    def _(x: Tensor, scale: float) -> Tensor:
        return torch.empty_like(x, dtype=torch.int8)

    # Step 2: Define out variant
    @torch.library.custom_op("demo::my_quant_out", mutates_args=("out",))
    def my_quant_out(out: Tensor, x: Tensor, scale: float) -> None:
        out.copy_((x * scale).to(torch.int8))

    @my_quant_out.register_fake
    def _(out: Tensor, x: Tensor, scale: float) -> None:
        pass

    # Step 3: Register the mapping
    register_functional_to_out(
        functional_op=torch.ops.demo.my_quant,
        out_op=torch.ops.demo.my_quant_out,
        out_arg_positions=(0,),
    )

    print("✓ Registered my_quant → my_quant_out")

    # Step 4: Use in compiled code
    def forward(x):
        return torch.ops.demo.my_quant(x, 100.0)

    x = torch.randn(4, 8)
    print(f"Input shape: {x.shape}")

    # Without compilation
    ref = forward(x)
    print(f"Reference output shape: {ref.shape}, dtype: {ref.dtype}")

    # With compilation (uses decompose pass)
    compiled_forward = torch.compile(forward, backend="inductor")
    result = compiled_forward(x)
    print(f"Compiled output shape: {result.shape}, dtype: {result.dtype}")

    torch.testing.assert_close(result, ref)
    print("✓ Results match!")

    clear_registry()
    print()


if __name__ == "__main__":
    # Run demo
    demo_basic_usage()

    # Run tests
    unittest.main(verbosity=2)
