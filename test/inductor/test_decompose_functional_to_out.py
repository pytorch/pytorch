"""
Tests for the functional-to-out variant decomposition pass.

This module tests:
1. Registry API (register, unregister, lookup)
2. TensorSpec allocation
3. Output spec inference from fake tensors
4. Basic decomposition pass
5. CUDA-specific tests
6. CUDAGraph compatibility tests
7. Multiple output decomposition tests
8. Edge cases and error handling
"""

import functools
import operator
import unittest

import torch
import torch.fx as fx
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


# Check CUDA availability
HAS_CUDA = torch.cuda.is_available()

# Skip if CUDA is not available (most tests need it)
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires CUDA")


class TestFunctionalToOutRegistry(TestCase):
    """Tests for the functional_to_out registry API."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        # Clear registry before each test
        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        # Clean up after each test
        clear_registry()
        # Clean up any test ops we created
        self._cleanup_test_ops()

    def _cleanup_test_ops(self):
        """Clean up test custom ops if they exist."""
        for ns in ["test_f2o"]:
            try:
                lib = getattr(torch.ops, ns, None)
                if lib is not None:
                    # Can't easily delete ops, but we can clear the registry
                    pass
            except Exception:
                pass

    def _create_test_ops(self):
        """Create test functional and out variant ops."""

        # Define a simple functional op
        @torch.library.custom_op("test_f2o::add_functional", mutates_args=())
        def add_functional(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        @add_functional.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # Define the out variant
        @torch.library.custom_op("test_f2o::add_out", mutates_args=("out",))
        def add_out(out: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
            out.copy_(x + y)

        @add_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
            pass

        return (
            torch.ops.test_f2o.add_functional,
            torch.ops.test_f2o.add_out,
        )

    def test_register_and_lookup(self):
        """Test basic registration and lookup."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
            has_out_variant,
        )

        func_op, out_op = self._create_test_ops()

        # Not registered yet
        self.assertFalse(has_out_variant(func_op))
        self.assertIsNone(get_out_variant(func_op))

        # Register
        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0,),
        )

        # Now should be registered
        self.assertTrue(has_out_variant(func_op))
        mapping = get_out_variant(func_op)
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.functional_op, func_op)
        self.assertEqual(mapping.out_op, out_op)
        self.assertEqual(mapping.out_arg_positions, (0,))

    def test_unregister(self):
        """Test unregistration."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            unregister_functional_to_out,
            has_out_variant,
        )

        func_op, out_op = self._create_test_ops()

        # Register
        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0,),
        )
        self.assertTrue(has_out_variant(func_op))

        # Unregister
        result = unregister_functional_to_out(func_op)
        self.assertTrue(result)
        self.assertFalse(has_out_variant(func_op))

        # Unregister again should return False
        result = unregister_functional_to_out(func_op)
        self.assertFalse(result)

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        from torch._library.functional_to_out import register_functional_to_out

        func_op, out_op = self._create_test_ops()

        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0,),
        )

        with self.assertRaises(ValueError):
            register_functional_to_out(
                functional_op=func_op,
                out_op=out_op,
                out_arg_positions=(0,),
            )

    def test_clear_registry(self):
        """Test clearing the registry."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            clear_registry,
            has_out_variant,
            get_all_registered_ops,
        )

        func_op, out_op = self._create_test_ops()

        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0,),
        )
        self.assertEqual(len(get_all_registered_ops()), 1)

        clear_registry()
        self.assertEqual(len(get_all_registered_ops()), 0)
        self.assertFalse(has_out_variant(func_op))


class TestTensorSpec(TestCase):
    """Tests for TensorSpec allocation."""

    def test_basic_allocation(self):
        """Test basic tensor allocation from TensorSpec."""
        from torch._library.functional_to_out import TensorSpec

        spec = TensorSpec(
            shape=(2, 3, 4),
            dtype=torch.float32,
            device="cpu",
            requires_grad=False,
        )

        tensor = spec.allocate()
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertFalse(tensor.requires_grad)

    @requires_cuda()
    def test_cuda_allocation(self):
        """Test CUDA tensor allocation from TensorSpec."""
        from torch._library.functional_to_out import TensorSpec

        spec = TensorSpec(
            shape=(64, 128),
            dtype=torch.float16,
            device="cuda",
            requires_grad=False,
        )

        tensor = spec.allocate()
        self.assertEqual(tensor.shape, (64, 128))
        self.assertEqual(tensor.dtype, torch.float16)
        self.assertEqual(tensor.device.type, "cuda")

    def test_requires_grad(self):
        """Test allocation with requires_grad."""
        from torch._library.functional_to_out import TensorSpec

        spec = TensorSpec(
            shape=(10,),
            dtype=torch.float32,
            device="cpu",
            requires_grad=True,
        )

        tensor = spec.allocate()
        self.assertTrue(tensor.requires_grad)


class TestOutputSpecsInference(TestCase):
    """Tests for inferring output specs from fake tensors."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def _create_multi_output_ops(self):
        """Create ops with multiple outputs for testing."""

        # Functional op returning tuple
        @torch.library.custom_op("test_f2o::multi_out_func", mutates_args=())
        def multi_out_func(
            x: torch.Tensor, scale: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x * scale, x + scale

        @multi_out_func.register_fake
        def _(x: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(x)

        # Out variant
        @torch.library.custom_op(
            "test_f2o::multi_out_out", mutates_args=("out1", "out2")
        )
        def multi_out_out(
            out1: torch.Tensor,
            out2: torch.Tensor,
            x: torch.Tensor,
            scale: float,
        ) -> None:
            out1.copy_(x * scale)
            out2.copy_(x + scale)

        @multi_out_out.register_fake
        def _(
            out1: torch.Tensor,
            out2: torch.Tensor,
            x: torch.Tensor,
            scale: float,
        ) -> None:
            pass

        return (
            torch.ops.test_f2o.multi_out_func,
            torch.ops.test_f2o.multi_out_out,
        )

    def test_infer_single_output(self):
        """Test inferring specs for single output op."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
            TensorSpec,
        )

        # Create simple ops
        @torch.library.custom_op("test_f2o::scale_func", mutates_args=())
        def scale_func(x: torch.Tensor, s: float) -> torch.Tensor:
            return x * s

        @scale_func.register_fake
        def _(x: torch.Tensor, s: float) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::scale_out", mutates_args=("out",))
        def scale_out(out: torch.Tensor, x: torch.Tensor, s: float) -> None:
            out.copy_(x * s)

        @scale_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor, s: float) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.scale_func,
            out_op=torch.ops.test_f2o.scale_out,
            out_arg_positions=(0,),
        )

        mapping = get_out_variant(torch.ops.test_f2o.scale_func)

        # Test spec inference
        x = torch.randn(4, 8)
        specs = mapping.get_output_specs((x, 2.0), {})

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape, (4, 8))
        self.assertEqual(specs[0].dtype, torch.float32)

    def test_infer_multi_output(self):
        """Test inferring specs for multiple output op."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
        )

        func_op, out_op = self._create_multi_output_ops()

        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0, 1),
        )

        mapping = get_out_variant(func_op)

        x = torch.randn(3, 5)
        specs = mapping.get_output_specs((x, 2.0), {})

        self.assertEqual(len(specs), 2)
        for spec in specs:
            self.assertEqual(spec.shape, (3, 5))
            self.assertEqual(spec.dtype, torch.float32)


class TestDecomposeFunctionalToOut(TestCase):
    """Tests for the main decomposition pass."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def _create_simple_ops(self):
        """Create simple test ops."""

        @torch.library.custom_op("test_f2o::simple_func", mutates_args=())
        def simple_func(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @simple_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::simple_out", mutates_args=("out",))
        def simple_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 2)

        @simple_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        return (
            torch.ops.test_f2o.simple_func,
            torch.ops.test_f2o.simple_out,
        )

    def test_decompose_single_output(self):
        """Test decomposition of single-output functional op."""
        from torch._library.functional_to_out import register_functional_to_out
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        func_op, out_op = self._create_simple_ops()

        register_functional_to_out(
            functional_op=func_op,
            out_op=out_op,
            out_arg_positions=(0,),
        )

        # Create a simple traced module
        def fn(x):
            return func_op(x)

        # Trace with fake tensor mode
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4)
            traced = torch.fx.symbolic_trace(fn)
            # Set fake tensor values in metadata
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function" and node.target == func_op:
                    node.meta["val"] = x

        # Run decomposition
        modified = decompose_functional_to_out(traced.graph)
        self.assertTrue(modified)

        # Verify the graph structure
        found_empty = False
        found_out_op = False
        for node in traced.graph.nodes:
            if node.op == "call_function":
                if node.target == torch.empty:
                    found_empty = True
                # Check for out variant by looking at the string representation
                # The target could be OpOverload like "test_f2o.simple_out.default"
                elif "simple_out" in str(node.target):
                    found_out_op = True

        self.assertTrue(found_empty, "Should have torch.empty allocation")
        self.assertTrue(found_out_op, "Should have out variant call")

    def test_decompose_multi_output(self):
        """Test decomposition of multi-output functional op."""
        from torch._library.functional_to_out import register_functional_to_out
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        # Create multi-output ops
        @torch.library.custom_op("test_f2o::dual_func", mutates_args=())
        def dual_func(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x + 1

        @dual_func.register_fake
        def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(x)

        @torch.library.custom_op("test_f2o::dual_out", mutates_args=("out1", "out2"))
        def dual_out(out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor) -> None:
            out1.copy_(x * 2)
            out2.copy_(x + 1)

        @dual_out.register_fake
        def _(out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.dual_func,
            out_op=torch.ops.test_f2o.dual_out,
            out_arg_positions=(0, 1),
        )

        # Create traced module with getitem pattern
        def fn(x):
            result = torch.ops.test_f2o.dual_func(x)
            return result[0], result[1]

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4)
            traced = torch.fx.symbolic_trace(fn)
            # Set fake tensor values
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function":
                    if node.target == torch.ops.test_f2o.dual_func:
                        node.meta["val"] = (x, x)
                    elif node.target is operator.getitem:
                        node.meta["val"] = x

        modified = decompose_functional_to_out(traced.graph)
        self.assertTrue(modified)

        # Verify: should have 2 empty allocations
        empty_count = 0
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.empty:
                empty_count += 1

        self.assertEqual(empty_count, 2, "Should have 2 torch.empty allocations")

    def test_no_decompose_without_registration(self):
        """Test that unregistered ops are not decomposed."""
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        func_op, _ = self._create_simple_ops()
        # Note: NOT registering the mapping

        def fn(x):
            return func_op(x)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4)
            traced = torch.fx.symbolic_trace(fn)
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function" and node.target == func_op:
                    node.meta["val"] = x

        modified = decompose_functional_to_out(traced.graph)
        self.assertFalse(modified)


class TestDecomposeFunctionalToOutCUDA(TestCase):
    """CUDA-specific tests for the decomposition pass."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    @requires_cuda()
    def test_cuda_decomposition(self):
        """Test decomposition preserves CUDA device."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            TensorSpec,
        )
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        @torch.library.custom_op("test_f2o::cuda_func", mutates_args=())
        def cuda_func(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @cuda_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::cuda_out", mutates_args=("out",))
        def cuda_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 2)

        @cuda_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.cuda_func,
            out_op=torch.ops.test_f2o.cuda_out,
            out_arg_positions=(0,),
        )

        def fn(x):
            return torch.ops.test_f2o.cuda_func(x)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4, device="cuda")
            traced = torch.fx.symbolic_trace(fn)
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function":
                    if node.target == torch.ops.test_f2o.cuda_func:
                        node.meta["val"] = x

        modified = decompose_functional_to_out(traced.graph)
        self.assertTrue(modified)

        # Verify allocation has CUDA device
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.empty:
                device = node.kwargs.get("device")
                self.assertIsNotNone(device)
                if hasattr(device, "type"):
                    self.assertEqual(device.type, "cuda")


@requires_cuda()
class TestCUDAGraphCompatibility(TestCase):
    """Tests for CUDAGraph compatibility."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_cudagraph_with_decomposed_op(self):
        """Test that decomposed ops work with CUDAGraph capture."""
        from torch._library.functional_to_out import register_functional_to_out

        @torch.library.custom_op("test_f2o::cg_func", mutates_args=())
        def cg_func(x: torch.Tensor) -> torch.Tensor:
            return x * 3

        @cg_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::cg_out", mutates_args=("out",))
        def cg_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 3)

        @cg_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.cg_func,
            out_op=torch.ops.test_f2o.cg_out,
            out_arg_positions=(0,),
        )

        # Test that the out variant can be captured in CUDAGraph
        x = torch.randn(4, 4, device="cuda")
        out = torch.empty_like(x)

        # Warmup
        torch.ops.test_f2o.cg_out(out, x)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.ops.test_f2o.cg_out(out, x)

        # Replay
        g.replay()

        expected = x * 3
        torch.testing.assert_close(out, expected)


class TestThreeOutputDecomposition(TestCase):
    """Test decomposition with three outputs."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_three_outputs(self):
        """Test decomposition with three outputs."""
        from torch._library.functional_to_out import register_functional_to_out
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        @torch.library.custom_op("test_f2o::triple_func", mutates_args=())
        def triple_func(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return x, x * 2, x + 1

        @triple_func.register_fake
        def _(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)

        @torch.library.custom_op(
            "test_f2o::triple_out", mutates_args=("out1", "out2", "out3")
        )
        def triple_out(
            out1: torch.Tensor,
            out2: torch.Tensor,
            out3: torch.Tensor,
            x: torch.Tensor,
        ) -> None:
            out1.copy_(x)
            out2.copy_(x * 2)
            out3.copy_(x + 1)

        @triple_out.register_fake
        def _(
            out1: torch.Tensor,
            out2: torch.Tensor,
            out3: torch.Tensor,
            x: torch.Tensor,
        ) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.triple_func,
            out_op=torch.ops.test_f2o.triple_out,
            out_arg_positions=(0, 1, 2),
        )

        def fn(x):
            result = torch.ops.test_f2o.triple_func(x)
            return result[0], result[1], result[2]

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4)
            traced = torch.fx.symbolic_trace(fn)
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function":
                    if node.target == torch.ops.test_f2o.triple_func:
                        node.meta["val"] = (x, x, x)
                    elif node.target is operator.getitem:
                        node.meta["val"] = x

        modified = decompose_functional_to_out(traced.graph)
        self.assertTrue(modified)

        # Verify: should have 3 empty allocations
        empty_count = 0
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.empty:
                empty_count += 1

        self.assertEqual(empty_count, 3)


class TestEdgeCases(TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_different_dtypes(self):
        """Test that different output dtypes are handled correctly."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
        )

        @torch.library.custom_op("test_f2o::mixed_dtype_func", mutates_args=())
        def mixed_dtype_func(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x.to(torch.float16)

        @mixed_dtype_func.register_fake
        def _(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(x, dtype=torch.float16)

        @torch.library.custom_op(
            "test_f2o::mixed_dtype_out", mutates_args=("out1", "out2")
        )
        def mixed_dtype_out(
            out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor
        ) -> None:
            out1.copy_(x)
            out2.copy_(x.to(torch.float16))

        @mixed_dtype_out.register_fake
        def _(out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.mixed_dtype_func,
            out_op=torch.ops.test_f2o.mixed_dtype_out,
            out_arg_positions=(0, 1),
        )

        mapping = get_out_variant(torch.ops.test_f2o.mixed_dtype_func)
        x = torch.randn(4, 4, dtype=torch.float32)
        specs = mapping.get_output_specs((x,), {})

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].dtype, torch.float32)
        self.assertEqual(specs[1].dtype, torch.float16)


class TestGraphInspection(TestCase):
    """Tests that inspect the generated graph structure."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_metadata_preserved(self):
        """Test that metadata is preserved in transformed nodes."""
        from torch._library.functional_to_out import register_functional_to_out
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        @torch.library.custom_op("test_f2o::meta_func", mutates_args=())
        def meta_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @meta_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::meta_out", mutates_args=("out",))
        def meta_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @meta_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.meta_func,
            out_op=torch.ops.test_f2o.meta_out,
            out_arg_positions=(0,),
        )

        def fn(x):
            return torch.ops.test_f2o.meta_func(x)

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.randn(4, 4)
            traced = torch.fx.symbolic_trace(fn)
            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    node.meta["val"] = x
                elif node.op == "call_function":
                    if node.target == torch.ops.test_f2o.meta_func:
                        node.meta["val"] = x

        decompose_functional_to_out(traced.graph)

        # Check that out_call has metadata about what it was decomposed from
        for node in traced.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.test_f2o.meta_out
            ):
                self.assertIn("decomposed_from", node.meta)
                self.assertEqual(
                    node.meta["decomposed_from"],
                    torch.ops.test_f2o.meta_func,
                )

        # Check allocation nodes have metadata
        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.empty:
                self.assertIn("allocation_for", node.meta)


class TestConfigOption(TestCase):
    """Tests for the config option controlling the pass."""

    def test_config_option_exists(self):
        """Test that the config option exists."""
        from torch._inductor import config

        self.assertTrue(hasattr(config, "decompose_functional_to_out"))

    def test_should_decompose_check(self):
        """Test the should_decompose_functional_to_out helper."""
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            should_decompose_functional_to_out,
        )
        from torch._inductor import config

        original = config.decompose_functional_to_out

        try:
            config.decompose_functional_to_out = True
            self.assertTrue(should_decompose_functional_to_out())

            config.decompose_functional_to_out = False
            self.assertFalse(should_decompose_functional_to_out())
        finally:
            config.decompose_functional_to_out = original


class TestBuildOutArgs(TestCase):
    """Tests for build_out_args functionality."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_build_out_args_single(self):
        """Test building out args for single output."""
        from torch._library.functional_to_out import FunctionalToOutMapping

        # Create a mock mapping
        mapping = FunctionalToOutMapping(
            functional_op=None,  # type: ignore
            out_op=None,  # type: ignore
            out_arg_positions=(0,),
        )

        out_buffer = torch.empty(4, 4)
        func_args = (torch.randn(4, 4), 2.0)
        func_kwargs = {"scale": 1.0}

        out_args, out_kwargs = mapping.build_out_args(
            [out_buffer], func_args, func_kwargs
        )

        self.assertEqual(len(out_args), 3)
        self.assertIs(out_args[0], out_buffer)
        self.assertEqual(out_kwargs, func_kwargs)

    def test_build_out_args_multi(self):
        """Test building out args for multiple outputs."""
        from torch._library.functional_to_out import FunctionalToOutMapping

        mapping = FunctionalToOutMapping(
            functional_op=None,  # type: ignore
            out_op=None,  # type: ignore
            out_arg_positions=(0, 1),
        )

        out1 = torch.empty(4, 4)
        out2 = torch.empty(4, 4)
        func_args = (torch.randn(4, 4),)
        func_kwargs = {}

        out_args, out_kwargs = mapping.build_out_args(
            [out1, out2], func_args, func_kwargs
        )

        self.assertEqual(len(out_args), 3)
        self.assertIs(out_args[0], out1)
        self.assertIs(out_args[1], out2)


class TestFunctionalToOutDecorator(TestCase):
    """Tests for the @functional_to_out decorator API."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_decorator_with_op_overload(self):
        """Test decorator when applied to an OpOverload directly."""
        from torch._library.functional_to_out import (
            functional_to_out,
            has_out_variant,
            get_out_variant,
        )

        # First create the out variant
        @torch.library.custom_op("test_f2o::dec_out_v1", mutates_args=("out",))
        def dec_out_v1(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 2)

        @dec_out_v1.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        # Create functional op and apply decorator
        @torch.library.custom_op("test_f2o::dec_func_v1", mutates_args=())
        def dec_func_v1(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @dec_func_v1.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # Get the actual OpOverload (not packet)
        func_op = torch.ops.test_f2o.dec_func_v1.default
        out_op = torch.ops.test_f2o.dec_out_v1.default

        # Apply decorator to the OpOverload
        decorated = functional_to_out(
            out_op=out_op,
            out_arg_positions=(0,),
        )(func_op)

        # Verify registration
        self.assertTrue(has_out_variant(func_op))
        mapping = get_out_variant(func_op)
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.out_arg_positions, (0,))

    def test_decorator_with_string_op_name(self):
        """Test decorator with string op name for out variant."""
        from torch._library.functional_to_out import (
            functional_to_out,
            has_out_variant,
            get_out_variant,
        )

        # Create out variant first
        @torch.library.custom_op("test_f2o::string_out", mutates_args=("out",))
        def string_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @string_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        # Create functional op
        @torch.library.custom_op("test_f2o::string_func", mutates_args=())
        def string_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @string_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # Get the OpOverload
        func_op = torch.ops.test_f2o.string_func.default

        # Apply decorator with string name
        decorated = functional_to_out(
            out_op="test_f2o::string_out",
            out_arg_positions=(0,),
        )(func_op)

        self.assertTrue(has_out_variant(func_op))
        mapping = get_out_variant(func_op)
        self.assertIsNotNone(mapping)

    def test_decorator_invalid_op_name_format(self):
        """Test that invalid op name format raises ValueError."""
        from torch._library.functional_to_out import functional_to_out

        @torch.library.custom_op("test_f2o::invalid_test", mutates_args=())
        def invalid_test(x: torch.Tensor) -> torch.Tensor:
            return x

        @invalid_test.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        # Get the OpOverload
        func_op = torch.ops.test_f2o.invalid_test.default

        with self.assertRaises(ValueError) as ctx:
            functional_to_out(
                out_op="invalid_format_no_double_colon",
                out_arg_positions=(0,),
            )(func_op)

        self.assertIn("Invalid op name format", str(ctx.exception))

    def test_decorator_deferred_registration(self):
        """Test decorator stores metadata for deferred registration."""
        from torch._library.functional_to_out import functional_to_out

        # Apply decorator to a plain function (not an op)
        @functional_to_out(
            out_op="test_f2o::deferred_out",
            out_arg_positions=(0, 1),
        )
        def plain_function(x):
            return x * 2

        # Should store metadata for later
        self.assertTrue(hasattr(plain_function, "_functional_to_out_metadata"))
        metadata = plain_function._functional_to_out_metadata
        self.assertEqual(metadata["out_op"], "test_f2o::deferred_out")
        self.assertEqual(metadata["out_arg_positions"], (0, 1))


class TestCustomOutputSpecsFn(TestCase):
    """Tests for custom output_specs_fn functionality."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_custom_output_specs_fn(self):
        """Test registration with custom output_specs_fn."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
            TensorSpec,
        )

        @torch.library.custom_op("test_f2o::custom_spec_func", mutates_args=())
        def custom_spec_func(x: torch.Tensor, scale: float) -> torch.Tensor:
            return x * scale

        @custom_spec_func.register_fake
        def _(x: torch.Tensor, scale: float) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::custom_spec_out", mutates_args=("out",))
        def custom_spec_out(out: torch.Tensor, x: torch.Tensor, scale: float) -> None:
            out.copy_(x * scale)

        @custom_spec_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor, scale: float) -> None:
            pass

        # Define custom output specs function
        def custom_specs_fn(x: torch.Tensor, scale: float) -> list[TensorSpec]:
            # Custom logic: always use float64 regardless of input
            return [
                TensorSpec(
                    shape=tuple(x.shape),
                    dtype=torch.float64,
                    device=x.device,
                )
            ]

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.custom_spec_func,
            out_op=torch.ops.test_f2o.custom_spec_out,
            out_arg_positions=(0,),
            output_specs_fn=custom_specs_fn,
        )

        mapping = get_out_variant(torch.ops.test_f2o.custom_spec_func)
        self.assertIsNotNone(mapping)

        # Test that custom specs function is used
        x = torch.randn(4, 4, dtype=torch.float32)
        specs = mapping.get_output_specs((x, 2.0), {})

        self.assertEqual(len(specs), 1)
        # Custom function should override to float64
        self.assertEqual(specs[0].dtype, torch.float64)
        self.assertEqual(specs[0].shape, (4, 4))

    def test_custom_output_specs_fn_multi_output(self):
        """Test custom output_specs_fn with multiple outputs."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
            TensorSpec,
        )

        @torch.library.custom_op("test_f2o::multi_custom_func", mutates_args=())
        def multi_custom_func(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x, x.sum(dim=-1)

        @multi_custom_func.register_fake
        def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty(x.shape[:-1], device=x.device)

        @torch.library.custom_op(
            "test_f2o::multi_custom_out", mutates_args=("out1", "out2")
        )
        def multi_custom_out(
            out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor
        ) -> None:
            out1.copy_(x)
            out2.copy_(x.sum(dim=-1))

        @multi_custom_out.register_fake
        def _(out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor) -> None:
            pass

        def custom_multi_specs(x: torch.Tensor) -> list[TensorSpec]:
            return [
                TensorSpec(shape=tuple(x.shape), dtype=x.dtype, device=x.device),
                TensorSpec(shape=tuple(x.shape[:-1]), dtype=x.dtype, device=x.device),
            ]

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.multi_custom_func,
            out_op=torch.ops.test_f2o.multi_custom_out,
            out_arg_positions=(0, 1),
            output_specs_fn=custom_multi_specs,
        )

        mapping = get_out_variant(torch.ops.test_f2o.multi_custom_func)
        x = torch.randn(3, 4, 5)
        specs = mapping.get_output_specs((x,), {})

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].shape, (3, 4, 5))
        self.assertEqual(specs[1].shape, (3, 4))


class TestDecomposeFunctionalCall(TestCase):
    """Tests for the decompose_functional_call utility function."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_decompose_functional_call_basic(self):
        """Test basic decompose_functional_call usage."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            decompose_functional_call,
        )

        @torch.library.custom_op("test_f2o::decompose_func", mutates_args=())
        def decompose_func(x: torch.Tensor) -> torch.Tensor:
            return x * 3

        @decompose_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::decompose_out", mutates_args=("out",))
        def decompose_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 3)

        @decompose_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.decompose_func,
            out_op=torch.ops.test_f2o.decompose_out,
            out_arg_positions=(0,),
        )

        x = torch.randn(4, 4)
        outputs, _ = decompose_functional_call(
            torch.ops.test_f2o.decompose_func,
            (x,),
            {},
        )

        self.assertEqual(len(outputs), 1)
        expected = x * 3
        torch.testing.assert_close(outputs[0], expected)

    def test_decompose_functional_call_multi_output(self):
        """Test decompose_functional_call with multiple outputs."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            decompose_functional_call,
        )

        @torch.library.custom_op("test_f2o::decompose_multi_func", mutates_args=())
        def decompose_multi_func(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return x * 2, x + 1

        @decompose_multi_func.register_fake
        def _(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.empty_like(x), torch.empty_like(x)

        @torch.library.custom_op(
            "test_f2o::decompose_multi_out", mutates_args=("out1", "out2")
        )
        def decompose_multi_out(
            out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor
        ) -> None:
            out1.copy_(x * 2)
            out2.copy_(x + 1)

        @decompose_multi_out.register_fake
        def _(out1: torch.Tensor, out2: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.decompose_multi_func,
            out_op=torch.ops.test_f2o.decompose_multi_out,
            out_arg_positions=(0, 1),
        )

        x = torch.randn(3, 3)
        outputs, _ = decompose_functional_call(
            torch.ops.test_f2o.decompose_multi_func,
            (x,),
            {},
        )

        self.assertEqual(len(outputs), 2)
        torch.testing.assert_close(outputs[0], x * 2)
        torch.testing.assert_close(outputs[1], x + 1)

    def test_decompose_functional_call_unregistered_raises(self):
        """Test that decompose_functional_call raises for unregistered op."""
        from torch._library.functional_to_out import decompose_functional_call

        @torch.library.custom_op("test_f2o::unregistered_func", mutates_args=())
        def unregistered_func(x: torch.Tensor) -> torch.Tensor:
            return x

        @unregistered_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        x = torch.randn(4, 4)

        with self.assertRaises(ValueError) as ctx:
            decompose_functional_call(
                torch.ops.test_f2o.unregistered_func,
                (x,),
                {},
            )

        self.assertIn("No out variant registered", str(ctx.exception))

    def test_decompose_functional_call_with_kwargs(self):
        """Test decompose_functional_call with keyword arguments."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            decompose_functional_call,
        )

        @torch.library.custom_op("test_f2o::kwargs_func", mutates_args=())
        def kwargs_func(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @kwargs_func.register_fake
        def _(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::kwargs_out", mutates_args=("out",))
        def kwargs_out(out: torch.Tensor, x: torch.Tensor, scale: float = 1.0) -> None:
            out.copy_(x * scale)

        @kwargs_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor, scale: float = 1.0) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.kwargs_func,
            out_op=torch.ops.test_f2o.kwargs_out,
            out_arg_positions=(0,),
        )

        x = torch.randn(4, 4)
        outputs, _ = decompose_functional_call(
            torch.ops.test_f2o.kwargs_func,
            (x,),
            {"scale": 5.0},
        )

        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], x * 5.0)


class TestErrorHandling(TestCase):
    """Tests for error handling in the decomposition pass."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_decompose_graceful_failure(self):
        """Test that decomposition failures are handled gracefully."""
        from torch._library.functional_to_out import register_functional_to_out
        from torch._inductor.fx_passes.decompose_functional_to_out import (
            decompose_functional_to_out,
        )

        @torch.library.custom_op("test_f2o::fail_func", mutates_args=())
        def fail_func(x: torch.Tensor) -> torch.Tensor:
            return x

        @fail_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::fail_out", mutates_args=("out",))
        def fail_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x)

        @fail_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.fail_func,
            out_op=torch.ops.test_f2o.fail_out,
            out_arg_positions=(0,),
        )

        # Create a traced module but DON'T set metadata
        # This should cause _get_output_specs_from_node to fail
        def fn(x):
            return torch.ops.test_f2o.fail_func(x)

        traced = torch.fx.symbolic_trace(fn)
        # Don't set node.meta["val"] - this will cause the pass to fail gracefully

        # Should not raise, just return False
        modified = decompose_functional_to_out(traced.graph)
        # Pass should handle the error gracefully
        self.assertFalse(modified)

    def test_num_outputs_property(self):
        """Test the num_outputs property of FunctionalToOutMapping."""
        from torch._library.functional_to_out import FunctionalToOutMapping

        mapping_single = FunctionalToOutMapping(
            functional_op=None,  # type: ignore
            out_op=None,  # type: ignore
            out_arg_positions=(0,),
        )
        self.assertEqual(mapping_single.num_outputs, 1)

        mapping_double = FunctionalToOutMapping(
            functional_op=None,  # type: ignore
            out_op=None,  # type: ignore
            out_arg_positions=(0, 1),
        )
        self.assertEqual(mapping_double.num_outputs, 2)

        mapping_triple = FunctionalToOutMapping(
            functional_op=None,  # type: ignore
            out_op=None,  # type: ignore
            out_arg_positions=(0, 1, 2),
        )
        self.assertEqual(mapping_triple.num_outputs, 3)


class TestInferOutputSpecsFromFake(TestCase):
    """Tests for _infer_output_specs_from_fake functionality."""

    def setUp(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def tearDown(self):
        from torch._library.functional_to_out import clear_registry

        clear_registry()

    def test_infer_specs_creates_fake_mode(self):
        """Test that spec inference creates fake mode when not present."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
        )

        @torch.library.custom_op("test_f2o::infer_func", mutates_args=())
        def infer_func(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        @infer_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::infer_out", mutates_args=("out",))
        def infer_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x * 2)

        @infer_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.infer_func,
            out_op=torch.ops.test_f2o.infer_out,
            out_arg_positions=(0,),
        )

        mapping = get_out_variant(torch.ops.test_f2o.infer_func)

        # Use real tensors (not fake) - this should trigger fake mode creation
        x = torch.randn(5, 5)
        specs = mapping._infer_output_specs_from_fake((x,), {})

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].shape, (5, 5))
        self.assertEqual(specs[0].dtype, torch.float32)

    def test_infer_specs_with_existing_fake_mode(self):
        """Test spec inference when already in fake mode."""
        from torch._library.functional_to_out import (
            register_functional_to_out,
            get_out_variant,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode

        @torch.library.custom_op("test_f2o::fake_mode_func", mutates_args=())
        def fake_mode_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        @fake_mode_func.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        @torch.library.custom_op("test_f2o::fake_mode_out", mutates_args=("out",))
        def fake_mode_out(out: torch.Tensor, x: torch.Tensor) -> None:
            out.copy_(x + 1)

        @fake_mode_out.register_fake
        def _(out: torch.Tensor, x: torch.Tensor) -> None:
            pass

        register_functional_to_out(
            functional_op=torch.ops.test_f2o.fake_mode_func,
            out_op=torch.ops.test_f2o.fake_mode_out,
            out_arg_positions=(0,),
        )

        mapping = get_out_variant(torch.ops.test_f2o.fake_mode_func)

        # Use existing fake mode
        with FakeTensorMode():
            x = torch.randn(6, 7)
            specs = mapping._infer_output_specs_from_fake((x,), {})

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].shape, (6, 7))


if __name__ == "__main__":
    run_tests()
