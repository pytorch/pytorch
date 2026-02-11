"""
Tests for inductor lowering of functional custom ops to out-variant via ExternKernelOut.

This tests the approach suggested by eellison: instead of a post-grad pass that
decomposes functional→empty+out-variant (ExternKernelAlloc, should_allocate=False),
we lower at inductor time to ExternKernelOut (should_allocate=True) which participates
in Inductor's AllocateLine.plan() buffer reuse.

Test strategy:
  1. Register test ops with both functional and .out overloads
  2. Compile with lower_custom_ops_to_out_variant=True
  3. Verify generated code calls the .out overload
  4. Verify correctness (output matches eager)
  5. Verify buffer reuse via code inspection (no extra allocations)
"""

import unittest

import torch
import torch._dynamo.testing
import torch._inductor.config as inductor_config
import torch._inductor.test_case
from torch._inductor.utils import run_and_get_code

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


class TestCustomOpOutLowering(torch._inductor.test_case.TestCase):
    """Tests for lowering functional custom ops to out-variant ExternKernelOut."""

    def _register_silu_and_mul_ops(self, lib):
        """Register a silu_and_mul op with functional + .out overloads.

        Mimics the vLLM silu_and_mul pattern:
        - Functional: silu_and_mul(Tensor input) -> Tensor
        - Out variant: silu_and_mul.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)
        """
        # Define the functional overload
        lib.define("silu_and_mul(Tensor input) -> Tensor")

        # Define the .out overload (follows PyTorch convention: out is a kwarg)
        lib.define("silu_and_mul.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)")

        def _silu_and_mul_impl(input: torch.Tensor) -> torch.Tensor:
            d = input.shape[-1] // 2
            return torch.nn.functional.silu(input[..., :d]) * input[..., d:]

        def _silu_and_mul_out_impl(
            input: torch.Tensor, *, out: torch.Tensor
        ) -> torch.Tensor:
            d = input.shape[-1] // 2
            result = torch.nn.functional.silu(input[..., :d]) * input[..., d:]
            out.copy_(result)
            return out

        lib.impl("silu_and_mul", _silu_and_mul_impl, "CompositeExplicitAutograd")
        lib.impl(
            "silu_and_mul.out", _silu_and_mul_out_impl, "CompositeExplicitAutograd"
        )

        # Fake tensor impl for the functional variant
        @torch.library.register_fake("mylib::silu_and_mul", lib=lib)
        def _silu_and_mul_fake(input):
            d = input.shape[-1] // 2
            return input.new_empty(*input.shape[:-1], d)

        return torch.ops.mylib.silu_and_mul, torch.ops.mylib.silu_and_mul.out

    def _register_add_one_ops(self, lib):
        """Register a simple add_one op with functional + .out overloads.

        Simple single-tensor-in, single-tensor-out pattern for basic testing.
        """
        lib.define("add_one(Tensor x) -> Tensor")
        lib.define("add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)")

        def _add_one_impl(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def _add_one_out_impl(x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
            out.copy_(x + 1)
            return out

        lib.impl("add_one", _add_one_impl, "CompositeExplicitAutograd")
        lib.impl("add_one.out", _add_one_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::add_one", lib=lib)
        def _add_one_fake(x):
            return x.new_empty(x.shape)

        return torch.ops.mylib.add_one, torch.ops.mylib.add_one.out

    def _register_rms_norm_ops(self, lib):
        """Register a rms_norm op with functional + .out overloads.

        Follows the vLLM rms_norm pattern with named 'result' out arg:
        - Functional: rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor
        - Out variant: rms_norm.out(Tensor input, Tensor weight, float epsilon, *, Tensor(a!) result) -> Tensor(a!)
        """
        lib.define("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor")
        lib.define(
            "rms_norm.out(Tensor input, Tensor weight, float epsilon, *, Tensor(a!) result) -> Tensor(a!)"
        )

        def _rms_norm_impl(
            input: torch.Tensor, weight: torch.Tensor, epsilon: float
        ) -> torch.Tensor:
            variance = input.pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + epsilon)
            return input * weight

        def _rms_norm_out_impl(
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
            *,
            result: torch.Tensor,
        ) -> torch.Tensor:
            variance = input.pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + epsilon)
            result.copy_(input * weight)
            return result

        lib.impl("rms_norm", _rms_norm_impl, "CompositeExplicitAutograd")
        lib.impl("rms_norm.out", _rms_norm_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::rms_norm", lib=lib)
        def _rms_norm_fake(input, weight, epsilon):
            return input.new_empty(input.shape)

        return torch.ops.mylib.rms_norm, torch.ops.mylib.rms_norm.out

    def _register_split_add_ops(self, lib):
        """Register a split_add op that returns two tensors.

        - Functional: split_add(Tensor x, float a, float b) -> (Tensor, Tensor)
          returns (x + a, x + b)
        - Out variant: split_add.out(Tensor x, float a, float b, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
        """
        lib.define("split_add(Tensor x, float a, float b) -> (Tensor, Tensor)")
        lib.define(
            "split_add.out(Tensor x, float a, float b, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))"
        )

        def _split_add_impl(x, a, b):
            return (x + a, x + b)

        def _split_add_out_impl(x, a, b, *, out0, out1):
            out0.copy_(x + a)
            out1.copy_(x + b)
            return (out0, out1)

        lib.impl("split_add", _split_add_impl, "CompositeExplicitAutograd")
        lib.impl("split_add.out", _split_add_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::split_add", lib=lib)
        def _split_add_fake(x, a, b):
            return (x.new_empty(x.shape), x.new_empty(x.shape))

        return torch.ops.mylib.split_add, torch.ops.mylib.split_add.out

    # ---- _out_variant.py unit tests ----

    def test_to_out_variant_finds_silu_and_mul(self):
        """Test that to_out_variant() correctly finds the .out overload."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_silu_and_mul_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    def test_to_out_variant_finds_add_one(self):
        """Test that to_out_variant() finds the .out overload for add_one."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    def test_to_out_variant_finds_rms_norm(self):
        """Test that to_out_variant() finds the .out overload for rms_norm."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_rms_norm_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    def test_to_out_variant_returns_none_for_mutable_op(self):
        """Test that to_out_variant() returns None for ops that are already mutable."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("inplace_sin(Tensor(a!) x) -> ()")

            def _inplace_sin_impl(x):
                x.sin_()

            lib.impl("inplace_sin", _inplace_sin_impl, "CompositeImplicitAutograd")

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(torch.ops.mylib.inplace_sin.default)
            self.assertIsNone(found)

    def test_check_out_variant(self):
        """Test check_out_variant() for debugging/validation."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            from torch._library._out_variant import check_out_variant

            # Should not raise
            check_out_variant(func_op.default, out_op)

    def test_to_out_variant_finds_split_add(self):
        """Test that to_out_variant() finds .out for a two-output op."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    # ---- Inductor lowering integration tests ----

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_add_one_lowered_to_out(self):
        """Test that a simple functional op gets lowered to its out-variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            # Verify the generated code calls the .out variant, not .default
            self.assertIn(".out(", code)
            self.assertNotIn(".default(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_silu_and_mul_lowered_to_out(self):
        """Test silu_and_mul (vLLM-style) lowered to out-variant.

        This is the key vLLM use case: silu_and_mul.default → silu_and_mul.out.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_silu_and_mul_ops(lib)

            def f(x):
                return torch.ops.mylib.silu_and_mul(x)

            x = torch.randn(2, 8)  # Last dim must be even for silu_and_mul
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            # Verify the generated code calls the .out variant, not .default
            self.assertIn(".out(", code)
            self.assertNotIn(".default(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_rms_norm_lowered_to_out(self):
        """Test rms_norm with named 'result' out arg gets lowered correctly.

        This tests the flexible out-arg naming (not hardcoded 'out=').
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_rms_norm_ops(lib)

            def f(x, weight):
                return torch.ops.mylib.rms_norm(x, weight, 1e-5)

            x = torch.randn(2, 4)
            weight = torch.randn(4)
            eager_out = f(x, weight)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x, weight
            )
            self.assertEqual(compiled_out, eager_out)

            # Verify the generated code calls the .out variant with named arg
            self.assertIn(".out(", code)
            self.assertIn("result=", code)
            self.assertNotIn(".default(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_chained_ops_buffer_reuse(self):
        """Test that chained custom ops participate in Inductor's buffer reuse.

        Pattern: x → add_one → (*2) → add_one → (+1) → add_one → output

        With ExternKernelOut (should_allocate=True), the pre-allocated output
        buffers participate in AllocateLine.plan() buffer reuse. Fused pointwise
        kernels between custom ops can reuse the dead ExternKernelOut buffers,
        producing '# reuse' comments in the generated code.

        Without this feature (ExternKernelAlloc, should_allocate=False), the
        custom op allocates internally and the buffer is opaque to the planner.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                y1 = torch.ops.mylib.add_one(x)
                y2 = y1 * 2.0
                y3 = torch.ops.mylib.add_one(y2)
                y4 = y3 + 1.0
                y5 = torch.ops.mylib.add_one(y4)
                return y5

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            # Correctness
            self.assertEqual(compiled_out, eager_out)

            # Out-variant lowering happened: .out( calls in generated code
            self.assertIn(".out(", code)

            # Buffer reuse via AllocateLine.plan(): fused pointwise kernels
            # reuse dead ExternKernelOut buffers, indicated by '# reuse'
            self.assertIn("# reuse", code)

            # Pre-allocation visible to memory planner: empty_strided used for
            # custom op output buffers (ExternKernelOut.should_allocate()=True)
            self.assertIn("empty_strided", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=False)
    def test_chained_ops_no_out_variant_without_feature(self):
        """Verify that without the feature, custom ops use .default (opaque alloc).

        This is the baseline: FallbackKernel → ExternKernelAlloc, which calls
        the functional op directly. The output is allocated inside the kernel,
        invisible to Inductor's memory planner.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                y1 = torch.ops.mylib.add_one(x)
                y2 = y1 * 2.0
                y3 = torch.ops.mylib.add_one(y2)
                y4 = y3 + 1.0
                y5 = torch.ops.mylib.add_one(y4)
                return y5

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            # Without the feature: no .out( calls, uses .default instead
            self.assertNotIn(".out(", code)
            self.assertIn(".default(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=False)
    def test_disabled_falls_back_to_fallback_kernel(self):
        """Test that disabling the config causes the op to go through FallbackKernel."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_op_without_out_variant_falls_through(self):
        """Test that ops without an out-variant fall through to FallbackKernel."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # Only define functional, no .out overload
            lib.define("no_out_op(Tensor x) -> Tensor")

            def _impl(x):
                return x + 1

            lib.impl("no_out_op", _impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("mylib::no_out_op", lib=lib)
            def _fake(x):
                return x.new_empty(x.shape)

            def f(x):
                return torch.ops.mylib.no_out_op(x)

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_multi_output_lowered_to_out(self):
        """Test a two-output functional op gets lowered to its .out variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            def f(x):
                a, b = torch.ops.mylib.split_add(x, 1.0, 2.0)
                return a + b

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            # Out-variant lowering happened
            self.assertIn(".out(", code)
            # Both out args referenced
            self.assertIn("out0=", code)
            self.assertIn("out1=", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_multi_output_both_outputs_used_separately(self):
        """Test that both outputs of a multi-output op are usable."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            def f(x):
                a, b = torch.ops.mylib.split_add(x, 3.0, 7.0)
                return a, b

            x = torch.randn(4, 4)
            eager_a, eager_b = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            compiled_a, compiled_b = compiled_out
            self.assertEqual(compiled_a, eager_a)
            self.assertEqual(compiled_b, eager_b)
            self.assertIn(".out(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_multi_output_buffer_reuse(self):
        """Test buffer reuse for multi-output ops.

        Pattern: split_add → use only one output → another op with same shape
        The unused output buffer should be reusable.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)
            self._register_add_one_ops(lib)

            def f(x):
                a, b = torch.ops.mylib.split_add(x, 1.0, 2.0)
                c = torch.sin(a)  # consume a, a's buffer dies
                d = torch.ops.mylib.add_one(c)  # same shape → reuse a's buffer?
                return d + b

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertIn(".out(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_with_mixed_ops(self):
        """Test a graph mixing custom ops (lowered to out) with built-in ops."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                y = torch.ops.mylib.add_one(x)
                z = torch.sin(y)
                w = torch.ops.mylib.add_one(z)
                return w + x

            x = torch.randn(4, 4)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)


@requires_cuda
class TestVLLMSiluAndMulGPU(torch._inductor.test_case.TestCase):
    """End-to-end tests with real vLLM silu_and_mul CUDA kernel.

    These tests verify that the inductor out-variant lowering works with
    production vLLM custom ops on GPU, not just mock Python ops on CPU.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import vllm._custom_ops  # noqa: F401

            cls.has_vllm = True
        except ImportError:
            cls.has_vllm = False

    def setUp(self):
        super().setUp()
        if not self.has_vllm:
            self.skipTest("vLLM not installed")
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def test_to_out_variant_finds_vllm_silu_and_mul(self):
        """Verify to_out_variant() discovers _C.silu_and_mul.out from .default."""
        from torch._library._out_variant import get_out_arg_names, to_out_variant

        func_op = torch.ops._C.silu_and_mul.default
        out_op = to_out_variant(func_op)
        self.assertIsNotNone(out_op)
        self.assertEqual(out_op, torch.ops._C.silu_and_mul.out)
        self.assertEqual(get_out_arg_names(out_op), ["out"])

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_single_silu_and_mul_lowered_to_out(self):
        """Single vLLM silu_and_mul lowered to .out variant on CUDA."""

        def f(x):
            return torch.ops._C.silu_and_mul(x)

        x = torch.randn(4, 16, device="cuda")
        eager_out = f(x)

        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, backend="inductor", fullgraph=True), x
        )

        # Correctness: real CUDA kernel output matches
        self.assertTrue(torch.allclose(compiled_out, eager_out))

        # Out-variant lowering happened
        self.assertIn(".out(", code)
        self.assertIn("out=", code)
        self.assertNotIn(".default(", code)

        # Inductor pre-allocates the output (should_allocate=True)
        self.assertIn("empty_strided", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_chained_silu_and_mul_buffer_reuse(self):
        """Chained silu_and_mul ops with buffer reuse on CUDA.

        Pattern: x(4×16) → silu_and_mul → y(4×8) → cat(y,y) → y2(4×16)
                 → silu_and_mul → z(4×8)

        The first silu_and_mul's pre-allocated buf0 (4×8) dies after the
        cat kernel consumes it. The second silu_and_mul's output (also 4×8)
        should reuse buf0's storage via AllocateLine.plan().
        """

        def f(x):
            y = torch.ops._C.silu_and_mul(x)
            y2 = torch.cat([y, y], dim=-1)
            z = torch.ops._C.silu_and_mul(y2)
            return z

        x = torch.randn(4, 16, device="cuda")
        eager_out = f(x)

        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, backend="inductor", fullgraph=True), x
        )

        # Correctness
        self.assertTrue(torch.allclose(compiled_out, eager_out))

        # Both calls lowered to .out variant
        self.assertEqual(code.count(".out("), 2)

        # Buffer reuse: first silu_and_mul's output buffer reused
        self.assertIn("# reuse", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=False)
    def test_baseline_no_out_variant(self):
        """Baseline: feature OFF uses .default (opaque alloc), no buffer reuse."""

        def f(x):
            y = torch.ops._C.silu_and_mul(x)
            y2 = torch.cat([y, y], dim=-1)
            z = torch.ops._C.silu_and_mul(y2)
            return z

        x = torch.randn(4, 16, device="cuda")
        eager_out = f(x)

        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, backend="inductor", fullgraph=True), x
        )

        # Correctness still holds
        self.assertTrue(torch.allclose(compiled_out, eager_out))

        # Uses .default, not .out
        self.assertNotIn(".out(", code)
        self.assertEqual(code.count(".default("), 2)

        # No buffer reuse for custom op outputs
        self.assertNotIn("# reuse", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_silu_and_mul_realistic_shapes(self):
        """Test with realistic LLM shapes (batch=32, hidden=4096)."""

        def f(x):
            return torch.ops._C.silu_and_mul(x)

        x = torch.randn(32, 4096 * 2, device="cuda", dtype=torch.float16)
        eager_out = f(x)

        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, backend="inductor", fullgraph=True), x
        )

        self.assertTrue(torch.allclose(compiled_out, eager_out, atol=1e-3, rtol=1e-3))
        self.assertIn(".out(", code)
        self.assertIn("empty_strided", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    def test_silu_and_mul_with_other_ops(self):
        """silu_and_mul mixed with standard ops in a compiled graph."""

        def f(x, weight):
            y = torch.ops._C.silu_and_mul(x)
            z = torch.matmul(y, weight)
            return z

        hidden = 8
        x = torch.randn(4, hidden * 2, device="cuda")
        weight = torch.randn(hidden, hidden, device="cuda")
        eager_out = f(x, weight)

        compiled_out, (code,) = run_and_get_code(
            torch.compile(f, backend="inductor", fullgraph=True), x, weight
        )

        self.assertTrue(torch.allclose(compiled_out, eager_out, atol=1e-4, rtol=1e-4))
        self.assertIn(".out(", code)


class TestOutVariantUtilities(torch._inductor.test_case.TestCase):
    """Tests specifically for the _out_variant.py utility module."""

    def test_get_out_arg_names(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("test_op.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)")

            def _impl(x, *, out):
                out.copy_(x)
                return out

            lib.impl("test_op.out", _impl, "CompositeImplicitAutograd")

            from torch._library._out_variant import get_out_arg_names

            names = get_out_arg_names(torch.ops.mylib.test_op.out)
            self.assertEqual(names, ["out"])

    def test_get_out_arg_names_custom_name(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("test_op2.out(Tensor x, *, Tensor(a!) result) -> Tensor(a!)")

            def _impl(x, *, result):
                result.copy_(x)
                return result

            lib.impl("test_op2.out", _impl, "CompositeImplicitAutograd")

            from torch._library._out_variant import get_out_arg_names

            names = get_out_arg_names(torch.ops.mylib.test_op2.out)
            self.assertEqual(names, ["result"])

    def test_get_out_arg_count(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "test_op3.out(Tensor x, *, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))"
            )

            def _impl(x, *, out1, out2):
                out1.copy_(x)
                out2.copy_(x)
                return out1, out2

            lib.impl("test_op3.out", _impl, "CompositeImplicitAutograd")

            from torch._library._out_variant import get_out_arg_count

            count = get_out_arg_count(torch.ops.mylib.test_op3.out)
            self.assertEqual(count, 2)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
