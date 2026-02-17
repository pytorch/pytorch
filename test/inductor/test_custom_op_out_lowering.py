# Owner(s): ["module: inductor"]
"""
Tests for inductor lowering of functional custom ops to out-variant via ExternKernelOut.

Verifies that custom ops with both functional and .out overloads are lowered
to ExternKernelOut (should_allocate=True) for buffer reuse, and that the config
flag properly gates the behavior.
"""
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")

DEVICES = ("cpu", GPU_TYPE) if HAS_GPU else ("cpu",)


@instantiate_parametrized_tests
class TestCustomOpOutLowering(InductorTestCase):
    """Tests for lowering functional custom ops to out-variant ExternKernelOut."""

    def _register_add_one_ops(self, lib):
        """Register a simple add_one op with functional + .out overloads."""
        lib.define("add_one(Tensor x) -> Tensor")
        lib.define(
            "add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)",
            tags=(torch.Tag.out_variant,),
        )

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

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_add_one_lowered_to_out(self, device):
        """Test that a simple functional op gets lowered to its out-variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            self.assertIn(".out(", code)
            self.assertNotIn(".default(", code)

    def _register_split_add_ops(self, lib):
        """Register a split_add op returning two tensors with functional + .out overloads."""
        lib.define("split_add(Tensor x, float a, float b) -> (Tensor, Tensor)")
        lib.define(
            "split_add.out(Tensor x, float a, float b, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))",
            tags=(torch.Tag.out_variant,),
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

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_multi_output_lowered_to_out(self, device):
        """Test a two-output functional op gets lowered to its .out variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            def f(x):
                a, b = torch.ops.mylib.split_add(x, 1.0, 2.0)
                return a + b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertIn(".out(", code)
            self.assertIn("out0=", code)
            self.assertIn("out1=", code)

    # ---- _out_variant.py unit tests ----

    def test_to_out_variant_finds_add_one(self):
        """Test that to_out_variant() finds the .out overload for single-output op."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    def test_to_out_variant_finds_split_add(self):
        """Test that to_out_variant() finds .out for a two-output op."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)

            from torch._library._out_variant import to_out_variant

            found = to_out_variant(func_op.default)
            self.assertIsNotNone(found)
            self.assertEqual(found, out_op)

    def test_to_out_variant_raises_for_mutable_op(self):
        """Test that to_out_variant() raises RuntimeError for mutable ops."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("inplace_sin(Tensor(a!) x) -> ()")

            def _inplace_sin_impl(x):
                x.sin_()

            lib.impl("inplace_sin", _inplace_sin_impl, "CompositeImplicitAutograd")

            from torch._library._out_variant import to_out_variant

            with self.assertRaises(RuntimeError):
                to_out_variant(torch.ops.mylib.inplace_sin.default)

    # ---- Additional integration tests ----

    def _register_rms_norm_ops(self, lib):
        """Register rms_norm with named 'result' out arg (follows vLLM pattern)."""
        lib.define("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor")
        lib.define(
            "rms_norm.out(Tensor input, Tensor weight, float epsilon, *, Tensor(a!) result) -> Tensor(a!)",
            tags=(torch.Tag.out_variant,),
        )

        def _rms_norm_impl(input, weight, epsilon):
            variance = input.pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + epsilon)
            return input * weight

        def _rms_norm_out_impl(input, weight, epsilon, *, result):
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

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_rms_norm_lowered_to_out(self, device):
        """Test rms_norm with named 'result' out arg (flexible out-arg naming)."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_rms_norm_ops(lib)

            def f(x, weight):
                return torch.ops.mylib.rms_norm(x, weight, 1e-5)

            x = torch.randn(2, 4, device=device)
            weight = torch.randn(4, device=device)
            eager_out = f(x, weight)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x, weight
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertIn(".out(", code)
            self.assertIn("result=", code)
            self.assertNotIn(".default(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_chained_ops_buffer_reuse(self, device):
        """Test that chained custom ops participate in buffer reuse.

        Pattern: x -> add_one -> (*2) -> add_one -> (+1) -> add_one -> output.
        With ExternKernelOut, fused pointwise kernels between custom ops can
        reuse dead buffers, producing '# reuse' comments in generated code.
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

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertIn(".out(", code)
            self.assertIn("# reuse", code)
            self.assertIn("empty_strided", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_op_without_out_variant_falls_through(self, device):
        """Test that ops without an out-variant fall through to FallbackKernel."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("no_out_op(Tensor x) -> Tensor")

            def _impl(x):
                return x + 1

            lib.impl("no_out_op", _impl, "CompositeExplicitAutograd")

            @torch.library.register_fake("mylib::no_out_op", lib=lib)
            def _fake(x):
                return x.new_empty(x.shape)

            def f(x):
                return torch.ops.mylib.no_out_op(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

    @inductor_config.patch(lower_custom_ops_to_out_variant=True)
    @parametrize("device", DEVICES)
    def test_multi_output_buffer_reuse(self, device):
        """Test buffer reuse for multi-output ops.

        Pattern: split_add -> use only one output -> another op with same shape.
        The unused output buffer should be reusable.
        """
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_split_add_ops(lib)
            self._register_add_one_ops(lib)

            def f(x):
                a, b = torch.ops.mylib.split_add(x, 1.0, 2.0)
                c = torch.sin(a)
                d = torch.ops.mylib.add_one(c)
                return d + b

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertIn(".out(", code)

    @inductor_config.patch(lower_custom_ops_to_out_variant=False)
    @parametrize("device", DEVICES)
    def test_disabled_falls_back_to_fallback_kernel(self, device):
        """Test that disabling the config causes the op to go through FallbackKernel."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            func_op, out_op = self._register_add_one_ops(lib)

            def f(x):
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)
            self.assertNotIn(".out(", code)
            self.assertIn(".default(", code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
