"""
Tests for wrapper helper function wiring in pythonify code generation.

These tests verify that when wrapper helper functions are emitted (e.g.,
_inject_effect_tokens, _remove_dupe_args), they are properly wired into
the generated CompiledFunction.forward() and backward() methods.
"""

import unittest

from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
from torch._dynamo.pythonify.ir import (
    AOTAutogradWrapperNode,
    AOTDedupeWrapperNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    KernelLoadNode,
    KernelType,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def make_forward_kernel(kernel_id="kernel_0"):
    """Create a forward kernel node with proper metadata for inductor."""
    return KernelLoadNode(
        kernel_id=kernel_id,
        kernel_type=KernelType.INLINE,
        inline_content="def call(args): return (args[0] * 2,)",
        metadata={"source": "inductor"},
    )


def make_backward_kernel(kernel_id="kernel_1"):
    """Create a backward kernel node with proper metadata for inductor."""
    return KernelLoadNode(
        kernel_id=kernel_id,
        kernel_type=KernelType.INLINE,
        inline_content="def call(args): return (args[0],)",
        metadata={"source": "inductor", "is_backward": True},
    )


class TestForwardWrapperWiring(TestCase):
    """Tests for wrapper wiring in forward() method."""

    def test_effect_tokens_wired_in_forward_pre_call(self):
        """Verify effect tokens wrapper is called before compiled_fn in forward()."""
        ir = RuntimeWrapperIR()

        effect_node = EffectTokensWrapperNode(token_count=2)
        ir.add_node(effect_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _inject_effect_tokens(args):", code)
        self.assertIn("def _strip_effect_tokens(outputs):", code)
        self.assertIn("_processed_args = list(args)", code)
        self.assertIn("_processed_args = _inject_effect_tokens(_processed_args)", code)

    def test_dedupe_wrapper_wired_in_forward_pre_call(self):
        """Verify dedupe wrapper is called before compiled_fn in forward()."""
        ir = RuntimeWrapperIR()

        dedupe_node = AOTDedupeWrapperNode(
            needs_post_compile=True,
            keep_arg_mask=[True, False, True],
            add_dupe_map=[0, 0, 1],
        )
        ir.add_node(dedupe_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=3,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _remove_dupe_args(args):", code)
        self.assertIn("_processed_args = list(args)", code)
        self.assertIn("_processed_args = _remove_dupe_args(_processed_args)", code)

    def test_runtime_wrapper_wired_in_forward_pre_call(self):
        """Verify runtime wrapper (detach) is called before compiled_fn in forward()."""
        ir = RuntimeWrapperIR()

        runtime_node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[0, 2],
            disable_amp=False,
        )
        ir.add_node(runtime_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=3,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _detach_inputs(args):", code)
        self.assertIn("_processed_args = list(args)", code)
        self.assertIn("_processed_args = _detach_inputs(_processed_args)", code)

    def test_debug_assert_wired_in_forward_pre_call(self):
        """Verify debug assert wrapper is called before compiled_fn in forward()."""
        ir = RuntimeWrapperIR()

        debug_node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False, True],
        )
        ir.add_node(debug_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=3,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _assert_requires_grad(args):", code)
        self.assertIn("_processed_args = list(args)", code)
        self.assertIn("_assert_requires_grad(_processed_args)", code)

    def test_no_wrappers_uses_args_directly(self):
        """Verify that without wrappers, forward() uses args directly."""
        ir = RuntimeWrapperIR()

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertNotIn("_processed_args", code)
        self.assertIn("compiled_fn(list(args))", code)


class TestForwardOutputWrapperWiring(TestCase):
    """Tests for wrapper wiring on forward() outputs."""

    def test_effect_tokens_stripped_from_output(self):
        """Verify effect tokens are stripped from forward outputs."""
        ir = RuntimeWrapperIR()

        effect_node = EffectTokensWrapperNode(token_count=2)
        ir.add_node(effect_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("_result = _strip_effect_tokens(_result)", code)
        self.assertIn("return _result", code)

    def test_fakified_out_wrapper_fixes_strides(self):
        """Verify fakified out wrapper fixes output strides."""
        ir = RuntimeWrapperIR()

        fakified_node = FakifiedOutWrapperNode(
            out_metas=[{"shape": (2, 3), "dtype": "torch.float32"}],
            fwd_output_strides=[(3, 1)],
        )
        ir.add_node(fakified_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def _fix_output_strides(outputs", code)
        self.assertIn("_result = _fix_output_strides(_result)", code)


class TestBackwardWrapperWiring(TestCase):
    """Tests for wrapper wiring in backward() method."""

    def test_effect_tokens_wired_in_backward(self):
        """Verify effect tokens wrapper is applied in backward()."""
        ir = RuntimeWrapperIR()

        effect_node = EffectTokensWrapperNode(token_count=2)
        ir.add_node(effect_node)

        ir.add_node(make_forward_kernel())
        ir.add_node(make_backward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def backward(ctx, *grad_outputs):", code)
        self.assertIn("backward_inputs = _inject_effect_tokens(backward_inputs)", code)
        self.assertIn("_bw_result = _strip_effect_tokens(_bw_result)", code)

    def test_dedupe_reinstates_grads_in_backward(self):
        """Verify dedupe wrapper reinstates duplicate gradients in backward()."""
        ir = RuntimeWrapperIR()

        dedupe_node = AOTDedupeWrapperNode(
            needs_post_compile=True,
            keep_arg_mask=[True, False, True],
            add_dupe_map=[0, 0, 1],
        )
        ir.add_node(dedupe_node)

        ir.add_node(make_forward_kernel())
        ir.add_node(make_backward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=3,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("def backward(ctx, *grad_outputs):", code)
        self.assertIn("def _add_dupe_args(args):", code)
        self.assertIn("_bw_result = _add_dupe_args(_bw_result)", code)


class TestMultipleWrappersWiring(TestCase):
    """Tests for multiple wrappers being wired together."""

    def test_multiple_wrappers_in_correct_order(self):
        """Verify multiple wrappers are applied in correct order in forward()."""
        ir = RuntimeWrapperIR()

        debug_node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False],
        )
        ir.add_node(debug_node)

        dedupe_node = AOTDedupeWrapperNode(
            needs_post_compile=True,
            keep_arg_mask=[True, True],
            add_dupe_map=[0, 1],
        )
        ir.add_node(dedupe_node)

        effect_node = EffectTokensWrapperNode(token_count=1)
        ir.add_node(effect_node)

        runtime_node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[0],
            disable_amp=False,
        )
        ir.add_node(runtime_node)

        ir.add_node(make_forward_kernel())

        aot_node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=2,
        )
        ir.add_node(aot_node)

        visitor = PythonCodeGenVisitor()
        visitor.prescan_ir(ir)
        ir.accept_all(visitor)
        code = visitor.get_code()

        self.assertIn("_processed_args = list(args)", code)

        debug_pos = code.find("_assert_requires_grad(_processed_args)")
        dedupe_pos = code.find("_processed_args = _remove_dupe_args(_processed_args)")
        effect_pos = code.find("_processed_args = _inject_effect_tokens(_processed_args)")
        detach_pos = code.find("_processed_args = _detach_inputs(_processed_args)")

        self.assertTrue(debug_pos < dedupe_pos, "Debug assert should come before dedupe")
        self.assertTrue(dedupe_pos < effect_pos, "Dedupe should come before effect tokens")
        self.assertTrue(effect_pos < detach_pos, "Effect tokens should come before detach")


class TestWrapperWiringProperties(TestCase):
    """Tests for helper properties that determine wrapper wiring behavior."""

    def test_has_forward_wrappers_property(self):
        """Verify _has_forward_wrappers property works correctly."""
        visitor = PythonCodeGenVisitor()
        self.assertFalse(visitor._has_forward_wrappers)

        visitor._effect_token_count = 2
        self.assertTrue(visitor._has_forward_wrappers)

    def test_has_forward_output_wrappers_property(self):
        """Verify _has_forward_output_wrappers property works correctly."""
        visitor = PythonCodeGenVisitor()
        self.assertFalse(visitor._has_forward_output_wrappers)

        visitor._has_fakified_wrapper = True
        self.assertTrue(visitor._has_forward_output_wrappers)


if __name__ == "__main__":
    run_tests()
