"""
Tests for the binary code generation backend of pythonify.

This module tests that BinaryCodeGenVisitor correctly handles all IR node types,
including the new wrapper nodes that model AOTAutograd post-compile wrappers.
The binary backend should ignore wrapper nodes without affecting existing behavior.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch._dynamo.pythonify.gen_binary import (
    BinaryCodeGenVisitor,
    generate_binary_wrapper,
    ArgumentExtractor,
    GuardChecker,
    CompiledWrapper,
)
from torch._dynamo.pythonify.ir import (
    AOTDedupeWrapperNode,
    AOTDispatchSubclassWrapperNode,
    AOTSyntheticBaseWrapperNode,
    ArgumentExtractionNode,
    ArgumentSource,
    AOTAutogradWrapperNode,
    CallableInvocationNode,
    CUDAGraphSetupNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    GuardCheckNode,
    GuardType,
    KernelLoadNode,
    KernelType,
    ReturnResultNode,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
    WrapperStackSegment,
)


class TestBinaryCodeGenVisitorWrapperNodes(TestCase):
    """
    Tests verifying that BinaryCodeGenVisitor correctly ignores wrapper nodes.

    The wrapper nodes are metadata for Python codegen and should be safely
    ignored by the binary backend without affecting the compiled wrapper.
    """

    def test_effect_tokens_wrapper_ignored(self):
        """EffectTokensWrapperNode should return None and not affect wrapper."""
        visitor = BinaryCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=3)
        result = visitor.visit_effect_tokens_wrapper(node)
        self.assertIsNone(result)
        wrapper = visitor.get_wrapper()
        self.assertEqual(len(wrapper.extractors), 0)

    def test_aot_dispatch_subclass_wrapper_ignored(self):
        """AOTDispatchSubclassWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            subclass_inp_meta={"some": "data"},
            num_fw_outs_saved_for_bw=2,
        )
        result = visitor.visit_aot_dispatch_subclass_wrapper(node)
        self.assertIsNone(result)

    def test_functionalized_rng_wrapper_ignored(self):
        """FunctionalizedRngRuntimeWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=2,
        )
        result = visitor.visit_functionalized_rng_runtime_wrapper(node)
        self.assertIsNone(result)

    def test_fakified_out_wrapper_ignored(self):
        """FakifiedOutWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas=["meta1", "meta2"],
            fwd_output_strides=[(2, 1), (4, 2, 1)],
        )
        result = visitor.visit_fakified_out_wrapper(node)
        self.assertIsNone(result)

    def test_runtime_wrapper_ignored(self):
        """RuntimeWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[0, 2, 3],
            disable_amp=True,
        )
        result = visitor.visit_runtime_wrapper(node)
        self.assertIsNone(result)

    def test_aot_dedupe_wrapper_ignored(self):
        """AOTDedupeWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            keep_arg_mask=[True, False, True],
            add_dupe_map=[(1, 0)],
            needs_post_compile=True,
        )
        result = visitor.visit_aot_dedupe_wrapper(node)
        self.assertIsNone(result)

    def test_aot_synthetic_base_wrapper_ignored(self):
        """AOTSyntheticBaseWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            synthetic_base_info={"base_info": "data"},
            aliased_arg_idx_with_metadata_mutations=[1, 2],
            needs_post_compile=True,
        )
        result = visitor.visit_aot_synthetic_base_wrapper(node)
        self.assertIsNone(result)

    def test_debug_assert_wrapper_ignored(self):
        """DebugAssertWrapperNode should return None."""
        visitor = BinaryCodeGenVisitor()
        node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False, True],
        )
        result = visitor.visit_debug_assert_wrapper(node)
        self.assertIsNone(result)


class TestBinaryCodeGenMixedIR(TestCase):
    """
    Tests verifying that binary codegen works correctly with mixed IR containing
    both wrapper nodes and standard nodes (extraction, guards, invocation, etc).
    """

    def test_wrapper_nodes_do_not_affect_argument_extraction(self):
        """
        Wrapper nodes interspersed with argument extraction nodes should not
        affect the extracted arguments in the compiled wrapper.
        """
        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="x",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        ))

        ir.add_node(EffectTokensWrapperNode(token_count=2))

        ir.add_node(ArgumentExtractionNode(
            name="y",
            source=ArgumentSource.F_LOCALS,
            access_path="y",
        ))

        ir.add_node(RuntimeWrapperNode(indices_of_inps_to_detach=[0]))

        ir.add_node(ArgumentExtractionNode(
            name="z",
            source=ArgumentSource.F_GLOBALS,
            access_path="z",
        ))

        wrapper = generate_binary_wrapper(ir)
        self.assertEqual(len(wrapper.extractors), 3)
        self.assertEqual(wrapper.extractors[0].name, "x")
        self.assertEqual(wrapper.extractors[1].name, "y")
        self.assertEqual(wrapper.extractors[2].name, "z")

    def test_wrapper_nodes_do_not_affect_guards(self):
        """
        Wrapper nodes interspersed with guard nodes should not affect guards.
        """
        ir = RuntimeWrapperIR()

        ir.add_node(GuardCheckNode(
            guard_type=GuardType.SHAPE,
            target_name="x",
            condition="x.shape[0] == 4",
            expected_value=4,
            dimension=0,
        ))

        ir.add_node(DebugAssertWrapperNode(flat_requires_grad=[True]))

        ir.add_node(GuardCheckNode(
            guard_type=GuardType.DTYPE,
            target_name="x",
            condition="x.dtype == torch.float32",
            expected_value=torch.float32,
        ))

        wrapper = generate_binary_wrapper(ir)
        self.assertEqual(len(wrapper.guards), 2)
        self.assertEqual(wrapper.guards[0].guard_type, GuardType.SHAPE)
        self.assertEqual(wrapper.guards[1].guard_type, GuardType.DTYPE)

    def test_full_ir_with_all_wrapper_types(self):
        """
        Test an IR containing all wrapper types alongside standard nodes.
        The wrapper nodes should be silently ignored.
        """
        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="input",
            source=ArgumentSource.F_LOCALS,
            access_path="input",
        ))

        ir.add_node(GuardCheckNode(
            guard_type=GuardType.SHAPE,
            target_name="input",
            condition="input.shape[0] == 8",
            expected_value=8,
            dimension=0,
        ))

        ir.add_node(EffectTokensWrapperNode(token_count=1))
        ir.add_node(AOTDispatchSubclassWrapperNode())
        ir.add_node(FunctionalizedRngRuntimeWrapperNode(is_rng_op_functionalized=True))
        ir.add_node(FakifiedOutWrapperNode())
        ir.add_node(RuntimeWrapperNode(indices_of_inps_to_detach=[]))
        ir.add_node(AOTDedupeWrapperNode(needs_post_compile=True))
        ir.add_node(AOTSyntheticBaseWrapperNode(needs_post_compile=True))
        ir.add_node(DebugAssertWrapperNode())

        ir.add_node(AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            num_inputs=1,
            num_outputs=1,
        ))

        ir.add_node(CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["input"],
            result_name="result",
        ))

        ir.add_node(ReturnResultNode(
            result_name="result",
            expose_as="y",
        ))

        wrapper = generate_binary_wrapper(ir)
        self.assertEqual(len(wrapper.extractors), 1)
        self.assertEqual(len(wrapper.guards), 1)
        self.assertEqual(wrapper.result_name, "result")
        self.assertEqual(wrapper.expose_as, "y")
        self.assertFalse(wrapper.is_autograd_function)


class TestBinaryCodeGenWrapperMetadata(TestCase):
    """
    Tests verifying that wrapper_stack_order and wrapper_stack_metadata in
    RuntimeWrapperIR are preserved and accessible, even though the binary
    codegen doesn't emit code for them.
    """

    def test_wrapper_metadata_preserved_after_codegen(self):
        """
        RuntimeWrapperIR wrapper metadata should remain accessible after
        binary codegen traversal.
        """
        ir = RuntimeWrapperIR()

        ir.record_wrapper(
            WrapperStackSegment.FORWARD_INFERENCE,
            "EffectTokensWrapper",
            {"token_count": 2},
        )
        ir.record_wrapper(
            WrapperStackSegment.DISPATCH,
            "AOTDedupeWrapper",
            {"keep_arg_mask": [True, False]},
        )

        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        visitor = BinaryCodeGenVisitor()
        ir.accept_all(visitor)

        order_forward = ir.get_wrapper_order(WrapperStackSegment.FORWARD_INFERENCE)
        self.assertIn("EffectTokensWrapper", order_forward)

        order_dispatch = ir.get_wrapper_order(WrapperStackSegment.DISPATCH)
        self.assertIn("AOTDedupeWrapper", order_dispatch)

        self.assertEqual(
            ir.wrapper_stack_metadata["EffectTokensWrapper"]["token_count"],
            2,
        )
        self.assertEqual(
            ir.wrapper_stack_metadata["AOTDedupeWrapper"]["keep_arg_mask"],
            [True, False],
        )

    def test_empty_wrapper_metadata_does_not_break_codegen(self):
        """Binary codegen should work fine with empty wrapper metadata."""
        ir = RuntimeWrapperIR()

        ir.add_node(ArgumentExtractionNode(
            name="x",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        ))
        ir.add_node(ReturnResultNode(result_name="result", expose_as="y"))

        wrapper = generate_binary_wrapper(ir)
        self.assertEqual(len(wrapper.extractors), 1)
        self.assertIsNotNone(wrapper)


class TestBinaryCodeGenExistingBehavior(TestCase):
    """
    Regression tests verifying that existing binary codegen behavior is unchanged.
    """

    def test_argument_extraction_f_locals(self):
        """Argument extraction from f_locals should work correctly."""
        visitor = BinaryCodeGenVisitor()
        node = ArgumentExtractionNode(
            name="x",
            source=ArgumentSource.F_LOCALS,
            access_path="x",
        )
        visitor.visit_argument_extraction(node)
        wrapper = visitor.get_wrapper()
        self.assertEqual(len(wrapper.extractors), 1)
        extractor = wrapper.extractors[0]
        self.assertEqual(extractor.name, "x")
        self.assertEqual(extractor.source, ArgumentSource.F_LOCALS)
        self.assertEqual(extractor.access_path, "x")

    def test_guard_check_shape(self):
        """Shape guard checks should work correctly."""
        visitor = BinaryCodeGenVisitor()
        node = GuardCheckNode(
            guard_type=GuardType.SHAPE,
            target_name="x",
            condition="x.shape[0] == 4",
            expected_value=4,
            dimension=0,
            error_message="Shape mismatch",
        )
        visitor.visit_guard_check(node)
        wrapper = visitor.get_wrapper()
        self.assertEqual(len(wrapper.guards), 1)
        guard = wrapper.guards[0]
        self.assertEqual(guard.guard_type, GuardType.SHAPE)
        self.assertEqual(guard.target_name, "x")
        self.assertEqual(guard.expected_value, 4)
        self.assertEqual(guard.dimension, 0)

    def test_aot_autograd_wrapper_sets_flag(self):
        """AOTAutogradWrapperNode with backward_graph should set flag."""
        visitor = BinaryCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            backward_graph="placeholder",
            num_inputs=2,
            num_outputs=1,
        )
        visitor.visit_aot_autograd_wrapper(node)
        wrapper = visitor.get_wrapper()
        self.assertTrue(wrapper.is_autograd_function)

    def test_cuda_graph_setup_config(self):
        """CUDAGraphSetupNode should configure CUDA graph settings."""
        visitor = BinaryCodeGenVisitor()
        node = CUDAGraphSetupNode(
            graph_id="graph_0",
            warmup_runs=3,
            static_inputs=True,
        )
        visitor.visit_cuda_graph_setup(node)
        wrapper = visitor.get_wrapper()
        self.assertIsNotNone(wrapper.cuda_graph_config)
        self.assertEqual(wrapper.cuda_graph_config.graph_id, "graph_0")
        self.assertEqual(wrapper.cuda_graph_config.warmup_runs, 3)
        self.assertTrue(wrapper.cuda_graph_config.static_inputs)

    def test_callable_invocation_stores_metadata(self):
        """CallableInvocationNode should store result_name."""
        visitor = BinaryCodeGenVisitor()
        node = CallableInvocationNode(
            callable_name="compiled_fn",
            argument_names=["a", "b"],
            result_name="output",
        )
        visitor.visit_callable_invocation(node)
        wrapper = visitor.get_wrapper()
        self.assertEqual(wrapper.result_name, "output")

    def test_return_result_sets_expose_as(self):
        """ReturnResultNode should set expose_as."""
        visitor = BinaryCodeGenVisitor()
        node = ReturnResultNode(
            result_name="result",
            expose_as="y_output",
        )
        visitor.visit_return_result(node)
        wrapper = visitor.get_wrapper()
        self.assertEqual(wrapper.expose_as, "y_output")


if __name__ == "__main__":
    run_tests()
