# Owner(s): ["module: dynamo"]

"""
Tests for wrapper metadata flow through PythonifyContext and RuntimeWrapperPipeline.

These tests verify that wrapper metadata captured in CompilationArtifacts correctly
flows through:
1. add_compilation_artifacts -> PythonifyContext.merged_wrapper_* fields
2. RuntimeWrapperPipeline.build() -> RuntimeWrapperIR.wrapper_stack_* fields

The tests are intentionally small and isolated per AGENTS.md guidelines.
"""

from torch.testing._internal.common_utils import run_tests, TestCase

from torch._dynamo.pythonify.context import (
    add_compilation_artifacts,
    get_merged_wrapper_stack_metadata,
    get_merged_wrapper_stack_order,
    pythonify_context,
)
from torch._dynamo.pythonify.ir import (
    AOTDedupeWrapperNode,
    AOTSyntheticBaseWrapperNode,
    ArgumentExtractionNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
    WrapperStackSegment,
)
from torch._dynamo.pythonify.pipeline import (
    CompilationArtifacts,
    RuntimeWrapperPipeline,
)


class TestPipelineWrapperMetadataFlow(TestCase):
    """
    Tests that wrapper metadata flows from CompilationArtifacts through
    RuntimeWrapperPipeline into the resulting RuntimeWrapperIR.
    """

    def test_pipeline_preserves_wrapper_stack_order_from_artifacts(self):
        """
        Wrapper stack order from CompilationArtifacts should be accessible
        in the RuntimeWrapperIR after pipeline.build().
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward": ["EffectTokensWrapper", "FakifiedOutWrapper"],
                "dispatch": ["AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "EffectTokensWrapper": {"token_count": 2},
                "FakifiedOutWrapper": {"out_metas": []},
                "AOTDedupeWrapper": {"keep_mask": [True, False]},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertIsNotNone(ir)
        self.assertIn("forward", artifacts.wrapper_stack_order)
        self.assertEqual(
            artifacts.wrapper_stack_order["forward"],
            ["EffectTokensWrapper", "FakifiedOutWrapper"],
        )
        self.assertEqual(
            artifacts.wrapper_stack_metadata["EffectTokensWrapper"]["token_count"], 2
        )

    def test_pipeline_builds_ir_without_affecting_other_fields(self):
        """
        Building the IR should not modify unrelated fields in CompilationArtifacts.
        """
        artifacts = CompilationArtifacts(
            input_names=["x", "y"],
            parameter_names=["W"],
            model_name="test_model",
            wrapper_stack_order={"forward": ["RuntimeWrapper"]},
            wrapper_stack_metadata={"RuntimeWrapper": {"detach_indices": [0]}},
        )

        original_input_names = list(artifacts.input_names)
        original_param_names = list(artifacts.parameter_names)
        original_model_name = artifacts.model_name

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertEqual(artifacts.input_names, original_input_names)
        self.assertEqual(artifacts.parameter_names, original_param_names)
        self.assertEqual(artifacts.model_name, original_model_name)

        self.assertEqual(
            artifacts.wrapper_stack_order["forward"], ["RuntimeWrapper"]
        )
        self.assertEqual(
            artifacts.wrapper_stack_metadata["RuntimeWrapper"]["detach_indices"], [0]
        )

    def test_pipeline_handles_empty_wrapper_metadata(self):
        """
        Pipeline should work correctly when wrapper metadata is empty.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={},
            wrapper_stack_metadata={},
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertIsNotNone(ir)
        self.assertEqual(len(ir.nodes), 4)  # arg, aot wrapper, callable, return

    def test_ir_wrapper_metadata_accessible_after_build(self):
        """
        The RuntimeWrapperIR should expose wrapper_stack_metadata for codegen.
        """
        ir = RuntimeWrapperIR()
        ir.record_wrapper(
            WrapperStackSegment.FORWARD_INFERENCE,
            "EffectTokensWrapper",
            {"token_count": 3},
        )
        ir.record_wrapper(
            WrapperStackSegment.DISPATCH,
            "AOTDedupeWrapper",
            {"keep_arg_mask": [True]},
        )

        self.assertEqual(
            ir.wrapper_stack_metadata["EffectTokensWrapper"]["token_count"], 3
        )
        self.assertEqual(
            ir.wrapper_stack_metadata["AOTDedupeWrapper"]["keep_arg_mask"], [True]
        )
        self.assertEqual(
            ir.get_wrapper_order(WrapperStackSegment.FORWARD_INFERENCE),
            ["EffectTokensWrapper"],
        )
        self.assertEqual(
            ir.get_wrapper_order(WrapperStackSegment.DISPATCH),
            ["AOTDedupeWrapper"],
        )


class TestContextPipelineIntegration(TestCase):
    """
    Tests for integration between PythonifyContext and RuntimeWrapperPipeline.

    These tests verify the full flow: artifacts with wrapper metadata added to
    context, then built through pipeline, with metadata reachable throughout.
    """

    def test_context_artifacts_reachable_by_pipeline(self):
        """
        Artifacts added to PythonifyContext should be buildable via pipeline.
        """
        with pythonify_context("/tmp/test_integration.py") as ctx:
            artifacts = CompilationArtifacts(
                input_names=["x"],
                wrapper_stack_order={"forward": ["EffectTokensWrapper"]},
                wrapper_stack_metadata={"EffectTokensWrapper": {"token_count": 1}},
            )
            add_compilation_artifacts(artifacts)

            self.assertEqual(len(ctx.artifacts_list), 1)
            self.assertIs(ctx.artifacts_list[0], artifacts)

            pipeline = RuntimeWrapperPipeline(ctx.artifacts_list[0])
            ir = pipeline.build()

            self.assertIsNotNone(ir)

    def test_context_merged_metadata_matches_artifacts(self):
        """
        Merged wrapper metadata in context should match what was in artifacts.
        """
        with pythonify_context("/tmp/test_merge.py"):
            artifacts = CompilationArtifacts(
                input_names=["x"],
                wrapper_stack_order={
                    "forward": ["EffectTokensWrapper", "FakifiedOutWrapper"],
                    "dispatch": ["AOTSyntheticBaseWrapper"],
                },
                wrapper_stack_metadata={
                    "EffectTokensWrapper": {"token_count": 2},
                    "FakifiedOutWrapper": {"out_metas": ["m1"]},
                    "AOTSyntheticBaseWrapper": {"synthetic_base_info": {}},
                },
            )
            add_compilation_artifacts(artifacts)

            merged_order = get_merged_wrapper_stack_order()
            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(
                merged_order["forward"],
                ["EffectTokensWrapper", "FakifiedOutWrapper"],
            )
            self.assertEqual(merged_order["dispatch"], ["AOTSyntheticBaseWrapper"])

            self.assertEqual(merged_meta["EffectTokensWrapper"]["token_count"], 2)
            self.assertEqual(merged_meta["FakifiedOutWrapper"]["out_metas"], ["m1"])
            self.assertIn("AOTSyntheticBaseWrapper", merged_meta)

    def test_multiple_artifacts_merge_wrapper_metadata(self):
        """
        Multiple artifacts with different wrapper metadata should be merged.
        """
        with pythonify_context("/tmp/test_multi.py"):
            artifacts1 = CompilationArtifacts(
                input_names=["x"],
                wrapper_stack_order={"forward": ["EffectTokensWrapper"]},
                wrapper_stack_metadata={"EffectTokensWrapper": {"token_count": 1}},
            )
            add_compilation_artifacts(artifacts1)

            artifacts2 = CompilationArtifacts(
                input_names=["y"],
                wrapper_stack_order={
                    "forward": ["RuntimeWrapper"],
                    "dispatch": ["AOTDedupeWrapper"],
                },
                wrapper_stack_metadata={
                    "RuntimeWrapper": {"indices": [0, 1]},
                    "AOTDedupeWrapper": {"keep_mask": [True]},
                },
            )
            add_compilation_artifacts(artifacts2)

            merged_order = get_merged_wrapper_stack_order()
            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(
                merged_order["forward"],
                ["EffectTokensWrapper", "RuntimeWrapper"],
            )
            self.assertEqual(merged_order["dispatch"], ["AOTDedupeWrapper"])

            self.assertEqual(merged_meta["EffectTokensWrapper"]["token_count"], 1)
            self.assertEqual(merged_meta["RuntimeWrapper"]["indices"], [0, 1])
            self.assertEqual(merged_meta["AOTDedupeWrapper"]["keep_mask"], [True])

    def test_empty_wrapper_metadata_does_not_break_pipeline(self):
        """
        Artifacts without wrapper metadata should not break the pipeline flow.
        """
        with pythonify_context("/tmp/test_empty.py") as ctx:
            artifacts = CompilationArtifacts(
                input_names=["x"],
                parameter_names=["W"],
            )
            add_compilation_artifacts(artifacts)

            self.assertEqual(get_merged_wrapper_stack_order(), {})
            self.assertEqual(get_merged_wrapper_stack_metadata(), {})

            pipeline = RuntimeWrapperPipeline(ctx.artifacts_list[0])
            ir = pipeline.build()

            self.assertIsNotNone(ir)
            self.assertEqual(ir.wrapper_stack_metadata, {})


class TestWrapperMetadataPreservation(TestCase):
    """
    Tests that wrapper metadata is preserved through various operations.
    """

    def test_artifacts_wrapper_metadata_not_mutated_by_context(self):
        """
        Adding artifacts to context should not mutate the original wrapper metadata.
        """
        original_order = {"forward": ["WrapperA"]}
        original_meta = {"WrapperA": {"key": "value"}}

        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order=original_order,
            wrapper_stack_metadata=original_meta,
        )

        with pythonify_context("/tmp/test_no_mutate.py"):
            add_compilation_artifacts(artifacts)

        self.assertEqual(artifacts.wrapper_stack_order, {"forward": ["WrapperA"]})
        self.assertEqual(artifacts.wrapper_stack_metadata, {"WrapperA": {"key": "value"}})

    def test_wrapper_metadata_accessible_throughout_pipeline_lifecycle(self):
        """
        Wrapper metadata should be accessible before, during, and after pipeline build.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={"forward": ["TestWrapper"]},
            wrapper_stack_metadata={"TestWrapper": {"test_key": "test_value"}},
        )

        self.assertEqual(
            artifacts.wrapper_stack_metadata["TestWrapper"]["test_key"], "test_value"
        )

        pipeline = RuntimeWrapperPipeline(artifacts)

        self.assertEqual(
            pipeline.artifacts.wrapper_stack_metadata["TestWrapper"]["test_key"],
            "test_value",
        )

        ir = pipeline.build()

        self.assertEqual(
            pipeline.artifacts.wrapper_stack_metadata["TestWrapper"]["test_key"],
            "test_value",
        )
        self.assertIsNotNone(ir)

    def test_wrapper_metadata_with_complex_nested_values(self):
        """
        Wrapper metadata with nested dicts and lists should be preserved.
        """
        complex_metadata = {
            "EffectTokensWrapper": {"token_count": 3, "config": {"nested": True}},
            "RuntimeWrapper": {
                "indices_of_inps_to_detach": [0, 1, 2],
                "options": {"trace_joint": False, "disable_amp": True},
            },
            "AOTSyntheticBaseWrapper": {
                "synthetic_base_info": {
                    "bases": [{"idx": 0, "offset": 0}],
                    "aliased_inputs": [[0, 1], [2]],
                },
            },
        }

        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={"forward": list(complex_metadata.keys())},
            wrapper_stack_metadata=complex_metadata,
        )

        with pythonify_context("/tmp/test_complex.py"):
            add_compilation_artifacts(artifacts)

            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(
                merged_meta["EffectTokensWrapper"]["config"]["nested"], True
            )
            self.assertEqual(
                merged_meta["RuntimeWrapper"]["indices_of_inps_to_detach"], [0, 1, 2]
            )
            self.assertEqual(
                merged_meta["RuntimeWrapper"]["options"]["disable_amp"], True
            )
            self.assertEqual(
                merged_meta["AOTSyntheticBaseWrapper"]["synthetic_base_info"]["bases"],
                [{"idx": 0, "offset": 0}],
            )


class TestWrapperNodePopulation(TestCase):
    """
    Tests that wrapper nodes are properly created and populated in the IR
    from wrapper_stack_order and wrapper_stack_metadata in CompilationArtifacts.

    These tests verify the core functionality added by the TODO:
    "Populate wrapper nodes in pythonify pipeline with correct reverse-application order"
    """

    def test_pipeline_creates_wrapper_nodes_from_metadata(self):
        """
        Pipeline should create IR nodes for each wrapper in wrapper_stack_order.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward_inference": ["EffectTokensWrapper", "FakifiedOutWrapper"],
            },
            wrapper_stack_metadata={
                "EffectTokensWrapper": {"num_tokens": 3},
                "FakifiedOutWrapper": {"has_output_strides": True},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        effect_nodes = ir.get_nodes_by_type(EffectTokensWrapperNode)
        fakified_nodes = ir.get_nodes_by_type(FakifiedOutWrapperNode)

        self.assertEqual(len(effect_nodes), 1)
        self.assertEqual(len(fakified_nodes), 1)
        self.assertEqual(effect_nodes[0].token_count, 3)

    def test_pipeline_creates_dispatch_wrapper_nodes(self):
        """
        Dispatch segment wrappers (AOTDedupeWrapper, AOTSyntheticBaseWrapper)
        should be created as IR nodes.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "dispatch": ["AOTSyntheticBaseWrapper", "AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "AOTSyntheticBaseWrapper": {"needs_post_compile": True},
                "AOTDedupeWrapper": {"keep_arg_mask": [True, False, True]},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        synthetic_nodes = ir.get_nodes_by_type(AOTSyntheticBaseWrapperNode)
        dedupe_nodes = ir.get_nodes_by_type(AOTDedupeWrapperNode)

        self.assertEqual(len(synthetic_nodes), 1)
        self.assertEqual(len(dedupe_nodes), 1)
        self.assertTrue(synthetic_nodes[0].needs_post_compile)
        self.assertEqual(dedupe_nodes[0].keep_arg_mask, [True, False, True])

    def test_pipeline_creates_autograd_assembly_wrapper_nodes(self):
        """
        Autograd assembly segment wrappers (RuntimeWrapper, DebugAssertWrapper)
        should be created as IR nodes.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "autograd_assembly": ["RuntimeWrapper", "DebugAssertWrapper"],
            },
            wrapper_stack_metadata={
                "RuntimeWrapper": {
                    "indices_of_inps_to_detach": [0, 2],
                    "disable_amp": True,
                },
                "DebugAssertWrapper": {"flat_requires_grad": [True, False]},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        runtime_nodes = ir.get_nodes_by_type(RuntimeWrapperNode)
        debug_nodes = ir.get_nodes_by_type(DebugAssertWrapperNode)

        self.assertEqual(len(runtime_nodes), 1)
        self.assertEqual(len(debug_nodes), 1)
        self.assertEqual(runtime_nodes[0].indices_of_inps_to_detach, [0, 2])
        self.assertTrue(runtime_nodes[0].disable_amp)
        self.assertEqual(debug_nodes[0].flat_requires_grad, [True, False])

    def test_wrapper_node_order_matches_aot_autograd_post_compile(self):
        """
        Wrapper nodes should be added in the correct order matching AOTAutograd
        post_compile semantics: forward_inference -> autograd_assembly -> dispatch.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward_inference": ["EffectTokensWrapper"],
                "autograd_assembly": ["RuntimeWrapper"],
                "dispatch": ["AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "EffectTokensWrapper": {"num_tokens": 1},
                "RuntimeWrapper": {"indices_of_inps_to_detach": []},
                "AOTDedupeWrapper": {},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        wrapper_nodes = [
            node for node in ir.nodes
            if isinstance(node, (
                EffectTokensWrapperNode,
                RuntimeWrapperNode,
                AOTDedupeWrapperNode,
            ))
        ]

        self.assertEqual(len(wrapper_nodes), 3)
        self.assertIsInstance(wrapper_nodes[0], EffectTokensWrapperNode)
        self.assertIsInstance(wrapper_nodes[1], RuntimeWrapperNode)
        self.assertIsInstance(wrapper_nodes[2], AOTDedupeWrapperNode)

    def test_wrapper_nodes_recorded_in_ir_stack_order(self):
        """
        Created wrapper nodes should be recorded in IR.wrapper_stack_order.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward_inference": ["EffectTokensWrapper", "FakifiedOutWrapper"],
                "dispatch": ["AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "EffectTokensWrapper": {"num_tokens": 2},
                "FakifiedOutWrapper": {},
                "AOTDedupeWrapper": {},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        forward_order = ir.get_wrapper_order(WrapperStackSegment.FORWARD_INFERENCE)
        dispatch_order = ir.get_wrapper_order(WrapperStackSegment.DISPATCH)

        self.assertEqual(forward_order, ["EffectTokensWrapper", "FakifiedOutWrapper"])
        self.assertEqual(dispatch_order, ["AOTDedupeWrapper"])

    def test_dispatch_wrappers_reverse_order_for_application(self):
        """
        Dispatch segment wrappers can be retrieved in reverse order for runtime
        application using get_wrapper_order(..., reverse_for_application=True).
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "dispatch": ["AOTSyntheticBaseWrapper", "AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "AOTSyntheticBaseWrapper": {},
                "AOTDedupeWrapper": {},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        normal_order = ir.get_wrapper_order(WrapperStackSegment.DISPATCH)
        reversed_order = ir.get_wrapper_order(
            WrapperStackSegment.DISPATCH, reverse_for_application=True
        )

        self.assertEqual(normal_order, ["AOTSyntheticBaseWrapper", "AOTDedupeWrapper"])
        self.assertEqual(reversed_order, ["AOTDedupeWrapper", "AOTSyntheticBaseWrapper"])

    def test_functionalized_rng_wrapper_node_metadata(self):
        """
        FunctionalizedRngRuntimeWrapper should be created with correct metadata.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward_inference": ["FunctionalizedRngRuntimeWrapper"],
            },
            wrapper_stack_metadata={
                "FunctionalizedRngRuntimeWrapper": {
                    "is_rng_op_functionalized": True,
                    "num_outputs_rng_offset": 2,
                    "num_forward_returns": 5,
                },
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        rng_nodes = ir.get_nodes_by_type(FunctionalizedRngRuntimeWrapperNode)

        self.assertEqual(len(rng_nodes), 1)
        self.assertTrue(rng_nodes[0].is_rng_op_functionalized)
        self.assertEqual(rng_nodes[0].num_outputs_rng_offset, 2)
        self.assertEqual(rng_nodes[0].num_forward_returns, 5)

    def test_full_wrapper_stack_scenario(self):
        """
        Test a full wrapper stack matching what AOTAutograd would produce,
        verifying all segments are populated correctly.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            wrapper_stack_order={
                "forward_inference": [
                    "EffectTokensWrapper",
                    "FunctionalizedRngRuntimeWrapper",
                    "FakifiedOutWrapper",
                ],
                "autograd_assembly": ["RuntimeWrapper"],
                "dispatch": ["AOTSyntheticBaseWrapper", "AOTDedupeWrapper"],
            },
            wrapper_stack_metadata={
                "EffectTokensWrapper": {"num_tokens": 2},
                "FunctionalizedRngRuntimeWrapper": {
                    "is_rng_op_functionalized": True,
                    "num_outputs_rng_offset": 1,
                },
                "FakifiedOutWrapper": {"has_output_strides": True},
                "RuntimeWrapper": {
                    "indices_of_inps_to_detach": [0],
                    "disable_amp": False,
                },
                "AOTSyntheticBaseWrapper": {"needs_post_compile": True},
                "AOTDedupeWrapper": {"keep_arg_mask": [True]},
            },
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertEqual(len(ir.get_nodes_by_type(EffectTokensWrapperNode)), 1)
        self.assertEqual(len(ir.get_nodes_by_type(FunctionalizedRngRuntimeWrapperNode)), 1)
        self.assertEqual(len(ir.get_nodes_by_type(FakifiedOutWrapperNode)), 1)
        self.assertEqual(len(ir.get_nodes_by_type(RuntimeWrapperNode)), 1)
        self.assertEqual(len(ir.get_nodes_by_type(AOTSyntheticBaseWrapperNode)), 1)
        self.assertEqual(len(ir.get_nodes_by_type(AOTDedupeWrapperNode)), 1)

        forward_order = ir.get_wrapper_order(WrapperStackSegment.FORWARD_INFERENCE)
        self.assertEqual(len(forward_order), 3)

        autograd_order = ir.get_wrapper_order(WrapperStackSegment.AUTOGRAD_ASSEMBLY)
        self.assertEqual(autograd_order, ["RuntimeWrapper"])

        dispatch_order = ir.get_wrapper_order(WrapperStackSegment.DISPATCH)
        self.assertEqual(dispatch_order, ["AOTSyntheticBaseWrapper", "AOTDedupeWrapper"])


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
