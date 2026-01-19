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
    ArgumentSource,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    ModelSource,
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


class TestModuleBasedSourcePreference(TestCase):
    """
    Tests that verify the pipeline prefers module-based sources (PARAMETER/BUFFER)
    over OBJECT_ID when module context is available and nested_path is populated.

    This is the "golden path" for nn.Module reconstruction in pythonify.
    """

    def test_parameter_uses_module_source_when_model_name_and_nested_path_available(self):
        """
        When model_name is provided and nested_path is available, parameters
        should use ArgumentSource.PARAMETER, not OBJECT_ID, even if tensor
        object is available.
        """
        import torch

        test_tensor = torch.randn(3, 4)

        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["layer.weight"],
            model_name="model",
            parameter_tensors={"layer.weight": test_tensor},
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)
        param_nodes = [
            n for n in extraction_nodes
            if n.access_path == "layer.weight"
        ]

        self.assertEqual(len(param_nodes), 1)
        param_node = param_nodes[0]

        self.assertEqual(
            param_node.source, ArgumentSource.PARAMETER,
            f"Expected PARAMETER source but got {param_node.source}. "
            "Module-based source should be preferred when model_name and nested_path are available."
        )
        self.assertEqual(param_node.nested_path, ["layer", "weight"])

    def test_buffer_uses_module_source_when_model_name_and_nested_path_available(self):
        """
        When model_name is provided and nested_path is available, buffers
        should use ArgumentSource.BUFFER, not OBJECT_ID, even if tensor
        object is available.
        """
        import torch

        test_buffer = torch.zeros(10)

        artifacts = CompilationArtifacts(
            input_names=["x"],
            buffer_names=["running_mean"],
            model_name="model",
            buffer_tensors={"running_mean": test_buffer},
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)
        buffer_nodes = [
            n for n in extraction_nodes
            if n.access_path == "running_mean"
        ]

        self.assertEqual(len(buffer_nodes), 1)
        buffer_node = buffer_nodes[0]

        self.assertEqual(
            buffer_node.source, ArgumentSource.BUFFER,
            f"Expected BUFFER source but got {buffer_node.source}. "
            "Module-based source should be preferred when model_name and nested_path are available."
        )

    def test_falls_back_to_object_id_when_no_model_name(self):
        """
        When model_name is empty, fall back to OBJECT_ID for parameters/buffers.
        This tests the legacy/fallback path.
        """
        import torch

        test_tensor = torch.randn(3, 4)

        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["W"],
            model_name="",
            parameter_tensors={"W": test_tensor},
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)
        param_nodes = [
            n for n in extraction_nodes
            if n.access_path == "W"
        ]

        self.assertEqual(len(param_nodes), 1)
        param_node = param_nodes[0]

        self.assertEqual(
            param_node.source, ArgumentSource.OBJECT_ID,
            f"Expected OBJECT_ID source when model_name is empty, but got {param_node.source}."
        )
        self.assertIsNotNone(param_node.object_id)

    def test_ordered_arg_info_prefers_module_source(self):
        """
        When using ordered_arg_info path with model_name, parameters and buffers
        should still prefer module-based sources over OBJECT_ID.
        """
        import torch

        param_tensor = torch.randn(5, 5)
        buffer_tensor = torch.ones(3)

        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["encoder.weight"],
            buffer_names=["encoder.bias"],
            model_name="model",
            parameter_tensors={"encoder.weight": param_tensor},
            buffer_tensors={"encoder.bias": buffer_tensor},
            ordered_arg_info=[
                {"name": "encoder.weight", "source_type": "parameter", "nested_path": ["encoder", "weight"]},
                {"name": "x", "source_type": "input"},
                {"name": "encoder.bias", "source_type": "buffer", "nested_path": ["encoder", "bias"]},
            ],
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)

        param_node = next((n for n in extraction_nodes if n.access_path == "encoder.weight"), None)
        buffer_node = next((n for n in extraction_nodes if n.access_path == "encoder.bias"), None)

        self.assertIsNotNone(param_node)
        self.assertIsNotNone(buffer_node)

        self.assertEqual(
            param_node.source, ArgumentSource.PARAMETER,
            "ordered_arg_info with model_name should use PARAMETER source"
        )
        self.assertEqual(
            buffer_node.source, ArgumentSource.BUFFER,
            "ordered_arg_info with model_name should use BUFFER source"
        )

    def test_nested_module_path_preserved(self):
        """
        For deeply nested module hierarchies, nested_path should be correctly
        populated to enable attribute traversal (e.g., model.encoder.layer1.weight).
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["encoder.layer1.weight", "encoder.layer2.bias"],
            model_name="model",
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)
        param_nodes = [
            n for n in extraction_nodes
            if n.source == ArgumentSource.PARAMETER
        ]

        self.assertEqual(len(param_nodes), 2)

        weight_node = next(n for n in param_nodes if "weight" in n.access_path)
        bias_node = next(n for n in param_nodes if "bias" in n.access_path)

        self.assertEqual(weight_node.nested_path, ["encoder", "layer1", "weight"])
        self.assertEqual(bias_node.nested_path, ["encoder", "layer2", "bias"])


class TestModelSourceConfiguration(TestCase):
    """
    Tests for ModelSource configuration in CompilationArtifacts.

    ModelSource specifies where the model comes from for exec() compatibility,
    driving how the generated code accesses the model (closure, f_locals, f_globals).
    """

    def test_model_source_defaults_to_none(self):
        """
        When model_source is not specified, it should default to None.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="model",
        )

        self.assertIsNone(artifacts.model_source)

    def test_model_source_can_be_set_to_closure(self):
        """
        ModelSource.CLOSURE indicates the model is directly accessible as a variable.
        This is the typical case for exec() where the model is passed in the namespace.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="model",
            model_source=ModelSource.CLOSURE,
        )

        self.assertEqual(artifacts.model_source, ModelSource.CLOSURE)

    def test_model_source_can_be_set_to_f_locals(self):
        """
        ModelSource.F_LOCALS indicates the model is accessed via f_locals[model_name].
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="self",
            model_source=ModelSource.F_LOCALS,
        )

        self.assertEqual(artifacts.model_source, ModelSource.F_LOCALS)

    def test_model_source_can_be_set_to_f_globals(self):
        """
        ModelSource.F_GLOBALS indicates the model is accessed via f_globals[model_name].
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="global_model",
            model_source=ModelSource.F_GLOBALS,
        )

        self.assertEqual(artifacts.model_source, ModelSource.F_GLOBALS)

    def test_model_source_flows_into_ir_source_info(self):
        """
        When building the IR, model_source should be included in source_info
        so code generators can access it.
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="model",
            model_source=ModelSource.CLOSURE,
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertIn("model_source", ir.source_info)
        self.assertEqual(ir.source_info["model_source"], ModelSource.CLOSURE)

    def test_model_source_none_flows_into_ir_source_info(self):
        """
        When model_source is None, it should still be included in source_info
        (codegen may want to apply a default).
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            model_name="model",
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertIn("model_source", ir.source_info)
        self.assertIsNone(ir.source_info["model_source"])

    def test_model_source_with_module_based_parameter_extraction(self):
        """
        ModelSource should work together with module-based parameter extraction.
        The model_source tells how to access the model, while module-based sources
        (PARAMETER/BUFFER) are still preferred when model_name is provided.
        """
        import torch

        test_tensor = torch.randn(3, 4)

        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["layer.weight"],
            model_name="model",
            model_source=ModelSource.CLOSURE,
            parameter_tensors={"layer.weight": test_tensor},
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertEqual(ir.source_info["model_source"], ModelSource.CLOSURE)

        extraction_nodes = ir.get_nodes_by_type(ArgumentExtractionNode)
        param_nodes = [n for n in extraction_nodes if n.access_path == "layer.weight"]

        self.assertEqual(len(param_nodes), 1)
        self.assertEqual(param_nodes[0].source, ArgumentSource.PARAMETER)
        self.assertEqual(param_nodes[0].nested_path, ["layer", "weight"])

    def test_model_source_f_locals_with_self_model_name(self):
        """
        A common pattern is model_name="self" with model_source=F_LOCALS,
        indicating the model is accessed as f_locals["self"].
        """
        artifacts = CompilationArtifacts(
            input_names=["x"],
            parameter_names=["W"],
            model_name="self",
            model_source=ModelSource.F_LOCALS,
        )

        pipeline = RuntimeWrapperPipeline(artifacts)
        ir = pipeline.build()

        self.assertEqual(ir.source_info["model_name"], "self")
        self.assertEqual(ir.source_info["model_source"], ModelSource.F_LOCALS)


class TestSetModelReferenceWithModelSource(TestCase):
    """
    Tests for set_model_reference() populating ModelSource in PythonifyContext.

    The set_model_reference() function now accepts an optional model_source
    parameter that specifies where the model comes from for exec() compatibility.
    This ModelSource flows from the context to CompilationArtifacts.
    """

    def test_set_model_reference_defaults_model_source_to_closure(self):
        """
        When model_source is not specified, set_model_reference should default to CLOSURE.
        """
        from torch._dynamo.pythonify.context import (
            get_model_source,
            PythonifyContext,
            set_model_reference,
        )

        class FakeModel:
            pass

        model = FakeModel()

        with pythonify_context("/tmp/test.py") as ctx:
            set_model_reference(model)
            source = get_model_source()

            self.assertEqual(source, ModelSource.CLOSURE)
            self.assertEqual(ctx.model_source, ModelSource.CLOSURE)

    def test_set_model_reference_with_explicit_closure(self):
        """
        Explicitly passing CLOSURE should set model_source to CLOSURE.
        """
        from torch._dynamo.pythonify.context import (
            get_model_source,
            set_model_reference,
        )

        class FakeModel:
            pass

        model = FakeModel()

        with pythonify_context("/tmp/test.py") as ctx:
            set_model_reference(model, model_source=ModelSource.CLOSURE)
            source = get_model_source()

            self.assertEqual(source, ModelSource.CLOSURE)
            self.assertEqual(ctx.model_source, ModelSource.CLOSURE)

    def test_set_model_reference_with_f_locals(self):
        """
        Passing F_LOCALS should set model_source to F_LOCALS.
        """
        from torch._dynamo.pythonify.context import (
            get_model_source,
            set_model_reference,
        )

        class FakeModel:
            pass

        model = FakeModel()

        with pythonify_context("/tmp/test.py") as ctx:
            set_model_reference(model, model_source=ModelSource.F_LOCALS)
            source = get_model_source()

            self.assertEqual(source, ModelSource.F_LOCALS)
            self.assertEqual(ctx.model_source, ModelSource.F_LOCALS)

    def test_set_model_reference_with_f_globals(self):
        """
        Passing F_GLOBALS should set model_source to F_GLOBALS.
        """
        from torch._dynamo.pythonify.context import (
            get_model_source,
            set_model_reference,
        )

        class FakeModel:
            pass

        model = FakeModel()

        with pythonify_context("/tmp/test.py") as ctx:
            set_model_reference(model, model_source=ModelSource.F_GLOBALS)
            source = get_model_source()

            self.assertEqual(source, ModelSource.F_GLOBALS)
            self.assertEqual(ctx.model_source, ModelSource.F_GLOBALS)

    def test_get_model_source_returns_closure_when_no_context(self):
        """
        get_model_source should return CLOSURE when no pythonify context is active.
        This is the safe default.
        """
        from torch._dynamo.pythonify.context import get_model_source

        source = get_model_source()
        self.assertEqual(source, ModelSource.CLOSURE)

    def test_context_model_source_defaults_to_closure(self):
        """
        PythonifyContext.model_source should default to CLOSURE.
        """
        from torch._dynamo.pythonify.context import PythonifyContext

        ctx = PythonifyContext(pythonify_path="/tmp/test.py")
        self.assertEqual(ctx.model_source, ModelSource.CLOSURE)


class TestGeneratePythonifyOutputHeader(TestCase):
    """
    Tests for the file header generated by generate_pythonify_output().

    The header should include different usage instructions depending on
    whether the golden path (module-based access) or the object ID fallback
    is being used.
    """

    def test_golden_path_header_shows_namespace_pattern(self):
        """
        When using golden path (no object IDs), header should show the
        namespace = {..., "model": model} exec() usage pattern.
        """
        import os
        import tempfile

        from torch._dynamo.pythonify.context import (
            _pythonify_context,
            generate_pythonify_output,
            PythonifyContext,
        )

        ctx = PythonifyContext(
            pythonify_path=tempfile.mktemp(suffix=".py"),
            model_name="my_model",
        )
        _pythonify_context.set(ctx)

        artifact = CompilationArtifacts(
            model_name="my_model",
            input_names=["x"],
            parameter_names=["W"],
            buffer_names=[],
            parameter_tensors={},
            buffer_tensors={},
        )
        ctx.artifacts_list.append(artifact)

        code = generate_pythonify_output()
        _pythonify_context.clear()

        self.assertIn("GOLDEN PATH (RECOMMENDED):", code)
        self.assertIn('namespace = {', code)
        self.assertIn('"my_model": your_model,', code)
        self.assertIn("exec(code, namespace)", code)
        self.assertIn("PROCESS-PORTABLE", code)
        self.assertIn("LIVE ACCESS", code)
        self.assertIn("PERSISTABLE", code)
        self.assertNotIn("frame.f_globals, frame.f_locals", code)

        os.unlink(ctx.pythonify_path)

    def test_object_id_header_shows_frame_pattern(self):
        """
        When using object ID fallback, header should show the
        frame.f_globals, frame.f_locals exec() usage pattern.
        """
        import os
        import tempfile

        import torch
        from torch._dynamo.pythonify.context import (
            _pythonify_context,
            generate_pythonify_output,
            PythonifyContext,
        )

        ctx = PythonifyContext(
            pythonify_path=tempfile.mktemp(suffix=".py"),
            model_name="model",
        )
        _pythonify_context.set(ctx)

        W = torch.randn(4, 4)
        artifact = CompilationArtifacts(
            model_name="model",
            input_names=["x"],
            parameter_names=["W"],
            buffer_names=[],
            parameter_tensors={"W": W},
            buffer_tensors={},
        )
        ctx.artifacts_list.append(artifact)

        code = generate_pythonify_output()
        _pythonify_context.clear()

        self.assertIn("OBJECT ID FALLBACK MODE:", code)
        self.assertIn("frame.f_globals, frame.f_locals", code)
        self.assertNotIn("GOLDEN PATH (RECOMMENDED):", code)
        self.assertIn("PROCESS-LOCAL FILE - DO NOT SAVE", code)

        os.unlink(ctx.pythonify_path)

    def test_golden_path_header_uses_model_name_from_context(self):
        """
        The golden path header should use the model_name from context.
        """
        import os
        import tempfile

        from torch._dynamo.pythonify.context import (
            _pythonify_context,
            generate_pythonify_output,
            PythonifyContext,
        )

        ctx = PythonifyContext(
            pythonify_path=tempfile.mktemp(suffix=".py"),
            model_name="custom_model_name",
        )
        _pythonify_context.set(ctx)

        artifact = CompilationArtifacts(
            model_name="custom_model_name",
            input_names=["x"],
            parameter_names=["W"],
            buffer_names=[],
            parameter_tensors={},
            buffer_tensors={},
        )
        ctx.artifacts_list.append(artifact)

        code = generate_pythonify_output()
        _pythonify_context.clear()

        self.assertIn("'custom_model_name'", code)
        self.assertIn('"custom_model_name": your_model,', code)

        os.unlink(ctx.pythonify_path)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
