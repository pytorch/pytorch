# Owner(s): ["module: dynamo"]

import unittest

from torch._dynamo.pythonify.ir import (
    AOTAutogradWrapperNode,
    AOTDedupeWrapperNode,
    AOTDispatchSubclassWrapperNode,
    AOTSyntheticBaseWrapperNode,
    ArgumentExtractionNode,
    ArgumentSource,
    CallableInvocationNode,
    CodeGenVisitor,
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


class _RecordingVisitor(CodeGenVisitor):
    """Visitor that records which visit_* methods were invoked."""

    def __init__(self):
        self.visited = []

    def _record(self, node, name):
        self.visited.append((name, type(node).__name__))
        return name

    def visit_argument_extraction(self, node: ArgumentExtractionNode):
        return self._record(node, "visit_argument_extraction")

    def visit_guard_check(self, node: GuardCheckNode):
        return self._record(node, "visit_guard_check")

    def visit_effect_tokens_wrapper(self, node: EffectTokensWrapperNode):
        return self._record(node, "visit_effect_tokens_wrapper")

    def visit_aot_dispatch_subclass_wrapper(
        self, node: AOTDispatchSubclassWrapperNode
    ):
        return self._record(node, "visit_aot_dispatch_subclass_wrapper")

    def visit_functionalized_rng_runtime_wrapper(
        self, node: FunctionalizedRngRuntimeWrapperNode
    ):
        return self._record(node, "visit_functionalized_rng_runtime_wrapper")

    def visit_fakified_out_wrapper(self, node: FakifiedOutWrapperNode):
        return self._record(node, "visit_fakified_out_wrapper")

    def visit_runtime_wrapper(self, node: RuntimeWrapperNode):
        return self._record(node, "visit_runtime_wrapper")

    def visit_aot_dedupe_wrapper(self, node: AOTDedupeWrapperNode):
        return self._record(node, "visit_aot_dedupe_wrapper")

    def visit_aot_synthetic_base_wrapper(
        self, node: AOTSyntheticBaseWrapperNode
    ):
        return self._record(node, "visit_aot_synthetic_base_wrapper")

    def visit_debug_assert_wrapper(self, node: DebugAssertWrapperNode):
        return self._record(node, "visit_debug_assert_wrapper")

    def visit_aot_autograd_wrapper(self, node: AOTAutogradWrapperNode):
        return self._record(node, "visit_aot_autograd_wrapper")

    def visit_cuda_graph_setup(self, node):  # pragma: no cover - not used here
        return self._record(node, "visit_cuda_graph_setup")

    def visit_callable_invocation(self, node: CallableInvocationNode):
        return self._record(node, "visit_callable_invocation")

    def visit_kernel_load(self, node: KernelLoadNode):
        return self._record(node, "visit_kernel_load")

    def visit_return_result(self, node: ReturnResultNode):
        return self._record(node, "visit_return_result")

    def visit_compiled_region(self, node):  # pragma: no cover - unused
        return self._record(node, "visit_compiled_region")

    def visit_multi_region_dispatch(self, node):  # pragma: no cover - unused
        return self._record(node, "visit_multi_region_dispatch")


class TestWrapperIRNodes(unittest.TestCase):
    def test_wrapper_nodes_call_expected_visitor_methods(self):
        visitor = _RecordingVisitor()

        nodes = [
            EffectTokensWrapperNode(token_count=2),
            AOTDispatchSubclassWrapperNode(subclass_inp_meta={"foo": "bar"}),
            FunctionalizedRngRuntimeWrapperNode(is_rng_op_functionalized=True),
            FakifiedOutWrapperNode(out_metas=["m0"]),
            RuntimeWrapperNode(indices_of_inps_to_detach=[0, 2]),
            AOTDedupeWrapperNode(keep_arg_mask=[True, False]),
            AOTSyntheticBaseWrapperNode(synthetic_base_info={"bases": [0]}),
            DebugAssertWrapperNode(flat_requires_grad=[True, False]),
            AOTAutogradWrapperNode(class_name="CompiledFunction"),
        ]

        expected = [
            "visit_effect_tokens_wrapper",
            "visit_aot_dispatch_subclass_wrapper",
            "visit_functionalized_rng_runtime_wrapper",
            "visit_fakified_out_wrapper",
            "visit_runtime_wrapper",
            "visit_aot_dedupe_wrapper",
            "visit_aot_synthetic_base_wrapper",
            "visit_debug_assert_wrapper",
            "visit_aot_autograd_wrapper",
        ]

        for node in nodes:
            result = node.accept(visitor)
            self.assertEqual(result, expected[len(visitor.visited) - 1])

        visited_names = [name for name, _ in visitor.visited]
        self.assertEqual(visited_names, expected)

    def test_accept_all_preserves_node_order(self):
        visitor = _RecordingVisitor()
        ir = RuntimeWrapperIR()

        ir.add_node(
            ArgumentExtractionNode(
                name="arg0", source=ArgumentSource.F_LOCALS, access_path="x"
            )
        )
        ir.add_node(
            GuardCheckNode(
                guard_type=GuardType.SHAPE,
                target_name="x",
                condition="x.shape[0] == 2",
                expected_value=2,
            )
        )
        ir.add_node(
            CallableInvocationNode(
                callable_name="compiled_fn",
                argument_names=["arg0"],
                result_name="res",
            )
        )
        ir.add_node(ReturnResultNode(result_name="res", expose_as="y"))

        results = ir.accept_all(visitor)

        expected_visit_sequence = [
            "visit_argument_extraction",
            "visit_guard_check",
            "visit_callable_invocation",
            "visit_return_result",
        ]

        self.assertEqual([name for name, _ in visitor.visited], expected_visit_sequence)
        self.assertEqual(results, expected_visit_sequence)


class TestWrapperStackMetadata(unittest.TestCase):
    def test_wrapper_stack_order_and_metadata_round_trip(self):
        ir = RuntimeWrapperIR()

        ir.record_wrapper(
            WrapperStackSegment.FORWARD_INFERENCE,
            "EffectTokensWrapper",
            {"token_count": 1},
        )
        ir.record_wrapper(
            WrapperStackSegment.FORWARD_INFERENCE,
            "FakifiedOutWrapper",
        )
        ir.record_wrapper(
            WrapperStackSegment.AUTOGRAD_ASSEMBLY,
            "RuntimeWrapper",
            {"indices_of_inps_to_detach": [0]},
        )
        ir.record_wrapper(
            WrapperStackSegment.DISPATCH,
            "AOTSyntheticBaseWrapper",
            {"synthetic_base_info": {"bases": [0]}},
        )
        ir.record_wrapper(
            WrapperStackSegment.DISPATCH,
            "AOTDedupeWrapper",
            {"keep_arg_mask": [True, False]},
        )

        self.assertEqual(
            ir.get_wrapper_order(WrapperStackSegment.FORWARD_INFERENCE),
            ["EffectTokensWrapper", "FakifiedOutWrapper"],
        )
        self.assertEqual(
            ir.get_wrapper_order(WrapperStackSegment.AUTOGRAD_ASSEMBLY),
            ["RuntimeWrapper"],
        )

        # Dispatch wrappers should be reversed for runtime application
        self.assertEqual(
            ir.get_wrapper_order(
                WrapperStackSegment.DISPATCH, reverse_for_application=True
            ),
            ["AOTDedupeWrapper", "AOTSyntheticBaseWrapper"],
        )

        self.assertEqual(
            ir.wrapper_stack_metadata["EffectTokensWrapper"], {"token_count": 1}
        )
        self.assertEqual(
            ir.wrapper_stack_metadata["RuntimeWrapper"],
            {"indices_of_inps_to_detach": [0]},
        )
        self.assertEqual(
            ir.wrapper_stack_metadata["AOTDedupeWrapper"],
            {"keep_arg_mask": [True, False]},
        )


class TestValidateWrapperMetadataMerge(unittest.TestCase):
    """Tests for the validate_wrapper_metadata_merge helper function."""

    def test_validates_preservation_of_prior_order_entries(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {"forward": ["WrapperA", "WrapperB"]}
        prior_meta = {"WrapperA": {"token_count": 2}, "WrapperB": {"out_metas": []}}

        merged_order = {"forward": ["WrapperA", "WrapperB", "WrapperC"]}
        merged_meta = {
            "WrapperA": {"token_count": 2},
            "WrapperB": {"out_metas": []},
            "WrapperC": {"new_field": True},
        }

        self.assertTrue(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_detects_missing_segment_after_merge(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {"forward": ["WrapperA"], "dispatch": ["WrapperB"]}
        prior_meta = {}

        merged_order = {"forward": ["WrapperA"]}
        merged_meta = {}

        self.assertFalse(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_detects_missing_wrapper_in_segment(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {"forward": ["WrapperA", "WrapperB"]}
        prior_meta = {}

        merged_order = {"forward": ["WrapperA"]}
        merged_meta = {}

        self.assertFalse(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_detects_wrapper_order_changed(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {"forward": ["WrapperA", "WrapperB"]}
        prior_meta = {}

        merged_order = {"forward": ["WrapperB", "WrapperA"]}
        merged_meta = {}

        self.assertFalse(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_detects_missing_wrapper_metadata(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {}
        prior_meta = {"WrapperA": {"token_count": 1}}

        merged_order = {}
        merged_meta = {}

        self.assertFalse(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_detects_missing_metadata_key(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {}
        prior_meta = {"WrapperA": {"key1": "val1", "key2": "val2"}}

        merged_order = {}
        merged_meta = {"WrapperA": {"key1": "val1"}}

        self.assertFalse(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )

    def test_accepts_empty_prior_state(self):
        from torch._dynamo.pythonify.context import validate_wrapper_metadata_merge

        prior_order = {}
        prior_meta = {}

        merged_order = {"forward": ["WrapperA"]}
        merged_meta = {"WrapperA": {"key": "value"}}

        self.assertTrue(
            validate_wrapper_metadata_merge(
                merged_order, merged_meta, prior_order, prior_meta
            )
        )


class TestAddCompilationArtifactsMerge(unittest.TestCase):
    """Tests that add_compilation_artifacts correctly merges wrapper metadata."""

    def test_add_compilation_artifacts_preserves_prior_wrapper_order(self):
        from torch._dynamo.pythonify.context import (
            add_compilation_artifacts,
            get_merged_wrapper_stack_metadata,
            get_merged_wrapper_stack_order,
            pythonify_context,
            validate_wrapper_metadata_merge,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test.py") as ctx:
            artifact1 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["TokensWrapper", "FakifiedWrapper"]},
                wrapper_stack_metadata={
                    "TokensWrapper": {"token_count": 2},
                    "FakifiedWrapper": {"out_metas": ["m1"]},
                },
            )
            add_compilation_artifacts(artifact1)

            prior_order = dict(ctx.merged_wrapper_stack_order)
            prior_order = {k: list(v) for k, v in prior_order.items()}
            prior_meta = {
                k: dict(v) for k, v in ctx.merged_wrapper_stack_metadata.items()
            }

            artifact2 = CompilationArtifacts(
                wrapper_stack_order={
                    "forward": ["RuntimeWrapper"],
                    "dispatch": ["DedupeWrapper"],
                },
                wrapper_stack_metadata={
                    "RuntimeWrapper": {"detach_indices": [0]},
                    "DedupeWrapper": {"keep_mask": [True]},
                },
            )
            add_compilation_artifacts(artifact2)

            merged_order = get_merged_wrapper_stack_order()
            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertTrue(
                validate_wrapper_metadata_merge(
                    merged_order, merged_meta, prior_order, prior_meta
                )
            )

            self.assertEqual(
                merged_order["forward"],
                ["TokensWrapper", "FakifiedWrapper", "RuntimeWrapper"],
            )
            self.assertIn("dispatch", merged_order)
            self.assertEqual(merged_order["dispatch"], ["DedupeWrapper"])

            self.assertIn("TokensWrapper", merged_meta)
            self.assertIn("RuntimeWrapper", merged_meta)
            self.assertEqual(merged_meta["TokensWrapper"]["token_count"], 2)


class TestPythonifyContextWrapperMetadataMerging(unittest.TestCase):
    """
    Tests for PythonifyContext merging wrapper metadata through
    add_compilation_artifacts and _merge_inductor_outputs_into_artifacts.

    This test class verifies the full flow of wrapper metadata propagation
    through the pythonify context, ensuring that:
    1. Wrapper stack order is merged correctly across multiple artifacts
    2. Wrapper metadata is accumulated without loss
    3. The merge helpers work together end-to-end
    """

    def test_context_merges_wrapper_order_from_multiple_artifacts(self):
        from torch._dynamo.pythonify.context import (
            add_compilation_artifacts,
            get_merged_wrapper_stack_metadata,
            get_merged_wrapper_stack_order,
            pythonify_context,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test_context_merge.py"):
            artifact1 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["EffectTokensWrapper"]},
                wrapper_stack_metadata={"EffectTokensWrapper": {"token_count": 2}},
            )
            add_compilation_artifacts(artifact1)

            artifact2 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["FakifiedOutWrapper"]},
                wrapper_stack_metadata={"FakifiedOutWrapper": {"out_metas": []}},
            )
            add_compilation_artifacts(artifact2)

            artifact3 = CompilationArtifacts(
                wrapper_stack_order={"dispatch": ["AOTDedupeWrapper"]},
                wrapper_stack_metadata={"AOTDedupeWrapper": {"keep_mask": [True, False]}},
            )
            add_compilation_artifacts(artifact3)

            merged_order = get_merged_wrapper_stack_order()
            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(
                merged_order["forward"],
                ["EffectTokensWrapper", "FakifiedOutWrapper"],
            )
            self.assertEqual(merged_order["dispatch"], ["AOTDedupeWrapper"])

            self.assertEqual(
                merged_meta["EffectTokensWrapper"]["token_count"], 2
            )
            self.assertEqual(merged_meta["FakifiedOutWrapper"]["out_metas"], [])
            self.assertEqual(
                merged_meta["AOTDedupeWrapper"]["keep_mask"], [True, False]
            )

    def test_context_avoids_duplicate_wrappers_in_same_segment(self):
        from torch._dynamo.pythonify.context import (
            add_compilation_artifacts,
            get_merged_wrapper_stack_order,
            pythonify_context,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test_no_dupes.py"):
            artifact1 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["WrapperA", "WrapperB"]},
            )
            add_compilation_artifacts(artifact1)

            artifact2 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["WrapperB", "WrapperC"]},
            )
            add_compilation_artifacts(artifact2)

            merged_order = get_merged_wrapper_stack_order()

            self.assertEqual(
                merged_order["forward"],
                ["WrapperA", "WrapperB", "WrapperC"],
            )

    def test_context_preserves_metadata_on_duplicate_wrapper(self):
        from torch._dynamo.pythonify.context import (
            add_compilation_artifacts,
            get_merged_wrapper_stack_metadata,
            pythonify_context,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test_meta_update.py"):
            artifact1 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["WrapperA"]},
                wrapper_stack_metadata={"WrapperA": {"key1": "val1", "key2": "old"}},
            )
            add_compilation_artifacts(artifact1)

            artifact2 = CompilationArtifacts(
                wrapper_stack_order={"forward": ["WrapperA"]},
                wrapper_stack_metadata={"WrapperA": {"key2": "new", "key3": "val3"}},
            )
            add_compilation_artifacts(artifact2)

            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(merged_meta["WrapperA"]["key1"], "val1")
            self.assertEqual(merged_meta["WrapperA"]["key2"], "new")
            self.assertEqual(merged_meta["WrapperA"]["key3"], "val3")

    def test_context_end_to_end_with_inductor_merge(self):
        from torch._dynamo.pythonify.context import (
            _merge_inductor_outputs_into_artifacts,
            add_compilation_artifacts,
            get_merged_wrapper_stack_metadata,
            get_merged_wrapper_stack_order,
            get_pythonify_context,
            pythonify_context,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test_e2e.py"):
            artifact = CompilationArtifacts(
                wrapper_stack_order={"forward": ["EffectTokensWrapper"]},
                wrapper_stack_metadata={"EffectTokensWrapper": {"token_count": 1}},
            )
            add_compilation_artifacts(artifact)

            ctx = get_pythonify_context()
            ctx.forward_inductor_output = {
                "source_code": "def call(args): return args[0]",
                "graph_str": "test",
                "wrapper_stack_order": {"forward": ["RuntimeWrapper"]},
                "wrapper_stack_metadata": {"RuntimeWrapper": {"detach": [0, 1]}},
            }

            _merge_inductor_outputs_into_artifacts(ctx)

            self.assertEqual(
                artifact.wrapper_stack_order["forward"],
                ["EffectTokensWrapper", "RuntimeWrapper"],
            )
            self.assertIn("EffectTokensWrapper", artifact.wrapper_stack_metadata)
            self.assertIn("RuntimeWrapper", artifact.wrapper_stack_metadata)
            self.assertEqual(
                artifact.wrapper_stack_metadata["RuntimeWrapper"]["detach"],
                [0, 1],
            )

    def test_empty_context_returns_empty_dicts(self):
        from torch._dynamo.pythonify.context import (
            get_merged_wrapper_stack_metadata,
            get_merged_wrapper_stack_order,
        )

        order = get_merged_wrapper_stack_order()
        meta = get_merged_wrapper_stack_metadata()

        self.assertEqual(order, {})
        self.assertEqual(meta, {})

    def test_context_with_no_wrapper_metadata_artifacts(self):
        from torch._dynamo.pythonify.context import (
            add_compilation_artifacts,
            get_merged_wrapper_stack_metadata,
            get_merged_wrapper_stack_order,
            pythonify_context,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        with pythonify_context("/tmp/test_no_wrapper.py"):
            artifact = CompilationArtifacts(
                input_names=["x"],
                parameter_names=["W"],
            )
            add_compilation_artifacts(artifact)

            merged_order = get_merged_wrapper_stack_order()
            merged_meta = get_merged_wrapper_stack_metadata()

            self.assertEqual(merged_order, {})
            self.assertEqual(merged_meta, {})


class TestMergeInductorOutputsPreservesWrapperMetadata(unittest.TestCase):
    """Tests that _merge_inductor_outputs_into_artifacts preserves wrapper metadata."""

    def test_merge_preserves_existing_wrapper_metadata_on_artifact(self):
        from torch._dynamo.pythonify.context import (
            _merge_inductor_outputs_into_artifacts,
            PythonifyContext,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        ctx = PythonifyContext(pythonify_path="/tmp/test.py")

        artifact = CompilationArtifacts(
            wrapper_stack_order={"forward": ["TokensWrapper"]},
            wrapper_stack_metadata={"TokensWrapper": {"token_count": 3}},
        )
        ctx.artifacts_list.append(artifact)

        ctx.forward_inductor_output = {
            "source_code": "def call(args): pass",
            "graph_str": "test graph",
        }

        _merge_inductor_outputs_into_artifacts(ctx)

        self.assertEqual(artifact.inductor_source_code, "def call(args): pass")
        self.assertEqual(artifact.wrapper_stack_order, {"forward": ["TokensWrapper"]})
        self.assertEqual(
            artifact.wrapper_stack_metadata,
            {"TokensWrapper": {"token_count": 3}},
        )

    def test_merge_adds_wrapper_metadata_from_inductor_output(self):
        from torch._dynamo.pythonify.context import (
            _merge_inductor_outputs_into_artifacts,
            PythonifyContext,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        ctx = PythonifyContext(pythonify_path="/tmp/test.py")

        artifact = CompilationArtifacts(
            wrapper_stack_order={"forward": ["TokensWrapper"]},
            wrapper_stack_metadata={"TokensWrapper": {"token_count": 1}},
        )
        ctx.artifacts_list.append(artifact)

        ctx.forward_inductor_output = {
            "source_code": "def call(args): pass",
            "graph_str": "test graph",
            "wrapper_stack_order": {"forward": ["RuntimeWrapper"]},
            "wrapper_stack_metadata": {"RuntimeWrapper": {"detach": [0]}},
        }

        _merge_inductor_outputs_into_artifacts(ctx)

        self.assertEqual(
            artifact.wrapper_stack_order,
            {"forward": ["TokensWrapper", "RuntimeWrapper"]},
        )
        self.assertIn("TokensWrapper", artifact.wrapper_stack_metadata)
        self.assertIn("RuntimeWrapper", artifact.wrapper_stack_metadata)
        self.assertEqual(
            artifact.wrapper_stack_metadata["TokensWrapper"],
            {"token_count": 1},
        )
        self.assertEqual(
            artifact.wrapper_stack_metadata["RuntimeWrapper"],
            {"detach": [0]},
        )

    def test_merge_with_multiple_artifacts_preserves_each_wrapper_metadata(self):
        from torch._dynamo.pythonify.context import (
            _merge_inductor_outputs_into_artifacts,
            PythonifyContext,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        ctx = PythonifyContext(pythonify_path="/tmp/test.py")

        artifact1 = CompilationArtifacts(
            wrapper_stack_order={"forward": ["WrapperA"]},
            wrapper_stack_metadata={"WrapperA": {"key": "val1"}},
        )
        artifact2 = CompilationArtifacts(
            wrapper_stack_order={"forward": ["WrapperB"]},
            wrapper_stack_metadata={"WrapperB": {"key": "val2"}},
        )
        ctx.artifacts_list = [artifact1, artifact2]

        ctx.inductor_outputs = [
            {
                "source_code": "def call(args): return 1",
                "graph_str": "graph1",
                "is_backward": False,
                "wrapper_stack_order": {"forward": ["WrapperX"]},
                "wrapper_stack_metadata": {"WrapperX": {"extra": True}},
            },
            {
                "source_code": "def call(args): return 2",
                "graph_str": "graph2",
                "is_backward": False,
                "wrapper_stack_order": {"forward": ["WrapperY"]},
                "wrapper_stack_metadata": {"WrapperY": {"extra": False}},
            },
        ]

        _merge_inductor_outputs_into_artifacts(ctx)

        self.assertEqual(artifact1.inductor_source_code, "def call(args): return 1")
        self.assertEqual(
            artifact1.wrapper_stack_order,
            {"forward": ["WrapperA", "WrapperX"]},
        )
        self.assertIn("WrapperA", artifact1.wrapper_stack_metadata)
        self.assertIn("WrapperX", artifact1.wrapper_stack_metadata)

        self.assertEqual(artifact2.inductor_source_code, "def call(args): return 2")
        self.assertEqual(
            artifact2.wrapper_stack_order,
            {"forward": ["WrapperB", "WrapperY"]},
        )
        self.assertIn("WrapperB", artifact2.wrapper_stack_metadata)
        self.assertIn("WrapperY", artifact2.wrapper_stack_metadata)

    def test_merge_without_wrapper_metadata_is_noop(self):
        from torch._dynamo.pythonify.context import (
            _merge_inductor_outputs_into_artifacts,
            PythonifyContext,
        )
        from torch._dynamo.pythonify.pipeline import CompilationArtifacts

        ctx = PythonifyContext(pythonify_path="/tmp/test.py")

        artifact = CompilationArtifacts(
            wrapper_stack_order={"forward": ["TokensWrapper"]},
            wrapper_stack_metadata={"TokensWrapper": {"token_count": 5}},
        )
        ctx.artifacts_list.append(artifact)

        ctx.forward_inductor_output = {
            "source_code": "def call(args): pass",
            "graph_str": "test graph",
        }

        _merge_inductor_outputs_into_artifacts(ctx)

        self.assertEqual(artifact.wrapper_stack_order, {"forward": ["TokensWrapper"]})
        self.assertEqual(
            artifact.wrapper_stack_metadata,
            {"TokensWrapper": {"token_count": 5}},
        )


class TestAOTAutogradWrapperNodeLazyBackwardFields(unittest.TestCase):
    """
    Tests for the lazy backward and fw/bw stitching fields in AOTAutogradWrapperNode.

    These tests verify that the IR node correctly models the lazy backward
    compilation and forward/backward stitching behavior from
    AOTDispatchAutograd.post_compile.
    """

    def test_default_values_for_lazy_backward_fields(self):
        """Verify all lazy backward fields have safe defaults."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")

        self.assertFalse(node.has_lazy_backward)
        self.assertIsNone(node.lazy_bw_module)
        self.assertIsNone(node.lazy_bw_placeholder_list)
        self.assertIsNone(node.lazy_bw_saved_context)
        self.assertIsNone(node.lazy_bw_saved_compile_context)

    def test_default_values_for_saved_tensor_slice_fields(self):
        """Verify saved tensor slice fields have safe defaults."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")

        self.assertIsNone(node.tensors_saved_for_bw_with_vc_check_slice)
        self.assertIsNone(node.tensors_saved_for_bw_no_vc_check_slice)
        self.assertIsNone(node.symints_saved_for_bw_slice)
        self.assertEqual(node.num_symints_saved_for_bw, 0)
        self.assertEqual(node.dynamic_saved_tensors_idxs, {})

    def test_default_values_for_rng_state_fields(self):
        """Verify RNG state pairing fields have safe defaults."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")

        self.assertEqual(node.num_graphsafe_rng_states, 0)
        self.assertIsNone(node.graphsafe_rng_state_index)
        self.assertFalse(node.is_rng_op_functionalized)

    def test_default_values_for_autograd_assembly_fields(self):
        """Verify autograd assembly fields have safe defaults."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")

        self.assertEqual(node.backward_state_indices, [])
        self.assertEqual(node.indices_of_inps_to_detach, [])
        self.assertFalse(node.disable_amp)
        self.assertIsNone(node.maybe_subclass_meta)
        self.assertIsNone(node.fw_metadata)
        self.assertFalse(node.try_save_cache_entry_present)

    def test_is_lazy_backward_enabled_false_by_default(self):
        """Verify is_lazy_backward_enabled returns False by default."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")
        self.assertFalse(node.is_lazy_backward_enabled())

    def test_is_lazy_backward_enabled_true_when_configured(self):
        """Verify is_lazy_backward_enabled returns True when lazy bw is set up."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        self.assertTrue(node.is_lazy_backward_enabled())

    def test_is_lazy_backward_enabled_false_without_module(self):
        """Verify is_lazy_backward_enabled returns False if no module is set."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            has_lazy_backward=True,
            lazy_bw_module=None,
        )
        self.assertFalse(node.is_lazy_backward_enabled())

    def test_requires_rng_pairing_false_by_default(self):
        """Verify requires_rng_pairing returns False by default."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")
        self.assertFalse(node.requires_rng_pairing())

    def test_requires_rng_pairing_true_with_rng_states(self):
        """Verify requires_rng_pairing returns True with RNG states."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            num_graphsafe_rng_states=2,
        )
        self.assertTrue(node.requires_rng_pairing())

    def test_requires_saved_tensor_slicing_false_by_default(self):
        """Verify requires_saved_tensor_slicing returns False by default."""
        node = AOTAutogradWrapperNode(class_name="TestFunction")
        self.assertFalse(node.requires_saved_tensor_slicing())

    def test_requires_saved_tensor_slicing_with_vc_check_slice(self):
        """Verify requires_saved_tensor_slicing with VC check slice set."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            tensors_saved_for_bw_with_vc_check_slice=(0, 3),
        )
        self.assertTrue(node.requires_saved_tensor_slicing())

    def test_requires_saved_tensor_slicing_with_no_vc_check_slice(self):
        """Verify requires_saved_tensor_slicing with no VC check slice set."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            tensors_saved_for_bw_no_vc_check_slice=(3, 5),
        )
        self.assertTrue(node.requires_saved_tensor_slicing())

    def test_requires_saved_tensor_slicing_with_symints_slice(self):
        """Verify requires_saved_tensor_slicing with symints slice set."""
        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            symints_saved_for_bw_slice=(5, 7),
        )
        self.assertTrue(node.requires_saved_tensor_slicing())

    def test_full_lazy_backward_configuration(self):
        """Test a fully configured lazy backward node."""
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="backward_module",
            lazy_bw_placeholder_list=["ph1", "ph2", "ph3"],
            lazy_bw_saved_context={"tracing_key": "value"},
            lazy_bw_saved_compile_context={"compile_key": "value2"},
            tensors_saved_for_bw_with_vc_check_slice=(0, 2),
            tensors_saved_for_bw_no_vc_check_slice=(2, 4),
            symints_saved_for_bw_slice=(4, 5),
            num_symints_saved_for_bw=1,
            dynamic_saved_tensors_idxs={0: {0, 1}},
            num_graphsafe_rng_states=1,
            graphsafe_rng_state_index=0,
            is_rng_op_functionalized=True,
            backward_state_indices=[0],
            indices_of_inps_to_detach=[1, 2],
            disable_amp=True,
            maybe_subclass_meta={"grad_input_metas": []},
            fw_metadata={"num_outputs": 2},
            try_save_cache_entry_present=True,
        )

        self.assertTrue(node.is_lazy_backward_enabled())
        self.assertTrue(node.requires_rng_pairing())
        self.assertTrue(node.requires_saved_tensor_slicing())

        self.assertEqual(node.lazy_bw_placeholder_list, ["ph1", "ph2", "ph3"])
        self.assertEqual(node.tensors_saved_for_bw_with_vc_check_slice, (0, 2))
        self.assertEqual(node.tensors_saved_for_bw_no_vc_check_slice, (2, 4))
        self.assertEqual(node.symints_saved_for_bw_slice, (4, 5))
        self.assertEqual(node.dynamic_saved_tensors_idxs, {0: {0, 1}})
        self.assertEqual(node.backward_state_indices, [0])
        self.assertEqual(node.indices_of_inps_to_detach, [1, 2])

    def test_visitor_still_works_with_new_fields(self):
        """Verify visitor pattern still works with new lazy backward fields."""
        visitor = _RecordingVisitor()

        node = AOTAutogradWrapperNode(
            class_name="TestFunction",
            has_lazy_backward=True,
            lazy_bw_module="bw_module",
            num_graphsafe_rng_states=2,
            indices_of_inps_to_detach=[0, 1],
        )

        result = node.accept(visitor)

        self.assertEqual(result, "visit_aot_autograd_wrapper")
        self.assertEqual(len(visitor.visited), 1)
        self.assertEqual(
            visitor.visited[0],
            ("visit_aot_autograd_wrapper", "AOTAutogradWrapperNode"),
        )

    def test_node_in_ir_with_lazy_backward_fields(self):
        """Verify nodes with lazy backward fields work in RuntimeWrapperIR."""
        ir = RuntimeWrapperIR()

        node = AOTAutogradWrapperNode(
            class_name="LazyBackwardFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
            lazy_bw_placeholder_list=["x", "y"],
            tensors_saved_for_bw_with_vc_check_slice=(0, 2),
            num_symints_saved_for_bw=3,
            num_graphsafe_rng_states=1,
            backward_state_indices=[0],
        )

        ir.add_node(node)

        nodes = ir.get_nodes_by_type(AOTAutogradWrapperNode)
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes[0].is_lazy_backward_enabled())
        self.assertTrue(nodes[0].requires_rng_pairing())
        self.assertTrue(nodes[0].requires_saved_tensor_slicing())


if __name__ == "__main__":
    unittest.main()
