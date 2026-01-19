"""
Tests for pythonify wrapper code generation.

This module tests that the PythonCodeGenVisitor correctly generates code for
the wrapper IR nodes that mirror AOTAutograd post-compile wrapper behavior.

Each test verifies that:
1. Wrapper nodes with appropriate metadata generate helper functions
2. The generated code is syntactically valid Python
3. The generated helper functions match the semantics of runtime_wrappers.py
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch._dynamo.pythonify.gen_python import PythonCodeGenVisitor
from torch._dynamo.pythonify.ir import (
    AOTDedupeWrapperNode,
    AOTDispatchSubclassWrapperNode,
    AOTSyntheticBaseWrapperNode,
    DebugAssertWrapperNode,
    EffectTokensWrapperNode,
    FakifiedOutWrapperNode,
    FunctionalizedRngRuntimeWrapperNode,
    RuntimeWrapperIR,
    RuntimeWrapperNode,
)


class TestEffectTokensWrapperCodegen(TestCase):
    """Tests for EffectTokensWrapper code generation."""

    def test_no_tokens_generates_no_code(self):
        """When token_count is 0, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=0)
        result = visitor.visit_effect_tokens_wrapper(node)
        self.assertEqual(result, "")
        self.assertEqual(visitor.get_code(), "")

    def test_positive_tokens_generates_helpers(self):
        """When token_count > 0, inject/strip helpers should be generated."""
        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=3)
        visitor.visit_effect_tokens_wrapper(node)
        code = visitor.get_code()

        # Verify key elements are present
        self.assertIn("_effect_token_count = 3", code)
        self.assertIn("def _inject_effect_tokens(args):", code)
        self.assertIn("def _strip_effect_tokens(outputs):", code)
        self.assertIn("[None] * _effect_token_count", code)
        self.assertIn("outputs[_effect_token_count:]", code)

    def test_generated_inject_function_works(self):
        """The generated _inject_effect_tokens function works correctly."""
        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=2)
        visitor.visit_effect_tokens_wrapper(node)
        code = visitor.get_code()

        # Execute and verify
        namespace = {}
        exec(code, namespace)
        result = namespace["_inject_effect_tokens"]([1, 2, 3])
        self.assertEqual(result, [None, None, 1, 2, 3])

    def test_generated_strip_function_works(self):
        """The generated _strip_effect_tokens function works correctly."""
        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=2)
        visitor.visit_effect_tokens_wrapper(node)
        code = visitor.get_code()

        # Execute and verify
        namespace = {}
        exec(code, namespace)
        result = namespace["_strip_effect_tokens"]([None, None, 10, 20])
        self.assertEqual(result, [10, 20])

    def test_strip_handles_none_output(self):
        """_strip_effect_tokens handles None outputs gracefully."""
        visitor = PythonCodeGenVisitor()
        node = EffectTokensWrapperNode(token_count=2)
        visitor.visit_effect_tokens_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)
        result = namespace["_strip_effect_tokens"](None)
        self.assertIsNone(result)


class TestAOTDedupeWrapperCodegen(TestCase):
    """Tests for AOTDedupeWrapper code generation."""

    def test_no_dedupe_generates_no_code(self):
        """When needs_post_compile=False, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(needs_post_compile=False)
        result = visitor.visit_aot_dedupe_wrapper(node)
        self.assertEqual(result, "")

    def test_empty_mask_generates_no_code(self):
        """When keep_arg_mask is empty, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            needs_post_compile=True,
            keep_arg_mask=None,
        )
        result = visitor.visit_aot_dedupe_wrapper(node)
        self.assertEqual(result, "")

    def test_dedupe_generates_helpers(self):
        """When dedupe is needed, helper functions should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            keep_arg_mask=[True, True, False, True],
            add_dupe_map=[0, 1, 0, 2],
            needs_post_compile=True,
        )
        visitor.visit_aot_dedupe_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_dedupe_keep_arg_mask", code)
        self.assertIn("_dedupe_add_dupe_map", code)
        self.assertIn("def _remove_dupe_args(args):", code)
        self.assertIn("def _add_dupe_args(args):", code)

    def test_remove_dupe_args_works(self):
        """The generated _remove_dupe_args function works correctly."""
        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            keep_arg_mask=[True, True, False, True],
            add_dupe_map=[0, 1, 0, 2],
            needs_post_compile=True,
        )
        visitor.visit_aot_dedupe_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        # Test removal: [a, b, a, c] with mask [T, T, F, T] -> [a, b, c]
        args = ["a", "b", "a", "c"]
        result = namespace["_remove_dupe_args"](args)
        self.assertEqual(result, ["a", "b", "c"])

    def test_add_dupe_args_works(self):
        """The generated _add_dupe_args function works correctly."""
        visitor = PythonCodeGenVisitor()
        node = AOTDedupeWrapperNode(
            keep_arg_mask=[True, True, False, True],
            add_dupe_map=[0, 1, 0, 2],
            needs_post_compile=True,
        )
        visitor.visit_aot_dedupe_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        # Test reinsertion: [a, b, c] with map [0, 1, 0, 2] -> [a, b, a, c]
        deduped = ["a", "b", "c"]
        result = namespace["_add_dupe_args"](deduped)
        self.assertEqual(result, ["a", "b", "a", "c"])


class TestRuntimeWrapperCodegen(TestCase):
    """Tests for RuntimeWrapper code generation."""

    def test_no_detach_generates_no_code(self):
        """When no inputs to detach and no AMP, no code generated."""
        visitor = PythonCodeGenVisitor()
        node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[],
            disable_amp=False,
        )
        result = visitor.visit_runtime_wrapper(node)
        self.assertEqual(result, "")

    def test_detach_indices_generates_helper(self):
        """When inputs to detach, helper function should be generated."""
        visitor = PythonCodeGenVisitor()
        node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[0, 2],
        )
        visitor.visit_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_indices_to_detach = [0, 2]", code)
        self.assertIn("def _detach_inputs(args):", code)

    def test_detach_inputs_works(self):
        """The generated _detach_inputs function works correctly."""
        visitor = PythonCodeGenVisitor()
        node = RuntimeWrapperNode(
            indices_of_inps_to_detach=[0, 2],
        )
        visitor.visit_runtime_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        # Create tensors with requires_grad
        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=True)
        t3 = torch.tensor([3.0], requires_grad=True)

        args = [t1, t2, t3]
        result = namespace["_detach_inputs"](args)

        # Check that indices 0 and 2 are detached (requires_grad=False)
        self.assertFalse(result[0].requires_grad)
        self.assertTrue(result[1].requires_grad)  # Not detached
        self.assertFalse(result[2].requires_grad)


class TestDebugAssertWrapperCodegen(TestCase):
    """Tests for DebugAssertWrapper code generation."""

    def test_no_requires_grad_generates_no_code(self):
        """When flat_requires_grad is None, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(flat_requires_grad=None)
        result = visitor.visit_debug_assert_wrapper(node)
        self.assertEqual(result, "")

    def test_empty_requires_grad_generates_no_code(self):
        """When flat_requires_grad is empty, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(flat_requires_grad=[])
        result = visitor.visit_debug_assert_wrapper(node)
        self.assertEqual(result, "")

    def test_requires_grad_generates_assert_function(self):
        """When flat_requires_grad is set, assertion function is generated."""
        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False, True],
        )
        visitor.visit_debug_assert_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_expected_requires_grad = [True, False, True]", code)
        self.assertIn("def _assert_requires_grad(args):", code)

    def test_assert_requires_grad_passes_correct_inputs(self):
        """_assert_requires_grad passes when requirements match."""
        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False],
        )
        visitor.visit_debug_assert_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        # Create tensors with matching requires_grad
        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=False)

        # Should not raise
        namespace["_assert_requires_grad"]([t1, t2])

    def test_assert_requires_grad_raises_on_mismatch(self):
        """_assert_requires_grad raises when requirements don't match."""
        visitor = PythonCodeGenVisitor()
        node = DebugAssertWrapperNode(
            flat_requires_grad=[True, False],
        )
        visitor.visit_debug_assert_wrapper(node)
        code = visitor.get_code()

        namespace = {}
        exec(code, namespace)

        # Create tensors with mismatched requires_grad
        t1 = torch.tensor([1.0], requires_grad=False)  # Expected True
        t2 = torch.tensor([2.0], requires_grad=False)

        # Should raise AssertionError
        with self.assertRaises(AssertionError):
            namespace["_assert_requires_grad"]([t1, t2])


class TestFunctionalizedRngWrapperCodegen(TestCase):
    """Tests for FunctionalizedRngRuntimeWrapper code generation."""

    def test_no_rng_functionalization_generates_no_code(self):
        """When is_rng_op_functionalized=False, no code generated."""
        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=False,
        )
        result = visitor.visit_functionalized_rng_runtime_wrapper(node)
        self.assertEqual(result, "")

    def test_rng_functionalization_generates_helpers(self):
        """When RNG is functionalized, helper functions should be generated."""
        visitor = PythonCodeGenVisitor()
        node = FunctionalizedRngRuntimeWrapperNode(
            is_rng_op_functionalized=True,
            num_outputs_rng_offset=1,
            num_forward_returns=2,
        )
        visitor.visit_functionalized_rng_runtime_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_rng_num_outputs_offset = 1", code)
        self.assertIn("def _append_rng_state(args):", code)
        self.assertIn("def _handle_rng_outputs(outputs):", code)


class TestSubclassWrapperCodegen(TestCase):
    """Tests for AOTDispatchSubclassWrapper code generation."""

    def test_no_subclass_meta_generates_no_code(self):
        """When maybe_subclass_meta is None, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(maybe_subclass_meta=None)
        result = visitor.visit_aot_dispatch_subclass_wrapper(node)
        self.assertEqual(result, "")

    def test_subclass_meta_generates_unwrap_wrap_functions(self):
        """When subclass metadata is present, unwrap/wrap functions are generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
            num_fw_outs_saved_for_bw=3,
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("AOT dispatch subclass wrapper", code)
        self.assertIn("_num_fw_outs_saved_for_bw = 3", code)
        self.assertIn("def _unwrap_tensor_subclasses(args, subclass_metas=None):", code)
        self.assertIn("def _wrap_tensor_subclasses(unwrapped_outputs, subclass_metas, num_fw_outs_saved=None):", code)

    def test_unwrap_function_uses_tensor_flatten(self):
        """The unwrap function uses __tensor_flatten__ for subclasses."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("is_traceable_wrapper_subclass", code)
        self.assertIn("__tensor_flatten__", code)
        self.assertIn("def _flatten_subclass(x, meta, out):", code)

    def test_wrap_function_uses_creation_fn(self):
        """The wrap function uses creation_fn for SubclassCreationMeta."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        self.assertIn("creation_fn", code)
        self.assertIn("PlainTensorMeta", code)
        self.assertIn("unwrapped_idx", code)

    def test_generated_unwrap_function_is_syntactically_valid(self):
        """The generated _unwrap_tensor_subclasses function is valid Python."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch.utils._python_dispatch import is_traceable_wrapper_subclass", namespace)
        exec(code, namespace)

        self.assertIn("_unwrap_tensor_subclasses", namespace)
        self.assertTrue(callable(namespace["_unwrap_tensor_subclasses"]))

    def test_generated_wrap_function_is_syntactically_valid(self):
        """The generated _wrap_tensor_subclasses function is valid Python."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch.utils._python_dispatch import is_traceable_wrapper_subclass", namespace)
        exec(code, namespace)

        self.assertIn("_wrap_tensor_subclasses", namespace)
        self.assertTrue(callable(namespace["_wrap_tensor_subclasses"]))

    def test_unwrap_function_flattens_regular_tensors(self):
        """_unwrap_tensor_subclasses passes through regular tensors unchanged."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch.utils._python_dispatch import is_traceable_wrapper_subclass", namespace)
        exec(code, namespace)

        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        result = namespace["_unwrap_tensor_subclasses"]([t1, t2])

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t1)
        self.assertIs(result[1], t2)

    def test_wrap_function_with_none_metas_passthrough(self):
        """_wrap_tensor_subclasses with None metas returns outputs unchanged."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch.utils._python_dispatch import is_traceable_wrapper_subclass", namespace)
        exec(code, namespace)

        outputs = [torch.randn(2, 3), torch.randn(4, 5)]
        result = namespace["_wrap_tensor_subclasses"](outputs, None)

        self.assertIs(result, outputs)

    def test_imports_are_added(self):
        """The required imports are added to the generated code."""
        visitor = PythonCodeGenVisitor()
        node = AOTDispatchSubclassWrapperNode(
            maybe_subclass_meta={"some": "meta"},
        )
        visitor.visit_aot_dispatch_subclass_wrapper(node)

        imports = visitor._emitter.get_imports_block()
        self.assertIn("import torch", imports)
        self.assertIn(
            "from torch.utils._python_dispatch import is_traceable_wrapper_subclass",
            imports
        )


class TestSyntheticBaseWrapperCodegen(TestCase):
    """Tests for AOTSyntheticBaseWrapper code generation."""

    def test_no_synthetic_base_generates_no_code(self):
        """When synthetic_base_info is None, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info=None,
        )
        result = visitor.visit_aot_synthetic_base_wrapper(node)
        self.assertEqual(result, "")

    def test_no_post_compile_generates_no_code(self):
        """When needs_post_compile is False, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=False,
            synthetic_base_info={"some": "info"},
        )
        result = visitor.visit_aot_synthetic_base_wrapper(node)
        self.assertEqual(result, "")

    def test_synthetic_base_generates_helper_functions(self):
        """When synthetic base info is present, helper functions are generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[1, 3],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("AOT synthetic base wrapper", code)
        self.assertIn("_aliased_arg_idx_with_metadata_mutations = [1, 3]", code)
        self.assertIn("def _merge_aliased_inputs_to_synthetic_bases", code)
        self.assertIn("def _unpack_synthetic_bases", code)
        self.assertIn("def _apply_metadata_mutations", code)

    def test_unpack_function_uses_gen_alias_from_base(self):
        """The unpack function uses gen_alias_from_base for view reconstruction."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("gen_alias_from_base", code)
        self.assertIn("inner_base_idx, view_tensor = inner_idx_or_tuple", code)
        self.assertIn("replay_views", code)

    def test_merge_function_uses_merge_view_inputs(self):
        """The merge function calls merge_view_inputs from runtime_wrappers."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("merge_view_inputs", code)
        self.assertIn("old_input_info", code)
        self.assertIn("is_inference=_synthetic_base_is_inference", code)

    def test_apply_metadata_mutations_uses_as_strided(self):
        """The apply_metadata_mutations function uses as_strided_ for mutations."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[0, 2],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("as_strided_", code)
        self.assertIn("mutated_inp.size()", code)
        self.assertIn("mutated_inp.stride()", code)
        self.assertIn("mutated_inp.storage_offset()", code)

    def test_no_metadata_mutations_skips_apply_function(self):
        """When there are no metadata mutations, _apply_metadata_mutations is not generated."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("def _merge_aliased_inputs_to_synthetic_bases", code)
        self.assertIn("def _unpack_synthetic_bases", code)
        self.assertNotIn("def _apply_metadata_mutations", code)

    def test_inference_mode_sets_is_inference_true(self):
        """When trace_joint is False (inference), is_inference is True."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            trace_joint=False,
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_synthetic_base_is_inference = True", code)

    def test_training_mode_sets_is_inference_false(self):
        """When trace_joint is True (training), is_inference is False."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            trace_joint=True,
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_synthetic_base_is_inference = False", code)

    def test_generated_code_is_syntactically_valid(self):
        """The generated code compiles without syntax errors."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
            aliased_arg_idx_with_metadata_mutations=[0, 1],
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch._functorch._aot_autograd.runtime_wrappers import merge_view_inputs", namespace)
        exec("from torch._functorch._aot_autograd.functional_utils import gen_alias_from_base", namespace)
        exec("from torch._functorch import config as functorch_config", namespace)
        exec(code, namespace)

        self.assertIn("_merge_aliased_inputs_to_synthetic_bases", namespace)
        self.assertIn("_unpack_synthetic_bases", namespace)
        self.assertIn("_apply_metadata_mutations", namespace)

    def test_unpack_synthetic_bases_direct_index(self):
        """_unpack_synthetic_bases handles direct int indices correctly."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch._functorch._aot_autograd.runtime_wrappers import merge_view_inputs", namespace)
        exec("from torch._functorch._aot_autograd.functional_utils import gen_alias_from_base", namespace)
        exec("from torch._functorch import config as functorch_config", namespace)
        exec(code, namespace)

        t0 = torch.randn(2, 3)
        t1 = torch.randn(4, 5)
        result = namespace["_unpack_synthetic_bases"]([t0, t1], [0, 1])

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t0)
        self.assertIs(result[1], t1)

    def test_unpack_synthetic_bases_none_info(self):
        """_unpack_synthetic_bases with None info returns list of inputs."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec("from torch._functorch._aot_autograd.runtime_wrappers import merge_view_inputs", namespace)
        exec("from torch._functorch._aot_autograd.functional_utils import gen_alias_from_base", namespace)
        exec("from torch._functorch import config as functorch_config", namespace)
        exec(code, namespace)

        t0 = torch.randn(2, 3)
        t1 = torch.randn(4, 5)
        result = namespace["_unpack_synthetic_bases"]([t0, t1], None)

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], t0)
        self.assertIs(result[1], t1)

    def test_imports_are_added(self):
        """The required imports are added to the generated code."""
        visitor = PythonCodeGenVisitor()
        node = AOTSyntheticBaseWrapperNode(
            needs_post_compile=True,
            synthetic_base_info={"some": "info"},
        )
        visitor.visit_aot_synthetic_base_wrapper(node)

        imports = visitor._emitter.get_imports_block()
        self.assertIn("import torch", imports)
        self.assertIn("merge_view_inputs", imports)
        self.assertIn("gen_alias_from_base", imports)
        self.assertIn("functorch_config", imports)


class TestFakifiedOutWrapperCodegen(TestCase):
    """Tests for FakifiedOutWrapper code generation."""

    def test_no_out_metas_generates_no_code(self):
        """When out_metas is None, no code should be generated."""
        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(out_metas=None)
        result = visitor.visit_fakified_out_wrapper(node)
        self.assertEqual(result, "")

    def test_out_metas_generates_comments(self):
        """When out_metas is present, comments are generated."""
        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(out_metas={"some": "meta"})
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        self.assertIn("Fakified output wrapper", code)

    def test_fwd_output_strides_generates_function(self):
        """When fwd_output_strides is provided, _fix_output_strides is generated."""
        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[4, 1], [8, 2, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        self.assertIn("_fwd_output_strides = [[4, 1], [8, 2, 1]]", code)
        self.assertIn("def _fix_output_strides(outputs, expected_strides=None):", code)
        self.assertIn("as_strided", code)
        self.assertIn("actual_strides = result[i].stride()", code)

    def test_fix_output_strides_function_works_matching_strides(self):
        """The generated _fix_output_strides returns tensor unchanged when strides match."""
        import torch

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[4, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        result = namespace["_fix_output_strides"]([t])
        self.assertEqual(result[0].shape, t.shape)
        self.assertEqual(result[0].stride(), t.stride())

    def test_fix_output_strides_function_corrects_mismatched_strides(self):
        """The generated _fix_output_strides applies as_strided for mismatched strides."""
        import torch

        visitor = PythonCodeGenVisitor()
        expected_strides = [1, 2]
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[expected_strides],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        original_strides = t.stride()
        self.assertNotEqual(list(original_strides), expected_strides)
        result = namespace["_fix_output_strides"]([t])
        self.assertEqual(list(result[0].stride()), expected_strides)

    def test_fix_output_strides_handles_none_strides(self):
        """The generated _fix_output_strides skips tensors with None expected strides."""
        import torch

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[None, [4, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.randn(3, 3)
        t2 = torch.randn(2, 4)
        original_t1_stride = t1.stride()
        result = namespace["_fix_output_strides"]([t1, t2])
        self.assertEqual(result[0].stride(), original_t1_stride)
        self.assertEqual(list(result[1].stride()), [4, 1])

    def test_fix_output_strides_handles_single_output(self):
        """The generated _fix_output_strides handles single tensor (not list) output."""
        import torch

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[1, 2]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t = torch.randn(2, 4)
        result = namespace["_fix_output_strides"](t)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(list(result.stride()), [1, 2])

    def test_fix_output_strides_handles_non_tensor_outputs(self):
        """The generated _fix_output_strides skips non-tensor outputs."""
        import torch

        visitor = PythonCodeGenVisitor()
        node = FakifiedOutWrapperNode(
            out_metas={"some": "meta"},
            fwd_output_strides=[[4, 1], None, [2, 1]],
        )
        visitor.visit_fakified_out_wrapper(node)
        code = visitor.get_code()

        namespace = {"torch": torch}
        exec(code, namespace)

        t1 = torch.randn(2, 4)
        non_tensor = 42
        t2 = torch.randn(3, 2)
        result = namespace["_fix_output_strides"]([t1, non_tensor, t2])
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[1], 42)
        self.assertIsInstance(result[2], torch.Tensor)


class TestMultipleWrappersCodegen(TestCase):
    """Tests for generating code with multiple wrappers."""

    def test_multiple_wrappers_share_section_header(self):
        """Multiple wrapper nodes should share the section header."""
        visitor = PythonCodeGenVisitor()

        # Visit multiple wrapper nodes
        node1 = EffectTokensWrapperNode(token_count=2)
        node2 = AOTDedupeWrapperNode(
            keep_arg_mask=[True, False],
            add_dupe_map=[0, 0],
            needs_post_compile=True,
        )
        node3 = RuntimeWrapperNode(indices_of_inps_to_detach=[0])

        visitor.visit_effect_tokens_wrapper(node1)
        visitor.visit_aot_dedupe_wrapper(node2)
        visitor.visit_runtime_wrapper(node3)

        code = visitor.get_code()

        # Only one section header should appear
        self.assertEqual(code.count("Wrapper Helper Functions"), 1)

        # All wrappers should have their code
        self.assertIn("_inject_effect_tokens", code)
        self.assertIn("_remove_dupe_args", code)
        self.assertIn("_detach_inputs", code)


class TestWrapperCodegenIntegration(TestCase):
    """Integration tests for wrapper code generation with IR."""

    def test_ir_with_wrapper_nodes(self):
        """Test that IR with wrapper nodes generates correct code."""
        ir = RuntimeWrapperIR()

        # Add wrapper nodes to IR
        ir.add_node(EffectTokensWrapperNode(token_count=1))
        ir.add_node(RuntimeWrapperNode(indices_of_inps_to_detach=[0, 1]))
        ir.add_node(DebugAssertWrapperNode(flat_requires_grad=[True, True]))

        # Visit all nodes
        visitor = PythonCodeGenVisitor()
        ir.accept_all(visitor)
        code = visitor.get_code()

        # Verify all wrapper code is present
        self.assertIn("_effect_token_count = 1", code)
        self.assertIn("_indices_to_detach = [0, 1]", code)
        self.assertIn("_expected_requires_grad = [True, True]", code)


class TestLazyBackwardCodegen(TestCase):
    """Tests for lazy backward compilation code generation."""

    def test_lazy_backward_state_variables_emitted(self):
        """When has_lazy_backward=True, state variables should be emitted."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",  # Non-None to trigger codegen
        )
        # Need to set up for backward kernel to exist
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify lazy backward state variables are present
        self.assertIn("_lazy_bw_module_id", code)
        self.assertIn("_lazy_bw_placeholder_list_id", code)
        self.assertIn("_lazy_bw_saved_context_id", code)
        self.assertIn("_lazy_bw_saved_compile_context_id", code)
        self.assertIn("_compiled_bw = None", code)

    def test_lazy_backward_compilation_logic_emitted(self):
        """When has_lazy_backward=True, compilation logic should be in backward()."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify lazy backward compilation logic is present
        self.assertIn("global _compiled_bw", code)
        self.assertIn("if _compiled_bw is None:", code)
        self.assertIn("obj_from_id(_lazy_bw_module_id)", code)

    def test_no_lazy_backward_when_flag_false(self):
        """When has_lazy_backward=False, no lazy backward code should be emitted."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=False,
        )
        visitor._has_inductor_kernel = True
        visitor._has_inductor_backward_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify no lazy backward code
        self.assertNotIn("_lazy_bw_module_id", code)
        self.assertNotIn("global _compiled_bw", code)

    def test_lazy_backward_uses_contexts(self):
        """Lazy backward compilation should use saved tracing/compile contexts."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify context restoration code is present
        self.assertIn("from torch._guards import tracing, compile_context", code)
        self.assertIn("tracing(_saved_context)", code)
        self.assertIn("compile_context(_saved_compile_context)", code)

    def test_lazy_backward_calls_compiled_bw(self):
        """Lazy backward should call _compiled_bw after compilation."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify compiled backward is called
        self.assertIn("_bw_result = _compiled_bw(backward_inputs)", code)

    def test_lazy_backward_with_object_ids(self):
        """Lazy backward state should store object IDs when objects provided."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        class FakeModule:
            pass

        fake_module = FakeModule()
        fake_placeholders = ["placeholder_1", "placeholder_2"]

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module=fake_module,
            lazy_bw_placeholder_list=fake_placeholders,
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # The object IDs should be present (actual values depend on memory)
        self.assertIn(f"_lazy_bw_module_id = {id(fake_module)}", code)
        self.assertIn(f"_lazy_bw_placeholder_list_id = {id(fake_placeholders)}", code)

    def test_lazy_backward_has_backward_true(self):
        """has_backward should be True when has_lazy_backward is True."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Should have a backward method that doesn't raise RuntimeError
        self.assertIn("def backward(ctx, *grad_outputs):", code)
        self.assertNotIn(
            "Backward pass not available: this model was compiled in inference mode",
            code
        )

    def test_lazy_backward_generated_code_is_valid_python(self):
        """The generated lazy backward code should be syntactically valid Python."""
        from torch._dynamo.pythonify.ir import AOTAutogradWrapperNode

        visitor = PythonCodeGenVisitor()
        node = AOTAutogradWrapperNode(
            class_name="CompiledFunction",
            has_lazy_backward=True,
            lazy_bw_module="fake_module",
        )
        visitor._has_inductor_kernel = True
        visitor.visit_aot_autograd_wrapper(node)
        code = visitor.get_code()

        # Verify code can be compiled (syntax check)
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}\nCode:\n{code}")


if __name__ == "__main__":
    run_tests()
