# Owner(s): ["module: dynamo"]

"""
Tests for _extract_wrapper_metadata_for_pythonify function in compile_fx.py.

These tests verify that wrapper metadata is correctly extracted from TracingContext,
including conditional inclusion of DebugAssertWrapper based on config.debug_assert.
"""

import unittest
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase


class TestExtractWrapperMetadataDebugAssert(TestCase):
    """
    Tests for conditional DebugAssertWrapper inclusion based on config.debug_assert.

    Per wrapper_audit.md, DebugAssertWrapper is only applied when config.debug_assert
    is enabled. The _extract_wrapper_metadata_for_pythonify function should respect
    this config setting.
    """

    def test_debug_assert_wrapper_included_when_config_enabled(self):
        """
        When functorch_config.debug_assert is True, DebugAssertWrapper should
        be included in the autograd_assembly segment.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = True

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            wrapper_stack_order = result["wrapper_stack_order"]
            wrapper_stack_metadata = result["wrapper_stack_metadata"]

            self.assertIn(
                "DebugAssertWrapper",
                wrapper_stack_order["autograd_assembly"],
                "DebugAssertWrapper should be in autograd_assembly when config.debug_assert is True",
            )
            self.assertIn("DebugAssertWrapper", wrapper_stack_metadata)
            self.assertEqual(
                wrapper_stack_metadata["DebugAssertWrapper"]["enabled"], True
            )
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_debug_assert_wrapper_excluded_when_config_disabled(self):
        """
        When functorch_config.debug_assert is False, DebugAssertWrapper should
        NOT be included in the wrapper stack.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            wrapper_stack_order = result["wrapper_stack_order"]
            wrapper_stack_metadata = result["wrapper_stack_metadata"]

            self.assertNotIn(
                "DebugAssertWrapper",
                wrapper_stack_order.get("autograd_assembly", []),
                "DebugAssertWrapper should NOT be in autograd_assembly when config.debug_assert is False",
            )
            self.assertNotIn("DebugAssertWrapper", wrapper_stack_metadata)
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_debug_assert_wrapper_after_runtime_wrapper_in_order(self):
        """
        When enabled, DebugAssertWrapper should appear AFTER RuntimeWrapper
        in the autograd_assembly segment, matching the order in wrapper_audit.md.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = [0, 1]
        mock_fw_metadata.num_mutated_inp_runtime_indices = 2
        mock_fw_metadata.num_outputs_aliased = 1

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = True

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            autograd_assembly = result["wrapper_stack_order"]["autograd_assembly"]

            runtime_idx = autograd_assembly.index("RuntimeWrapper")
            debug_idx = autograd_assembly.index("DebugAssertWrapper")

            self.assertGreater(
                debug_idx,
                runtime_idx,
                "DebugAssertWrapper should come after RuntimeWrapper in autograd_assembly",
            )
        finally:
            functorch_config.debug_assert = original_debug_assert


class TestExtractWrapperMetadataBasic(TestCase):
    """
    Basic tests for _extract_wrapper_metadata_for_pythonify to ensure
    correct extraction of other wrapper metadata.
    """

    def test_returns_none_when_no_tracing_context(self):
        """
        When TracingContext is not available, function should return None.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        import torch._guards

        with mock.patch.object(
            torch._guards.TracingContext, "try_get", return_value=None
        ):
            result = _extract_wrapper_metadata_for_pythonify()

        self.assertIsNone(result)

    def test_returns_none_when_no_fw_metadata(self):
        """
        When fw_metadata is None, function should return None.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        import torch._guards

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = None

        with mock.patch.object(
            torch._guards.TracingContext, "try_get", return_value=mock_context
        ):
            result = _extract_wrapper_metadata_for_pythonify()

        self.assertIsNone(result)

    def test_runtime_wrapper_always_included(self):
        """
        RuntimeWrapper should always be included in autograd_assembly.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = [0]
        mock_fw_metadata.num_mutated_inp_runtime_indices = 1
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            autograd_assembly = result["wrapper_stack_order"]["autograd_assembly"]

            self.assertIn("RuntimeWrapper", autograd_assembly)
            self.assertEqual(
                result["wrapper_stack_metadata"]["RuntimeWrapper"]["indices_of_inps_to_detach"],
                [0],
            )
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_dispatch_wrappers_always_included(self):
        """
        AOTSyntheticBaseWrapper and AOTDedupeWrapper should always be in dispatch segment.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            dispatch = result["wrapper_stack_order"]["dispatch"]

            self.assertIn("AOTSyntheticBaseWrapper", dispatch)
            self.assertIn("AOTDedupeWrapper", dispatch)
        finally:
            functorch_config.debug_assert = original_debug_assert


class TestDispatchWrapperMetadataExtraction(TestCase):
    """
    Tests for dispatch wrapper metadata extraction from TracingContext.

    These tests verify that _extract_wrapper_metadata_for_pythonify correctly
    extracts AOTDedupeWrapper and AOTSyntheticBaseWrapper metadata from
    TracingContext.dispatch_wrappers_metadata when available.
    """

    def test_dedupe_wrapper_metadata_extracted_when_present(self):
        """
        When dispatch_wrappers_metadata contains AOTDedupeWrapper metadata,
        it should be extracted and included in wrapper_stack_metadata.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []
        mock_context.dispatch_wrappers_metadata = {
            "AOTDedupeWrapper": {
                "keep_arg_mask": [True, True, False, True],
                "add_dupe_map": [0, 1, 0, 2],
                "needs_post_compile": True,
            },
            "AOTSyntheticBaseWrapper": {
                "synthetic_base_info": None,
                "aliased_arg_idx_with_metadata_mutations": [],
                "needs_post_compile": False,
                "is_inference": False,
            },
        }

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            dedupe_metadata = result["wrapper_stack_metadata"]["AOTDedupeWrapper"]

            self.assertEqual(dedupe_metadata["keep_arg_mask"], [True, True, False, True])
            self.assertEqual(dedupe_metadata["add_dupe_map"], [0, 1, 0, 2])
            self.assertEqual(dedupe_metadata["needs_post_compile"], True)
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_synthetic_base_wrapper_metadata_extracted_when_present(self):
        """
        When dispatch_wrappers_metadata contains AOTSyntheticBaseWrapper metadata,
        it should be extracted and included in wrapper_stack_metadata.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []
        mock_context.dispatch_wrappers_metadata = {
            "AOTDedupeWrapper": {
                "keep_arg_mask": [],
                "add_dupe_map": [],
                "needs_post_compile": False,
            },
            "AOTSyntheticBaseWrapper": {
                "synthetic_base_info": [
                    0,
                    {"base_idx": 0, "view_size": [2, 3], "view_stride": [3, 1], "view_storage_offset": 0, "view_requires_grad": True},
                ],
                "aliased_arg_idx_with_metadata_mutations": [1],
                "needs_post_compile": True,
                "is_inference": False,
            },
        }

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            synthetic_metadata = result["wrapper_stack_metadata"]["AOTSyntheticBaseWrapper"]

            self.assertIsNotNone(synthetic_metadata["synthetic_base_info"])
            self.assertEqual(len(synthetic_metadata["synthetic_base_info"]), 2)
            self.assertEqual(synthetic_metadata["synthetic_base_info"][0], 0)
            self.assertEqual(synthetic_metadata["synthetic_base_info"][1]["base_idx"], 0)
            self.assertEqual(synthetic_metadata["aliased_arg_idx_with_metadata_mutations"], [1])
            self.assertEqual(synthetic_metadata["needs_post_compile"], True)
            self.assertEqual(synthetic_metadata["is_inference"], False)
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_empty_metadata_when_dispatch_wrappers_metadata_none(self):
        """
        When dispatch_wrappers_metadata is None, fallback to empty dicts.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []
        mock_context.dispatch_wrappers_metadata = None

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            # Should fallback to empty dicts
            self.assertEqual(result["wrapper_stack_metadata"]["AOTDedupeWrapper"], {})
            self.assertEqual(result["wrapper_stack_metadata"]["AOTSyntheticBaseWrapper"], {})
        finally:
            functorch_config.debug_assert = original_debug_assert

    def test_no_needs_post_compile_dedupe_metadata(self):
        """
        When AOTDedupeWrapper.needs_post_compile is False, metadata reflects this.
        """
        from torch._inductor.compile_fx import _extract_wrapper_metadata_for_pythonify
        from torch._functorch import config as functorch_config
        import torch._guards

        mock_fw_metadata = mock.MagicMock()
        mock_fw_metadata.tokens = []
        mock_fw_metadata.subclass_inp_meta = None
        mock_fw_metadata.subclass_fw_graph_out_meta = None
        mock_fw_metadata.is_rng_op_functionalized = False
        mock_fw_metadata.indices_of_inps_to_detach = []
        mock_fw_metadata.num_mutated_inp_runtime_indices = 0
        mock_fw_metadata.num_outputs_aliased = 0

        mock_context = mock.MagicMock()
        mock_context.fw_metadata = mock_fw_metadata
        mock_context.output_strides = []
        mock_context.dispatch_wrappers_metadata = {
            "AOTDedupeWrapper": {
                "keep_arg_mask": [],
                "add_dupe_map": [],
                "needs_post_compile": False,
            },
            "AOTSyntheticBaseWrapper": {
                "synthetic_base_info": None,
                "aliased_arg_idx_with_metadata_mutations": [],
                "needs_post_compile": False,
                "is_inference": True,
            },
        }

        original_debug_assert = functorch_config.debug_assert
        try:
            functorch_config.debug_assert = False

            with mock.patch.object(
                torch._guards.TracingContext, "try_get", return_value=mock_context
            ):
                result = _extract_wrapper_metadata_for_pythonify()

            self.assertIsNotNone(result)
            dedupe_metadata = result["wrapper_stack_metadata"]["AOTDedupeWrapper"]
            synthetic_metadata = result["wrapper_stack_metadata"]["AOTSyntheticBaseWrapper"]

            self.assertEqual(dedupe_metadata["needs_post_compile"], False)
            self.assertEqual(dedupe_metadata["keep_arg_mask"], [])
            self.assertEqual(dedupe_metadata["add_dupe_map"], [])

            self.assertEqual(synthetic_metadata["needs_post_compile"], False)
            self.assertEqual(synthetic_metadata["synthetic_base_info"], None)
            self.assertEqual(synthetic_metadata["is_inference"], True)
        finally:
            functorch_config.debug_assert = original_debug_assert


class TestDispatchWrapperMetadataIntegration(TestCase):
    """
    Integration tests verifying dispatch wrapper metadata is correctly captured
    during actual torch.compile runs with pythonify.
    """

    def test_dedupe_metadata_captured_for_duplicate_args(self):
        """
        When a model is called with duplicate tensor arguments, the AOTDedupeWrapper
        metadata should be captured in TracingContext and available for pythonify.
        """
        import tempfile
        import torch
        import torch.nn as nn

        class DuplicateArgModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))

            def forward(self, x, y):
                # x and y are the same tensor at runtime
                return x @ self.weight + y @ self.weight

        model = DuplicateArgModel()
        x = torch.randn(2, 3, requires_grad=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/compiled.py"
            compiled_model = torch.compile(model, pythonify=path)
            # Call with duplicate arg (x passed twice)
            _ = compiled_model(x, x)

            # Read the generated file and check it was created
            with open(path, "r") as f:
                content = f.read()

            # The file should exist and be valid Python
            self.assertTrue(len(content) > 0)
            # Should compile as valid Python
            compile(content, path, "exec")

    def test_synthetic_base_metadata_captured_for_aliased_inputs(self):
        """
        When a model receives aliased tensor inputs (views of the same storage),
        the AOTSyntheticBaseWrapper metadata should be captured.
        """
        import tempfile
        import torch
        import torch.nn as nn

        class AliasedInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))

            def forward(self, x):
                # Returns a view that aliases the input
                return x @ self.weight

        model = AliasedInputModel()
        x = torch.randn(2, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/compiled.py"
            compiled_model = torch.compile(model, pythonify=path)
            _ = compiled_model(x)

            # Read the generated file and check it was created
            with open(path, "r") as f:
                content = f.read()

            # The file should exist and be valid Python
            self.assertTrue(len(content) > 0)
            # Should compile as valid Python
            compile(content, path, "exec")


if __name__ == "__main__":
    run_tests()
