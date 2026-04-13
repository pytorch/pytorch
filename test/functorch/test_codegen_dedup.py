# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing AOTDedupeWrapper.

The codegen'd remove_dupe_args emits straight-line index selections like
[args[0], args[2], args[5]] with all indices baked in as literals, replacing
the closure-based zip + filter over keep_arg_mask.

Strategy 2 (the dedup post_compile path) is triggered when duplicate args
have mutations, so strategy 1 (leafification) can't handle them. Dynamo
already deduplicates inputs, so these tests use aot_function directly.

Tests verify that a "dedup_wrapper" artifact is emitted via trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")


def _nop_compiler(gm, example_inputs):  # type: ignore[no-untyped-def]
    return gm.forward


class TestCodegenDedup(TestCase):
    @contextmanager
    def _capture_codegen_source(self, artifact_name):
        """Capture codegen artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == artifact_name
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_duplicate_args_with_mutation(self):
        """
        When the same tensor is passed as two args and one position mutates,
        strategy 2 dedup kicks in. The codegen should emit straight-line
        arg selection.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b):
                b.mul_(2)
                return a + b

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            x_ref = x.clone()
            out = compiled_f(x, x)

        self.assertEqual(x, x_ref * 2)
        self.assertEqual(out, x_ref * 4)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)

    def test_three_way_duplicate_with_mutation(self):
        """
        Three-way duplication where the last position mutates.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b, c):
                c.add_(1)
                return a + b + c

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            x_ref = x.clone()
            out = compiled_f(x, x, x)

        self.assertEqual(x, x_ref + 1)
        self.assertEqual(out, (x_ref + 1) * 3)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)
        self.assertNotIn("args[2]", source)

    def test_partial_duplicate_with_mutation(self):
        """
        Two args are duplicates (with mutation), third is distinct.
        Codegen should select [args[0], args[2]] dropping the duplicate.
        """
        with self._capture_codegen_source("dedup_wrapper") as captured:

            def f(a, b, c):
                b.mul_(3)
                return a + b + c

            compiled_f = aot_function(f, _nop_compiler)
            x = torch.randn(4)
            y = torch.randn(4)
            x_ref = x.clone()
            y_ref = y.clone()
            out = compiled_f(x, x, y)

        self.assertEqual(x, x_ref * 3)
        self.assertEqual(out, x_ref * 3 + x_ref * 3 + y_ref)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected dedup_wrapper codegen artifact to be emitted",
        )
        source = captured[0]
        self.assertIn("args[0]", source)
        self.assertNotIn("args[1]", source)
        self.assertIn("args[2]", source)


if __name__ == "__main__":
    run_tests()
