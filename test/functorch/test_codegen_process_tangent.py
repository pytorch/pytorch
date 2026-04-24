# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the process_runtime_tangent in aot_autograd.

The codegen'd tangent processing replaces the per-tangent
process_runtime_tangent loop with straight-line code that inlines
the memory format coercion for each tangent with baked-in format
metadata. For PlainTensorMeta tangents (the common case), this
eliminates multiple isinstance checks and function calls per tangent.

Tests verify that a "process_tangents" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._dynamo
from torch.testing._internal.common_utils import run_tests, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenProcessTangent(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    @contextmanager
    def _capture_codegen_source(self, artifact_name):
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

    def test_codegen_emitted_for_training(self):
        """
        Training path should emit a process_tangents codegen artifact.
        """
        with self._capture_codegen_source("process_tangents") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            x = torch.randn(4, requires_grad=True)
            out = f(x)
            out.sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _process_tangents", source)
        self.assertIn("_coerce_", source)

    def test_no_codegen_for_inference(self):
        """
        Inference path should not emit a process_tangents artifact.
        """
        with self._capture_codegen_source("process_tangents") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            f(torch.randn(4))

        self.assertEqual(
            len(captured),
            0,
            "No codegen should be emitted for inference",
        )

    def test_backward_correctness_single_input(self):
        """
        Verify backward correctness through the codegen'd tangent
        processing with a single input.
        """

        @torch.compile(backend="aot_eager")
        def f(x):
            return x * 3 + 1

        x = torch.randn(4, requires_grad=True)
        out = f(x)
        out.sum().backward()

        self.assertEqual(x.grad, torch.full((4,), 3.0))

    def test_backward_correctness_multiple_inputs(self):
        """
        Verify backward correctness with multiple inputs. Each tangent
        gets its own inline coercion in the codegen.
        """

        @torch.compile(backend="aot_eager")
        def f(x, y):
            return x * y + x

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        out = f(x, y)
        out.sum().backward()

        self.assertEqual(x.grad, y + 1)
        self.assertEqual(y.grad, x)

    def test_backward_correctness_multiple_outputs(self):
        """
        Multiple outputs produce multiple tangents. The codegen should
        handle all of them.
        """

        @torch.compile(backend="aot_eager")
        def f(x):
            return x * 2, x * 3

        x = torch.randn(4, requires_grad=True)
        out1, out2 = f(x)
        (out1.sum() + out2.sum()).backward()

        self.assertEqual(x.grad, torch.full((4,), 5.0))

    def test_per_tangent_format_baked(self):
        """
        Each tangent should have its memory format baked into the
        codegen as a separate constant.
        """
        with self._capture_codegen_source("process_tangents") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y, x * y

            x = torch.randn(3, requires_grad=True)
            y = torch.randn(3, requires_grad=True)
            o1, o2 = f(x, y)
            (o1.sum() + o2.sum()).backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("_fmt_0", source)

    def test_backward_with_non_differentiable_output(self):
        """
        When some outputs don't require grad, there should be fewer
        tangents. Codegen should handle this correctly.
        """

        @torch.compile(backend="aot_eager")
        def f(x, y):
            return x * 2, y.detach()

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        out1, out2 = f(x, y)

        self.assertTrue(out1.requires_grad)
        self.assertFalse(out2.requires_grad)

        out1.sum().backward()
        self.assertEqual(x.grad, torch.full((4,), 2.0))

    def test_codegen_source_structure(self):
        """
        Verify the structure of the generated source code.
        """
        with self._capture_codegen_source("process_tangents") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x + 1

            x = torch.randn(4, requires_grad=True)
            f(x).sum().backward()

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("all_args[:start]", source)
        self.assertIn("all_args[end:]", source)
        self.assertIn("isinstance", source)


if __name__ == "__main__":
    run_tests()
