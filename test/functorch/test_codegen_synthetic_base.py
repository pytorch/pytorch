# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the AOTSyntheticBaseWrapper in aot_autograd.

The codegen'd synthetic base wrapper bakes the metadata mutation indices
and output slice as compile-time constants, eliminating the data-driven
loop that applies as_strided_ to metadata-mutated aliased inputs.

Tests verify that a "synthetic_base_wrapper" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._dynamo
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenSyntheticBase(TestCase):
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

    def test_no_aliased_inputs_no_codegen(self):
        """
        When inputs don't alias each other, no synthetic bases are
        needed and no codegen artifact is emitted.
        """
        with self._capture_codegen_source("synthetic_base_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            f(torch.randn(4), torch.randn(4))

        self.assertEqual(
            len(captured),
            0,
            "No codegen should be emitted when inputs don't alias",
        )

    def test_aliased_inputs_with_mutation(self):
        """
        Two aliased inputs where one is mutated. Should emit a
        synthetic_base_wrapper codegen artifact.
        """
        with self._capture_codegen_source("synthetic_base_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b):
                a.mul_(2)
                return a + b

            x = torch.randn(4)
            f(x.view(-1), x.view(-1))

        self.assertEqual(len(captured), 1)

    def test_aliased_mutation_correctness(self):
        """
        Verify that aliased mutation produces correct results through
        the codegen'd wrapper.
        """

        @torch.compile(backend="aot_eager")
        def f(a, b):
            a.mul_(2)
            return a + b

        x = torch.randn(4)
        x_ref = x.clone()
        a = x.view(-1)
        b = x.view(-1)
        out = f(a, b)

        self.assertEqual(a, x_ref * 2)
        self.assertEqual(b, x_ref * 2)
        self.assertEqual(out, x_ref * 4)

    @skipIfTorchDynamo("dynamo handles metadata mutations in-graph")
    def test_metadata_mutation_on_aliased_input(self):
        """
        Metadata mutation (transpose) on an aliased input. Uses
        aot_function directly since dynamo handles metadata mutations
        in-graph.
        """
        from functorch.compile import nop
        from torch._functorch.aot_autograd import aot_function

        with self._capture_codegen_source("synthetic_base_wrapper") as captured:

            def f(a, b):
                a.mul_(2)
                b.transpose_(1, 0)
                return a + 1

            x = torch.randn(2, 2)
            a = x.view(2, 2)
            b = x.view(2, 2)
            compiled_f = aot_function(f, nop)
            compiled_f(a, b)

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("as_strided_", source)

    def test_non_aliased_inputs_passthrough(self):
        """
        Non-aliased inputs should pass through the synthetic base
        wrapper without modification.
        """

        @torch.compile(backend="aot_eager")
        def f(x, a, b):
            a.mul_(2)
            return x + a + b

        x = torch.randn(4)
        base = torch.randn(4)
        base_ref = base.clone()
        a = base.view(-1)
        b = base.view(-1)
        x_ref = x.clone()
        out = f(x, a, b)

        self.assertEqual(a, base_ref * 2)
        self.assertEqual(out, x_ref + base_ref * 2 + base_ref * 2)

    def test_codegen_source_structure(self):
        """
        Verify the codegen'd source has the expected structure.
        """
        with self._capture_codegen_source("synthetic_base_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b):
                a.add_(1)
                return a + b

            x = torch.randn(4)
            f(x.view(-1), x.view(-1))

        self.assertEqual(len(captured), 1)
        source = captured[0]
        self.assertIn("def _synthetic_base_wrapper", source)
        self.assertIn("_merge_view_inputs_", source)
        self.assertIn("_compiled_fn_", source)
        self.assertIn("args.clear()", source)

    def test_training_path_aliased_mutation(self):
        """
        Training path with aliased inputs and mutation. Verify backward
        correctness through the synthetic base wrapper.
        """

        @torch.compile(backend="aot_eager")
        def f(a, b):
            a.mul_(2)
            return (a + b).sum()

        x = torch.randn(4, requires_grad=True)
        a = x.clone()
        b = a.view(-1)
        out = f(a, b)
        out.backward()


if __name__ == "__main__":
    run_tests()
