# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing the mutation epilogue in _create_runtime_wrapper.

The codegen'd mutation epilogue emits one of as_strided_(), copy_(),
detach().copy_(), or copy_mutated_input() per mutated input, with the branch
resolved at codegen time from each input's mutation metadata.

Tests that exercise data-only mutations use torch.compile (dynamo handles
metadata mutations in-graph, so only data mutations reach the epilogue).

Tests that exercise metadata mutations (metadata-only, data+metadata)
use aot_function directly so metadata mutations flow through the epilogue.

Tests verify that a "mutation_epilogue" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from functorch.compile import nop
from torch._functorch.aot_autograd import aot_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


trace_log = logging.getLogger("torch.__trace")


class TestCodegenMutationEpilogue(TestCase):
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

    def test_single_data_mutation(self):
        """
        Single input data mutation via mul_. Codegen should emit a direct
        copy_() for this input.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                x.mul_(2)
                return x + y

            x = torch.randn(4, requires_grad=True).clone()
            x.retain_grad()
            y = torch.randn(4)
            x_ref = x.detach().clone()
            y_ref = y.clone()
            out = f(x, y)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + y_ref)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("copy_", captured[0])

    def test_multiple_data_mutations(self):
        """
        Multiple inputs mutated. Codegen should emit a copy_() per mutated
        input, with non-mutated inputs skipped entirely.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(a, b, c):
                a.mul_(2)
                c.add_(1)
                return a + b + c

            a = torch.randn(4, requires_grad=True).clone()
            a.retain_grad()
            b = torch.randn(4)
            c = torch.randn(4, requires_grad=True).clone()
            c.retain_grad()
            a_ref, c_ref = a.detach().clone(), c.detach().clone()
            out = f(a, b, c)

        self.assertEqual(a.detach(), a_ref * 2)
        self.assertEqual(c.detach(), c_ref + 1)
        self.assertEqual(out, a_ref * 2 + b + c_ref + 1)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("copy_", captured[0])

    def test_leaf_mutation_under_no_grad(self):
        """
        Leaf tensor mutated under no_grad (e.g. via detach().mul_()).
        Codegen should emit detach().copy_() for this case.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                x.detach().mul_(2)
                return x + 1

            x = torch.randn(4, requires_grad=True)
            x_ref = x.detach().clone()
            out = f(x)

        self.assertEqual(x.detach(), x_ref * 2)
        self.assertEqual(out, x_ref * 2 + 1)

        self.assertEqual(
            len(captured),
            1,
            "Expected mutation_epilogue codegen artifact to be emitted",
        )
        self.assertIn("orig_inputs[0].detach().copy_(updated_inputs[0])", captured[0])
        self.assertNotIn("copy_mutated_input", captured[0])

    def test_storage_copy_leaf_mutation_autograd(self):
        def f(x):
            with torch.no_grad():
                x.as_strided((10,), (1,), 5).add_(1)
            return x.sin()

        ref_storage = torch.arange(20.0)
        ref_x = ref_storage[:10].detach().requires_grad_()
        ref_out = f(ref_x)
        ref_out.sum().backward()

        storage = torch.arange(20.0)
        x = storage[:10].detach().requires_grad_()
        with self._capture_codegen_source("mutation_epilogue") as captured:
            out = torch.compile(f, backend="aot_eager")(x)
        out.sum().backward()

        self.assertEqual(out, ref_out)
        self.assertEqual(storage, ref_storage)
        self.assertEqual(x.grad, ref_x.grad)
        self.assertEqual(len(captured), 1)
        self.assertIn(
            "copy_mutated_input(orig_inputs[0].detach(), updated_inputs[0])",
            captured[0],
        )

    def test_storage_copy_nonleaf_mutation_autograd(self):
        def f(x):
            with torch.no_grad():
                x.as_strided((10,), (1,), 5).add_(1)
            return x.sin()

        ref_source = torch.arange(20.0, requires_grad=True)
        ref_storage = ref_source * 1
        ref_out = f(ref_storage[:10])
        ref_out.sum().backward()

        source = torch.arange(20.0, requires_grad=True)
        storage = source * 1
        with self._capture_codegen_source("mutation_epilogue") as captured:
            out = torch.compile(f, backend="aot_eager")(storage[:10])
        out.sum().backward()

        self.assertEqual(out, ref_out)
        self.assertEqual(storage, ref_storage)
        self.assertEqual(source.grad, ref_source.grad)
        self.assertEqual(len(captured), 1)
        self.assertIn(
            "copy_mutated_input(orig_inputs[0], updated_inputs[0])", captured[0]
        )
        self.assertNotIn("orig_inputs[0].detach()", captured[0])

    def test_storage_copy_tracked_nonleaf_mutations_in_bounds(self):
        def f(x):
            # Both absolute storage ranges are within x's logical range [5, 14].
            x.as_strided((3,), (1,), 6).add_(1)
            x.as_strided((2,), (1,), 11).mul_(2)
            return x.sin()

        ref_source = torch.arange(20.0, requires_grad=True)
        ref_storage = ref_source * 1
        ref_out = f(ref_storage[5:15])
        ref_out.sum().backward()

        source = torch.arange(20.0, requires_grad=True)
        storage = source * 1
        with self._capture_codegen_source("mutation_epilogue") as captured:
            # Keep the input length symbolic while proving the fixed mutation
            # footprints are contained in it.
            out = torch.compile(f, backend="aot_eager", dynamic=True)(storage[5:15])
        out.sum().backward()

        self.assertEqual(out, ref_out)
        self.assertEqual(storage, ref_storage)
        self.assertEqual(source.grad, ref_source.grad)
        self.assertEqual(len(captured), 1)
        self.assertRegex(
            captured[0],
            r"orig_inputs\[\d+\]\.copy_\(updated_inputs\[0\]\)",
        )
        self.assertNotIn("copy_mutated_input", captured[0])

    def test_storage_copy_tracked_nonleaf_mutation_errors(self):
        def f(x):
            x.as_strided((10,), (1,), 5).add_(1)
            return x.sin()

        source = torch.arange(20.0, requires_grad=True)
        storage = source * 1
        storage_before = storage.detach().clone()
        with self.assertRaisesRegex(
            RuntimeError,
            "differentiable input mutations that access storage outside",
        ):
            torch.compile(f, backend="aot_eager")(storage[:10])
        self.assertEqual(storage, storage_before)

    def test_storage_copy_tracked_noncontiguous_mutation_errors(self):
        def f(x):
            x.as_strided((2,), (2,), 2).mul_(2)
            return x.sin()

        source = torch.arange(10.0, requires_grad=True)
        storage = source * 1
        storage_before = storage.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "non-contiguous inputs"):
            torch.compile(f, backend="aot_eager")(storage[::2])
        self.assertEqual(storage, storage_before)

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_metadata_only_mutation(self):
        """
        Metadata-only mutation via transpose_(). Codegen should emit
        as_strided_() without copy_(). Uses aot_function directly because
        dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(a, b):
                a.transpose_(1, 0)
                return a + b

            a = torch.randn(3, 4, requires_grad=True).add(0)
            b = torch.randn(4, 3)
            compiled_f = aot_function(f, nop)
            out = compiled_f(a, b)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(out.shape, (4, 3))

        self.assertEqual(len(captured), 1)
        self.assertIn("as_strided_", captured[0])
        self.assertNotIn("copy_", captured[0])

    @skipIfTorchDynamo(
        "aot_function uses FX tracing which conflicts with dynamo wrapping"
    )
    def test_data_and_metadata_mutation(self):
        """
        Both data and metadata mutated (transpose_ then mul_). Codegen
        should emit as_strided_() followed by copy_(). Uses aot_function
        directly because dynamo handles metadata mutations in-graph.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            def f(a):
                a.transpose_(1, 0)
                a.mul_(2)
                return a + 1

            a = torch.randn(3, 4, requires_grad=True).add(0)
            a_ref = a.detach().clone()
            compiled_f = aot_function(f, nop)
            out = compiled_f(a)

        self.assertEqual(a.shape, (4, 3))
        self.assertEqual(a.detach(), a_ref.transpose(1, 0) * 2)
        self.assertEqual(out, a_ref.transpose(1, 0) * 2 + 1)

        self.assertEqual(len(captured), 1)
        self.assertIn("as_strided_", captured[0])
        self.assertIn("copy_", captured[0])

    def test_no_mutation_no_epilogue(self):
        """
        No mutations at all. No mutation_epilogue artifact should be
        emitted.
        """
        with self._capture_codegen_source("mutation_epilogue") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x + y

            x = torch.randn(4, requires_grad=True)
            y = torch.randn(4)
            out = f(x, y)

        self.assertEqual(out, x + y)
        self.assertEqual(len(captured), 0)


if __name__ == "__main__":
    run_tests()
