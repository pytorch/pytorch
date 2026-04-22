# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing _backward_epilogue_functional subclass wrapping.

_backward_epilogue_functional wraps backward outputs (grad inputs) back
into tensor subclasses using a codegen'd wrap_fn. At compile time,
codegen_backward_subclass_wrap() generates straight-line wrapping code
with all subclass types and attr names baked in, replacing the generic
wrap_tensor_subclasses() loop.

Tests verify that a "backward_subclass_wrapper" artifact is emitted via
trace_structured, analogous to the "subclass_wrapper" artifact emitted
by the forward path.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


trace_log = logging.getLogger("torch.__trace")


class TestCodegenBackwardEpilogue(TestCase):
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

    def test_simple_backward_wraps_grad_inputs(self):
        """
        f(TwoTensor) -> TwoTensor, backward should wrap grad inputs back
        into TwoTensor via codegen'd epilogue.
        """
        with self._capture_codegen_source("backward_subclass_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            tt = TwoTensor(a, b)

            # Run ref (eager) and test (compiled) to compare gradients
            tt_ref = TwoTensor(
                a.clone().detach().requires_grad_(True),
                b.clone().detach().requires_grad_(True),
            )
            out_ref = tt_ref * 2
            out_ref.sum().backward()

            out = f(tt)
            out.sum().backward()

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(tt.grad, TwoTensor)
        self.assertEqual(tt.grad.a, tt_ref.grad.a)
        self.assertEqual(tt.grad.b, tt_ref.grad.b)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected backward_subclass_wrapper codegen artifact to be emitted",
        )

    def test_multi_input_backward_wraps_grad_inputs(self):
        """
        f(TwoTensor, TwoTensor) -> TwoTensor, backward should wrap grad
        inputs for each subclass input.
        """
        with self._capture_codegen_source("backward_subclass_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x * y

            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            c = torch.randn(4, requires_grad=True)
            d = torch.randn(4, requires_grad=True)

            tt1_ref = TwoTensor(
                a.clone().detach().requires_grad_(True),
                b.clone().detach().requires_grad_(True),
            )
            tt2_ref = TwoTensor(
                c.clone().detach().requires_grad_(True),
                d.clone().detach().requires_grad_(True),
            )
            out_ref = tt1_ref * tt2_ref
            out_ref.sum().backward()

            tt1 = TwoTensor(a, b)
            tt2 = TwoTensor(c, d)
            out = f(tt1, tt2)
            out.sum().backward()

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(tt1.grad, TwoTensor)
        self.assertIsInstance(tt2.grad, TwoTensor)
        self.assertEqual(tt1.grad.a, tt1_ref.grad.a)
        self.assertEqual(tt1.grad.b, tt1_ref.grad.b)
        self.assertEqual(tt2.grad.a, tt2_ref.grad.a)
        self.assertEqual(tt2.grad.b, tt2_ref.grad.b)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected backward_subclass_wrapper codegen artifact to be emitted",
        )

    def test_nested_subclass_backward_wraps_grad_inputs(self):
        """
        Nested TwoTensor backward should recursively wrap grad inputs.
        """
        with self._capture_codegen_source("backward_subclass_wrapper") as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.sin()

            a1 = torch.randn(4, requires_grad=True)
            a2 = torch.randn(4, requires_grad=True)
            a3 = torch.randn(4, requires_grad=True)
            a4 = torch.randn(4, requires_grad=True)

            inner_a_ref = TwoTensor(
                a1.clone().detach().requires_grad_(True),
                a2.clone().detach().requires_grad_(True),
            )
            inner_b_ref = TwoTensor(
                a3.clone().detach().requires_grad_(True),
                a4.clone().detach().requires_grad_(True),
            )
            tt_ref = TwoTensor(inner_a_ref, inner_b_ref)
            out_ref = tt_ref.sin()
            out_ref.sum().backward()

            inner_a = TwoTensor(a1, a2)
            inner_b = TwoTensor(a3, a4)
            tt = TwoTensor(inner_a, inner_b)
            out = f(tt)
            out.sum().backward()

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(out.a, TwoTensor)
        self.assertIsInstance(out.b, TwoTensor)
        self.assertIsInstance(tt.grad, TwoTensor)
        self.assertEqual(tt.grad.a.a, tt_ref.grad.a.a)
        self.assertEqual(tt.grad.a.b, tt_ref.grad.a.b)
        self.assertEqual(tt.grad.b.a, tt_ref.grad.b.a)
        self.assertEqual(tt.grad.b.b, tt_ref.grad.b.b)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected backward_subclass_wrapper codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
