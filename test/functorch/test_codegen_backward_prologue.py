# Owner(s): ["module: functorch"]

"""
Tests for codegen'ing _backward_prologue_functional non-tangent subclass
unwrapping.

_backward_prologue_functional unwraps saved tensors (non-tangent subclass
inputs) before passing them to the compiled backward. The tangent processing
(process_runtime_tangent) is runtime-dependent and NOT a codegen candidate,
but the non-tangent unwrapping is pure compile-time-determined subclass
unwrapping, identical to the forward input unwrapping already codegen'd.

Tests verify that a "backward_subclass_unwrap" artifact is emitted via
trace_structured.
"""

import logging
from contextlib import contextmanager

import torch
import torch._functorch.config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


trace_log = logging.getLogger("torch.__trace")


class TestCodegenBackwardPrologue(TestCase):
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

    def test_saved_subclass_tensors_unwrapped(self):
        """
        When subclass tensors are saved for backward, the prologue should
        codegen their unwrapping (non-tangent path).
        """
        with self._capture_codegen_source("backward_subclass_unwrap") as captured:

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

        self.assertEqual(tt1.grad.a, tt1_ref.grad.a)
        self.assertEqual(tt1.grad.b, tt1_ref.grad.b)
        self.assertEqual(tt2.grad.a, tt2_ref.grad.a)
        self.assertEqual(tt2.grad.b, tt2_ref.grad.b)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected backward_subclass_unwrap codegen artifact to be emitted",
        )

    def test_mixed_subclass_and_plain_saved_tensors(self):
        """
        When both subclass and plain tensors are saved, the prologue should
        codegen unwrapping only for the subclass ones.
        """
        with self._capture_codegen_source("backward_subclass_unwrap") as captured:

            @torch.compile(backend="aot_eager")
            def f(x, y):
                return x * y

            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)

            tt_ref = TwoTensor(
                a.clone().detach().requires_grad_(True),
                b.clone().detach().requires_grad_(True),
            )
            plain_ref = torch.randn(4, requires_grad=True)
            out_ref = tt_ref * plain_ref
            out_ref.sum().backward()

            tt = TwoTensor(a, b)
            plain = plain_ref.clone().detach().requires_grad_(True)
            out = f(tt, plain)
            out.sum().backward()

        self.assertEqual(tt.grad.a, tt_ref.grad.a)
        self.assertEqual(tt.grad.b, tt_ref.grad.b)
        self.assertEqual(plain.grad, plain_ref.grad)

        self.assertGreaterEqual(
            len(captured),
            1,
            "Expected backward_subclass_unwrap codegen artifact to be emitted",
        )


if __name__ == "__main__":
    run_tests()
