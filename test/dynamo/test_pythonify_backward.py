# Owner(s): ["module: dynamo"]
"""
Tests for backward kernel integration in pythonify.

These tests verify that when pythonify is enabled, the backward kernel
is compiled eagerly (not lazily) so it can be captured and included
in the generated Python file.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPythonifyBackwardIntegration(TestCase):
    """
    Tests for backward kernel integration in pythonify.

    These tests verify that when pythonify is enabled, the backward kernel
    is compiled eagerly (not lazily) so it can be captured and included
    in the generated Python file. This is achieved by setting
    force_non_lazy_backward_lowering=True when pythonify is active.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_backward_kernel_captured_in_pythonify_output(self):
        """
        Test that backward kernel is captured when pythonify is enabled.

        When pythonify=path is passed to torch.compile, the backward kernel
        should be compiled eagerly so it can be included in the generated
        Python file along with the forward kernel.
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.zeros(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x.matmul(self.W)

        torch.manual_seed(0)
        features = 4
        batch = 3
        model = Model(features)
        x = torch.randn(batch, features, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            y = torch.compile(model, pythonify=path)(x)

            with open(path) as f:
                code = f.read()

            self.assertIn(
                "def backward",
                code,
                "Expected backward method to be present in generated code",
            )
            self.assertIn(
                "compiled_fn_backward",
                code,
                "Expected compiled_fn_backward to be present in generated code",
            )

            backward_kernel_count = code.count("# AOT ID:")
            self.assertEqual(
                backward_kernel_count,
                2,
                f"Expected 2 AOT IDs (forward + backward), got {backward_kernel_count}",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pythonify_backward_produces_correct_gradients(self):
        """
        Test that backward pass via pythonify produces correct gradients.

        This is an end-to-end test that verifies:
        1. Forward output from exec'd pythonify code matches original
        2. Backward gradients from exec'd pythonify code match original
        """
        torch.set_default_device("cuda")

        class Model(nn.Module):
            def __init__(self, features: int):
                super().__init__()
                self.W = nn.Parameter(torch.zeros(features, features))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + x.matmul(self.W)

        torch.manual_seed(42)
        features = 4
        batch = 3
        model = Model(features)

        x1 = torch.randn(batch, features, requires_grad=True)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            y1 = torch.compile(model, pythonify=path)(x1)
            loss1 = y1.mean()
            loss1.backward()
            original_grad = x1.grad.clone()

            torch.manual_seed(42)
            x2 = torch.randn(batch, features, requires_grad=True)

            with open(path) as f:
                code = f.read()

            namespace = {"x": x2, "model": model, "torch": torch}
            exec(code, namespace)
            y2 = namespace["y"]

            self.assertTrue(
                torch.allclose(y1.detach(), y2.detach()),
                "Forward outputs should match",
            )

            loss2 = y2.mean()
            loss2.backward()

            self.assertTrue(
                torch.allclose(original_grad, x2.grad),
                f"Gradients should match. Expected {original_grad}, got {x2.grad}",
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    run_tests()
