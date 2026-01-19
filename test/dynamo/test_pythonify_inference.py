# Owner(s): ["module: dynamo"]
"""
Tests for inference-only mode in pythonify.

Tests verify that pythonify handles models without backward pass (inference mode)
correctly by either not generating a backward method or generating one that raises
a clear error.
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPythonifyInference(TestCase):
    """
    Tests for inference-only mode in pythonify.

    Verifies that models without requires_grad work correctly and that
    the generated code handles absence of backward kernel gracefully.
    """

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pythonify_inference_no_requires_grad(self):
        """
        Test that inference mode (requires_grad=False) works correctly.

        When inputs don't require gradients, the backward kernel is not
        compiled. The generated code should work correctly without errors.
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

        # Create input WITHOUT requires_grad - inference mode
        x = torch.randn(batch, features, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            y1 = torch.compile(model, pythonify=path)(x)

            # Verify the generated code exists and is valid Python
            with open(path) as f:
                code = f.read()

            # The code should have forward method
            self.assertIn(
                "def forward",
                code,
                "Expected forward method to be present",
            )

            # Execute the generated code
            namespace = {"x": x, "model": model, "torch": torch}
            exec(code, namespace)
            y2 = namespace["y"]

            # Verify outputs match
            self.assertTrue(
                torch.allclose(y1, y2),
                "Forward outputs should match in inference mode",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pythonify_inference_with_torch_no_grad(self):
        """
        Test that torch.no_grad() context works correctly.

        Even if the model has parameters that could have gradients,
        when executed within torch.no_grad(), no backward kernel should
        be needed.
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

        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            # Compile within no_grad context
            with torch.no_grad():
                y1 = torch.compile(model, pythonify=path)(x)

            # Execute the generated code
            with open(path) as f:
                code = f.read()

            namespace = {"x": x, "model": model, "torch": torch}
            with torch.no_grad():
                exec(code, namespace)
                y2 = namespace["y"]

            # Verify outputs match
            self.assertTrue(
                torch.allclose(y1, y2),
                "Forward outputs should match with no_grad",
            )
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pythonify_backward_not_called_raises_no_error(self):
        """
        Test that even if backward is not available, forward still works.

        This verifies that the absence of backward kernel doesn't break
        forward pass execution.
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

        # Use requires_grad=False to avoid backward kernel compilation
        x = torch.randn(batch, features, requires_grad=False)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()
            y1 = torch.compile(model, pythonify=path)(x)

            # Execute generated code multiple times to ensure no state issues
            with open(path) as f:
                code = f.read()

            for _ in range(3):
                x_test = torch.randn(batch, features, requires_grad=False)
                namespace = {"x": x_test, "model": model, "torch": torch}
                exec(code, namespace)
                y_test = namespace["y"]

                # Should execute without errors
                self.assertEqual(y_test.shape, (batch, features))
        finally:
            os.unlink(path)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_pythonify_inference_backward_raises_clear_error(self):
        """
        Test that backward on inference-compiled model raises clear error.

        When a model is compiled in inference mode (e.g., with torch.no_grad()),
        there's no backward kernel. If the user later tries to call backward
        with requires_grad inputs, they should get a clear error message
        explaining the issue, not a generic PyTorch NotImplementedError.
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

        x = torch.randn(batch, features)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            path = f.name

        try:
            torch._dynamo.reset()

            # Compile with no_grad - no backward kernel will be compiled
            with torch.no_grad():
                torch.compile(model, pythonify=path)(x)

            # Verify the generated code has a backward method with fallback
            with open(path) as f:
                code = f.read()

            self.assertIn(
                "def backward",
                code,
                "Expected backward method to be present for fallback",
            )
            self.assertIn(
                "Backward pass not available",
                code,
                "Expected fallback error message in generated code",
            )

            # Now try to run with requires_grad input and call backward
            x2 = torch.randn(batch, features, requires_grad=True)
            namespace = {"x": x2, "model": model, "torch": torch}
            exec(code, namespace)
            y2 = namespace["y"]

            # Forward should work
            self.assertEqual(y2.shape, (batch, features))
            self.assertTrue(y2.requires_grad)

            # Backward should raise a clear RuntimeError
            loss = y2.mean()
            with self.assertRaises(RuntimeError) as context:
                loss.backward()

            error_msg = str(context.exception)
            self.assertIn(
                "Backward pass not available",
                error_msg,
                "Expected clear error message about missing backward kernel",
            )
            self.assertIn(
                "inference mode",
                error_msg,
                "Expected error message to mention inference mode",
            )

        finally:
            os.unlink(path)


if __name__ == "__main__":
    run_tests()
