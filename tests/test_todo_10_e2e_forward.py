# Owner(s): ["oncall: pt2"]
"""
End-to-end test: forward pass with Precompile API.

This test verifies TODO 10: End-to-end test with simple model (forward only).
Tests the complete flow: Precompile.dynamo() -> Precompile.aot_autograd() -> run inference.
Verifies output matches eager execution.
"""

import torch
import torch.nn as nn

from torch.testing._internal.common_utils import run_tests, TestCase


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestE2EForward(TestCase):
    def test_mlp_forward_pass(self):
        """Test end-to-end forward pass with an MLP model."""
        torch.manual_seed(42)
        model = MLP()
        example_input = torch.randn(4, 10)

        # Trace with Precompile.dynamo()
        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)

        # Compile with Precompile.aot_autograd()
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # Run inference and compare
        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-5),
            f"Forward pass should match. Max diff: {(compiled_out - eager_out).abs().max()}",
        )

    def test_simple_model_forward(self):
        """Test with a simpler single-layer model."""
        torch.manual_seed(123)
        model = nn.Linear(5, 3)
        example_input = torch.randn(2, 5)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-5),
            "Simple model forward pass should match",
        )

    def test_model_with_multiple_ops(self):
        """Test with a model that has multiple different operations."""

        class MultiOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                x = self.linear(x)
                x = torch.relu(x)
                x = x * 2 + 1
                x = torch.sigmoid(x)
                return x

        torch.manual_seed(456)
        model = MultiOpModel()
        example_input = torch.randn(3, 8)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-5),
            "Multi-op model forward pass should match",
        )


if __name__ == "__main__":
    run_tests()
