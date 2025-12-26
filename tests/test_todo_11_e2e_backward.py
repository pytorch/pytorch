# Owner(s): ["oncall: pt2"]
"""
End-to-end test: backward pass with Precompile API.

This test verifies TODO 11: End-to-end test with backward pass (training).
Tests that gradients flow correctly through the compiled function.
Verifies gradients match eager execution.
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


class TestE2EBackward(TestCase):
    def test_mlp_backward_pass(self):
        """Test end-to-end backward pass with an MLP model."""
        torch.manual_seed(42)
        model = MLP()
        example_input = torch.randn(4, 10)

        # Trace with Precompile.dynamo()
        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)

        # Compile with Precompile.aot_autograd()
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # Forward + backward with compiled
        compiled_out = compiled_fn(example_inputs)
        compiled_out.sum().backward()
        compiled_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Reset gradients
        model.zero_grad()

        # Forward + backward with eager
        eager_out = model(example_input)
        eager_out.sum().backward()
        eager_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Verify gradients match
        for name in compiled_grads:
            self.assertTrue(
                torch.allclose(compiled_grads[name], eager_grads[name], atol=1e-5),
                f"Gradient mismatch for {name}. "
                f"Max diff: {(compiled_grads[name] - eager_grads[name]).abs().max()}",
            )

    def test_simple_linear_backward(self):
        """Test backward pass with a simple linear model."""
        torch.manual_seed(123)
        model = nn.Linear(5, 3)
        example_input = torch.randn(2, 5)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # Forward + backward with compiled
        compiled_out = compiled_fn(example_inputs)
        compiled_out.sum().backward()
        compiled_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Reset gradients
        model.zero_grad()

        # Forward + backward with eager
        eager_out = model(example_input)
        eager_out.sum().backward()
        eager_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        for name in compiled_grads:
            self.assertTrue(
                torch.allclose(compiled_grads[name], eager_grads[name], atol=1e-5),
                f"Gradient mismatch for {name}",
            )

    def test_model_with_nonlinear_ops_backward(self):
        """Test backward pass with a model that has various nonlinear operations."""

        class NonLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(8, 16)
                self.linear2 = nn.Linear(16, 8)

            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = x * 2 + 1
                x = torch.tanh(x)
                x = self.linear2(x)
                return x

        torch.manual_seed(456)
        model = NonLinearModel()
        example_input = torch.randn(3, 8)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # Forward + backward with compiled
        compiled_out = compiled_fn(example_inputs)
        loss = compiled_out.sum()
        loss.backward()
        compiled_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Reset gradients
        model.zero_grad()

        # Forward + backward with eager
        eager_out = model(example_input)
        eager_loss = eager_out.sum()
        eager_loss.backward()
        eager_grads = {name: p.grad.clone() for name, p in model.named_parameters()}

        for name in compiled_grads:
            self.assertTrue(
                torch.allclose(compiled_grads[name], eager_grads[name], atol=1e-5),
                f"Gradient mismatch for {name} in nonlinear model",
            )

    def test_multiple_backward_passes(self):
        """Test that multiple backward passes work correctly."""
        torch.manual_seed(789)
        model = nn.Linear(4, 2)
        example_input = torch.randn(2, 4)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # First backward pass with compiled
        out1 = compiled_fn(example_inputs)
        out1.sum().backward()
        grads1 = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Reset and second backward pass
        model.zero_grad()
        out2 = compiled_fn(example_inputs)
        out2.sum().backward()
        grads2 = {name: p.grad.clone() for name, p in model.named_parameters()}

        # Gradients should be the same for the same input
        for name in grads1:
            self.assertTrue(
                torch.allclose(grads1[name], grads2[name], atol=1e-6),
                f"Gradients should be consistent across backward passes for {name}",
            )


if __name__ == "__main__":
    run_tests()
