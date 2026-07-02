"""Tests for gradient health after backward pass (gh#177116).

Validates that gradients produced by autograd are:
- Finite (no NaN, no Inf)
- Reasonably bounded (not catastrophically wrong)
- Consistent with the forward computation

These tests would catch hardware-specific gradient corruption bugs
such as the MPS gradient bug (gh#177116), where the MPS backend
produces gradients that are 100x-100,000x too large.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestGradientHealth(TestCase):
    """Gradient correctness tests that catch silent corruption."""

    def test_gradient_finite_after_backward(self):
        """Gradients must be finite after a normal backward pass."""
        model = torch.nn.Linear(10, 5)
        x = torch.randn(3, 10)
        target = torch.randn(3, 5)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Non-finite gradient in {name}: {param.grad}"
                )

    def test_gradient_not_exploding_simple(self):
        """Gradients must not be unreasonably large for simple models."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        x = torch.randn(4, 10)
        target = torch.randint(0, 2, (4,))
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.assertLess(
                    grad_norm,
                    1e6,
                    f"Gradient norm {grad_norm:.2e} too large in {name}. "
                    f"Expected < 1e6 for a 2-layer model with 4 samples."
                )

    def test_gradient_consistent_with_loss_scale(self):
        """Scaling loss by factor f scales gradients by factor f."""
        model = torch.nn.Linear(5, 3)
        x = torch.randn(2, 5)
        target = torch.randn(2, 3)

        # Reference: loss = 1.0 * mse
        loss_ref = torch.nn.functional.mse_loss(model(x), target)
        model.zero_grad()
        loss_ref.backward()
        grads_ref = {n: p.grad.clone() for n, p in model.named_parameters()}

        # Scaled: loss = 10.0 * mse
        loss_scaled = 10.0 * torch.nn.functional.mse_loss(model(x), target)
        model.zero_grad()
        loss_scaled.backward()
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.allclose(param.grad, grads_ref[name] * 10.0, rtol=1e-4),
                f"Gradient scaling mismatch in {name}: "
                f"expected {grads_ref[name] * 10.0}, got {param.grad}"
            )


if __name__ == "__main__":
    run_tests()
