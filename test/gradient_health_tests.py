"""Gradient health validation tests â€” catches hardware-specific gradient corruption (gh#177116)."""
import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestGradientHealth(TestCase):
    def test_gradient_finite_after_backward(self):
        """Gradients must be finite after a normal backward pass."""
        model = torch.nn.Linear(8, 2)
        x = torch.randn(4, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertTrue(
                param.grad.isfinite().all(),
                f'{name}.grad is not finite: {param.grad}'
            )

    def test_gradient_not_exploding_simple(self):
        """Gradients must not explode on scaled inputs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )
        x = torch.randn(4, 8) * 10.0
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            grad_norm = param.grad.norm().item()
            self.assertLess(grad_norm, 1000.0,
                f'{name}.grad norm={grad_norm:.1f} exceeds threshold')

    def test_gradient_consistent_with_loss_scale(self):
        """Gradients must scale linearly with loss scaling."""
        model = torch.nn.Linear(4, 2)
        x = torch.randn(4, 4)
        out = model(x)

        loss_1x = out.sum()
        loss_1x.backward(retain_graph=True)
        g_1x = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad()

        loss_2x = (out * 2.0).sum()
        loss_2x.backward()
        for name, param in model.named_parameters():
            self.assertTrue(
                torch.allclose(param.grad, g_1x[name] * 2.0, rtol=1e-4),
                f'{name}.grad mismatch'
            )


if __name__ == '__main__':
    run_tests()
