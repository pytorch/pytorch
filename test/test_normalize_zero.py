import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests


class TestNormalizeZeroInput(TestCase):
    '''Tests for F.normalize correctness at zero input (gh#184575).'''

    def test_normalize_zero_input_forward_nan(self):
        '''F.normalize at zero input must return NaN in forward pass.'''
        x = torch.zeros(3)
        y = F.normalize(x, dim=0)
        self.assertTrue(
            torch.isnan(y).any(),
            f'F.normalize(zeros) must return NaN, got {y}'
        )

    def test_normalize_zero_input_backward_nan(self):
        '''F.normalize at zero input must return NaN gradient, not finite.'''
        x = torch.zeros(3, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertTrue(
            torch.isnan(x.grad).any(),
            f'F.normalize(zeros) grad must be NaN, got {x.grad}'
        )

    def test_normalize_zero_input_no_finite_gradient(self):
        '''F.normalize at zero input must NOT return finite gradient (gh#184575).'''
        x = torch.zeros(5, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertFalse(
            x.grad.isfinite().all().item(),
            f'Expected non-finite gradient, got {x.grad}'
        )

    def test_normalize_normal_input_unchanged(self):
        '''F.normalize on normal input must still work correctly (regression).'''
        x = torch.randn(3, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertTrue(x.grad.isfinite().all().item())
        self.assertFalse(torch.isnan(x.grad).any().item())


if __name__ == '__main__':
    run_tests()
