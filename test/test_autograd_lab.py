import torch
from torch.autograd.function import Function
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    TestCase,
    run_tests,
)

# mypy: ignore-errors

class Attention(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
            raise ValueError(
                f"Attention: Expected inputs to be 2D, got q = {q.dim()}D, k = {k.dim()}D, v = {v.dim()}D instead."
            )
        if q.size(0) != k.size(0) or q.size(0) != v.size(0):
            raise ValueError(
                f"Attention: Expected inputs to have the same first dimension, got q = {q.size(0)}, k = {k.size(0)}, v = {v.size(0)}."
            )

        if q.size(1) != k.size(1):
            raise ValueError(
                f"Attention: Expected q and k to have the same second dimension, got q = {q.size(1)}, k = {k.size(1)}."
            )

        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        ctx.save_for_backward(q, k, v, a)
        return o, a

    @staticmethod
    def backward(ctx, grad_o, grad_a):
        q, k, v, a = ctx.saved_tensors
        grad_a_local = grad_o @ v.transpose(0, 1)
        grad_v = a.transpose(0, 1) @ grad_o
        # We have to add grad_a and ga together here because grad_a contains contributions
        # from functions upstream which compute their own gradients w.r.t a, while grad_a_local
        # is the contribution from this function
        grad_x = (grad_a + grad_a_local) * (1 - a ** 2)
        grad_q = grad_x @ k
        grad_k = grad_x.transpose(0, 1) @ q
        return grad_q, grad_k, grad_v

class TestAutogradLab(TestCase):
    def test_attention(self):
        q = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        k = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        v = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)

        gradcheck(Attention.apply, (q, k, v))
        gradgradcheck(Attention.apply, (q, k, v))

    def test_attention_mismatched_dims(self):
        test_cases = [
            ((3, 5), (4, 5), (3, 7)),  # q and k have different first dimensions
            ((3, 5), (3, 4), (3, 7)),  # q and k have different second dimensions
            ((3, 5), (3, 5), (4, 7)),  # q and v have different first dimensions
        ]
        for q_shape, k_shape, v_shape in test_cases:
            q = torch.randn(*q_shape, dtype=torch.float64, requires_grad=True)
            k = torch.randn(*k_shape, dtype=torch.float64, requires_grad=True)
            v = torch.randn(*v_shape, dtype=torch.float64, requires_grad=True)

            self.assertRaises(ValueError, Attention.apply, q, k, v)
        
    def test_attention_native(self):
        q = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        k = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        v = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)

        gradcheck(torch.ops.aten.attention, (q, k, v))
        gradgradcheck(torch.ops.aten.attention, (q, k, v))

    def test_attention_native_mismatched_dims(self):
        test_cases = [
            ((3, 5), (4, 5), (3, 7)),  # q and k have different first dimensions
            ((3, 5), (3, 4), (3, 7)),  # q and k have different second dimensions
            ((3, 5), (3, 5), (4, 7)),  # q and v have different first dimensions
        ]
        for q_shape, k_shape, v_shape in test_cases:
            q = torch.randn(*q_shape, dtype=torch.float64, requires_grad=True)
            k = torch.randn(*k_shape, dtype=torch.float64, requires_grad=True)
            v = torch.randn(*v_shape, dtype=torch.float64, requires_grad=True)

            self.assertRaises(RuntimeError, torch.ops.aten.attention, q, k, v)
if __name__ == "__main__":
    run_tests()
