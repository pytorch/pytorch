from torch.autograd import Function
from torch import Tensor
import torch
from torch.autograd import gradcheck, gradgradcheck
from torch.testing._internal.common_utils import run_tests, TestCase


class Attn(Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor):
        # Tensors must be 2d. The first dim is always the same, the second dim for q and k must be the same.
        assert q.dim() == 2
        assert k.dim() == 2
        assert v.dim() == 2
        assert q.size(1) == k.size(1)
        assert q.size(0) == v.size(0)
        assert k.size(0) == v.size(0)

        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)

        ctx.save_for_backward(q, k, v, a)
        ctx.set_materialize_grads(False)

        return o, a

    @staticmethod
    def backward(ctx, grad_o, grad_a):
        q, k, v, a = ctx.saved_tensors

        grad_v = a.T @ grad_o if grad_o is not None else None
        grad_a_total = (grad_o @ v.T if grad_o is not None else 0) + (
            grad_a if grad_a is not None else 0
        )

        grad_x = grad_a_total * (1 - a * a)
        grad_q = grad_x @ k
        grad_k = grad_x.T @ q

        return grad_q, grad_k, grad_v


class TestAttn(TestCase):
    def test_gradcheck(self):
        input_q = torch.randn(2, 3, requires_grad=True, dtype=torch.double)
        input_k = torch.randn(2, 3, requires_grad=True, dtype=torch.double)
        input_v = torch.randn(2, 5, requires_grad=True, dtype=torch.double)
        assert gradcheck(Attn.apply, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)
        assert gradgradcheck(Attn.apply, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)

    def test_zero_tensors(self):
        input_q = torch.zeros(2, 3, requires_grad=True, dtype=torch.double)
        input_k = torch.zeros(2, 3, requires_grad=True, dtype=torch.double)
        input_v = torch.zeros(2, 5, requires_grad=True, dtype=torch.double)
        assert gradcheck(Attn.apply, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)
        assert gradgradcheck(Attn.apply, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)

    def test_native_attn(self):
        input_q = torch.zeros(2, 3, requires_grad=True, dtype=torch.double)
        input_k = torch.zeros(2, 3, requires_grad=True, dtype=torch.double)
        input_v = torch.zeros(2, 5, requires_grad=True, dtype=torch.double)

        assert gradcheck(torch.attn, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)
        assert gradgradcheck(torch.attn, (input_q, input_k, input_v), eps=1e-6, atol=1e-4)


if __name__ == "__main__":
    run_tests()
