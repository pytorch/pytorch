# Owner(s): ["module: autograd"]

import torch
from torch.autograd import Function, gradcheck, gradgradcheck
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class Attn(Function):
    @staticmethod
    def forward(ctx, q, k, v):
        x = q @ k.t()
        a = torch.tanh(x)
        o = a @ v
        ctx.save_for_backward(q, k, v, a)
        return o, a

    @staticmethod
    def backward(ctx, grad_o, grad_a):
        q, k, v, a = ctx.saved_tensors
        grad_v = a.t() @ grad_o
        grad_a_total = grad_a + grad_o @ v.t()
        grad_x = grad_a_total * (1 - a * a)
        grad_q = grad_x @ k
        grad_k = grad_x.t() @ q
        return grad_q, grad_k, grad_v


class TestAttnFunction(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_gradcheck(self):
        q = torch.rand(2, 3, dtype=torch.double, requires_grad=True)
        k = torch.rand(2, 3, dtype=torch.double, requires_grad=True)
        v = torch.rand(2, 4, dtype=torch.double, requires_grad=True)
        self.assertTrue(gradcheck(Attn.apply, (q, k, v)))
        self.assertTrue(gradgradcheck(Attn.apply, (q, k, v)))


if __name__ == "__main__":
    run_tests()
