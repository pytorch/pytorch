import unittest
import torch

from torch.autograd import gradcheck
from torch.testing._internal.common_utils import (
    TestCase
)

def attn_lab(q, k, v):
    x = torch.matmul(q, k.t())
    a = torch.tanh(x)
    o = torch.matmul(a, v)

    return o, x

class AttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        x = torch.matmul(q, k.t())
        a = torch.tanh(x)
        o = torch.matmul(a, v)

        ctx.save_for_backward(q, k, v, a)
        ctx.mark_non_differentiable(a)

        return o, a

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_o, ignore):
        q, k, v, a = ctx.saved_tensors

        # bprop both sides of mm(a, V)
        # where a = tanh(Q*K^T)
        #   get dL/dV directly
        #   get dL/da for next part
        grad_v = torch.matmul(a.t(), grad_o)
        grad_a = torch.matmul(grad_o, v.t())

        # bprop tanh to get dL/dx where
        # x = Q * K^T
        grad_x = grad_a * (1. - a**2)

        # bprop both sides of Q * K^T
        grad_q = torch.matmul(grad_x, k)
        grad_k = torch.matmul(q.t(), grad_x).t()

        return grad_q, grad_k, grad_v

Attn = AttnFn.apply

class TestAttentionLab(TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def testAttnFn(self):
        qc = torch.randn(2, 3, requires_grad=True)
        kc = torch.randn(2, 3, requires_grad=True)
        vc = torch.randn(2, 4, requires_grad=True)

        qf = qc.clone().detach().requires_grad_(True)
        kf = kc.clone().detach().requires_grad_(True)
        vf = vc.clone().detach().requires_grad_(True)

        def run(fn, q, k, v):
            o, a = fn(q, k, v)

            loss_o = torch.sum(o)
            loss = loss_o

            loss.backward()

        run(attn_lab, qc, kc, vc)
        run(Attn, qf, kf, vf)

        assert(torch.allclose(qc.grad, qf.grad))
        assert(torch.allclose(kc.grad, kf.grad))
        assert(torch.allclose(vc.grad, vf.grad))

    def testGradCheck(self):
        q = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        k = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        v = torch.randn(2, 4, dtype=torch.double, requires_grad=True)

        input = (q, k, v)
        test = gradcheck(Attn, input, eps=1e-6, atol=1e-4)
        assert(test)

