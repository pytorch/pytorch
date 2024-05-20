import torch
from torch.testing._internal.common_utils import TestCase
from torch.autograd import gradcheck, gradgradcheck


class AttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        # ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, k, v, x, a)
        return o, a

    @staticmethod
    def backward(ctx, grad_out_o, grad_out_a):
        q, k, v, x, a = ctx.saved_tensors
        da_dx = 1. - (a ** 2)

        # Gradients of the loss w.r.t. a
        grad_a = torch.matmul(grad_out_o, v.transpose(0, 1)) + grad_out_a

        # Gradients of the loss w.r.t. x
        grad_out_x = grad_a * da_dx

        # Gradients of the loss w.r.t. q
        grad_q = torch.matmul(grad_out_x, k)

        # Gradients of the loss w.r.t. k
        grad_k = torch.matmul(grad_out_x.transpose(0, 1), q)

        # Gradients of the loss w.r.t. v
        grad_v = torch.matmul(a.transpose(0, 1), grad_out_o)

        return grad_q, grad_k, grad_v

class TestAttn(TestCase):
    def test_attn(self):
        dtype = torch.float64
        q = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        k = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        v = torch.randn(10, 8, dtype=dtype, requires_grad=True)
        inputs = (q, k, v)
        assert gradcheck(AttnFunc.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-2)
        assert gradgradcheck(AttnFunc.apply, inputs, eps=1e-6, atol=1e-4, rtol=1e-2)

    def test_native_method(self):
        dtype = torch.float64
        q = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        k = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        v = torch.randn(10, 8, dtype=dtype, requires_grad=True)
        inputs = (q, k, v)
        assert gradcheck(torch.attention_function, inputs, eps=1e-6, atol=1e-4, rtol=1e-2)
        assert gradgradcheck(torch.attention_function, inputs, eps=1e-6, atol=1e-4, rtol=1e-2)
