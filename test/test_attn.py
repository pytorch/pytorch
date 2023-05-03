import torch
from torch.testing._internal.common_utils import (TestCase, gradcheck, gradgradcheck)
from torch.autograd import Function

class TestAutogradAttn(TestCase):

    def test_autograd_attn(self):
        class Attn(Function):

            @staticmethod
            def forward(q, k, v):
                if q.dim() != 2:
                    raise RuntimeError('attention function is currently supported only for 2D tensors')
                if k.dim() != 2:
                    raise RuntimeError('attention function is currently supported only for 2D tensors')
                if v.dim() != 2:
                    raise RuntimeError('attention function is currently supported only for 2D tensors')

                if q.shape[-1] != k.shape[-1]:
                    raise RuntimeError(f'Expected size of k to be {q.shape[-1]} but got: {k.shape[-1]}')
                x = torch.matmul(q, k.transpose(1, 0))
                a = torch.tanh(x)
                o = torch.matmul(a, v)
                return o, a, x

            @staticmethod
            def setup_context(ctx, inputs, outputs):
                q, k, v = inputs
                o, a, x = outputs
                ctx.save_for_backward(q, k, v, a, x)

            @staticmethod
            def backward(ctx, grad_o, grad_a, grad_x):
                q, k, v, a, x = ctx.saved_tensors
                # TODO Figure out why we need to add `grad_a` here or `grad_x` in next step
                # More boradly, what is `grad_a` and how gradcheck works.
                dL_da = grad_a + torch.matmul(grad_o, v.transpose(1, 0))
                dL_dv = torch.matmul(a.transpose(1, 0), grad_o)

                dL_dx = grad_x + (dL_da / ((torch.cosh(x)) ** 2))

                dL_dq = torch.matmul(dL_dx, k)
                dL_dk = torch.matmul(q.transpose(1, 0), dL_dx).transpose(1, 0)
                return dL_dq, dL_dk, dL_dv

        attn = Attn.apply

        q = torch.rand(2, 3, requires_grad=True, dtype=torch.double)
        k = torch.rand(2, 3, requires_grad=True, dtype=torch.double)
        v = torch.rand(2, 4, requires_grad=True, dtype=torch.double)

        assert gradcheck(attn, (q, k, v), eps=1e-6, atol=1e-4)
        assert gradgradcheck(attn, (q, k, v), eps=1e-6, atol=1e-4)
