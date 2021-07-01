import torch

class Attn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        assert q.size() == k.size()
        assert q.size()[0] == v.size()[0]

        attn = torch.matmul(q, k.swapaxes(0, 1))
        attn = torch.tanh(attn)
        output = torch.matmul(attn, v)

        ctx.save_for_backward(q, k, v, attn)
        return output, attn

    @staticmethod
    def backward(ctx, grad_output, grad_attn):
        (q, k, v, attn) = ctx.saved_tensors

        # computes and stores sech^2(qk')
        partial = 1 / torch.cosh(q.matmul(k.swapaxes(0, 1))).pow(2)

        grad_output_vt_partial = grad_output.matmul(v.swapaxes(0, 1)).mul(partial)

        # computes grad_output terms
        grad_q = grad_output_vt_partial.matmul(k)
        grad_k = grad_output_vt_partial.swapaxes(0, 1).matmul(q)
        grad_v = attn.swapaxes(0, 1).matmul(grad_output)

        # adds grad_attn terms
        grad_q.add_(grad_attn.mul(partial).matmul(k))
        grad_k.add_(grad_attn.mul(partial).swapaxes(0, 1).matmul(q))

        return grad_q, grad_k, grad_v


attn = Attn.apply

from torch.autograd import gradcheck
from torch.autograd import gradgradcheck

q = torch.rand(2, 3, requires_grad=True, dtype=torch.float64)
k = torch.rand(2, 3, requires_grad=True, dtype=torch.float64)
v = torch.rand(2, 4, requires_grad=True, dtype=torch.float64)

test = gradcheck(attn, (q, k, v), eps=1e-6, atol=1e-4)
print(test)

test = gradgradcheck(attn, (q, k, v), eps=1e-6, atol=1e-4)
print(test)
