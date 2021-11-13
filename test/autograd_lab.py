import torch
from torch.testing._internal.common_utils import TestCase
from torch.autograd import gradcheck
from torch.autograd import gradgradcheck

class AutogradAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        assert q.dim() == k.dim() == v.dim() == 2, "all the inputs must be 2D tensors"
        assert q.size()[1] == k.size()[1], "q and k must have the same number of columns"
        assert k.size()[0] == v.size()[0], "k and v must have the same number of rows"

        x = torch.mm(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.mm(a, v)
        ctx.save_for_backward(q, k, v, a)
        
        return o, a
    
    @staticmethod
    def backward(ctx, grad_out_o, grad_out_a):
        q, k, v, a = ctx.saved_tensors
        x = torch.mm(q, k.transpose(0, 1))
        da_dx = 1 - (torch.tanh(x) ** 2)

        # for dLa_d<>
        grad1_dl_dx = grad_out_a * da_dx

        # for dLo_d<>
        grad2_dl_da = torch.mm(grad_out_o, v.transpose(0, 1))
        grad2_dl_dx = grad2_dl_da * da_dx 

        if ctx.needs_input_grad[0]: # q
            dla_dq = torch.mm(grad1_dl_dx, k)
            dlo_dq = torch.mm(grad2_dl_dx, k)
            grad_q = dla_dq + dlo_dq 

        if ctx.needs_input_grad[1]: # k
            dla_dk = torch.mm(grad1_dl_dx.transpose(0, 1), q)
            dlo_dk = torch.mm(grad2_dl_dx.transpose(0, 1), q)
            grad_k = dla_dk + dlo_dk

        if ctx.needs_input_grad[2]: # v
            grad_v = torch.mm(a.transpose(0, 1), grad_out_o)
        
        return grad_q, grad_k, grad_v

class OnboardingLab(TestCase):
    def test_onboarding_autograd_lab(self):
        attn_fn = AutogradAttn.apply

        A = 10
        B = 20
        C = 30
        D = 40

        q = torch.randn(A, B, requires_grad=True, dtype=torch.float64)
        k = torch.randn(C, B, requires_grad=True, dtype=torch.float64)
        v = torch.randn(C, D, requires_grad=True, dtype=torch.float64)

        gradcheck(attn_fn, (q, k, v))
        gradgradcheck(attn_fn, (q, k, v))

if __name__ == '__main__':
    OnboardingLab().test_onboarding_autograd_lab()
