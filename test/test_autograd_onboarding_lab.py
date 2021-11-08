import torch
from torch.autograd import Function, gradcheck, gradgradcheck
from torch.testing._internal.common_utils import TestCase, run_tests

class TestOnboardinngAttn(TestCase):
    class Onboarding_Attn(Function):

        @staticmethod
        def forward(ctx, q, k, v):
            ''' Inputs
                q: m * n
                k: m * n
                v: m * p
                Outputs
                a: m * m
                o: m * p
            '''
            # check that inputs are 2D and check sizes of input
            assert q.dim() == 2 and k.dim() == 2 and v.dim() == 2, "all inputs must have dim=2"
            assert q.size()[1] == k.size()[1], "1st dim of q and 1st dim of k must have the same size"
            assert k.size()[0] == v.size()[0], "0th dim of k and 0th dim of v must have the same size"

            x = torch.matmul(q, k.transpose(0, 1))
            a = torch.tanh(x)
            o = torch.matmul(a, v)
            ctx.save_for_backward(q, k, v, a)
            return o, a

        @staticmethod
        def backward(ctx, *grads):
            grad_output_o = grads[0]
            grad_output_a = grads[1]
            q, k, v, a = ctx.saved_tensors
            x = torch.matmul(q, k.transpose(0, 1))
            grad_q = grad_k = grad_v = None

            if ctx.needs_input_grad[0]:
                do_da = torch.mm(grad_output_o, v.t())
                da_dx = (1 / torch.cosh(x))**2
                grad_q_o = torch.mm(do_da * da_dx, k)
                grad_q_a = torch.mm(grad_output_a * da_dx, k)
                grad_q = grad_q_o + grad_q_a
            if ctx.needs_input_grad[1]:
                do_da = torch.mm(grad_output_o, v.t())
                da_dx = (1 / torch.cosh(x))**2
                grad_k_o = torch.mm((do_da * da_dx).t(), q)
                grad_k_a = torch.mm((grad_output_a * da_dx).t(), q)
                grad_k = grad_k_o + grad_k_a
            if ctx.needs_input_grad[2]:
                grad_v_o = torch.mm(a.t(), grad_output_o)
                grad_v_a = torch.zeros_like(grad_v_o)
                grad_v = grad_v_o + grad_v_a

            return grad_q, grad_k, grad_v

    def test_onboarding_aten(self):
        M = 20
        N = 30
        P = 15
        attn = self.Onboarding_Attn.apply
        input = (torch.randn(M, N, dtype=torch.double, requires_grad=True),
                 torch.randn(M, N, dtype=torch.double, requires_grad=True),
                 torch.randn(M, P, dtype=torch.double, requires_grad=True))
        test_gradcheck = gradcheck(attn, input)
        test_gradgradcheck = gradgradcheck(attn, input)

if __name__ == '__main__':
    run_tests()
