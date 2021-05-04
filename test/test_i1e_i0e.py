import scipy.special
import numpy as np

import torch

from torch.autograd import gradcheck, gradgradcheck
torch.manual_seed(42)
S = 5
# t = torch.randn((S,), device='cpu', dtype=torch.float64, requires_grad=True)
t = torch.zeros((S,), device='cpu', dtype=torch.float64, requires_grad=True)

class special_i1e(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = torch.special.i1e(i)
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_tensors
        # zero_g = torch.special.i0e(i) - (result * i.sign()) - 1 * (i+eps).reciprocal()
        def f1(i):
            i = i + 1e-6
            g = special_i0e.apply(i) - result * (i.sign() + (i).reciprocal())
            return g
        
        def f2(i):
            return 0.5 # * i.exp()
        
        g = f1(i)
        zero_g = f2(i)
        # out_g = zero_g * (i == 0) + g * (i != 0)
        # print(out_g, zero_g, g)
        out_g = torch.where(i == 0, zero_g, g)
        return grad_output * out_g

class special_i0e(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = torch.special.i0e(i)
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_tensors
        return grad_output * (special_i1e.apply(i) - i.sgn() * result)

#Use it by calling the apply method:
output = special_i0e.apply(t)

gradcheck(lambda x: special_i1e.apply(x), t)
gradgradcheck(lambda x: special_i1e.apply(x), t)

gradcheck(lambda x: special_i0e.apply(x), t)
gradgradcheck(lambda x: special_i0e.apply(x), t)

# from fdm import gradient, jacobian, jvp, hvp, central_fdm

# print(central_fdm(order=12, deriv=1)(scipy.special.i0, 1) + scipy.special.i0(1))
