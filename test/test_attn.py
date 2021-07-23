import torch
from torch.autograd import gradcheck, Function
from typing import Tuple


# This file contains the custom autograd function of 'attn' for the purpose
# of verifying that the manually derived gradients are correct

class AttentionFunction(Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_t = k.t()
        mm_output = torch.mm(q, k_t)
        tanh_output = torch.tanh(mm_output)
        output = torch.mm(tanh_output, v)

        ctx.save_for_backward(q, k, v, mm_output, tanh_output)

        return output, tanh_output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_tanh_out: torch.Tensor):

        q, k, v, mm_output, tanh_output = ctx.saved_tensors
        grad_q = grad_k = grad_v = None

        if ctx.needs_input_grad[0]:  # q
            dtanh_out = 1. - (torch.tanh(mm_output) ** 2)

            grad_tanh_out_1 = torch.mm(grad_out, v.t())  # This is different from the given grad_tanh_out
            grad_mm_out_1 = grad_tanh_out_1 * dtanh_out
            grad_q_1 = torch.mm(grad_mm_out_1, k)

            grad_mm_out_2 = grad_tanh_out * dtanh_out
            grad_q_2 = torch.mm(grad_mm_out_2, k)

            grad_q = grad_q_1 + grad_q_2

        if ctx.needs_input_grad[1]:  # k
            dtanh_out = 1. - (torch.tanh(mm_output) ** 2)

            grad_tanh_out_1 = torch.mm(grad_out, v.t())  # This is different from the given grad_tanh_out
            grad_mm_out_1 = grad_tanh_out_1 * dtanh_out
            grad_k_t_1 = torch.mm(q.t(), grad_mm_out_1)
            grad_k_1 = grad_k_t_1.t()

            grad_mm_out_2 = grad_tanh_out * dtanh_out
            grad_k_t_2 = torch.mm(q.t(), grad_mm_out_2)
            grad_k_2 = grad_k_t_2.t()

            grad_k = grad_k_1 + grad_k_2

        if ctx.needs_input_grad[2]:  # v
            grad_v = torch.mm(tanh_output.t(), grad_out)

        return grad_q, grad_k, grad_v


attn = AttentionFunction.apply

q_1 = torch.randn((2,2), dtype=torch.double, requires_grad=True)
k_1 = torch.randn((2,2), dtype=torch.double, requires_grad=True)
v_1 = torch.randn((2,2), dtype=torch.double, requires_grad=True)

test = gradcheck(attn, (q_1, k_1, v_1))
if (test):
    print("The gradients have been verified as correct.")
