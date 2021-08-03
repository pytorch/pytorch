import torch
from torch.autograd import Function, gradcheck, gradgradcheck
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

        ctx.save_for_backward(q, k, v, tanh_output)

        return output, tanh_output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_tanh_out: torch.Tensor):

        q, k, v, tanh_output = ctx.saved_tensors
        mm_output = torch.mm(q, k.t())
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

q_1 = torch.randn((2, 2), dtype=torch.double, requires_grad=True)
k_1 = torch.randn((2, 2), dtype=torch.double, requires_grad=True)
v_1 = torch.randn((2, 2), dtype=torch.double, requires_grad=True)

q_2 = torch.randn((10, 10), dtype=torch.double, requires_grad=True)
k_2 = torch.randn((10, 10), dtype=torch.double, requires_grad=True)
v_2 = torch.randn((10, 10), dtype=torch.double, requires_grad=True)

test1 = gradcheck(attn, (q_1, k_1, v_1))
test2 = gradcheck(attn, (q_2, k_2, v_2))
test_gg1 = gradgradcheck(attn, (q_1, k_1, v_1))
test_gg2 = gradgradcheck(attn, (q_2, k_2, v_2))

if test1 and test2:
    print("The gradients have been verified as correct.")
if test_gg1 and test_gg2:
    print("The 2nd order gradients have been verified as correct.")


q_0 = torch.tensor([[0, -1], [2, 3]], dtype=torch.double, requires_grad=True)
k_0 = torch.tensor([[0, -1], [2, 3]], dtype=torch.double, requires_grad=True)
v_0 = torch.tensor([[0, -1], [2, 3]], dtype=torch.double, requires_grad=True)


def loss_fn(x):
    a, b = x[0], x[1]
    return a.sum() + b.sum()

x = torch.attn(q_0, k_0, v_0)
loss = loss_fn(x)
loss.backward()

print(f"torch.attn(q_0, k_0, v_0):\n{x}")
print(f"q_0.grad:\n{q_0.grad}")
print(f"k_0.grad:\n{k_0.grad}")
print(f"v_0.grad:\n{v_0.grad}")

print(gradcheck(torch.attn, (q_0, k_0, v_0)))
