import torch

class AttentionGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        (q, k, v) = args
        ctx.set_materialize_grads(False)
        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        ctx.save_for_backward(q, k, v, a)
        return o, a

    @staticmethod
    def backward(ctx, grad_o, grad_a):
        (q, k, v, a) = ctx.saved_tensors

        out = [None, None, None]
        if grad_o is not None:
            intermediate_for_grad_wrt_o = (grad_o @ v.transpose(0, 1)) * (1 - a**2)
            grad_q_from_o = intermediate_for_grad_wrt_o @ k
            grad_k_from_o = (intermediate_for_grad_wrt_o).transpose(0, 1) @ q

        if grad_a is not None:
            intermediate_for_grad_wrt_a = grad_a * (1 - a**2)
            grad_q_from_a = intermediate_for_grad_wrt_a @ k
            grad_k_from_a = (intermediate_for_grad_wrt_a).transpose(0, 1) @ q

        if q.requires_grad:
            if grad_o is not None and grad_a is not None:
                out[0] = grad_q_from_o + grad_q_from_a
            elif grad_o is not None:
                out[0] = grad_q_from_o
            elif grad_a is not None:
                out[0] = grad_q_from_a

        if k.requires_grad:
            if grad_o is not None and grad_a is not None:
                out[1] = grad_k_from_o + grad_k_from_a
            elif grad_o is not None:
                out[1] = grad_k_from_o
            elif grad_a is not None:
                out[1] = grad_k_from_a

        if v.requires_grad and grad_o is not None:
            out[2] = a.transpose(0, 1) @ grad_o

        return tuple(out)

attn = AttentionGrad.apply
