import torch
from torch.testing._internal.common_utils import TestCase
from torch.autograd import gradcheck
from torch.autograd import gradgradcheck

class Attn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):

        if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
            raise ValueError(
                f"Expected 2D inputs for attention, but got shapes: q={q.shape}, k={k.shape}, v={v.shape}"
            )

        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError(
                f"Mismatch in dim 0: q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
            )

        if q.shape[1] != k.shape[1]:
            raise ValueError(
                f"Mismatch in dim 1 for q and k: q={q.shape[1]}, k={k.shape[1]}"
            )

        b = torch.mm(q, k.transpose(0,1))
        a = torch.tanh(b)
        o = torch.mm(a, v)
        ctx.save_for_backward(q, k, v, a)

        return o, a
    
    @staticmethod
    def backward(ctx, grad_o, grad_a):
        q, k, v, a = ctx.saved_tensors
        da_db = 1 - torch.tanh(torch.mm(q, k.transpose(0,1))) ** 2
        dl_do = torch.mm(grad_o, v.transpose(0,1)) * da_db
        dl_db = grad_a * da_db

        dlo_dq = torch.mm(dl_do, k)
        dla_dq = torch.mm(dl_db, k)
        dl_dq = dlo_dq + dla_dq

        dlo_dk = torch.mm(dl_do.transpose(0,1), q)
        dla_dk = torch.mm(dl_db.transpose(0,1), q)
        dl_dk = dlo_dk + dla_dk

        dl_dv = torch.mm(a.transpose(0,1), grad_o)

        return dl_dq, dl_dk, dl_dv

class AttnTest(TestCase):
    def test_attn(self):
        q = torch.rand((2, 3), requires_grad=True, dtype=torch.float64)
        k = torch.rand((2, 3), requires_grad=True, dtype=torch.float64)
        v = torch.rand((2, 4), requires_grad=True, dtype=torch.float64)

        attn_fn = Attn.apply

        gradcheck(attn_fn, (q, k, v))
        gradgradcheck(attn_fn, (q, k, v))

if __name__ == "__main__":
    AttnTest().test_attn()