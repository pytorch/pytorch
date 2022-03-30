import torch
from torch import Tensor
import copy

def attn(q, k, v):
    x = torch.matmul(q, k.t())
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a

def loss(o, a):
    return o.exp().sum() + a.sigmoid().sum()

class AttentionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor):
        assert q.dim() == 2, f"Expecting q's rank to be 2, got {q.dim()}"
        assert k.dim() == 2, f"Expecting k's rank to be 2, got {k.dim()}"
        assert v.dim() == 2, f"Expecting v's rank to be 2, got {v.dim()}"

        assert q.size(0) == k.size(0) == v.size(0), f"Expecting qkv's first dim to be same, got ({q.size(0)}, {k.size(0)}, {v.size(0)})"
        assert q.size(1) == k.size(1), f"Expecting qk's second dim to be same, got ({q.size(1)}, {k.size(1)})"

        o, a = attn(q, k, v)

        ctx.save_for_backward(q, k, v, a)
        ctx.set_materialize_grads(False)

        return o, a

    @staticmethod
    def backward(ctx, do: Tensor, da: Tensor):
        q, k, v, a = ctx.saved_tensors

        # o = a * v
        # da = do * v', dv = a' * do
        if do is not None:
            da_partial = torch.matmul(do, v.t())
            dv = torch.matmul(a.t(), do)
        else:
            da_partial = torch.zeros_like(a)
            dv = torch.zeros_like(v)

        if da is not None:
            da += da_partial
        else:
            da = da_partial

        # a = tanh(x)
        dx = da * (1 - a * a)

        # x = q * k'
        # dq = dx * k, dk = dx' * q
        dq = torch.matmul(dx, k)
        dk = torch.matmul(dx.t(), q)

        return dq, dk, dv


q = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
k = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
v = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
qq, kk, vv = (copy.deepcopy(x) for x in (q, k, v))

# my impl
o, a = AttentionLayer.apply(q, k, v)
l = loss(o, a)
l.backward()

# reference impl
oo, aa = attn(qq, kk, vv)
ll = loss(oo, aa)
ll.backward()

assert torch.allclose(q.grad, qq.grad), f"{q.grad}, {qq.grad}"
assert torch.allclose(k.grad, kk.grad)
assert torch.allclose(v.grad, vv.grad)

assert torch.autograd.gradcheck(AttentionLayer.apply, [q, k, v])

