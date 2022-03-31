import torch
from torch import Tensor
import copy

from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, instantiate_parametrized_tests

def attn(q, k, v):
    x = torch.matmul(q, k.t())
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a

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
        needs_dq, needs_dk, needs_dv = ctx.needs_input_grad
        dq, dk, dv = None, None, None

        # forward: o = mm(a, v)
        # backward: da = mm(do, v'), dv = mm(a', do)
        if do is not None:
            da_partial = torch.matmul(do, v.t())
            dv = torch.matmul(a.t(), do)
        else:
            da_partial = torch.zeros_like(a)
            dv = torch.zeros_like(v)

        if needs_dq or needs_dk:
            if da is not None:
                da += da_partial
            else:
                da = da_partial

            # forward: a = tanh(x)
            # backward: dx = da * (1 - a * a)
            dx = da * (1 - a * a)

            # forward: x = mm(q, k')
            # backward: dq = mm(dx, k); dk = mm(dx', q)
            if needs_dq:
                dq = torch.matmul(dx, k)
            if needs_dk:
                dk = torch.matmul(dx.t(), q)

        return dq, dk, dv


class TestAttnAutograd(TestCase):
    @parametrize("q_requires_grad", [True, False])
    @parametrize("k_requires_grad", [True, False])
    @parametrize("v_requires_grad", [True, False])
    # @parametrize("input_requires_grad", [True, False])
    def test_by_gradcheck(self, q_requires_grad, k_requires_grad, v_requires_grad):
        if not any([q_requires_grad, k_requires_grad, v_requires_grad]):
            return

        q = torch.randn(2, 3, dtype=torch.float64, requires_grad=q_requires_grad)
        k = torch.randn(2, 3, dtype=torch.float64, requires_grad=k_requires_grad)
        v = torch.randn(2, 4, dtype=torch.float64, requires_grad=v_requires_grad)

        assert torch.autograd.gradcheck(AttentionLayer.apply, [q, k, v])
        # assert torch.autograd.gradgradcheck(AttentionLayer.apply, [q, k, v])

    @parametrize("q_requires_grad", [True, False])
    @parametrize("k_requires_grad", [True, False])
    @parametrize("v_requires_grad", [True, False])
    def test_by_compare_with_torch(self, q_requires_grad, k_requires_grad, v_requires_grad):
        if not any([q_requires_grad, k_requires_grad, v_requires_grad]):
            return

        q = torch.randn(2, 3, dtype=torch.float64, requires_grad=q_requires_grad)
        k = torch.randn(2, 3, dtype=torch.float64, requires_grad=k_requires_grad)
        v = torch.randn(2, 4, dtype=torch.float64, requires_grad=v_requires_grad)
        qq, kk, vv = (copy.deepcopy(x) for x in (q, k, v))

        def loss(o, a):
            return o.exp().sum() + a.sigmoid().sum()

        # my impl
        o, a = AttentionLayer.apply(q, k, v)
        l = loss(o, a)
        l.backward()

        # reference impl
        oo, aa = attn(qq, kk, vv)
        ll = loss(oo, aa)
        ll.backward()

        def assert_is_none_or_close(t, tt):
            if tt.grad is not None:
                assert torch.allclose(t.grad, tt.grad)
            else:
                assert t.grad is None

        assert_is_none_or_close(q, qq)
        assert_is_none_or_close(k, kk)
        assert_is_none_or_close(v, vv)


instantiate_parametrized_tests(TestAttnAutograd)

if __name__ == '__main__':
    run_tests()
