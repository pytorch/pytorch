# Owner(s): ["module: higher order operators"]


import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.nn.functional as F
from torch import nn
from torch._dynamo.variables.higher_order_ops import LocalMapWrappedHigherOrderVariable
from torch.distributed._tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.triton_utils import requires_cuda_and_triton


nested_compile_region = torch.compiler.nested_compile_region


class MyTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x + 100

    @staticmethod
    def backward(ctx, grad):
        return grad + 100


def context_parallel_attention(query, key, value):
    out = F.scaled_dot_product_attention(
        query=query, key=key, value=value, is_causal=False
    )
    return out


@local_map(
    out_placements=((Shard(0), Shard(1), Shard(2)),),
    in_placements=(
        (Shard(0), Shard(1), Shard(2)),  # query
        (Shard(0), Shard(1), Replicate()),  # key
        (Shard(0), Shard(1), Replicate()),  # value
    ),
    redistribute_inputs=True,
    in_grad_placements=None,
    device_mesh=None,
)
def cp_decorated(query, key, value):
    return context_parallel_attention(query, key, value)


cp_function = local_map(
    context_parallel_attention,
    out_placements=(Shard(0), Shard(1), Shard(2)),
    in_placements=(
        (Shard(0), Shard(1), Shard(2)),  # query
        (Shard(0), Shard(1), Replicate()),  # key
        (Shard(0), Shard(1), Replicate()),  # value
    ),
    redistribute_inputs=True,
    in_grad_placements=None,
    device_mesh=None,
)


def create_model(attention_fn, nheads, dim1, dim2):
    class LocalMapTransformerBlock(nn.Module):
        def __init__(self, nheads, dim1, dim2):
            super().__init__()
            self.nheads = nheads
            bias = False
            self.wq = nn.Linear(dim1, dim1, bias=bias)
            self.wk = nn.Linear(dim1, dim1, bias=bias)
            self.wv = nn.Linear(dim1, dim1, bias=bias)
            self.wo = nn.Linear(dim1, dim1, bias=bias)
            self.w1 = nn.Linear(dim1, dim2, bias=bias)
            self.w2 = nn.Linear(dim2, dim1, bias=bias)

        def forward(self, x):
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
            v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

            o = attention_fn(q, k, v)
            o = o.permute(0, 2, 1, 3).flatten(-2)

            o = self.wo(o)

            o0 = o + x

            o = self.w1(o0)
            o = torch.nn.functional.relu(o)
            o = self.w2(o)

            o = o0 + o
            return o

    return LocalMapTransformerBlock(nheads, dim1, dim2)


class TestLocalMap(TestCase):
    @requires_cuda_and_triton
    def test_simple(self):
        bs = 8 * 32
        dim1 = 6144
        dim2 = dim1 * 4
        nheads = 48
        seq_len = 256

        model = create_model(cp_decorated, nheads, dim1, dim2).cuda()
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True).cuda(),)
        with LocalMapWrappedHigherOrderVariable.enable():
            out = torch.compile(model, fullgraph=False)(*inputs)
        out.sum().backward()


if __name__ == "__main__":
    run_tests()
