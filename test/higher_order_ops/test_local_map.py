# Owner(s): ["module: higher order operators"]

import unittest
import unittest.mock as mock

from parameterized import parameterized_class
import functools

import torch
from torch import nn
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.utils._pytree as pytree
from functorch.compile import aot_function, nop
from torch.distributed._tensor.experimental import local_map
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    InductorAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.schema import find_hop_schema
from torch._inductor import config as inductor_config
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton, requires_gpu

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch._higher_order_ops.local_map import apply_local_map

nested_compile_region = torch.compiler.nested_compile_region


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
        # @apply_local_map(
        @functools.partial(
            local_map,
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
        def _context_parallel_attention(query, key, value):
            out = F.scaled_dot_product_attention(
                query=query, key=key, value=value, is_causal=False
            )
            return (out,)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = _context_parallel_attention(q, k, v)[0]
        o = o.permute(0, 2, 1, 3).flatten(-2)

        o = self.wo(o)

        o0 = o + x

        o = self.w1(o0)
        o = torch.nn.functional.relu(o)
        o = self.w2(o)

        o = o0 + o
        return o

class TestLocalMap(TestCase):
    def test_simple(self):
        bs = 8 * 32
        dim1 = 6144
        dim2 = dim1 * 4
        nheads = 48
        seq_len = 256

        model = LocalMapTransformerBlock(nheads, dim1, dim2).cuda()
        inputs = (torch.randn(bs, seq_len, dim1, requires_grad=True).cuda(),)
        ep = torch.export.export(model, inputs, strict=True)
        breakpoint()

if __name__ == "__main__":
    run_tests()
