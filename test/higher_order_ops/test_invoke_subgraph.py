# Owner(s): ["module: functorch"]
# flake8: noqa: B950
import unittest

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


invoke_subgraph = torch._higher_order_ops.invoke_subgraph


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestInvokeSubgraph(TestCase):
    def setUp(self):
        init_torchbind_implementations()

    def test_simple(self):
        silu = torch.nn.SiLU().cuda()

        def gn(x, y):
            # return torch.mul(x, y)
            a = silu(x)
            a.add_(y)
            return a

        def fn(x, y):
            return (
                x
                * invoke_subgraph(gn, "start", x, y)
                * y
                * invoke_subgraph(gn, "start", x, y)
            )

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        # opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(8, requires_grad=True, device="cuda")
        y = torch.randn(8, requires_grad=True, device="cuda")
        out = opt_fn(x, y)
        print(out)
        out.sum().backward()


if __name__ == "__main__":
    run_tests()
