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
            return x * invoke_subgraph(gn, "start", None, (x, y))

        # opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(8, requires_grad=True, device="cuda")
        y = torch.randn(8, requires_grad=True, device="cuda")
        out = opt_fn(x, y)
        print(out)
        out.sum().backward()

    def test_multi_output(self):
        def gn(x, y):
            return torch.sin(x), torch.cos(y)

        def fn(x, y):
            a, b = invoke_subgraph(gn, "start", None, (x, y))
            return a + b

        # opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(8, requires_grad=True, device="cuda")
        y = torch.randn(8, requires_grad=True, device="cuda")
        out = opt_fn(x, y)
        print(out)
        out.sum().backward()

    def test_linear(self):
        n_layers = 12
        mods = [torch.nn.Linear(1, 1, bias=False).cuda() for _ in range(n_layers)]

        def fn(x):
            for mod in mods:
                x = invoke_subgraph(mod, "start", None, (x,))
            return x

        x = torch.randn(1, 1, requires_grad=True, device="cuda")

        ref = fn(x)

        # opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition")
        opt_fn = torch.compile(fn)

        res = opt_fn(x)

    def test_multiple(self):
        def gn(x):
            return torch.cos(x)

        def fn(x):
            a = invoke_subgraph(gn, "start", None, (x,))
            b = invoke_subgraph(gn, "start", None, (x,))
            return a + b

        # opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(8, requires_grad=True, device="cuda")
        out = opt_fn(x)
        print(out)

    def test_repro_gpt_fast(self):
        n_layers = 2

        def gn(x, y):
            return torch.matmul(x, y).sin()

        def fn(x, y):
            for _ in range(n_layers):
                x = invoke_subgraph(gn, "start", None, (x, y))
            return x

        # opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True, device="cuda")
        y = torch.randn(8, 8, requires_grad=True, device="cuda")
        out = opt_fn(x, y)


if __name__ == "__main__":
    run_tests()
