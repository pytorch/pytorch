# Owner(s): ["module: functorch"]
# flake8: noqa: B950

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import aot_function, nop
from functorch.experimental.control_flow import UnsupportedAliasMutationException
from torch.testing._internal.common_utils import run_tests, TestCase


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


invoke_subgraph = torch._higher_order_ops.invoke_subgraph


class TestInvokeSubgraph(TestCase):
    def test_simple(self):
        def gn(x, y):
            return (torch.mul(x, y),)

        def fn(x, y):
            return invoke_subgraph(gn, "start", (x, y))[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        res = fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function(self):
        def gn(x, y):
            return (torch.mul(x, y),)

        def fn(x, y):
            return invoke_subgraph(gn, "start", (x, y))[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)[0]

        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_multiple(self):
        n_layers = 2

        def cos(x):
            return (torch.cos(x),)

        def sin(x):
            return (torch.sin(x),)

        def fn(x):
            a = invoke_subgraph(cos, "start", (x,))[0]
            b = invoke_subgraph(sin, "start", (a,))[0]
            return invoke_subgraph(cos, "start", (b,))[0]

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        self.assertEqual(ref, res)

    def test_input_mutation(self):
        def gn(x, y):
            x.add_(1)
            return (torch.mul(x, y),)

        def fn(x, y):
            return invoke_subgraph(gn, "start", (x, y))[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of invoke_subgraph hop"
        ):
            aot_fn = aot_function(fn, nop)
            res = aot_fn(x, y)

    def test_input_aliasing(self):
        def gn(x, y):
            return (x, torch.mul(x, y))

        def fn(x, y):
            outs = invoke_subgraph(gn, "start", (x, y))
            return outs[0] * outs[1]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of invoke_subgraph hop"
        ):
            aot_fn = aot_function(fn, nop)
            res = aot_fn(x, y)


if __name__ == "__main__":
    run_tests()
