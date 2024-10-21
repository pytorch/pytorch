# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import aot_function, nop
from torch._higher_order_ops import invoke_subgraph
from torch.testing._internal.common_utils import run_tests, TestCase


class TestInvokeSubgraph(TestCase):
    def test_simple(self):
        def gn(x, y):
            return (torch.mul(x, y),)

        def fn(x, y):
            return invoke_subgraph(gn, None, (x, y))[0]

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
            return invoke_subgraph(gn, None, (x, y))[0]

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
            a = invoke_subgraph(cos, None, (x,))[0]
            b = invoke_subgraph(sin, None, (a,))[0]
            return invoke_subgraph(cos, None, (b,))[0]

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        self.assertEqual(ref, res)

    def test_differing_strides_for_grad_outs(self):
        class CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                a = grad_out.view(12, 5)
                return torch.cos(torch.reshape(a, (3, 4, 5)))

        def gn(x):
            return (CustomOp.apply(x),)

        def fn(x):
            a = invoke_subgraph(gn, None, (x,))[0]
            # Force stride changes so that backward view causes a failure if
            # contiguous not called.
            b = torch.permute(a, (0, 2, 1))
            return b

        x = torch.randn(3, 4, 5, requires_grad=True)
        ref = torch.permute(gn(x)[0], (0, 2, 1))

        x_clone = x.clone().detach().requires_grad_(True)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)


if __name__ == "__main__":
    run_tests()
