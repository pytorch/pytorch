# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import aot_function, nop
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch._higher_order_ops import invoke_subgraph
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)


@skipIfTorchDynamo("Not a torch._dynamo test")
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


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraphCompile(TestCase):
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
        res = torch.compile(fn, backend="eager", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_multiple(self):
        def gn(x, y):
            return (torch.mul(x, y),)

        def fn(x, y):
            a = invoke_subgraph(gn, None, (x, y))[0]
            return invoke_subgraph(gn, None, (a, y))[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Check that the Dynamo graph has just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        subgraph_attr_names = set()
        for node in backend.graphs[0].graph.nodes:
            if node.op == "get_attr":
                subgraph_attr_names.add(node.target)
        self.assertEqual(len(subgraph_attr_names), 1)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_, l_y_));  invoke_subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        invoke_subgraph_1 = self.invoke_subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_1, 'invoke_subgraph_0', (a, l_y_));  invoke_subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        return (getitem_1,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            child: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            return (child,)
""",
            )

    def test_nonlocal_update(self):
        counter = 2

        def gn(x, y):
            nonlocal counter
            return (torch.mul(x, y) * counter,)

        def fn(x, y):
            nonlocal counter
            counter = 2
            a = invoke_subgraph(gn, None, (x, y))[0]
            counter = 3
            return invoke_subgraph(gn, None, (a, y))[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.clone().detach().requires_grad_(True)
        y_clone = y.clone().detach().requires_grad_(True)
        res = torch.compile(fn, backend="eager", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        torch._dynamo.reset()
        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_, l_y_));  invoke_subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        invoke_subgraph_1 = self.invoke_subgraph_1
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_1, 'invoke_subgraph_1', (a, l_y_));  invoke_subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        return (getitem_1,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            child: "f32[8]" = mul * 2;  mul = None
            return (child,)

    class invoke_subgraph_1(torch.nn.Module):
        def forward(self, a: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(a, l_y_);  a = l_y_ = None
            child: "f32[8]" = mul * 3;  mul = None
            return (child,)
""",
            )

    def test_normalize_gm(self):
        def gn(x, y):
            # Different graph give different names to intermediate nodes
            for _ in range(5):
                x = x * y
            return x

        def fn(x, y):
            for _ in range(5):
                x = invoke_subgraph(gn, None, (x, y))
            return x

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        opt_fn(x, y)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_, l_y_));  invoke_subgraph_0 = l_x_ = None
        x: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        invoke_subgraph_1 = self.invoke_subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_1, 'invoke_subgraph_0', (x, l_y_));  invoke_subgraph_1 = x = None
        x_1: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        invoke_subgraph_3 = self.invoke_subgraph_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_3, 'invoke_subgraph_0', (x_1, l_y_));  invoke_subgraph_3 = x_1 = None
        x_2: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        invoke_subgraph_5 = self.invoke_subgraph_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_5, 'invoke_subgraph_0', (x_2, l_y_));  invoke_subgraph_5 = x_2 = None
        x_3: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        invoke_subgraph_7 = self.invoke_subgraph_0
        invoke_subgraph_8 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_7, 'invoke_subgraph_0', (x_3, l_y_));  invoke_subgraph_7 = x_3 = l_y_ = None
        x_4: "f32[8]" = invoke_subgraph_8[0];  invoke_subgraph_8 = None
        return (x_4,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            x: "f32[8]" = l_x_ * l_y_;  l_x_ = None
            x_1: "f32[8]" = x * l_y_;  x = None
            x_2: "f32[8]" = x_1 * l_y_;  x_1 = None
            x_3: "f32[8]" = x_2 * l_y_;  x_2 = None
            x_4: "f32[8]" = x_3 * l_y_;  x_3 = l_y_ = None
            return (x_4,)
""",
            )

    def test_input_mutation(self):
        def gn(x, y):
            x.add_(1)
            return (torch.mul(x, y),)

        def fn(x, y):
            return invoke_subgraph(gn, None, (x, y))[0]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "NYI: invoke_subgraph with aliasing"
        ):
            opt_fn(x, y)

    def test_input_aliasing(self):
        def gn(x, y):
            return (x, torch.mul(x, y))

        def fn(x, y):
            outs = invoke_subgraph(gn, None, (x, y))
            return outs[0] * outs[1]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "NYI: invoke_subgraph with aliasing"
        ):
            opt_fn(x, y)


if __name__ == "__main__":
    run_tests()
