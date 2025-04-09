# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950

import unittest

from parameterized import parameterized_class

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import aot_function, nop
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch._higher_order_ops.invoke_subgraph import mark_compile_region
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraph(TestCase):
    def test_simple(self):
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return mark_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function(self):
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return mark_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_multiple(self):
        @mark_compile_region
        def cos(x):
            return torch.cos(x)

        @mark_compile_region
        def sin(x):
            return torch.sin(x)

        def fn(x):
            a = cos(x)
            b = sin(a)
            return cos(b)

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        self.assertEqual(ref, res)


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraphCompile(TestCase):
    def count_unique_get_attr_nodes(self, gm, args, expected):
        subgraph_attr_names = set()
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                subgraph_attr_names.add(node.target)
        self.assertEqual(len(subgraph_attr_names), expected)

    def test_simple(self):
        @mark_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            return gn(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_list(self):
        @mark_compile_region
        def gn(x, y):
            return [torch.mul(x, y), torch.add(x, y)]

        def fn(x, y):
            lst = gn(x, y)
            lst.append(torch.sin(x))
            return lst[0] + lst[1] + lst[2]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_tuple_of_tuple(self):
        @mark_compile_region
        def gn(x, y):
            return ((torch.mul(x, y),), torch.add(x, y))

        def fn(x, y):
            tup = gn(x, y)
            return tup[0][0] + tup[1]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    @unittest.skip("FunctionCtx ops is not cacheable right now")
    def test_differing_strides_for_grad_outs(self):
        class CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                a = grad_out.view(12, 5)
                return torch.cos(torch.reshape(a, (3, 4, 5)))

        @mark_compile_region
        def gn(x):
            return CustomOp.apply(x)

        def fn(x):
            a = gn(x)
            # Force stride changes so that backward view causes a failure if
            # contiguous not called.
            b = torch.permute(a, (0, 2, 1))
            return b

        x = torch.randn(3, 4, 5, requires_grad=True)
        ref = torch.permute(gn(x), (0, 2, 1))

        x_clone = x.clone().detach().requires_grad_(True)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @requires_cuda
    def test_sdpa(self):
        @mark_compile_region
        def gn(q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        def fn(q, k, v):
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
            ):
                return gn(q, k, v)

        q = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        ref = fn(q, k, v)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(q, k, v)
        res.sum().backward()
        self.assertEqual(ref, res)

        res = opt_fn(q, k, v)
        res.sum().backward()

    def test_symint_from_fwd_to_bwd(self):
        @mark_compile_region
        def gn(x, y):
            a = torch.sum(x, (1,), keepdim=True).view(y.shape[1], y.shape[0])
            return torch.matmul(a, y)

        def fn(x, y):
            return gn(x, y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(64, 1, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

        x = torch.randn(256, 1, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

        x = torch.randn(16, 1, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

    def test_dropout(self):
        # `dropout` tests that joint graph passes (not just partitioner) is ran
        # on the hop graphs. Inductor rng functionalization happens in the joint
        # graph passes. Without running joint graph passes, we would get an
        # error like AssertionError: should have been handled in
        # replace_random.py
        @mark_compile_region
        def gn(x):
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        @mark_compile_region
        def hn(x):
            return torch.sin(x)

        def fn(x):
            return gn(x) + hn(x)

        x = torch.randn(8, requires_grad=True)
        # Difficult to check the results here because we random does not match
        # between eager and Triton.
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x)  # noqa: F841

    def test_dedupe(self):
        @mark_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x, y):
            a = gn(x, y)
            return gn(a, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 1)

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
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            return (mul,)
""",
            )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]", primals_2: "f32[8]"):
        ___forward_invoke_subgraph_0_post_graph = self.___forward_invoke_subgraph_0_post_graph

        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(___forward_invoke_subgraph_0_post_graph, '___forward_invoke_subgraph_0_post_graph', (primals_1, primals_2));  ___forward_invoke_subgraph_0_post_graph = primals_1 = None
        getitem_9: "f32[8]" = invoke_subgraph_4[2]
        getitem_8: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        ___forward_invoke_subgraph_0_post_graph_1 = self.___forward_invoke_subgraph_0_post_graph

        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(___forward_invoke_subgraph_0_post_graph_1, '___forward_invoke_subgraph_0_post_graph', (getitem, primals_2));  ___forward_invoke_subgraph_0_post_graph_1 = getitem = primals_2 = None
        getitem_11: "f32[8]" = invoke_subgraph_5[2]
        getitem_10: "f32[8]" = invoke_subgraph_5[1]
        getitem_1: "f32[8]" = invoke_subgraph_5[0];  invoke_subgraph_5 = None
        return (getitem_1, getitem_9, getitem_8, getitem_11, getitem_10)

    class ___forward_invoke_subgraph_0_post_graph(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            return (mul, primals_0, primals_1)
""",
        )

    def test_nonlocal_update(self):
        counter = 2

        @mark_compile_region
        def gn(x, y):
            nonlocal counter
            return (torch.mul(x, y) * counter,)

        def fn(x, y):
            nonlocal counter
            counter = 2
            a = gn(x, y)[0]
            counter = 3
            return gn(a, y)[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

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
        @mark_compile_region
        def gn(x, y):
            # Different graph give different names to intermediate nodes
            for _ in range(5):
                x = x * y
            return x

        def fn(x, y):
            for _ in range(5):
                x = gn(x, y)
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
        @mark_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            return gn(x, y)

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered input mutation during higher order op tracing for HOP - invoke_subgraph",
        ):
            opt_fn(x, y)

    def test_input_mutation_inference_mode(self):
        @mark_compile_region
        def gn(x, y):
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            z = torch.cos(x)
            with torch.inference_mode():
                return gn(torch.cos(z), y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered input mutation during higher order op tracing",
        ):
            opt_fn(x, y)

    def test_simple_module(self):
        mod = torch.nn.Linear(8, 8)

        @mark_compile_region
        def gn(x):
            return torch.cos(x), mod(x)

        def fn(x):
            out = gn(x)
            return out[0] + out[1]

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # requires_grad is False deliberately to force None the joint_graph
        # outputs
        x = torch.randn(8, 8, requires_grad=False)
        x_clone = x.detach().clone().requires_grad_(False)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    def test_fail_with_direct_invoke_subgraph(self):
        from torch._higher_order_ops import invoke_subgraph

        def gn(x):
            return torch.sin(x)

        def fn(x):
            return invoke_subgraph(gn, None, (x,))

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Directly using invoke_subgraph is not"
        ):
            opt_fn(x)

    def test_input_output_aliasing(self):
        @mark_compile_region
        def gn(x, y):
            return (x, torch.mul(x, y))

        def fn(x, y):
            outs = gn(x, y)
            return outs[0] * outs[1]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x, y)

    def test_input_input_aliasing(self):
        @mark_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        def fn(x):
            return gn(x, x.view(1, 8))

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_output_output_aliasing(self):
        @mark_compile_region
        def gn(x):
            z = torch.cos(x)
            return z, z.view(1, 8)

        def fn(x):
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_mod_attr_aliasing(self):
        class MutateParam(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.ones(8)

            def forward(self, x):
                self.a.add_(1)
                return torch.mul(x, self.a)

        @mark_compile_region
        def gn(x):
            return mod(x)

        def fn(x, y):
            return gn(x) * y

        mod = MutateParam()
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        fn(x, y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Encountered input mutation during higher order op tracing",
        ):
            opt_fn(x, y)

    def test_kwargs_only(self):
        @mark_compile_region
        def gn(x, *, y):
            return x * y

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        def fn(x, y):
            return gn(x, y=y)

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_module_method(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            @mark_compile_region
            def helper(self, x):
                return self.linear(x)

            def forward(self, x):
                return x + self.helper(x) * self.helper(x) + x

        mod = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_mod = torch.compile(mod, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_self_modules_linear_parameters_weight_: "f32[8, 8]", L_self_modules_linear_parameters_bias_: "f32[8]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_));  invoke_subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        invoke_subgraph_1 = self.invoke_subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_1, 'invoke_subgraph_0', (l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_));  invoke_subgraph_1 = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]", l_self_modules_linear_parameters_weight_: "f32[8, 8]", l_self_modules_linear_parameters_bias_: "f32[8]"):
            linear: "f32[8, 8]" = torch._C._nn.linear(l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  l_x_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
            return (linear,)
""",
            )

    def test_module(self):
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = mark_compile_region(SubMod())

            def forward(self, x):
                return x + self.submod(x) * self.submod(x) + x

        mod = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_mod = torch.compile(mod, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_,));  invoke_subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        invoke_subgraph_1 = self.invoke_subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_1, 'invoke_subgraph_0', (l_x_,));  invoke_subgraph_1 = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)
""",
            )

    @requires_cuda
    def test_return_none(self):
        from torch.nn import functional as F

        weight = torch.ones(
            1000, device="cuda:0", dtype=torch.float32, requires_grad=True
        )
        ones = torch.ones(1000, device="cuda:0", dtype=torch.float32)

        @mark_compile_region
        def fn(x, train):
            return F.dropout(x * weight, 0.33, train)

        @torch._dynamo.optimize_assert("inductor")
        def run(x, train=True):
            return fn(x, train)

        r1 = run(ones, train=False)
        r1.sum().backward()
        weight.grad.clone()

    def test_return_none_from_fwd(self):
        @mark_compile_region
        def gn(x):
            return x * 2, None, x * 3

        def fn(x):
            ys = gn(x)
            return ys[0] + ys[2]

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        res = opt_fn(x_clone)
        res.sum().backward()

        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 1)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        invoke_subgraph_0 = self.invoke_subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(invoke_subgraph_0, 'invoke_subgraph_0', (l_x_,));  invoke_subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph[2];  invoke_subgraph = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class invoke_subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            child: "f32[8, 8]" = l_x_ * 2
            child_1: "f32[8, 8]" = l_x_ * 3;  l_x_ = None
            return (child, None, child_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 8]"):
        ___forward_invoke_subgraph_0_post_graph = self.___forward_invoke_subgraph_0_post_graph

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(___forward_invoke_subgraph_0_post_graph, '___forward_invoke_subgraph_0_post_graph', (primals_1,));  ___forward_invoke_subgraph_0_post_graph = primals_1 = None
        getitem: "f32[8, 8]" = invoke_subgraph_2[0]
        getitem_2: "f32[8, 8]" = invoke_subgraph_2[2];  invoke_subgraph_2 = None

        add: "f32[8, 8]" = torch.ops.aten.add.Tensor(getitem, getitem_2);  getitem = getitem_2 = None
        return (add,)

    class ___forward_invoke_subgraph_0_post_graph(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]"):
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(primals_0, 2)
            mul_1: "f32[8, 8]" = torch.ops.aten.mul.Tensor(primals_0, 3);  primals_0 = None
            return (mul, None, mul_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 8]"):
        ___backward_invoke_subgraph_0_post_graph = self.___backward_invoke_subgraph_0_post_graph

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(___backward_invoke_subgraph_0_post_graph, '___backward_invoke_subgraph_0_post_graph', (tangents_1, tangents_1));  ___backward_invoke_subgraph_0_post_graph = tangents_1 = None
        getitem_3: "f32[8, 8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        return (getitem_3,)

    class ___backward_invoke_subgraph_0_post_graph(torch.nn.Module):
        def forward(self, tangents_0: "f32[8, 8]", tangents_1: "f32[8, 8]"):
            mul_2: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, 3)
            mul_3: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

            add: "f32[8, 8]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
            return (add,)
""",
            )

    def test_dynamic(self):
        @mark_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            return gn(x)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_pending_unbacked(self):
        @mark_compile_region
        def gn(x):
            u = x[0].item()
            return x * u

        def fn(x):
            return gn(x)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)
        opt_fn = torch.compile(
            fn, backend="eager", fullgraph=True
        )  # Inductor fails with cpp compilation error
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked(self):
        @mark_compile_region
        def gn(x, y):
            b = x.item()
            torch._check_is_size(b)
            torch._check(b < y.shape[0])
            return y[:b].clone()

        def fn(x, y):
            return gn(x, y)

        x = torch.tensor(4)
        y = torch.randn(8)
        ref = fn(x, y)
        opt_fn = torch.compile(
            fn, backend="eager", fullgraph=True
        )  # Inductor fails with assertion error when lowering aten.sym_constrain_range_for_size.default
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_bwd_partitioning(self):
        @mark_compile_region
        def gn(x, y):
            z = torch.matmul(x, y)
            return torch.sin(z)

        def fn(x, y):
            return torch.sin(gn(x, y))

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 8]", primals_2: "f32[8, 8]"):
        ___forward_invoke_subgraph_0_post_graph = self.___forward_invoke_subgraph_0_post_graph

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(___forward_invoke_subgraph_0_post_graph, '___forward_invoke_subgraph_0_post_graph', (primals_1, primals_2));  ___forward_invoke_subgraph_0_post_graph = primals_1 = primals_2 = None
        getitem_6: "f32[8, 8]" = invoke_subgraph_2[3]
        getitem_5: "f32[8, 8]" = invoke_subgraph_2[2]
        getitem_4: "f32[8, 8]" = invoke_subgraph_2[1]
        getitem: "f32[8, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        sin: "f32[8, 8]" = torch.ops.aten.sin.default(getitem)
        cos: "f32[8, 8]" = torch.ops.aten.cos.default(getitem);  getitem = None
        return (sin, getitem_6, getitem_5, getitem_4, cos)

    class ___forward_invoke_subgraph_0_post_graph(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]", primals_1: "f32[8, 8]"):
            mm: "f32[8, 8]" = torch.ops.aten.mm.default(primals_0, primals_1)

            sin: "f32[8, 8]" = torch.ops.aten.sin.default(mm)

            t: "f32[8, 8]" = torch.ops.aten.t.default(primals_0);  primals_0 = None
            t_1: "f32[8, 8]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
            return (sin, mm, t, t_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_6: "f32[8, 8]", getitem_5: "f32[8, 8]", getitem_4: "f32[8, 8]", cos: "f32[8, 8]", tangents_1: "f32[8, 8]"):
        mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None

        ___backward_invoke_subgraph_0_post_graph = self.___backward_invoke_subgraph_0_post_graph

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(___backward_invoke_subgraph_0_post_graph, '___backward_invoke_subgraph_0_post_graph', (getitem_4, getitem_5, getitem_6, mul));  ___backward_invoke_subgraph_0_post_graph = getitem_4 = getitem_5 = getitem_6 = mul = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_3[0]
        getitem_2: "f32[8, 8]" = invoke_subgraph_3[1];  invoke_subgraph_3 = None
        return (getitem_1, getitem_2)

    class ___backward_invoke_subgraph_0_post_graph(torch.nn.Module):
        def forward(self, mm: "f32[8, 8]", t: "f32[8, 8]", t_1: "f32[8, 8]", tangents_0: "f32[8, 8]"):
            cos: "f32[8, 8]" = torch.ops.aten.cos.default(mm);  mm = None
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_0, cos);  tangents_0 = cos = None

            mm_1: "f32[8, 8]" = torch.ops.aten.mm.default(t, mul);  t = None
            mm_2: "f32[8, 8]" = torch.ops.aten.mm.default(mul, t_1);  mul = t_1 = None
            return (mm_2, mm_1)
""",
            )

    def test_const_tensor(self):
        @mark_compile_region
        def gn(x):
            return torch.tensor(64, dtype=torch.float32) * x

        def fn(x):
            return gn(x)

        x = torch.randn(64, requires_grad=True)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)


@parameterized_class(
    [
        {"strict": False},
        {"strict": True},
    ],
    class_name_func=lambda cls, _, params: f"{cls.__name__}{'Strict' if params['strict'] else 'Nonstrict'}",
)
class TestInvokeSubgraphExport(TestCase):
    def test_simple_func(self):
        @mark_compile_region
        def gn(x, y):
            return torch.mul(x, y)

        class M(torch.nn.Module):
            def forward(self, x, y):
                x = gn(x, y)
                x = gn(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

        self.assertExpectedInline(
            normalize_gm(ep.graph_module.print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[8]", y: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', (x, y));  repeated_subgraph0 = x = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'invoke_subgraph_0', (getitem, y));  repeated_subgraph0_1 = getitem = y = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (mul,)
""",
        )

    def test_unbacked(self):
        @mark_compile_region
        def gn(x, y):
            b = x.item()
            torch._check_is_size(b)
            torch._check(b < y.shape[0])
            return y[:b].clone()

        class M(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for _ in range(10):
                    res.append(gn(x, y))
                return torch.cat(res)

        x = torch.tensor(4)
        y = torch.randn(8)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        ep = ep.run_decompositions()

        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    def test_pending_unbacked(self):
        class M(torch.nn.Module):
            @mark_compile_region
            def gn(self, x):
                u = x[0].item()
                return x * u

            def forward(self, x):
                for _ in range(4):
                    x = self.gn(x)
                return x

        ep = torch.export.export(
            M(),
            (torch.randn(8),),
            strict=self.strict,
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        )
        ep = ep.run_decompositions()

        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

        ep = torch.export.export(
            M(),
            (torch.randn(8, requires_grad=True),),
            strict=self.strict,
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        )
        ep = ep.run_decompositions()

        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    def test_simple_method(self):
        class M(torch.nn.Module):
            @mark_compile_region
            def gn(self, x, y):
                return torch.mul(x, y)

            def forward(self, x, y):
                x = self.gn(x, y)
                x = self.gn(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    def test_multiple_module(self):
        b = torch.randn(8)

        class N(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", b)

            @mark_compile_region
            def forward(self, x, y):
                return x * y + self.buf

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod_list = torch.nn.ModuleList(N() for _ in range(10))

            def forward(self, x, y):
                for m in self.mod_list:
                    x = m(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)


if __name__ == "__main__":
    run_tests()
