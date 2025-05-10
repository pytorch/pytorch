# Owner(s): ["module: higher order operators"]
# flake8: noqa: B950
# flake8: noqa: E731

import unittest

from parameterized import parameterized_class

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import aot_function, nop
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    InductorAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.invoke_subgraph import mark_compile_region
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda, requires_gpu


if HAS_GPU:
    import triton


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

    def test_module_forward(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = 5

            @mark_compile_region
            def forward(self, x, y):
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            return mod(x, y) + mod(x, y)

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

    def test_dropout_checks_joint_graph(self):
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

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x)
        res.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1);  partitioned_fw_subgraph_0_0 = None
        getitem_7: "b8[8]" = invoke_subgraph_4[2]
        getitem_6: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        partitioned_fw_subgraph_1_0 = self.partitioned_fw_subgraph_1_0

        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_1_0, 'partitioned_fw_subgraph_1_0', primals_1);  partitioned_fw_subgraph_1_0 = primals_1 = None
        getitem_8: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_7, getitem_6, getitem_8)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)

            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))
            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None

            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1, primals_0, gt)

    class partitioned_fw_subgraph_1_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)
            return (sin, primals_0)
""",
            )

    def test_dropout_checks_joint_graph_inference(self):
        # Checks that joint graph results in inductor seeds for just the inference graph
        @mark_compile_region
        def gn(x):
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        def fn(x):
            return gn(x)

        backend = InductorAndRecordGraphs()
        x = torch.randn(8, requires_grad=False)
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))
            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None

            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  gt = sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1,)
""",
            )

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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', a, l_y_);  subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
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
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = primals_1 = None
        getitem_9: "f32[8]" = invoke_subgraph_4[2]
        getitem_8: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', getitem, primals_2);  partitioned_fw_subgraph_0_1 = getitem = primals_2 = None
        getitem_11: "f32[8]" = invoke_subgraph_6[2]
        getitem_10: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        return (getitem_1, getitem_9, getitem_8, getitem_11, getitem_10)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            return (mul, primals_0, primals_1)
""",
        )

    def test_dce(self):
        @mark_compile_region
        def gn(x):
            x = torch.sin(x)
            # should be dce'd
            y = torch.cos(x)  # noqa: F841
            return x

        def fn(x):
            return gn(x)

        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.randn(4, requires_grad=False)
        )

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[4]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4]"):
            sin: "f32[4]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return (sin,)
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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        a: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', a, l_y_);  subgraph_1 = a = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            child: "f32[8]" = mul * 2;  mul = None
            return (child,)

    class subgraph_1(torch.nn.Module):
        def forward(self, a: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(a, l_y_);  a = l_y_ = None
            child: "f32[8]" = mul * 3;  mul = None
            return (child,)
""",
            )

    def test_view_to_reshape(self):
        @mark_compile_region
        def gn(x):
            x = torch.sin(x)
            x = x.view(1, 8)
            return torch.sin(x)

        def fn(x):
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        torch._dynamo.reset()
        backend = InductorAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[1, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

            view: "f32[1, 8]" = torch.ops.aten.reshape.default(sin, [1, 8]);  sin = None

            sin_1: "f32[1, 8]" = torch.ops.aten.sin.default(view);  view = None
            return (sin_1,)
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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        x: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', x, l_y_);  subgraph_1 = x = None
        x_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        subgraph_2 = self.subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_2, 'subgraph_0', x_1, l_y_);  subgraph_2 = x_1 = None
        x_2: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        subgraph_3 = self.subgraph_0
        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(subgraph_3, 'subgraph_0', x_2, l_y_);  subgraph_3 = x_2 = None
        x_3: "f32[8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        subgraph_4 = self.subgraph_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(subgraph_4, 'subgraph_0', x_3, l_y_);  subgraph_4 = x_3 = l_y_ = None
        x_4: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        return (x_4,)

    class subgraph_0(torch.nn.Module):
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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  subgraph_1 = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class subgraph_0(torch.nn.Module):
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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class subgraph_0(torch.nn.Module):
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

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph[2];  invoke_subgraph = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
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
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1);  partitioned_fw_subgraph_0_0 = primals_1 = None
        getitem: "f32[8, 8]" = invoke_subgraph_2[0]
        getitem_2: "f32[8, 8]" = invoke_subgraph_2[2];  invoke_subgraph_2 = None

        add: "f32[8, 8]" = torch.ops.aten.add.Tensor(getitem, getitem_2);  getitem = getitem_2 = None
        return (add,)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
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
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', tangents_1, tangents_1);  partitioned_bw_subgraph_0_0 = tangents_1 = None
        getitem_3: "f32[8, 8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        return (getitem_3,)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
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
            return gn(x) + gn(x)

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
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = primals_1 = primals_2 = None
        getitem_6: "f32[8, 8]" = invoke_subgraph_2[3]
        getitem_5: "f32[8, 8]" = invoke_subgraph_2[2]
        getitem_4: "f32[8, 8]" = invoke_subgraph_2[1]
        getitem: "f32[8, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None

        sin: "f32[8, 8]" = torch.ops.aten.sin.default(getitem)
        cos: "f32[8, 8]" = torch.ops.aten.cos.default(getitem);  getitem = None
        return (sin, getitem_6, getitem_5, getitem_4, cos)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
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

        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_4, getitem_5, getitem_6, mul);  partitioned_bw_subgraph_0_0 = getitem_4 = getitem_5 = getitem_6 = mul = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_3[0]
        getitem_2: "f32[8, 8]" = invoke_subgraph_3[1];  invoke_subgraph_3 = None
        return (getitem_1, getitem_2)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
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
            return gn(x) + gn(x)

        x = torch.randn(64, requires_grad=True)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_ac(self):
        def fn1(x):
            return torch.cos(x)

        @mark_compile_region
        def fn1_checkpoint(x):
            return torch.utils.checkpoint.checkpoint(fn1, x, use_reentrant=False)

        def fn2(x):
            return torch.sin(x)

        @mark_compile_region
        def fn2_checkpoint(x):
            return torch.utils.checkpoint.checkpoint(fn2, x, use_reentrant=False)

        def fn(x):
            return (
                fn1_checkpoint(x)
                # repeat the same fn1_checkpoint to see that we dedupe
                + fn1_checkpoint(x)
                # Check that a new fn2_checkpoint goes through a different HOP
                + fn2_checkpoint(x)
            )

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)

        x_clone = x.clone().detach().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 2)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 2)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 2)

        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone)
        self.assertEqual(ref, res)

    def test_fake_tensor_checking(self):
        @mark_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x, y):
            # x and y are different shapes, so we should use different graph
            return gn(x), gn(y)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)

        ref = fn(x, y)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_y_: "f32[16, 16]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', l_y_);  subgraph_1 = l_y_ = None
        getitem_1: "f32[16, 16]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem, getitem_1)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)

    class subgraph_1(torch.nn.Module):
        def forward(self, l_y_: "f32[16, 16]"):
            sin: "f32[16, 16]" = torch.sin(l_y_);  l_y_ = None
            return (sin,)
""",
            )

    def test_different_symint(self):
        """
        Tests check that the same subgraph called with different symints use different graphs
        """

        @mark_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            a = gn(x)
            # Get first half of the tensor
            b = torch.narrow(a, 0, 0, a.size()[0] // 2)
            return gn(b)

        opt_fn = torch.compile(fn, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)

        ref = fn(x)
        res = opt_fn(x)
        torch._dynamo.reset()

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)

        ref = fn(x)
        res = opt_fn(x)

        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', s77, l_x_);  subgraph_0 = l_x_ = None
        a: "f32[s77, 8]" = invoke_subgraph[0];  invoke_subgraph = None

        floordiv: "Sym((s77//2))" = s77 // 2
        b: "f32[(s77//2), 8]" = torch.narrow(a, 0, 0, floordiv);  a = floordiv = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', s77, b);  subgraph_1 = s77 = b = None
        getitem_3: "f32[(s77//2), 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_3,)

    class subgraph_0(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77, 8]"):
            sin: "f32[s77, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)

    class subgraph_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", b: "f32[(s77//2), 8]"):
            sin: "f32[(s77//2), 8]" = torch.sin(b);  b = None
            return (sin,)
""",
            )

    def test_autograd_function(self):
        class CustomOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                return x * torch.cos(grad_out)

        @mark_compile_region
        def gn(x):
            return CustomOp.apply(x)

        def fn(x):
            return gn(x) + gn(x)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = l_x_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            function_ctx = torch.autograd.function.FunctionCtx();  function_ctx = None
            fwd_body_0 = self.fwd_body_0
            bwd_body_0 = self.bwd_body_0
            autograd_function_apply: "f32[8, 8]" = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, args_tensor_mask = [True], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = l_x_ = None
            return (autograd_function_apply,)

        class fwd_body_0(torch.nn.Module):
            def forward(self, ctx : torch.autograd.function.Function, x: "f32[8, 8]"):
                _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

                sin: "f32[8, 8]" = torch.sin(x)

                _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
                return (sin, [x])

        class bwd_body_0(torch.nn.Module):
            def forward(self, ctx : torch.autograd.function.Function, grad_out: "f32[8, 8]", x: "f32[8, 8]"):
                _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

                cos: "f32[8, 8]" = torch.cos(grad_out);  grad_out = None
                mul: "f32[8, 8]" = x * cos;  x = cos = None

                _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
                return mul
""",
            )

    @requires_gpu
    def test_triton_kernel_native(self):
        from torch.testing._internal.triton_utils import add_kernel

        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            n_elements = output.numel()

            def grid_fn(meta):
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        @mark_compile_region
        def gn(x, y):
            o = torch.zeros_like(x)
            call_triton_add(x, y, o, 0)
            return o.sin()

        def fn(x, y):
            x = x.sin()
            y = y.sin()
            z = gn(x, y)
            return gn(z, y)

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)

        ref = fn(t1, t2)
        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        self.assertEqual(opt_fn(t1, t2), ref)

        # NOTE THAT THIS TEST DOES NOT REALLY WORK
        # We wanted one invoke_subgraph called twice, but because of
        # constant_args_idx changing in the grpah, the graph equivalence fails

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_y_: "f32[5]"):
        l_x_ = L_x_
        l_y_ = L_y_

        x: "f32[5]" = l_x_.sin();  l_x_ = None

        y: "f32[5]" = l_y_.sin();  l_y_ = None

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', x, y);  subgraph_0 = x = None
        z: "f32[5]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', z, y);  subgraph_1 = z = y = None
        getitem_1: "f32[5]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, x: "f32[5]", y: "f32[5]"):
            o: "f32[5]" = torch.zeros_like(x)

            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(5, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': x, 'in_ptr1': y, 'out_ptr': o});  x = y = triton_kernel_wrapper_mutation = None

            sin: "f32[5]" = o.sin();  o = None
            return (sin,)

    class subgraph_1(torch.nn.Module):
        def forward(self, z: "f32[5]", y: "f32[5]"):
            o: "f32[5]" = torch.zeros_like(z)

            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 1, grid = [(5, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': z, 'in_ptr1': y, 'out_ptr': o});  z = y = triton_kernel_wrapper_mutation = None

            sin: "f32[5]" = o.sin();  o = None
            return (sin,)
""",
            )

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unbacked_symbol(self):
        @mark_compile_region
        def gn(x):
            return torch.sin(torch.nonzero(x))

        def fn(x):
            return gn(x) + gn(x)

        x = torch.randn(64, 1, requires_grad=True)

        # Inductor fails with a lowering error
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_different_strides_in_backward(self):
        @mark_compile_region
        def gn(x):
            return torch.cos(x)

        def fn(x):
            a = gn(x)
            a2 = gn(a)
            b = torch.sin(a2)
            c = gn(b)
            c2 = gn(c)
            return c.sum() + c2.sum()

        opt_fn = torch.compile(fn, fullgraph=True)

        x = torch.randn(8, 16, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        x_clone = x.detach().clone().requires_grad_(True)
        torch._dynamo.mark_dynamic(x_clone, 0)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)

        torch.compiler.reset()
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 16, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        x_clone = x.detach().clone().requires_grad_(True)
        torch._dynamo.mark_dynamic(x_clone, 0)
        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", primals_2: "f32[s77, 16]"):
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_1

        invoke_subgraph_8 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_1', primals_1, primals_2);  partitioned_fw_subgraph_0_1 = primals_2 = None
        getitem_17: "Sym(s77)" = invoke_subgraph_8[2]
        getitem_16: "f32[s77, 16]" = invoke_subgraph_8[1]
        getitem: "f32[s77, 16]" = invoke_subgraph_8[0];  invoke_subgraph_8 = None

        partitioned_fw_subgraph_0_2 = self.partitioned_fw_subgraph_0_1

        invoke_subgraph_10 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_2, 'partitioned_fw_subgraph_0_1', primals_1, getitem);  partitioned_fw_subgraph_0_2 = getitem = None
        getitem_19: "Sym(s77)" = invoke_subgraph_10[2]
        getitem_18: "f32[s77, 16]" = invoke_subgraph_10[1]
        getitem_1: "f32[s77, 16]" = invoke_subgraph_10[0];  invoke_subgraph_10 = None

        sin: "f32[s77, 16]" = torch.ops.aten.sin.default(getitem_1)

        partitioned_fw_subgraph_0_3 = self.partitioned_fw_subgraph_0_1

        invoke_subgraph_12 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_3, 'partitioned_fw_subgraph_0_1', primals_1, sin);  partitioned_fw_subgraph_0_3 = sin = None
        getitem_21: "Sym(s77)" = invoke_subgraph_12[2]
        getitem_20: "f32[s77, 16]" = invoke_subgraph_12[1]
        getitem_2: "f32[s77, 16]" = invoke_subgraph_12[0];  invoke_subgraph_12 = None

        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_14 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, getitem_2);  partitioned_fw_subgraph_0_0 = None
        getitem_23: "Sym(s77)" = invoke_subgraph_14[2]
        getitem_22: "f32[s77, 16]" = invoke_subgraph_14[1]
        getitem_3: "f32[s77, 16]" = invoke_subgraph_14[0];  invoke_subgraph_14 = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_3);  getitem_3 = None
        add_15: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None

        cos: "f32[s77, 16]" = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None
        return (add_15, getitem_16, getitem_18, getitem_20, getitem_22, cos, primals_1, getitem_17, getitem_19, getitem_21, getitem_23)

    class partitioned_fw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]"):
            cos: "f32[s77, 16]" = torch.ops.aten.cos.default(primals_1)
            return (cos, primals_1, primals_0)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]"):
            cos: "f32[s77, 16]" = torch.ops.aten.cos.default(primals_1)
            return (cos, primals_1, primals_0)
""",
            )
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s77)", getitem_17: "Sym(s77)", getitem_19: "Sym(s77)", getitem_21: "Sym(s77)", getitem_23: "Sym(s77)", getitem_16: "f32[s77, 16]", getitem_18: "f32[s77, 16]", getitem_20: "f32[s77, 16]", getitem_22: "f32[s77, 16]", cos: "f32[s77, 16]", tangents_1: "f32[]"):
        expand: "f32[s77, 16]" = torch.ops.aten.expand.default(tangents_1, [primals_1, 16]);  tangents_1 = primals_1 = None

        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0

        invoke_subgraph_15 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_23, getitem_22, expand);  partitioned_bw_subgraph_0_0 = getitem_23 = getitem_22 = None
        getitem_5: "f32[s77, 16]" = invoke_subgraph_15[1];  invoke_subgraph_15 = None

        add_16: "f32[s77, 16]" = torch.ops.aten.add.Tensor(expand, getitem_5);  expand = getitem_5 = None

        partitioned_bw_subgraph_0_3 = self.partitioned_bw_subgraph_0_1

        invoke_subgraph_13 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_3, 'partitioned_bw_subgraph_0_1', getitem_21, getitem_20, add_16);  partitioned_bw_subgraph_0_3 = getitem_21 = getitem_20 = add_16 = None
        getitem_8: "f32[s77, 16]" = invoke_subgraph_13[1];  invoke_subgraph_13 = None

        mul_10: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(getitem_8, cos);  getitem_8 = cos = None

        partitioned_bw_subgraph_0_2 = self.partitioned_bw_subgraph_0_1

        invoke_subgraph_11 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_2, 'partitioned_bw_subgraph_0_1', getitem_19, getitem_18, mul_10);  partitioned_bw_subgraph_0_2 = getitem_19 = getitem_18 = mul_10 = None
        getitem_11: "f32[s77, 16]" = invoke_subgraph_11[1];  invoke_subgraph_11 = None

        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_1

        invoke_subgraph_9 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_1', getitem_17, getitem_16, getitem_11);  partitioned_bw_subgraph_0_1 = getitem_17 = getitem_16 = getitem_11 = None
        getitem_14: "f32[s77, 16]" = invoke_subgraph_9[1];  invoke_subgraph_9 = None
        return (None, getitem_14)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]", tangents_0: "f32[s77, 16]"):
            sin: "f32[s77, 16]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None
            neg: "f32[s77, 16]" = torch.ops.aten.neg.default(sin);  sin = None
            mul_9: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(tangents_0, neg);  tangents_0 = neg = None
            return (None, mul_9)

    class partitioned_bw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]", tangents_0: "f32[s77, 16]"):
            sin: "f32[s77, 16]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None
            neg: "f32[s77, 16]" = torch.ops.aten.neg.default(sin);  sin = None
            mul_10: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(tangents_0, neg);  tangents_0 = neg = None
            return (None, mul_10)
""",
            )

    @unittest.skip("Repro for an issue which is not fixed yet")
    def test_div(self):
        @mark_compile_region
        def gn(x):
            div = torch.div(1024, 256, rounding_mode="trunc")
            return div * torch.ones(64, div)

        def fn(x):
            return gn(x)

        x = torch.randn(64, 1, requires_grad=True)

        opt_fn = torch.compile(fn, fullgraph=True)

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
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', x, y);  repeated_subgraph0 = x = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', getitem, y);  repeated_subgraph0_1 = getitem = y = None
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
