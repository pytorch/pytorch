# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


def normalize_graph(gm):
    return normalize_gm(gm.print_readable(print_output=False))


class InvokeQuantTest(torch._higher_order_ops.PrimHOPBase):
    def __init__(self):
        super().__init__("invoke_quant_test")

    def __call__(self, subgraph, operands, *, scheme):
        return super().__call__(subgraph, operands, scheme=scheme)


invoke_quant_test = InvokeQuantTest()


class PrimHOPBaseTest(torch._dynamo.test_case.TestCase):
    # TODO: flip to False later, we're landing a refactor PR and don't want to merge conflict
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_dynamo(self):
        def inner(x, y):
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            return invoke_quant_test(inner, (x, y), scheme="nf4")

        out = f(x, y)
        self.assertEqual(out, inner(x, y))

        assert len(backend.graphs) == 1
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, (l_x_, l_y_), scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            matmul: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            cos: "f32[3, 3]" = sin.cos();  sin = None
            return (cos,)
""",  # NOQA: B950
        )

    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_aot_eager(self):
        def inner(x, y):
            return (x @ y).sin_().cos()

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)

        backend = AotEagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            return invoke_quant_test(inner, (x, y), scheme="nf4")

        out = f(x, y)
        result = torch.autograd.grad(out, x, y)
        out = inner(x, y)
        expected = torch.autograd.grad(out, x, y)
        self.assertEqual(result, expected)

        assert len(backend.fw_graphs) == 1
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]"):
        subgraph0 = self.subgraph0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph0, (primals_1, primals_2), scheme = 'nf4');  subgraph0 = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem, primals_1, primals_2)

    class subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(mm);  mm = None
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(sin);  sin = None
            return (cos,)
""",  # NOQA: B950
        )

        assert len(backend.bw_graphs) == 1
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        subgraph1 = self.subgraph1
        invoke_quant_test_1 = torch.ops.higher_order.invoke_quant_test(subgraph1, (primals_1, primals_2, tangents_1), scheme = 'nf4');  subgraph1 = primals_1 = primals_2 = tangents_1 = None
        getitem_1: "f32[3, 3]" = invoke_quant_test_1[0]
        getitem_2: "f32[3, 3]" = invoke_quant_test_1[1];  invoke_quant_test_1 = None
        return (getitem_1, getitem_2)

    class subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
            clone: "f32[3, 3]" = torch.ops.aten.clone.default(mm)
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(mm);  mm = None
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(sin);  cos = None
            sin_1: "f32[3, 3]" = torch.ops.aten.sin.default(sin);  sin = None
            neg: "f32[3, 3]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg2_1, neg);  arg2_1 = neg = None
            cos_1: "f32[3, 3]" = torch.ops.aten.cos.default(clone);  clone = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(mul, cos_1);  mul = cos_1 = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t, mul_1);  t = None
            t_1: "f32[3, 3]" = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
            mm_2: "f32[3, 3]" = torch.ops.aten.mm.default(mul_1, t_1);  mul_1 = t_1 = None
            return [mm_2, mm_1]
""",  # NOQA: B950
        )

    def test_aliasing_mutation_error(self):
        def inner(x, y):
            return x

        def inner2(x, y):
            x.sin_()
            return x + y

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(inner, x, y):
            return invoke_quant_test(inner, (x, y), scheme="nf4")

        with self.assertRaisesRegex(RuntimeError, "aliases of the inputs"):
            f(inner, x, y)

        with self.assertRaisesRegex(RuntimeError, "inputs are mutated"):
            f(inner2, x, y)

    def test_eager_call(self):
        def inner(x, y):
            return x + y

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with self.assertRaisesRegex(RuntimeError, "torch.fx.GraphModule"):
            invoke_quant_test(inner, (x, y), scheme="nf4")

        from functorch import make_fx

        result = make_fx(inner)(x, y)
        # smoke test
        invoke_quant_test(result, (x, y), scheme="nf4")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
