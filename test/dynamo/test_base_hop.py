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
from torch._higher_order_ops.schema import find_hop_schema
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


def normalize_graph(gm):
    return normalize_gm(gm.print_readable(print_output=False))


class InvokeQuantTest(torch._higher_order_ops.BaseHOP):
    def __init__(self):
        super().__init__("invoke_quant_test")

    def __call__(self, subgraph, *operands, scheme):
        return super().__call__(subgraph, *operands, scheme=scheme)

    def gen_schema(self, subgraph, *operands, scheme):
        # Idea 1: using inspect.signature and sample inputs to generate a schema
        # Idea 2: we still need to know how to call into subgraph/fn given the inputs.
        #       wrap_subgraphs gives two callable to call into subgraph.
        from torch._higher_order_ops.schema import (
            CFunctionSchemaGen,
            HopArgumentInfoGen,
        )
        from torch._higher_order_ops.utils import (
            check_input_alias_and_mutation_return_ouputs,
        )

        (
            mutated_inp_idx,
            inp_inp_alias,
            inp_out_alias,
            out_out_alias,
            output,
        ) = check_input_alias_and_mutation_return_ouputs(subgraph, operands)
        assert (
            len(inp_inp_alias) == 0
            and len(inp_out_alias) == 0
            and len(out_out_alias) == 0
        ), f"Aliasing is not suppported for HOP subgraph. {subgraph}"

        args = [
            HopArgumentInfoGen.from_example(
                subgraph, name="subgraph", default_value=None, is_mutated=False
            )
        ]
        for idx, arg in enumerate(operands):
            example_value = arg
            arg_name = f"operands{idx}"
            args.append(
                HopArgumentInfoGen.from_example(
                    example_value=example_value,
                    name=arg_name,
                    default_value=None,
                    is_mutated=idx in mutated_inp_idx,
                )
            )

        args.append(
            HopArgumentInfoGen.from_example(
                example_value=scheme,
                name="scheme",
                default_value=scheme,
                is_mutated=False,
                kw_only=True,
            )
        )
        output = HopArgumentInfoGen.from_example(
            example_value=output,
            name="output",
            default_value=None,
            is_mutated=False,
            kw_only=False,
        )
        return CFunctionSchemaGen.from_hop_argument_info(str(self), args, output)


invoke_quant_test = InvokeQuantTest()


class BaseHOPTest(torch._dynamo.test_case.TestCase):
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
            return invoke_quant_test(inner, x, y, scheme="nf4")

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
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
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

    def test_schema_gen_single_return(self):
        def inner(x, y):
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=False)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        out = f(x.clone(), y)
        self.assertEqual(out, inner(x.clone(), y))
        schemas = find_hop_schema(backend.graphs[0], invoke_quant_test)
        self.assertEqual(len(schemas), 1)
        self.assertExpectedInline(
            str(schemas[0]),
            """invoke_quant_test(Any subgraph, Tensor operands0, Tensor operands1, *, str scheme="nf4") -> ((Tensor))""",  # noqa: B950
        )

    def test_schema_gen_pytree_in_out(self):
        def inner(x_y):
            x, y = x_y
            return [
                (x @ y).sin().cos(),
                (x + y, x - y),
                {"out": (x @ y,)},
            ]

        # make x not require grad because we want to inplace mutate it
        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend)
        def f(x, y):
            return invoke_quant_test(inner, [x, y], scheme="nf4")

        out = f(x.clone(), y)
        self.assertEqual(out, inner([x.clone(), y]))
        schemas = find_hop_schema(backend.graphs[0], invoke_quant_test)
        self.assertEqual(len(schemas), 1)
        self.assertExpectedInline(
            str(schemas[0]),
            """invoke_quant_test(Any subgraph, Tensor operands0, Tensor operands1, *, str scheme="nf4") -> (Tensor, Tensor, Tensor, Tensor)""",  # noqa: B950
        )

    def test_schema_gen_single_return_with_mutation(self):
        def inner(x, y):
            x.add_(1)
            y.mul_(-1)
            return (x @ y).sin().cos()

        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=False)

        backend = EagerAndRecordGraphs()

        def f(x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        torch.compile(f, backend=backend, fullgraph=True)(x.clone(), y)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            add_: "f32[3, 3]" = l_x_.add_(1);  add_ = None

            mul_: "f32[3, 3]" = l_y_.mul_(-1);  mul_ = None

            matmul: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            cos: "f32[3, 3]" = sin.cos();  sin = None
            return (cos,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(find_hop_schema(backend.graphs[0], invoke_quant_test)[0]),
            """invoke_quant_test(Any subgraph, Tensor(a1!) operands0, Tensor(a2!) operands1, *, str scheme="nf4") -> ((Tensor))""",
        )

    def test_schema_gen_pytree_in_out_with_mutation(self):
        def inner(x_y):
            x, y = x_y
            x.add_(1)
            return [
                (x @ y).sin().cos(),
                (x + y, x - y),
                {"out": (x @ y,)},
            ]

        # make x not require grad because we want to inplace mutate it
        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=True)

        backend = EagerAndRecordGraphs()

        def f(x, y):
            return invoke_quant_test(inner, [x, y], scheme="nf4")

        torch.compile(f, backend=backend, fullgraph=True)(x.clone(), y)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');  subgraph_0 = l_x_ = l_y_ = None
        getitem: "f32[3, 3]" = invoke_quant_test[0]
        getitem_1: "f32[3, 3]" = invoke_quant_test[1]
        getitem_2: "f32[3, 3]" = invoke_quant_test[2]
        getitem_3: "f32[3, 3]" = invoke_quant_test[3];  invoke_quant_test = None
        return (getitem, getitem_1, getitem_2, getitem_3)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 3]", l_y_: "f32[3, 3]"):
            add_: "f32[3, 3]" = l_x_.add_(1);  add_ = None

            matmul: "f32[3, 3]" = l_x_ @ l_y_
            sin: "f32[3, 3]" = matmul.sin();  matmul = None
            child: "f32[3, 3]" = sin.cos();  sin = None

            child_1: "f32[3, 3]" = l_x_ + l_y_
            child_2: "f32[3, 3]" = l_x_ - l_y_

            child_3: "f32[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
            return (child, child_1, child_2, child_3)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(find_hop_schema(backend.graphs[0], invoke_quant_test)[0]),
            """invoke_quant_test(Any subgraph, Tensor(a1!) operands0, Tensor operands1, *, str scheme="nf4") -> (Tensor, Tensor, Tensor, Tensor)""",  # noqa: B950
        )

    def test_none_input(self):
        def inner(x, y):
            if x is not None:
                return y.sin()
            return y.cos()

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = None
        y = torch.randn(3, 4)
        out = f(x, y)
        self.assertEqual(out, inner(x, y))
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[3, 4]"):
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_y_, scheme = 'nf4');  subgraph_0 = l_y_ = None
        getitem: "f32[3, 4]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_y_: "f32[3, 4]"):
            cos: "f32[3, 4]" = l_y_.cos();  l_y_ = None
            return (cos,)
""",
        )

    def test_int_input(self):
        def inner(x, y):
            return x + y

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def f(x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = 1
        y = torch.randn(3, 4)
        out = f(x, y)
        self.assertEqual(out, inner(x, y))
        self.assertExpectedInline(
            normalize_graph(backend.graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[3, 4]"):
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_y_, scheme = 'nf4');  subgraph_0 = l_y_ = None
        getitem: "f32[3, 4]" = invoke_quant_test[0];  invoke_quant_test = None
        return (getitem,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_y_: "f32[3, 4]"):
            add: "f32[3, 4]" = 1 + l_y_;  l_y_ = None
            return (add,)
""",
        )

    def test_auto_functionalize(self):
        def inner(x, y):
            x.add_(1)
            return x + y

        backend = AotEagerAndRecordGraphs()

        def f(x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        x = torch.randn(3, 3, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(3, 3, requires_grad=True)
        compiled_out = torch.compile(f, backend=backend, fullgraph=True)(x, y)
        # assert x is not mutated
        self.assertEqual(x, x_clone)
        self.assertEqual(compiled_out, x + y + 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertExpectedInline(
            normalize_graph(backend.fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]"):
        functiona_schema_0 = self.functiona_schema_0
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.invoke_quant_test, subgraph = auto_functionalized_subgraph_0, operands1 = primals_2, scheme = 'nf4', _operands0_base_index = 0, _all_bases = [primals_1], _op_schema = functiona_schema_0);  auto_functionalized_subgraph_0 = functiona_schema_0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None
        return (getitem, primals_1, primals_2)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            add_: "f32[3, 3]" = torch.ops.aten.add_.Tensor(arg0_1, 1);  arg0_1 = None

            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(add_, arg1_1);  add_ = arg1_1 = None
            return (add,)
""",  # noqa: B950
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
            return invoke_quant_test(inner, x, y, scheme="nf4")

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
        functiona_schema_0 = self.functiona_schema_0
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.invoke_quant_test, subgraph = auto_functionalized_subgraph_0, operands0 = primals_1, operands1 = primals_2, scheme = 'nf4', _all_bases = [], _op_schema = functiona_schema_0);  auto_functionalized_subgraph_0 = functiona_schema_0 = None
        getitem: "f32[3, 3]" = auto_functionalized_v2[0];  auto_functionalized_v2 = None
        return (getitem, primals_1, primals_2)

    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            sin_: "f32[3, 3]" = torch.ops.aten.sin_.default(mm);  mm = None
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(sin_);  sin_ = None
            return (cos,)
""",  # NOQA: B950
        )

        assert len(backend.bw_graphs) == 1
        self.assertExpectedInline(
            normalize_graph(backend.bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        functiona_schema_1 = self.functiona_schema_1
        auto_functionalized_subgraph_1 = self.auto_functionalized_subgraph_1
        auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.invoke_quant_test, subgraph = auto_functionalized_subgraph_1, operands0 = primals_1, operands1 = primals_2, operands2 = tangents_1, scheme = 'nf4', _all_bases = [], _op_schema = functiona_schema_1);  auto_functionalized_subgraph_1 = primals_1 = primals_2 = tangents_1 = functiona_schema_1 = None
        getitem_1: "f32[3, 3]" = auto_functionalized_v2_1[0]
        getitem_2: "f32[3, 3]" = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
        return (getitem_1, getitem_2)

    class auto_functionalized_subgraph_1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            mm: "f32[3, 3]" = torch.ops.aten.mm.default(arg0_1, arg1_1)
            clone: "f32[3, 3]" = torch.ops.aten.clone.default(mm)
            sin_: "f32[3, 3]" = torch.ops.aten.sin_.default(mm);  mm = None
            cos: "f32[3, 3]" = torch.ops.aten.cos.default(sin_);  cos = None
            sin: "f32[3, 3]" = torch.ops.aten.sin.default(sin_);  sin_ = None
            neg: "f32[3, 3]" = torch.ops.aten.neg.default(sin);  sin = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg2_1, neg);  arg2_1 = neg = None
            cos_1: "f32[3, 3]" = torch.ops.aten.cos.default(clone);  clone = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(mul, cos_1);  mul = cos_1 = None
            t: "f32[3, 3]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t, mul_1);  t = None
            t_1: "f32[3, 3]" = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
            mm_2: "f32[3, 3]" = torch.ops.aten.mm.default(mul_1, t_1);  mul_1 = t_1 = None
            return (mm_2, mm_1)
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
        x_clone = x.clone()
        y_clone = y.clone()

        @torch.compile(backend="eager", fullgraph=True)
        def f(inner, x, y):
            return invoke_quant_test(inner, x, y, scheme="nf4")

        compiled_f = torch.compile(f, backend="eager", fullgraph=True)

        with self.assertRaisesRegex(
            RuntimeError, "Encountered aliasing during higher order op tracing for HOP"
        ):
            compiled_f(inner, x, y)

        compiled_out = compiled_f(inner2, x, y)
        self.assertEqual(compiled_out, f(inner2, x_clone, y_clone))

    def test_eager_call(self):
        def inner(x, y):
            return x + y

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with self.assertRaisesRegex(RuntimeError, "torch.fx.GraphModule"):
            invoke_quant_test(inner, x, y, scheme="nf4")

        from functorch import make_fx

        result = make_fx(inner)(x, y)
        # smoke test
        invoke_quant_test(result, x, y, scheme="nf4")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
