"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing import FileCheck
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import export, dynamic_dim
from torch._export.constraints import constrain_as_value, constrain_as_size
from torch._export.exported_program import ExportGraphSignature
from torch._export.passes import (
    ReplaceViewOpsWithViewCopyOpsPass,
)
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    is_view_op,
    get_view_copy_of_view_op,
)
from torch._export.passes.functionalize_side_effectful_ops_pass import (
    _FunctionalizeSideEffectfulOpsPass,
)
from functorch.experimental.control_flow import cond


def count_call_function(graph: torch.fx.Graph, target: torch.ops.OpOverload) -> int:
    count = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == target:
            count += 1
    return count


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    def test_replace_broken_ops(self) -> None:
        x = torch.randn([2, 3, 4, 5])
        model: torch.nn.Linear = torch.nn.Linear(5, 5)

        def f(inp: torch.Tensor) -> torch.Tensor:
            return model(inp)

        ep = export(f, (x,)).transform(ReplaceViewOpsWithViewCopyOpsPass())

        count_after = 0
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                count_after += 1
        self.assertEqual(count_after, 0)
        self.assertTrue(torch.allclose(ep(x), f(x)))

    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        ep = export(M(), (x,), constraints=[dynamic_dim(x, 1) >= 2, dynamic_dim(x, 1) <= 6])

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 3)
        self.assertEqual(num_scalar_tensor, 3)

        with self.assertRaisesRegex(RuntimeError, "Input arg0_1"):
            ep(torch.zeros(2, 7, 3))

        self.assertEqual(ep(torch.ones(2, 4, 3)), M().forward(torch.ones(2, 4, 3)))

    def test_runtime_assert_multiple_dims(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(x, 1) >= 2,
            dynamic_dim(x, 1) <= 6,
            dynamic_dim(y, 0) >= 3,
            dynamic_dim(x, 0) >= 3
        ]

        ep = export(M(), (x, y), constraints=constraints)

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input arg0_1"):
            ep(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(RuntimeError, "Input arg1_1"):
            ep(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

    def test_runtime_assert_some_dims_not_specified(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(x, 1) >= 2,
            dynamic_dim(x, 1) <= 6,
            dynamic_dim(x, 0) >= 3
        ]

        ep = export(M(), (x, y), constraints=constraints)

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # there are 3 asserts from y and 2 from dynamic x dims and 1 from static x dim
        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input arg0_1"):
            ep(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, r"Input arg1_1.shape\[0\] is specialized at 5"):
            ep(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep(torch.ones(3, 1, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.ones(3, 1, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_runtime_assert_some_inps_not_used(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return y.cos().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        constraints = [
            dynamic_dim(y, 1) >= 3,
            dynamic_dim(y, 1) <= 6,
        ]

        ep = export(M(), (x, y), constraints=constraints)

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # there are 4 asserts from y and 3 from x
        self.assertEqual(num_assert, 7)
        self.assertEqual(num_scalar_tensor, 7)

        with self.assertRaisesRegex(RuntimeError, "Input arg0_1"):
            ep(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, r"Input arg1_1.shape\[0\] is specialized at 5"):
            ep(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_view_to_view_copy(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                z = x.view(x.shape)
                return z.cos().sum()

        x = torch.zeros(4, 2, 3)

        ep = export(M(), (x,))
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 1)

        ep = ep.transform(ReplaceViewOpsWithViewCopyOpsPass())
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 0)

    def test_functionalization_with_view_copy(self) -> None:
        def foo(x):
            y = x + 4
            y.add_(4)
            z = y.view(y.shape)
            return x.cos() + z.cos()

        x = torch.zeros(4, 2, 3)

        ep = export(foo, (x,)).transform(ReplaceViewOpsWithViewCopyOpsPass())
        # After this pass, there shouldn't be any view nodes in the graph
        self.assertTrue(count_call_function(ep.graph, torch.ops.aten.view.default) == 0)
        self.assertTrue(count_call_function(ep.graph, torch.ops.aten.view_copy.default) > 0)

    def test_views_op_having_view_copy(self) -> None:
        schemas = torch._C._dispatch_get_registrations_for_dispatch_key("")
        aten_schemas = [s[6:] for s in schemas if s.startswith("aten::")]

        for aten_schema in aten_schemas:
            val = aten_schema.split(".")
            assert len(val) <= 2
            name = ""
            overload = ""
            if len(val) == 1:
                name = val[0]
                overload = "default"
            else:
                name, overload = val[0], val[1]

            op_overload = getattr(getattr(torch.ops.aten, name), overload)
            if torch.Tag.core in op_overload.tags and is_view_op(op_overload._schema):
                self.assertIsNotNone(get_view_copy_of_view_op(op_overload._schema))

    def test_runtime_assert_inline_constraints_for_item(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                b = x.item()
                constrain_as_value(b, min=2, max=5)
                return b

        x = torch.tensor([2])
        mod = M()
        ep = export(mod, (x,))

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)
        # 1 constraint for shape of x, 2 constraints for b
        self.assertEqual(num_assert, 3)
        self.assertEqual(num_scalar_tensor, 3)

        with self.assertRaisesRegex(RuntimeError, r"_local_scalar_dense_default is outside of inline constraint \[2, 5\]."):
            ep(torch.tensor([6]))

        new_inp = torch.tensor([5])
        self.assertEqual(mod(new_inp), ep(new_inp))

    def test_runtime_assert_inline_constraints_for_nonzero(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                b = x.nonzero()
                constrain_as_value(b.shape[0], min=3, max=5)
                return b

        x = torch.tensor([2, 1, 2, 3, 5, 0])

        mod = M()
        ep = export(mod, (x,), constraints=[dynamic_dim(x, 0) >= 2])

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # TODO: De-duplicate assertions for same symbol.
        self.assertEqual(num_assert, 4)
        self.assertEqual(num_scalar_tensor, 4)

        with self.assertRaisesRegex(RuntimeError, r"nonzero_default.shape\[0\] is outside of inline constraint \[3, 5\]."):
            ep(torch.tensor([1, 1, 0, 0, 0]))

        with self.assertRaisesRegex(RuntimeError, r"nonzero_default.shape\[0\] is outside of inline constraint \[3, 5\]."):
            ep(torch.ones(6))

        new_inp = torch.tensor([1, 1, 1, 1])
        self.assertEqual(mod(new_inp), ep(new_inp))

    def test_runtime_assert_inline_constraints_for_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    constrain_as_value(b, min=2, max=5)
                    return x - b

                def false_fn(x, y):
                    c = y.item()
                    constrain_as_value(c, min=2, max=5)
                    return y - c

                ret = cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        ep = export(mod, (torch.tensor(True), x, y))
        with self.assertRaisesRegex(RuntimeError, "is outside of inline constraint \\[2, 5\\]."):
            ep(torch.tensor(False), torch.tensor([6]), torch.tensor([6]))

    def test_runtime_assert_equality_constraint(self):
        class Adder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        m = Adder()
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        exported = torch._export.export(
            m, (x, y), constraints=[dynamic_dim(x, 1) == dynamic_dim(y, 1)]
        )

        x = torch.rand(3, 5)
        y = torch.rand(3, 6)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Input arg0_1.shape\[1\] is not equal to input arg1_1.shape\[1\]"
        ):
            exported(x, y)

        y = torch.rand(3, 5)
        dynamo_result = exported(x, y)
        real_result = m(x, y)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))

    def test_functionalize_inline_contraints(self) -> None:
        def f(x):
            a = x.item()
            constrain_as_size(a, 4, 7)
            return torch.empty((a, 4))

        ep = torch._export.export(f, (torch.tensor([7]),))
        gm = ep.graph_module
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default",
            1,
            exactly=True,
        ).run(gm.code)

        gm = ep.transform(_FunctionalizeSideEffectfulOpsPass()).graph_module

        with self.assertRaisesRegex(
            RuntimeError,
            r"_local_scalar_dense_default is outside of inline constraint \[4, 7\]",
        ) as cm:
            gm(torch.tensor([20]))

        inp = torch.tensor([5])
        res, dep_token = gm(inp)
        self.assertEqual(res.shape, torch.Size([5, 4]))
        self.assertEqual(dep_token.shape, torch.Size([]))

        FileCheck().check_count(
            "torch.ops.aten._functional_sym_constrain_range", 1, exactly=True
        ).run(gm.code)
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default", 0, exactly=True
        ).run(gm.code)

        dep_token_node = next(n for n in gm.graph.nodes if n.name == "dep_token3")
        constrain_node = next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten._functional_sym_constrain_range
        )
        self.assertEqual(constrain_node.kwargs["dep_token"], dep_token_node)

    def test_functionalize_input_constraints(self) -> None:
        def f(x):
            return x * 2

        inp = torch.zeros(4, 8)
        ep = torch._export.export(
            f,
            (inp,),
            constraints=[
                dynamic_dim(inp, 0) < 10,
                dynamic_dim(inp, 0) >= 3,
            ],
        )
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 3, exactly=True
        ).run(ep.graph_module.code)

        gm = ep.transform(_FunctionalizeSideEffectfulOpsPass()).graph_module
        with self.assertRaisesRegex(
            RuntimeError,
            r"Input arg0_1.shape\[0\] is outside of specified dynamic range \[3, 9\]",
        ):
            gm(torch.ones(11, 8))

        inp = torch.ones(6, 8)
        self.assertEqual(gm(inp)[0], f(inp))
        FileCheck().check_count(
            "torch.ops.aten._functional_assert_async.msg", 3, exactly=True
        ).run(gm.code)
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 0, exactly=True
        ).run(gm.code)

    def test_functionalization(self) -> None:
        def f(x, y):
            a = x.item()
            constrain_as_size(a, 4, 7)
            return x + 4, x + y * 2

        inps = (torch.tensor([5]), torch.zeros((3, 4)))
        ep = torch._export.export(
            f,
            inps,
            constraints=[dynamic_dim(inps[1], 1) < 6],
            _functionalize_runtime_assertions=True,
        )
        FileCheck().check_count(
            "torch.ops.aten._functional_sym_constrain_range", 1, exactly=True
        ).run(ep.graph_module.code)
        inps = (torch.tensor([7]), torch.ones((3, 5)))
        self.assertTrue(torch._dynamo.utils.same(ep(*inps), f(*inps)))

    def test_functionalization_with_native_python_assertion(self) -> None:
        def f(x):
            b = x.sin()
            assert x[0] == 3
            return x.cos() + b

        inp = torch.Tensor([3, 4, 5])
        ep = torch._export.export(f, (inp,), _functionalize_runtime_assertions=True)

        # Check native assertion has corresponding functional assertion nodes generated.
        select_int_node = next(
            n
            for n in ep.graph_module.graph.nodes
            if n.target == torch.ops.aten.select.int
        )
        equal_scalar_node = select_int_node.next
        dep_token_node = next(
            n
            for n in ep.graph_module.graph.nodes
            if (
                n.target == torch.ops.aten._functional_assert_async.msg
                and n.args[0] == equal_scalar_node
            )
        )
        self.assertIn(
            "call_function[target=torch.ops.aten._functional_assert_async.msg]"
            "(args = (%eq_scalar, assertion error), kwargs = {dep_token: %dep_token1}",
            dep_token_node.format_node(),
        )

    def test_functionalization_with_mutated_buffer(self) -> None:
        buf = torch.ones(6, 2)
        weight = 0.01
        bias = 0.2
        d_in = 3
        d_out = 4

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buf", buf)

                self.linear = torch.nn.Linear(d_in, d_out)
                self.linear.weight.data.fill_(weight)
                self.linear.bias.data.fill_(bias)

            def forward(self, x):
                self.buf.add_(5)
                return self.linear(x).cos() + self.buf.sum()

        inp = torch.ones(4, 3)
        ep = torch._export.export(
            Foo(),
            (inp,),
            constraints=[dynamic_dim(inp, 0) >= 3],
            _functionalize_runtime_assertions=True,
        )

        gs = ep.graph_signature
        self.assertEqual(
            gs,
            ExportGraphSignature(
                parameters=["L__self___linear.weight", "L__self___linear.bias"],
                buffers=["L__self___buf"],
                user_inputs=["arg3_1"],
                user_outputs=["add_tensor_1"],
                inputs_to_parameters={
                    "arg0_1": "L__self___linear.weight",
                    "arg1_1": "L__self___linear.bias",
                },
                inputs_to_buffers={"arg2_1": "L__self___buf"},
                buffers_to_mutate={"add_tensor": "L__self___buf"},
                backward_signature=None,
                assertion_dep_token={2: "dep_token7"},
            ),
        )
        outputs = next(n for n in ep.graph.nodes if n.op == "output").args[0]
        self.assertEqual(
            [str(o) for o in outputs],
            ["add_tensor", "add_tensor_1", "dep_token7"],
        )
        self.assertEqual(
            len(outputs), len(gs.buffers_to_mutate) + len(gs.user_outputs) + 1,
        )
        inp = torch.randn(5, 3)
        self.assertTrue(
            torch._dynamo.utils.same(
                # Directly check run output of `ep.graph_module` which is
                # functionalized.
                ep.graph_module(
                    torch.full((d_out, d_in), weight),
                    torch.full((d_out,), bias),
                    buf.clone(),
                    inp,
                ),
                (buf.add(5), Foo()(inp), torch.empty(0)),
            )
        )
        self.assertTrue(torch._dynamo.utils.same(ep(inp), Foo()(inp)))


if __name__ == '__main__':
    run_tests()
