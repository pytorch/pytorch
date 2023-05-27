# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import export, dynamic_dim, _export
from torch._export.constraints import constrain_as_value
from torch._export.passes import (
    ConstPropPass,
    ReplaceViewOpsWithViewCopyOpsPass,
)
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    is_view_op,
    get_view_copy_of_view_op,
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

        gm = export(f, (x,)).find_method("forward")

        new_gm = ReplaceViewOpsWithViewCopyOpsPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module

        count_after = 0
        for node in new_gm.graph.nodes:
            if node.target == torch.ops.aten.view.default:
                count_after += 1
        self.assertEqual(count_after, 0)
        self.assertTrue(torch.allclose(gm(x), f(x)))

    def test_const_prop_pass(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.ones(1, 2, 3))

            def forward(self, x):
                b = self.a + self.a
                c = torch.cat([self.a, b])
                return (c + c) + x

        def count_additions(gm) -> int:
            return sum(
                (node.target == torch.ops.aten.add.Tensor) for node in gm.graph.nodes
            )

        gm = export(M(), (torch.zeros(2, 2, 3),)).find_method("forward")
        self.assertEqual(count_additions(gm), 3)

        new_gm = ConstPropPass()(gm)
        self.assertIsNotNone(new_gm)
        new_gm = new_gm.graph_module
        self.assertEqual(count_additions(new_gm), 1)

    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        ep = export(M(), (x,), constraints=[dynamic_dim(x, 1) >= 2, dynamic_dim(x, 1) <= 6]).add_runtime_assertions()

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 3)
        self.assertEqual(num_scalar_tensor, 3)

        with self.assertRaisesRegex(RuntimeError, "Input arg0"):
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

        gm = export(M(), (x, y), constraints=constraints).add_runtime_assertions().find_method("forward")

        num_assert = count_call_function(gm.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(gm.graph, torch.ops.aten.scalar_tensor.default)

        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input arg0"):
            gm(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(RuntimeError, "Input arg1"):
            gm(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

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

        ep = export(M(), (x, y), constraints=constraints).add_runtime_assertions()

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # there are 3 asserts from y and 2 from dynamic x dims and 1 from static x dim
        self.assertEqual(num_assert, 6)
        self.assertEqual(num_scalar_tensor, 6)

        with self.assertRaisesRegex(RuntimeError, "Input arg0"):
            ep(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, "Input arg1's dimension #0 size is specialized at 5"):
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

        ep = export(M(), (x, y), constraints=constraints).add_runtime_assertions()

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # there are 4 asserts from y and 3 from x
        self.assertEqual(num_assert, 7)
        self.assertEqual(num_scalar_tensor, 7)

        with self.assertRaisesRegex(RuntimeError, "Input arg0"):
            ep(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(RuntimeError, "Input arg1's dimension #0 size is specialized at 5"):
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

        gm = _export(M(), (x,))
        self.assertEqual(count_call_function(gm.graph, torch.ops.aten.view.default), 1)

        pass_result = ReplaceViewOpsWithViewCopyOpsPass()(gm)
        self.assertTrue(pass_result.modified)
        self.assertEqual(count_call_function(pass_result.graph_module.graph, torch.ops.aten.view.default), 0)

    def test_functionalization_with_view_copy(self) -> None:
        def foo(x):
            x.add_(4)
            y = x.view(x.shape)
            return x.cos() + y.cos()

        x = torch.zeros(4, 2, 3)

        ep = export(foo, (x,)).transform(ReplaceViewOpsWithViewCopyOpsPass())
        # After this pass, there shouldn't be any view nodes in the graph
        self.assertTrue(count_call_function(ep.module.graph, torch.ops.aten.view.default) == 0)
        self.assertTrue(count_call_function(ep.module.graph, torch.ops.aten.view_copy.default) > 0)

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
        ep = export(mod, (x,)).add_runtime_assertions()

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
        ep = export(mod, (x,), constraints=[dynamic_dim(x, 0) >= 2]).add_runtime_assertions()

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(ep.graph, torch.ops.aten.scalar_tensor.default)

        # 2 constraints for b
        self.assertEqual(num_assert, 2)
        self.assertEqual(num_scalar_tensor, 2)

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
        ep = export(mod, (torch.tensor(True), x, y)).add_runtime_assertions()
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
        exported = exported.add_runtime_assertions()

        x = torch.rand(3, 5)
        y = torch.rand(3, 6)
        with self.assertRaisesRegex(
            RuntimeError,
            "Input arg1's dimension #1 size is not equal to input arg0's dimension #1",
        ):
            exported(x, y)

        y = torch.rand(3, 5)
        dynamo_result = exported(x, y)
        real_result = m(x, y)
        self.assertTrue(torch._dynamo.utils.same(real_result, dynamo_result))


if __name__ == '__main__':
    run_tests()
