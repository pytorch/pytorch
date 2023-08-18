"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# Owner(s): ["module: dynamo"]
import unittest
from typing import List, Set
import operator

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing import FileCheck
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import export, dynamic_dim
from torch._export.constraints import constrain_as_value
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
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.infra.partitioner import Partition
from torch.utils._pytree import tree_flatten


def count_call_function(graph: torch.fx.Graph, target: torch.ops.OpOverload) -> int:
    count = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == target:
            count += 1
    return count


class _AddOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in {operator.add}


class _AtenAddOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return node.op == "call_function" and node.target in {
            torch.ops.aten.add.Tensor
        }


def _to_partition_names(partitions: List[Partition]) -> List[Set[str]]:
    return [{n.name for n in p.nodes} for p in partitions]


def _get_output_names(gm: torch.fx.GraphModule) -> List[str]:
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    args = tree_flatten(output_node.args)[0]
    # if isinstance(args, tuple) and len(args) == 1:
    #     args = args[0]
    return [str(arg) for arg in args]


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
        self.assertTrue(torch.allclose(ep(x), f(x), atol=1e-3, rtol=0.01))

    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        ep = export(M(), (x,), constraints=[dynamic_dim(x, 1) >= 2, dynamic_dim(x, 1) <= 6])

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

        with self.assertRaisesRegex(RuntimeError, r"_local_scalar_dense is outside of inline constraint \[2, 5\]."):
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

        with self.assertRaisesRegex(RuntimeError, r"nonzero.shape\[0\] is outside of inline constraint \[3, 5\]."):
            ep(torch.tensor([1, 1, 0, 0, 0]))

        with self.assertRaisesRegex(RuntimeError, r"nonzero.shape\[0\] is outside of inline constraint \[3, 5\]."):
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
            constrain_as_value(a, 4, 7)
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
            r"_local_scalar_dense is outside of inline constraint \[4, 7\]",
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


if __name__ == '__main__':
    run_tests()
