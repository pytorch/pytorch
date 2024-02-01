"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# Owner(s): ["oncall: export"]
import math
import operator
import unittest
from typing import List, Set
from re import escape

import torch
from functorch.experimental.control_flow import cond
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.passes.functionalize_side_effectful_ops_pass import (
    _FunctionalizeSideEffectfulOpsPass,
)
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
    is_view_op,
    ReplaceViewOpsWithViewCopyOpsPass,
)
from torch.export import export, WrapperModule
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupport
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase, skipIfTorchDynamo
from torch.utils import _pytree as pytree


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
    args = pytree.tree_leaves(output_node.args)
    # if isinstance(args, tuple) and len(args) == 1:
    #     args = args[0]
    return [str(arg) for arg in args]


@skipIfTorchDynamo("recursively running dynamo on export is unlikely")
@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        ep = torch.export.export(M(), (x,), dynamic_shapes={"x": {1: dim1_x}})

        with self.assertRaisesRegex(RuntimeError, escape("Expected input at *args[0].shape[1] to be <= 6, but got 7")):
            ep.module()(torch.zeros(2, 7, 3))

        self.assertEqual(ep.module()(torch.ones(2, 4, 3)), M().forward(torch.ones(2, 4, 3)))

    def test_runtime_assert_multiple_dims(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x, dim0_y = torch.export.dims("dim0_x", "dim0_y", min=3)

        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": {0: dim0_y}}
        )

        with self.assertRaisesRegex(RuntimeError, escape("Expected input at *args[0].shape[1] to be <= 6, but got 7")):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(RuntimeError, escape("Expected input at *args[1].shape[0] to be >= 3, but got 2")):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

    def test_runtime_assert_some_dims_not_specified(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x = torch.export.Dim("dim0_x", min=3)

        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": None}
        )

        with self.assertRaisesRegex(RuntimeError, escape("Expected input at *args[0].shape[1] to be <= 6, but got 7")):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[1].shape[0] to be equal to 5, but got 2")
        ):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep.module()(torch.ones(3, 1, 3), torch.ones(5, 5, 5))
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

        dim1_y = torch.export.Dim("dim1_y", min=3, max=6)
        ep = torch.export.export(M(), (x, y), dynamic_shapes={"x": None, "y": {1: dim1_y}})

        with self.assertRaisesRegex(RuntimeError, escape("shape[1] to be equal to 2")):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(
            RuntimeError, escape("Expected input at *args[1].shape[0] to be equal to 5, but got 2")
        ):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep.module()(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))
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

        ep = ep._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 0)

    def test_functionalization_with_view_copy(self) -> None:
        def foo(x):
            y = x + 4
            y.add_(4)
            z = y.view(y.shape)
            return x.cos() + z.cos()

        x = torch.zeros(4, 2, 3)

        ep = export(WrapperModule(foo), (x,))._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
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
                torch._constrain_as_value(b, min=2, max=5)
                return b

        x = torch.tensor([2])
        mod = M()
        ep = export(mod, (x,))

        with self.assertRaisesRegex(RuntimeError, r"_local_scalar_dense is outside of inline constraint \[2, 5\]."):
            ep.module()(torch.tensor([6]))

        new_inp = torch.tensor([5])
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    def test_runtime_assert_inline_constraints_for_nonzero(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                b = x.nonzero()
                torch._constrain_as_value(b.shape[0], min=3, max=5)
                return b

        x = torch.tensor([2, 1, 2, 3, 5, 0])

        mod = M()
        dim0_x = torch.export.Dim("dim0_x")
        ep = torch.export.export(mod, (x,), dynamic_shapes={"x": {0: dim0_x}})

        num_assert = count_call_function(ep.graph, torch.ops.aten._assert_async.msg)
        num_scalar_tensor = count_call_function(
            ep.graph, torch.ops.aten.scalar_tensor.default
        )

        self.assertEqual(num_assert, 2)
        self.assertEqual(num_scalar_tensor, 2)

        with self.assertRaisesRegex(
            RuntimeError, r"nonzero.shape\[0\] is outside of inline constraint \[3, 5\]."
        ):
            ep.module()(torch.tensor([1, 1, 0, 0, 0]))

        with self.assertRaisesRegex(
            RuntimeError, r"nonzero.shape\[0\] is outside of inline constraint \[3, 5\]."
        ):
            ep.module()(torch.ones(6))

        new_inp = torch.tensor([1, 1, 1, 1])
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    def test_runtime_assert_inline_constraints_for_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    torch._constrain_as_value(b, min=2, max=5)
                    return x - b

                def false_fn(x, y):
                    c = y.item()
                    torch._constrain_as_value(c, min=2, max=5)
                    return y - c

                ret = cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        ep = export(mod, (torch.tensor(True), x, y))


        with self.assertRaisesRegex(RuntimeError, "is outside of inline constraint \\[2, 5\\]."):
            ep.module()(torch.tensor(False), torch.tensor([6]), torch.tensor([6]))

    def test_functionalize_inline_constraints(self) -> None:
        def f(x):
            a = x.item()
            torch._constrain_as_value(a, 4, 7)
            return torch.empty((a, 4))

        ep = torch._export.export(f, (torch.tensor([7]),))
        gm = ep.graph_module
        FileCheck().check_count(
            "torch.ops.aten.sym_constrain_range.default",
            1,
            exactly=True,
        ).run(gm.code)

        gm = _FunctionalizeSideEffectfulOpsPass()(ep.graph_module).graph_module

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

    def test_math_ops(self):
        def func(x):
            return (
                torch.tensor([math.ceil(x.item())]),
                torch.tensor([math.floor(x.item())]),
            )

        x = torch.randn(1, dtype=torch.float32)
        ep = torch.export.export(WrapperModule(func), args=(x,))
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)


if __name__ == '__main__':
    run_tests()
