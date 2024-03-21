"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# Owner(s): ["oncall: export"]
import math
import operator
import unittest
from re import escape
from typing import List, Set

import torch
from functorch.experimental.control_flow import cond
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.passes.functionalize_side_effectful_ops_pass import (
    _FunctionalizeSideEffectfulOpsPass,
)
from torch._export.passes.replace_set_grad_with_hop_pass import (
    _is_set_grad_enabled_node,
    _is_set_grad_enabled_sub_mod,
)
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
    is_view_op,
    ReplaceViewOpsWithViewCopyOpsPass,
)
from torch._export.utils import (
    node_inline_,
    nodes_count,
    nodes_filter,
    nodes_map,
    sequential_split,
)
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch.export import export
from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupport
from torch.library import impl, _scoped_library
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
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

def _set_grad_enabled_tests():
    from torch.export._trace import _export

    class SetGradOp(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            torch._C._set_grad_enabled(True)
            c = x.sin().sum()
            torch._C._set_grad_enabled(False)
            d = c + 1
            torch._C._set_grad_enabled(True)
            e = d - 1
            return d, e

    class SetGradCtxManager(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch.enable_grad():
                c = x.sin().sum()
            with torch.no_grad():
                d = c + 1
            with torch.enable_grad():
                e = d - 1
            return d, e

    class SetGradCtxManagerMultiDep(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch.enable_grad():
                c1 = x.sin().sum()
                c2 = x.cos().sum()
            with torch.no_grad():
                d1 = c1 + 1
                d2 = c2 + 1
            with torch.enable_grad():
                e1 = d1 - 1
                e2 = d2 - 1
            return d1, d2, e1, e2

    x = torch.randn(2, 2)

    def _get_predispatch_module(mod, args, ambient_grad_enabled=True):
        with torch.set_grad_enabled(ambient_grad_enabled):
            return _export(mod, args, pre_dispatch=True).module()

    return {
        "ctx_manager" : (_get_predispatch_module(SetGradCtxManager(), (x,)), (x,)),
        "ctx_manager_under_no_grad" : (_get_predispatch_module(SetGradCtxManager(), (x,), False), (x,)),
        "ctx_manager_multi_dep" : (_get_predispatch_module(SetGradCtxManagerMultiDep(), (x,)), (x,)),
        "ctx_manager_multi_dep_no_grad" : (_get_predispatch_module(SetGradCtxManagerMultiDep(), (x,), False), (x,)),
        "op" : (_get_predispatch_module(SetGradOp(), (x,)), (x,)),
        "op_under_no_grad" : (_get_predispatch_module(SetGradOp(), (x,), False), (x,))
    }


def _sequential_split_inline_tests():
    from torch.export._trace import _export

    class Simple(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            c = x.sin().sum()
            d = c + 1
            e = d - 1
            return d, e

    class MultiDep(torch.nn.Module):
        def forward(self, x1, x2):
            x1 = x1 + 1
            x2 = x2 + 1
            c1 = x1.sin()
            c2 = x2.cos()
            d1 = c1 + 1
            d2 = c2 + 1
            e1 = d1 - 1
            e2 = d2 - 1
            return d1, d2, e1, e2

    def _get_predispatch_module(mod, args):
        return _export(mod, args, pre_dispatch=True).module()

    def _insert_dilimiter_nodes(gm: torch.fx.GraphModule, step: int = 1):
        insert_locs = []
        for i, node in enumerate(nodes_filter(gm.graph.nodes, lambda n: n.op == "call_function")):
            if i % step == 0:
                insert_locs.append(node)

        for i, node in enumerate(insert_locs):
            with gm.graph.inserting_before(node):
                gm.graph.call_function(torch._C._set_grad_enabled, (True if i % 2 == 0 else False,), {})
        return gm

    x = torch.randn(2, 2)
    simple = _get_predispatch_module(Simple(), (x,))
    simple1 = _get_predispatch_module(Simple(), (x,))
    multi_dep = _get_predispatch_module(MultiDep(), (x, x.sin()))
    multi_dep1 = _get_predispatch_module(MultiDep(), (x, x.sin()))
    return {
        'simple_step1': (_insert_dilimiter_nodes(simple1, 1), (x,)),
        'simple_step2': (_insert_dilimiter_nodes(simple, 2), (x,)),
        'multi_dep_step2': (_insert_dilimiter_nodes(multi_dep, 2), (x, x.sin())),
        'multi_dep_step3': (_insert_dilimiter_nodes(multi_dep1, 3), (x, x.sin())),
    }


@skipIfTorchDynamo("recursively running dynamo on export is unlikely")
@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    def setUp(self):
        super().setUp()
        self.SEQUENTIAL_SPLIT_INLINE_TESTS = _sequential_split_inline_tests()
        self.SET_GRAD_ENABLED_TESTS = _set_grad_enabled_tests()

    def tearDown(self):
        self.SEQUENTIAL_SPLIT_INLINE_TESTS.clear()
        self.SET_GRAD_ENABLED_TESTS.clear()
        super().tearDown()

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
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x + 4
                y.add_(4)
                z = y.view(y.shape)
                return x.cos() + z.cos()

        x = torch.zeros(4, 2, 3)
        foo = Module()
        ep = export(foo, (x,))._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
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

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
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
        class Foo(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._constrain_as_value(a, 4, 7)
                return torch.empty((a, 4))

        f = Foo()

        ep = torch.export.export(f, (torch.tensor([7]),))
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
        class Module(torch.nn.Module):
            def forward(self, x):
                return (
                    torch.tensor([math.ceil(x.item())]),
                    torch.tensor([math.floor(x.item())]),
                )

        func = Module()
        x = torch.randn(1, dtype=torch.float32)
        ep = torch.export.export(func, args=(x,))
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)

    def test_predispatceh_set_grad(self):
        mod, args = self.SET_GRAD_ENABLED_TESTS["op"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_4 = self.submod_2
    add_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_4, sum_1);  submod_4 = sum_1 = None
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    """)
        mod, args = self.SET_GRAD_ENABLED_TESTS["op_under_no_grad"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_4 = self.submod_2
    add_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_4, sum_1);  submod_4 = sum_1 = None
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    """)

        mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_3 = self.submod_1
    add_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_3, sum_1);  submod_3 = sum_1 = None
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    """)
        mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager_under_no_grad"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    submod_5 = self.submod_1
    sum_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_5, add);  submod_5 = add = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    submod_6 = self.submod_3
    sub = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_6, add_1);  submod_6 = None
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    """)
        mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager_multi_dep"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    sin = torch.ops.aten.sin.default(add)
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    cos = torch.ops.aten.cos.default(add);  add = None
    sum_2 = torch.ops.aten.sum.default(cos);  cos = None
    submod_3 = self.submod_1
    wrap_with_set_grad_enabled = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_3, sum_1, sum_2);  submod_3 = sum_1 = sum_2 = None
    add_1 = wrap_with_set_grad_enabled[0]
    add_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    sub_1 = torch.ops.aten.sub.Tensor(add_2, 1)
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    """)  # noqa: B950
        mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager_multi_dep_no_grad"]
        self.assertExpectedInline(mod.code.strip("\n"), """\
def forward(self, arg_0):
    arg0_1, = fx_pytree.tree_flatten_spec(([arg_0], {}), self._in_spec)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    submod_5 = self.submod_1
    wrap_with_set_grad_enabled = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_5, add);  submod_5 = add = None
    sum_1 = wrap_with_set_grad_enabled[0]
    sum_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    add_2 = torch.ops.aten.add.Tensor(sum_2, 1);  sum_2 = None
    submod_6 = self.submod_3
    wrap_with_set_grad_enabled_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_6, add_1, add_2);  submod_6 = None
    sub = wrap_with_set_grad_enabled_1[0]
    sub_1 = wrap_with_set_grad_enabled_1[1];  wrap_with_set_grad_enabled_1 = None
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    """)  # noqa: B950

    def test_sequential_split(self):
        for gm, args in self.SEQUENTIAL_SPLIT_INLINE_TESTS.values():
            set_grad_counts = nodes_count(gm.graph.nodes, _is_set_grad_enabled_node)
            new_gm = sequential_split(gm, _is_set_grad_enabled_node)
            new_set_grad_counts = nodes_count(new_gm.graph.nodes, _is_set_grad_enabled_sub_mod)
            self.assertEqual(set_grad_counts, new_set_grad_counts)
            self.assertEqual(gm(*args), new_gm(*args))

    def test_sequential_split_graph(self):
        gm, args = self.SEQUENTIAL_SPLIT_INLINE_TESTS["multi_dep_step2"]

        new_gm = sequential_split(gm, _is_set_grad_enabled_node)
        self.assertEqual(gm(*args), new_gm(*args))
        self.assertExpectedInline(new_gm.code.strip("\n"), """\
def forward(self, arg_0, arg_1):
    arg0_1, arg1_1, = fx_pytree.tree_flatten_spec(([arg_0, arg_1], {}), self._in_spec)
    submod_1 = self.submod_1(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    getitem = submod_1[0]
    getitem_1 = submod_1[1];  submod_1 = None
    submod_2 = self.submod_2(getitem, getitem_1);  getitem = getitem_1 = None
    getitem_2 = submod_2[0]
    getitem_3 = submod_2[1];  submod_2 = None
    submod_3 = self.submod_3(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    getitem_4 = submod_3[0]
    getitem_5 = submod_3[1];  submod_3 = None
    submod_4 = self.submod_4(getitem_4, getitem_5)
    getitem_6 = submod_4[0]
    getitem_7 = submod_4[1];  submod_4 = None
    return pytree.tree_unflatten((getitem_4, getitem_5, getitem_6, getitem_7), self._out_spec)
    """)
        self.assertExpectedInline(new_gm.submod_1.code.strip("\n"), """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True)
    add = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None
    add_1 = torch.ops.aten.add.Tensor(arg1_1, 1);  arg1_1 = None
    return (add, add_1)
    """)
        self.assertExpectedInline(new_gm.submod_2.code.strip("\n"), """\
def forward(self, add, add_1):
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
    sin = torch.ops.aten.sin.default(add);  add = None
    cos = torch.ops.aten.cos.default(add_1);  add_1 = None
    return (sin, cos)
    """)
        self.assertExpectedInline(new_gm.submod_3.code.strip("\n"), """\
def forward(self, sin, cos):
    _set_grad_enabled_2 = torch._C._set_grad_enabled(True)
    add_2 = torch.ops.aten.add.Tensor(sin, 1);  sin = None
    add_3 = torch.ops.aten.add.Tensor(cos, 1);  cos = None
    return (add_2, add_3)
    """)

    def test_inline_(self):
        for gm, args in self.SEQUENTIAL_SPLIT_INLINE_TESTS.values():
            before_str = gm.print_readable(print_output=False)
            new_gm = sequential_split(gm, _is_set_grad_enabled_node)
            nodes_map(new_gm.graph.nodes, lambda node: node_inline_(node) if node.op == "call_module" else node)
            after_inline_str = new_gm.print_readable(print_output=False)
            self.assertEqual(before_str, after_inline_str)
            self.assertEqual(gm(*args), new_gm(*args))

    def test_remove_auto_functionalized_pass(self) -> None:
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:

            lib.define("custom_mutator(Tensor x, Tensor(a!) y) -> Tensor")

            @impl(lib, "custom_mutator", "Meta")
            def custom_mutator_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return torch.empty_like(x)


            @impl(lib, "custom_mutator", "CompositeExplicitAutograd")
            def custom_mutator(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return x + y.add_(1)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("state", torch.zeros(1))

                def forward(self, x):
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator(x, self.state)

            mod = M()
            x = torch.randn([3, 3])
            ep = export(mod, (x,))
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)
            nodes = inplace_ep.graph.nodes
            for node in nodes:
                if node.op == "call_function":
                    self.assertFalse(node.target is auto_functionalized)
                    self.assertFalse(node.target is operator.getitem)

            for spec in inplace_ep.graph_signature.output_specs:
                self.assertFalse("getitem" in spec.arg.name)

    def test_remove_auto_functionalized_pass_tuple(self) -> None:
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:

            lib.define("custom_mutator_tuple(Tensor x, Tensor(a!) y) -> (Tensor, Tensor)")

            @impl(lib, "custom_mutator_tuple", "Meta")
            def custom_mutator_tuple_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                return (torch.empty_like(x), torch.empty_like(x))


            @impl(lib, "custom_mutator_tuple", "CompositeExplicitAutograd")
            def custom_mutator_tuple(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                return (x, x + y.add_(1))

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("state", torch.zeros(1))

                def forward(self, x):
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator_tuple(
                        x, self.state
                    )

            mod = M()
            x = torch.randn([3, 3])
            ep = export(mod, (x,))
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)

            nodes = inplace_ep.graph.nodes
            getitems = 0
            for node in nodes:
                if node.op == "call_function":
                    self.assertFalse(node.target is auto_functionalized)
                    if node.target is operator.getitem:
                        getitems += 1
            self.assertEqual(getitems, 2)  # tuple return of len 2

            out_specs = inplace_ep.graph_signature.output_specs
            self.assertEqual(out_specs[0].arg.name, "arg0_1")  # state
            self.assertEqual(out_specs[1].arg.name, "getitem")  # tuple return 1
            self.assertEqual(out_specs[2].arg.name, "getitem_1")  # tuple return 2


if __name__ == '__main__':
    run_tests()
