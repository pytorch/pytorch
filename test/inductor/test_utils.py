# Owner(s): ["module: inductor"]

import unittest
from collections.abc import Callable
from copy import deepcopy

from sympy import Symbol, sympify

import torch
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.compile_fx import _get_subgraph_names
from torch._inductor.fx_utils import (
    count_flops_fx,
    countable_fx,
    FakeTensorUpdater,
    get_fake,
)
from torch._inductor.utils import get_device_tflops, sympy_str, sympy_subs
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestUtils(TestCase):
    def test_zip_schema(self):
        def foo(x: torch.Tensor) -> None:
            pass

        result = torch.library.custom_op("mylib::foo", foo, mutates_args={"x"})
        schema = result._opoverload._schema
        g = torch.tensor([11, 2])
        found = False
        for arg, val in torch._library.utils.zip_schema(schema, [], {"x": g}):
            if arg.name == "x":
                found = True

        self.assertTrue(found)

        found = False
        for arg, val in torch._library.utils.zip_schema(schema, [g], {}):
            if arg.name == "x":
                found = True
        self.assertTrue(found)

    def testSympySubs(self):
        # integer and nonnegetaive attributes are preserved.
        expr = Symbol("x")
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        expr = Symbol("x", integer=True, nonnegative=False)
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)

        # invalid replacement.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x"): Symbol("y")})
        self.assertEqual(result.name, "x")

        # valid replacement since properties match.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x", integer=True): Symbol("y")})
        self.assertEqual(result.name, "y")

        # invalid replacement.
        expr = Symbol("x", integer=None)
        result = sympy_subs(expr, {Symbol("x", integer=False): Symbol("y")})
        self.assertEqual(result.name, "x")

        # replaced can't be string
        self.assertRaises(AssertionError, sympy_subs, expr, {"x": "y"})

        # replaced can be an expression
        expr = Symbol("x")
        expr = abs(expr)
        self.assertEqual(expr.is_integer, None)
        self.assertEqual(expr.is_nonnegative, None)
        # replace abs(x) with y
        # propagate abs(x) sympy properties.
        result = sympy_subs(expr, {expr: Symbol("y")})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

    def test_sympy_str(self):
        self.assertEqual(sympy_str(sympify("a+b+c")), "a + b + c")
        self.assertEqual(sympy_str(sympify("a*b+c")), "c + a * b")
        self.assertEqual(sympy_str(sympify("a+b*(c+d)")), "a + b * (c + d)")
        self.assertEqual(sympy_str(sympify("(a+b)*(c+d)")), "(a + b) * (c + d)")
        self.assertEqual(sympy_str(sympify("-a")), "-a")
        self.assertEqual(sympy_str(sympify("a-b")), "a - b")
        self.assertEqual(sympy_str(sympify("a+-b")), "a - b")

    def test_flops_fx(self):
        def create_fx_node(
            aten, op_overload: torch._ops.OpOverload, args, kwargs
        ) -> tuple[torch.fx.Node, torch.fx.Node]:
            node1 = torch.fx.Node(
                graph=torch.fx.Graph(),
                name="",
                op="call_function",
                target=aten,
                args=args,
                kwargs=kwargs,
            )
            # name: str = aten.overloads()[0]
            # if aten == torch.ops.aten.addmm:
            #     name = "default"
            # print(aten)
            # print(aten.overloads())
            # print(name)
            # op_overload: torch._ops.OpOverload = getattr(aten, name)
            node2 = torch.fx.Node(
                graph=torch.fx.Graph(),
                name="",
                op="call_function",
                target=op_overload,
                args=args,
                kwargs=kwargs,
            )
            return node1, node2

        with V.set_fake_mode(
            torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        ):
            trues = [
                (
                    torch.ops.aten.addmm,
                    torch.ops.aten.addmm.default,
                    (torch.Tensor(4, 4), torch.Tensor(4, 5), torch.Tensor(5, 4)),
                    {},
                ),
                (
                    torch.ops.aten.bmm,
                    torch.ops.aten.bmm.default,
                    (torch.Tensor(10, 4, 5), torch.Tensor(10, 5, 4)),
                    {},
                ),
                (
                    torch.ops.aten.mm,
                    torch.ops.aten.mm.default,
                    (torch.Tensor(2, 3), torch.Tensor(3, 2)),
                    {},
                ),
                (
                    torch.ops.aten.convolution,
                    torch.ops.aten.convolution.default,
                    (
                        torch.Tensor(2, 2, 3),
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                    ),
                    {},
                ),
                (
                    torch.ops.aten._convolution,
                    torch.ops.aten._convolution.deprecated,
                    (
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                        False,
                        True,
                        False,
                    ),
                    {},
                ),
            ]
            # we don't support pointwise ops
            falses = [
                (
                    torch.ops.aten.add,
                    torch.ops.aten.add.Tensor,
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
                (
                    torch.ops.aten.mul,
                    torch.ops.aten.mul.Tensor,
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
            ]
            for t, t2, args, kwargs in trues:
                fx_node_1, fx_node_2 = create_fx_node(t, t2, args, kwargs)
                self.assertTrue(
                    countable_fx(fx_node_1), f"Expected true {t}: {fx_node_1}"
                )
                self.assertTrue(
                    countable_fx(fx_node_2), f"Expected true {t}: {fx_node_2}"
                )
                self.assertNotEqual(count_flops_fx(fx_node_1), None)
                self.assertNotEqual(count_flops_fx(fx_node_2), None)
            for f, f2, args, kwargs in falses:
                fx_node_1, fx_node_2 = create_fx_node(f, f2, args, kwargs)
                self.assertFalse(
                    countable_fx(fx_node_1), f"Expected false {f}: {fx_node_1}"
                )
                self.assertFalse(
                    countable_fx(fx_node_2), f"Expected false {f}: {fx_node_2}"
                )

    @unittest.skipIf(not torch.cuda.is_available(), "skip if no device")
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_get_device_tflops(self, dtype):
        ret = get_device_tflops(dtype)
        self.assertTrue(type(ret) is float)


instantiate_device_type_tests(TestUtils, globals(), allow_xpu=True)


class TestFakeTensorUpdater(TestCase):
    def _insert_clone(self, main_graph: torch.fx.GraphModule) -> None:
        updater = FakeTensorUpdater(main_graph)

        def recursively_test_graph_mod(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.find_nodes(op="placeholder"):
                with gm.graph.inserting_after(node):
                    clone_node = gm.graph.call_function(
                        torch.ops.aten.clone.default, (node,)
                    )

                node.replace_all_uses_with(
                    clone_node, delete_user_cb=lambda n: n != clone_node
                )

                # At minimum we should update the cloned node.  We may also update a
                # variable number of other nodes, so it's difficult to make hard
                # assertions here.
                with V.set_fake_mode(detect_fake_mode(get_fake(node, gm))):
                    self.assertGreaterEqual(updater.incremental_update(), 1, str(node))

            # iterate over subgraphs, updating *main_graph*
            for subgraph_name in _get_subgraph_names(gm):
                subgraph = getattr(gm, subgraph_name)
                self.assertIsInstance(subgraph, torch.fx.GraphModule)
                recursively_test_graph_mod(subgraph)

        recursively_test_graph_mod(main_graph)

    def _modify_node(self, main_graph: torch.fx.GraphModule) -> None:
        updater = FakeTensorUpdater(main_graph)

        def recursively_test_graph_mod(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.nodes:
                # If "val" isn't in the meta dict initially, an update will
                # likely skip the node anyway, so the logic of this test doesn't
                # work.
                if "val" not in node.meta:
                    continue

                val: torch.Tensor = node.meta["val"]
                dtype = val.dtype
                shape = val.size()
                strides = val.stride()
                del node.meta["val"], val

                self.assertEqual(updater.incremental_update(), 1)
                self.assertIn("val", node.meta)

                val: torch.Tensor = node.meta["val"]
                self.assertEqual(val.dtype, dtype)
                self.assertEqual(val.size(), shape)
                self.assertEqual(val.stride(), strides)

            # iterate over subgraphs, updating *main_graph*
            for subgraph_name in _get_subgraph_names(gm):
                subgraph = getattr(gm, subgraph_name)
                self.assertIsInstance(subgraph, torch.fx.GraphModule)
                recursively_test_graph_mod(subgraph)

        recursively_test_graph_mod(main_graph)

    def _common_test(
        self, fn: Callable[..., torch.Tensor], *args: torch.Tensor
    ) -> None:
        backend = AotEagerAndRecordGraphs()
        # populate the backend with a captured graph
        torch.compile(backend=backend, fullgraph=True)(fn)(*args)

        self._modify_node(deepcopy(backend.graphs[0]))
        self._insert_clone(deepcopy(backend.graphs[0]))

    # TODO: remove this XFAIL by resolving our failure to update into torch.cond
    # subgraphs.
    # @unittest.expectedFailure
    def test_hop_implicit_subgraph_inputs(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.cond(torch.sum(x) < 0, torch.sin, torch.cos, (x,))

        a = torch.randn((32, 32, 32))
        self._common_test(fn, a)

    def test_hop_subgraph_inputs(self):
        @torch.compiler.nested_compile_region
        def nested_section_inner(a: torch.Tensor) -> torch.Tensor:
            return torch.sin(a)

        @torch.compiler.nested_compile_region
        def nested_section_outer(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            return nested_section_inner(nested_section_inner(a)), nested_section_inner(
                b
            )

        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x, y = nested_section_outer(a, b)
            return x + y

        a = torch.randint(0, (1 << 16), (32, 32, 32), dtype=torch.int32)
        b = torch.randint(0, (1 << 16), (32, 32, 32), dtype=torch.int32)
        self._common_test(fn, a, b)


if __name__ == "__main__":
    run_tests()
