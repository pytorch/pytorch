# Owner(s): ["module: inductor"]

import unittest

from sympy import Symbol, sympify

import torch
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._export.utils import _detect_fake_mode_from_gm
from torch._inductor.fx_utils import count_flops_fx, countable_fx, FakeTensorUpdater
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
        # propagte abs(x) sympy properties.
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
            aten: torch._ops.OpOverloadPacket, args, kwargs
        ) -> tuple[torch.fx.Node, torch.fx.Node]:
            node1 = torch.fx.Node(
                graph=torch.fx.Graph(),
                name="",
                op="call_function",
                target=aten,
                args=args,
                kwargs=kwargs,
            )
            name: str = aten.overloads()[0]
            op_overload: torch._ops.OpOverload = getattr(aten, name)
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
                    (torch.Tensor(4, 4), torch.Tensor(4, 5), torch.Tensor(5, 4)),
                    {},
                ),
                (
                    torch.ops.aten.bmm,
                    (torch.Tensor(10, 4, 5), torch.Tensor(10, 5, 4)),
                    {},
                ),
                (torch.ops.aten.mm, (torch.Tensor(2, 3), torch.Tensor(3, 2)), {}),
                (
                    torch.ops.aten.convolution,
                    (
                        torch.Tensor(2, 3, 3),
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2),
                        (1, 1),
                        (0, 0),
                        (1, 1),
                        True,
                        (0, 0),
                        1,
                    ),
                    {},
                ),
                (
                    torch.ops.aten._convolution,
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
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
                (
                    torch.ops.aten.mul,
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
            ]
            for t, args, kwargs in trues:
                fx_node_1, fx_node_2 = create_fx_node(t, args, kwargs)
                self.assertTrue(
                    countable_fx(fx_node_1), f"Expected true {t}: {fx_node_1}"
                )
                self.assertTrue(
                    countable_fx(fx_node_2), f"Expected true {t}: {fx_node_2}"
                )
                self.assertNotEqual(count_flops_fx(fx_node_1), None)
                self.assertNotEqual(count_flops_fx(fx_node_2), None)
            for f, args, kwargs in falses:
                fx_node_1, fx_node_2 = create_fx_node(f, args, kwargs)
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
        self.assertTrue(type(ret) == float)


instantiate_device_type_tests(TestUtils, globals())


class TestFakeTensorUpdater(TestCase):
    def _common_impl(self, gm: torch.fx.GraphModule) -> None:
        """Assumes that gm is a GraphModule with a single-dimensioned tensor output
        whose size will grow proportionally to the input size."""

        def add_cat_to_inputs(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
            """Transforms input GraphModule by concatenating all inputs with
            themselves."""
            for node in gm.graph.find_nodes(op="placeholder"):
                with gm.graph.inserting_after(node):
                    cat_node = gm.graph.call_function(
                        torch.ops.aten.cat, ([node, node],)
                    )
                    node.replace_all_uses_with(cat_node, lambda n: n != cat_node)
            gm.graph.eliminate_dead_code()
            gm.graph.lint()
            return gm

        def get_output_size(gm: torch.fx.GraphModule) -> torch.Size:
            output_node: torch.fx.Node = gm.graph.output_node().args[0][0]  # type: ignore[arg-type]
            if "val" in output_node.meta:
                return output_node.meta["val"].size()
            return output_node.meta["example_value"].size()

        output_size = get_output_size(gm)
        updater = FakeTensorUpdater(gm)
        for _ in range(5):
            gm = add_cat_to_inputs(gm)
            with V.set_fake_mode(_detect_fake_mode_from_gm(gm)):
                updater.incremental_update()

            # We could check the graph more thoroughly, but it should be sufficient to
            # check the meta for the output node alone.
            output_size = torch.Size((output_size[0] * 2,))
            self.assertEqual(get_output_size(gm), output_size)

    def test_hop_no_subgraph_inputs(self):
        pass

    def test_hop_subgraph_inputs(self):
        """Test propagation of FakeTensor into the invoke_subgraph HOP.  Modifying the
        tested subgraph itself is not supported by the current implementation of
        invoke_subgraph FakeTensor caching."""

        @torch.compiler.nested_compile_region
        def nested_section(a: torch.Tensor) -> torch.Tensor:
            return torch.sin(a)

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            x = nested_section(a)
            y = nested_section(b)
            return x + y

        a = torch.randn(32)
        b = torch.randn(32)

        fn(a, b)

        # Test propagation of FakeTensor _into_ subgraph HOP.  Modifying the subgraph
        # itself is not supported by the current implementation of invoke_subgraph
        # FakeTensor caching.
        self._common_impl(backend.graphs[0])


if __name__ == "__main__":
    run_tests()
