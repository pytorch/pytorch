# Owner(s): ["module: inductor"]

import importlib.util
import unittest
from collections.abc import Callable, Iterator

from sympy import I, Max, Min, Symbol, sympify

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
from torch.ops import aten
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.utils._sympy.functions import Identity


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

    def testSympySubsIdentityNonComparable(self):
        q0 = Symbol("q0", integer=True, nonnegative=True)
        expr = Min(2, Max(0, Identity(q0)))
        result = sympy_subs(expr, {q0: I})
        self.assertTrue(result.has(I))

    def testIdentityComparisonNoRecursion(self):
        self.assertTrue(Identity(sympify("0")) >= 0)
        self.assertFalse(Identity(sympify("-6")) >= 0)
        self.assertTrue(0 >= Identity(sympify("-6")))

    def testIdentityComparableNumbersInMinMax(self):
        expr = Identity(sympify("-6"))
        self.assertTrue(expr.is_number)
        self.assertTrue(expr.is_comparable)
        self.assertEqual(Max(0, expr), 0)

    def testIdentityRationalComparisonNoRecursion(self):
        expr = Identity(sympify("1/7"))
        self.assertTrue(expr >= 0)
        self.assertTrue(Max(0, expr).has(expr))

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

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not torch.cuda.is_available(), "skip if no device")
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_get_device_tflops(self, dtype):
        ret = get_device_tflops(dtype)
        self.assertTrue(type(ret) is float)


instantiate_device_type_tests(TestUtils, globals(), allow_xpu=True)


class TestRuntimeEstimation(TestCase):
    def test_get_compute_time_units(self):
        """TFLOPS-to-FLOPS/s conversion must use 1e12, not 1e15."""
        from unittest.mock import patch

        from torch.utils._runtime_estimation import get_compute_time

        M, K, N = 64, 64, 64
        known_tflops = 1000.0
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        out = torch.mm(a, b)

        with patch(
            "torch.utils._runtime_estimation.get_device_tflops",
            return_value=known_tflops,
        ):
            result_ns = get_compute_time(
                torch.ops.aten.mm, (a, b), {}, out, {torch.float32}
            )

        # mm flops = 2*M*K*N, divided by 2 for MACs, then time = macs / (0.75 * peak) * 1e9
        expected_macs = 2 * M * K * N / 2
        expected_ns = (expected_macs / (0.75 * known_tflops * 1e12)) * 1e9
        self.assertAlmostEqual(result_ns, expected_ns)


class TestFP4Support(TestCase):
    """Tests for FP4 (float4_e2m1fn_x2) infrastructure support."""

    @unittest.skipIf(
        not torch.cuda.is_available()
        or importlib.util.find_spec("cutlass_api") is None,
        "requires CUDA and cutlass_api",
    )
    def test_ensure_fp4_dtype_registered(self):
        """_ensure_fp4_dtype_registered should patch cutlass_api for FP4."""
        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()
        import cutlass
        import cutlass_api.utils

        result = cutlass_api.utils.cutlass_type_from_torch_type(torch.float4_e2m1fn_x2)
        self.assertEqual(result, cutlass.Float4E2M1FN)

        result_fp32 = cutlass_api.utils.cutlass_type_from_torch_type(torch.float32)
        self.assertEqual(result_fp32, cutlass.Float32)

    def test_rand_strided_fp4(self):
        """rand_strided should produce valid FP4 tensors."""
        from torch._dynamo.testing import rand_strided

        t = rand_strided((4, 8), (8, 1), dtype=torch.float4_e2m1fn_x2, device="cpu")
        self.assertEqual(t.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(t.shape, (4, 8))
        self.assertEqual(t.stride(), (8, 1))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_rand_strided_fp4_cuda(self):
        from torch._dynamo.testing import rand_strided

        t = rand_strided((16, 32), (32, 1), dtype=torch.float4_e2m1fn_x2, device="cuda")
        self.assertEqual(t.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(t.shape, (16, 32))
        self.assertTrue(t.is_cuda)


class TestFakeTensorUpdater(TestCase):
    @staticmethod
    def _get_faketensormode(
        graph: torch.fx.GraphModule,
    ) -> torch._subclasses.FakeTensorMode:
        return (
            detect_fake_mode(get_fake(next(iter(graph.graph.nodes)), graph))
            or torch._subclasses.FakeTensorMode()
        )

    @staticmethod
    def _get_graph(
        fn: Callable[..., torch.Tensor], *args: torch.Tensor
    ) -> torch.fx.GraphModule:
        backend = AotEagerAndRecordGraphs()
        torch.compile(backend=backend, fullgraph=True)(fn)(*args)
        return backend.fw_graphs[0]

    @staticmethod
    def _get_call_function_nodes(
        graph: torch.fx.GraphModule,
    ) -> Iterator[tuple[torch.fx.GraphModule, torch.fx.Node]]:
        """Recursively yields all call_function nodes in a GraphModule.  These nodes are
        ideal to apply transformations to, since callables are the focus of
        FakeTensorUpdater."""
        for sn in _get_subgraph_names(graph):
            yield from TestFakeTensorUpdater._get_call_function_nodes(
                getattr(graph, sn)
            )

        yield from ((graph, n) for n in graph.graph.nodes if n.op == "call_function")

    def _add_delete_nodes_test(self, graph: torch.fx.GraphModule) -> None:
        updater = FakeTensorUpdater(graph)
        fake_mode = self._get_faketensormode(graph)

        for gm, fn in self._get_call_function_nodes(graph):
            fake_outputs = get_fake(fn, gm)
            self.assertIsNot(fake_outputs, fn, msg="No fake outputs for node!")

            # Since we're testing changes in subgraphs, we've explicitly disallowed
            # changes other than striding to anything input to a subgraph.  With
            # cascading changes, cloning is the most straightforward approach to ensure
            # that constraint is met.
            clone_function = (
                torch._foreach_clone if isinstance(fake_outputs, tuple) else torch.clone
            )
            with gm.graph.inserting_after(fn):
                # When tests use input tensors with dim == 4, shuffle striding order to
                # test that updating subgraphs handles striding changes.
                should_shuffle_strides = "val" in fn.meta and (
                    (
                        isinstance(fn.meta["val"], tuple)
                        and all(len(v.size()) == 4 for v in fn.meta["val"])
                    )
                    or len(fn.meta["val"].size()) == 4
                )
                if should_shuffle_strides:
                    cloned_node = gm.graph.call_function(
                        clone_function, (fn,), {"memory_format": torch.channels_last}
                    )
                else:
                    cloned_node = gm.graph.call_function(clone_function, (fn,))
            nodes_modified = fn.replace_all_uses_with(
                cloned_node, lambda n: n != cloned_node
            )

            with V.set_fake_mode(fake_mode):
                clone_num_updated = updater.incremental_update()

            # At a minimum, we have to update the newly inserted node and all the nodes
            # which had an input replaced.  There may be more nodes modified in
            # subgraphs, so we can't do a strict equality assertion here.
            self.assertGreaterEqual(clone_num_updated, len(nodes_modified) + 1)

            cloned_node.replace_all_uses_with(fn)
            gm.graph.erase_node(cloned_node)
            with V.set_fake_mode(fake_mode):
                erase_num_updated = updater.incremental_update()

            # Deleting the node should update the same number of nodes as previously,
            # excluding the reshaped node itself.
            self.assertEqual(clone_num_updated - 1, erase_num_updated)

    def test_hop_implicit_subgraph_inputs(self):
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.cond(torch.sum(x) < 0, torch.sin, torch.cos, (x,))

        # Use 4-D tensor so that we can test re-striding with channels_last.
        a = torch.randn((8, 4, 2, 1))
        graph = self._get_graph(fn, a)
        self._add_delete_nodes_test(graph)

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

        # Use 4-D tensor so that we can test re-striding with channels_last.
        a = torch.randint(0, (1 << 16), (8, 4, 2, 1), dtype=torch.int32)
        b = torch.randint(0, (1 << 16), (8, 4, 2, 1), dtype=torch.int32)
        graph = self._get_graph(fn, a, b)
        self._add_delete_nodes_test(graph)

    def test_reorder_nodes(self):
        def fn(*args: torch.Tensor) -> torch.Tensor:
            ret = torch.ones_like(args[0])
            for a in args:
                ret = a * ret
            return ret

        a = torch.rand((8,))
        b = torch.rand((8, 8))
        c = torch.rand((8, 8, 8))
        d = torch.rand((8, 8, 8, 8))
        graph = self._get_graph(fn, a, b, c, d)
        updater = FakeTensorUpdater(graph)

        reversed_placeholders: list[torch.fx.Node] = list(
            reversed(graph.graph.find_nodes(op="placeholder"))
        )
        mul_nodes: list[torch.fx.Node] = graph.graph.find_nodes(
            op="call_function", target=aten.mul.Tensor
        )
        for p, m in zip(reversed_placeholders, mul_nodes, strict=True):
            # The argument tensor is always at index zero.
            m.replace_input_with(m.all_input_nodes[0], p)

        with V.set_fake_mode(self._get_faketensormode(graph)):
            num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 4)
        # With reversed multiplication order, all the mul_nodes should output 4-D
        # tensors.
        for m in mul_nodes:
            self.assertEqual(len(m.meta["val"].size()), 4)


if __name__ == "__main__":
    run_tests()
