# Owner(s): ["module: inductor"]

import importlib.util
import operator
import unittest
from collections.abc import Callable, Iterator

from sympy import I, Max, Min, Symbol, sympify

import torch
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._dynamo.utils import detect_fake_mode
from torch._inductor.compile_fx import _get_subgraph_names
from torch._inductor.fx_utils import (
    _is_fake_tensor_same,
    count_flops_fx,
    countable_fx,
    FakeTensorUpdater,
    get_fake,
    get_fake_args_kwargs,
)
from torch._inductor.utils import get_device_tflops, sympy_str, sympy_subs
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
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

    def test_flops_fx_higher_order_op(self):
        """count_flops_fx must use the registered formula for HOP targets
        rather than invoking the HOP. flex_attention.__call__ requires a
        Dynamo/proxy tracing context (TransformGetItemToIndex) and raises
        TypeError when invoked on bare (fake) tensors.
        """
        from torch.utils.flop_counter import flop_registry

        flex_attention = torch.ops.higher_order.flex_attention
        self.assertIn(flex_attention, flop_registry)

        q_shape = (2, 16, 1024, 64)
        k_shape = (2, 4, 1024, 64)
        v_shape = (2, 4, 1024, 64)

        with V.set_fake_mode(
            torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        ):
            graph = torch.fx.Graph()
            q = graph.placeholder("q")
            k = graph.placeholder("k")
            v = graph.placeholder("v")
            q.meta["val"] = torch.randn(*q_shape, device="meta", dtype=torch.bfloat16)
            k.meta["val"] = torch.randn(*k_shape, device="meta", dtype=torch.bfloat16)
            v.meta["val"] = torch.randn(*v_shape, device="meta", dtype=torch.bfloat16)
            node = graph.call_function(flex_attention, args=(q, k, v))
            node.meta["val"] = (
                torch.randn(*q_shape, device="meta", dtype=torch.bfloat16),
                torch.randn(
                    q_shape[0],
                    q_shape[1],
                    q_shape[2],
                    device="meta",
                    dtype=torch.float32,
                ),
                torch.randn(
                    q_shape[0],
                    q_shape[1],
                    q_shape[2],
                    device="meta",
                    dtype=torch.float32,
                ),
            )

            self.assertTrue(countable_fx(node))
            flops = count_flops_fx(node)
            expected = flop_registry[flex_attention](
                q.meta["val"], k.meta["val"], v.meta["val"], out_val=node.meta["val"]
            )
            self.assertEqual(flops, expected)

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


class TestTritonTypeMapping(TestCase):
    """Tests for acc_type() dtype conversions."""

    def test_acc_type(self):
        from torch._inductor.kernel.mm_common import acc_type

        cases = {
            "half promotes to float32": (torch.float16, "tl.float32"),
            "bfloat16 promotes to float32": (torch.bfloat16, "tl.float32"),
            "float32 passthrough": (torch.float32, "tl.float32"),
            "fp8 e4m3fn promotes to float32": (torch.float8_e4m3fn, "tl.float32"),
            "fp8 e5m2 promotes to float32": (torch.float8_e5m2, "tl.float32"),
            "fp8 e4m3fnuz promotes to float32": (torch.float8_e4m3fnuz, "tl.float32"),
            "fp8 e5m2fnuz promotes to float32": (torch.float8_e5m2fnuz, "tl.float32"),
        }
        for desc, (dtype, expected) in cases.items():
            with self.subTest(desc=desc, dtype=dtype):
                self.assertEqual(acc_type(dtype), expected)


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
                aten._foreach_clone.default
                if isinstance(fake_outputs, tuple)
                else aten.clone.default
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
            fn.replace_all_uses_with(cloned_node, lambda n: n != cloned_node)

            with V.set_fake_mode(fake_mode):
                clone_num_updated = updater.incremental_update()

            # At a minimum, we have to update the newly inserted node.  The users
            # may not need updates if the replacement has equivalent metadata.
            self.assertGreaterEqual(clone_num_updated, 1)
            self.assertIn("val", cloned_node.meta)
            with V.set_fake_mode(fake_mode):
                self.assertEqual(updater.incremental_update(), 0)

            cloned_node.replace_all_uses_with(fn)
            gm.graph.erase_node(cloned_node)
            with V.set_fake_mode(fake_mode):
                updater.incremental_update()
                self.assertEqual(updater.incremental_update(), 0)

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

    def test_hop_subgraph_dtype_change(self):
        def true_fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def false_fn(x: torch.Tensor) -> torch.Tensor:
            return x - 1

        def fn(x: torch.Tensor) -> torch.Tensor:
            y = x.view(torch.int32)
            return torch.cond(x.sum() > 0, true_fn, false_fn, (y,))

        def tensor_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
            return [
                fake.dtype
                for node in gm.graph.nodes
                if isinstance((fake := node.meta.get("val")), torch.Tensor)
            ]

        graph = make_fx(fn, tracing_mode="fake")(torch.randn(4))
        view_dtype_node = next(
            n for n in graph.graph.nodes if n.target == torch.ops.aten.view.dtype
        )
        self.assertEqual(view_dtype_node.meta["val"].dtype, torch.int32)
        for subgraph_name in _get_subgraph_names(graph):
            self.assertEqual(
                tensor_dtypes(getattr(graph, subgraph_name)), [torch.int32] * 2
            )

        updater = FakeTensorUpdater(graph)
        view_dtype_node.args = (view_dtype_node.args[0], torch.float32)
        with V.set_fake_mode(self._get_faketensormode(graph)):
            updater.incremental_update()

        self.assertEqual(view_dtype_node.meta["val"].dtype, torch.float32)
        for subgraph_name in _get_subgraph_names(graph):
            self.assertEqual(
                tensor_dtypes(getattr(graph, subgraph_name)), [torch.float32] * 2
            )

    def test_auto_functionalized_dtype_change(self):
        with torch.library._scoped_library("fake_tensor_updater", "FRAGMENT") as lib:
            torch.library.define(
                "fake_tensor_updater::mutate_x",
                "(Tensor(a!) x) -> ()",
                lib=lib,
            )

            @torch.library.impl(
                "fake_tensor_updater::mutate_x", "CompositeExplicitAutograd", lib=lib
            )
            def mutate_x_impl(x: torch.Tensor) -> None:
                x.add_(1)

            @torch.library.register_fake("fake_tensor_updater::mutate_x", lib=lib)
            def mutate_x_fake(x: torch.Tensor) -> None:
                return None

            def fn(x: torch.Tensor) -> torch.Tensor:
                y = x.view(torch.int32)
                _, new_y = torch.ops.higher_order.auto_functionalized(
                    torch.ops.fake_tensor_updater.mutate_x.default, x=y
                )
                return new_y + 1

            graph = make_fx(fn, tracing_mode="fake")(torch.randn(4))

            view_dtype_node = next(
                n for n in graph.graph.nodes if n.target == torch.ops.aten.view.dtype
            )
            auto_functionalized_node = next(
                n
                for n in graph.graph.nodes
                if n.target == torch.ops.higher_order.auto_functionalized
            )
            add_node = next(
                n for n in graph.graph.nodes if n.target == torch.ops.aten.add.Tensor
            )

            def updated_tensor_dtype() -> torch.dtype:
                val = auto_functionalized_node.meta["val"]
                self.assertIsNone(val[0])
                self.assertIsInstance(val[1], torch.Tensor)
                return val[1].dtype

            self.assertEqual(view_dtype_node.meta["val"].dtype, torch.int32)
            self.assertEqual(updated_tensor_dtype(), torch.int32)
            self.assertEqual(add_node.meta["val"].dtype, torch.int32)

            updater = FakeTensorUpdater(graph)
            view_dtype_node.args = (view_dtype_node.args[0], torch.float32)

            hop = torch.ops.higher_order.auto_functionalized
            sentinel = object()
            old_lowering_marker = getattr(hop, "_inductor_lowering_function", sentinel)
            hop._inductor_lowering_function = True
            try:
                with V.set_fake_mode(self._get_faketensormode(graph)):
                    updater.incremental_update()
            finally:
                if old_lowering_marker is sentinel:
                    delattr(hop, "_inductor_lowering_function")
                else:
                    hop._inductor_lowering_function = old_lowering_marker

            self.assertEqual(view_dtype_node.meta["val"].dtype, torch.float32)
            self.assertEqual(updated_tensor_dtype(), torch.float32)
            self.assertEqual(add_node.meta["val"].dtype, torch.float32)

    def test_run_const_graph_dtype_change(self):
        def inner_fn(y: torch.Tensor) -> torch.Tensor:
            return y + 1

        inner_graph = make_fx(inner_fn, tracing_mode="fake")(
            torch.ones(4, dtype=torch.int32)
        )

        def fn(x: torch.Tensor) -> torch.Tensor:
            y = x.view(torch.int32)
            return torch.ops.higher_order.run_const_graph(inner_graph, (y,))

        def tensor_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
            return [
                fake.dtype
                for node in gm.graph.nodes
                if isinstance((fake := node.meta.get("val")), torch.Tensor)
            ]

        graph = make_fx(fn, tracing_mode="fake")(torch.randn(4))
        view_dtype_node = next(
            n for n in graph.graph.nodes if n.target == torch.ops.aten.view.dtype
        )
        run_const_graph_node = next(
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.higher_order.run_const_graph
        )
        subgraph = getattr(graph, next(_get_subgraph_names(graph)))

        self.assertEqual(view_dtype_node.meta["val"].dtype, torch.int32)
        self.assertEqual(run_const_graph_node.meta["val"].dtype, torch.int32)
        self.assertEqual(tensor_dtypes(subgraph), [torch.int32] * 2)

        updater = FakeTensorUpdater(graph)
        view_dtype_node.args = (view_dtype_node.args[0], torch.float32)
        with V.set_fake_mode(self._get_faketensormode(graph)):
            updater.incremental_update()

        self.assertEqual(view_dtype_node.meta["val"].dtype, torch.float32)
        self.assertEqual(run_const_graph_node.meta["val"].dtype, torch.float32)
        self.assertEqual(tensor_dtypes(subgraph), [torch.float32] * 2)

    def test_new_subgraph_after_updater_init_dtype_change(self):
        def inner_fn(y: torch.Tensor) -> torch.Tensor:
            return y + 1

        inner_graph = make_fx(inner_fn, tracing_mode="fake")(
            torch.ones(4, dtype=torch.int32)
        )

        def tensor_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
            return [
                fake.dtype
                for node in gm.graph.nodes
                if isinstance((fake := node.meta.get("val")), torch.Tensor)
            ]

        root = torch.nn.Module()
        outer_graph = torch.fx.Graph()
        x = outer_graph.placeholder("x")
        view = outer_graph.call_function(torch.ops.aten.view.dtype, (x, torch.int32))
        output = outer_graph.output(view)
        graph = torch.fx.GraphModule(root, outer_graph)

        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            x.meta["val"] = torch.randn(4)
            view.meta["val"] = view.target(*(get_fake(arg, graph) for arg in view.args))

        updater = FakeTensorUpdater(graph)

        graph.add_module("subgraph", inner_graph)
        with graph.graph.inserting_before(output):
            subgraph_attr = graph.graph.get_attr("subgraph")
        with graph.graph.inserting_before(output):
            run_const_graph = graph.graph.call_function(
                torch.ops.higher_order.run_const_graph,
                (subgraph_attr, (view,)),
            )
        output.args = (run_const_graph,)
        graph.graph.lint()

        with fake_mode:
            is_valid, args, kwargs = get_fake_args_kwargs(run_const_graph, graph)
            self.assertTrue(is_valid)
            run_const_graph.meta["val"] = run_const_graph.target(*args, **kwargs)

        self.assertEqual(view.meta["val"].dtype, torch.int32)
        self.assertEqual(run_const_graph.meta["val"].dtype, torch.int32)
        self.assertEqual(tensor_dtypes(inner_graph), [torch.int32] * 2)

        view.args = (x, torch.float32)
        with V.set_fake_mode(fake_mode):
            updater.incremental_update()

        self.assertEqual(view.meta["val"].dtype, torch.float32)
        self.assertEqual(run_const_graph.meta["val"].dtype, torch.float32)
        self.assertEqual(tensor_dtypes(inner_graph), [torch.float32] * 2)

    def test_with_effects_invoke_subgraph_dtype_change(self):
        def inner_fn(y: torch.Tensor) -> tuple[torch.Tensor]:
            return (y + 1,)

        inner_graph = make_fx(inner_fn, tracing_mode="fake")(
            torch.ones(4, dtype=torch.int32)
        )

        root = torch.nn.Module()
        root.subgraph = inner_graph
        outer_graph = torch.fx.Graph()
        x = outer_graph.placeholder("x")
        view = outer_graph.call_function(torch.ops.aten.view.dtype, (x, torch.int32))
        token = outer_graph.call_function(torch.ops.aten._make_dep_token.default, ())
        subgraph_attr = outer_graph.get_attr("subgraph")
        with_effects = outer_graph.call_function(
            torch.ops.higher_order.with_effects,
            (
                token,
                torch.ops.higher_order.invoke_subgraph,
                subgraph_attr,
                "fake_tensor_updater",
                view,
            ),
        )
        out = outer_graph.call_function(operator.getitem, (with_effects, 1))
        outer_graph.output(out)
        graph = torch.fx.GraphModule(root, outer_graph)

        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            x.meta["val"] = torch.randn(4)
            view.meta["val"] = view.target(*(get_fake(arg, graph) for arg in view.args))
            token.meta["val"] = token.target()
            with_effects.meta["val"] = with_effects.target(
                *(get_fake(arg, graph) for arg in with_effects.args)
            )
            out.meta["val"] = out.target(get_fake(with_effects, graph), 1)

        def tensor_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
            return [
                fake.dtype
                for node in gm.graph.nodes
                if isinstance((fake := node.meta.get("val")), torch.Tensor)
            ]

        self.assertEqual(view.meta["val"].dtype, torch.int32)
        self.assertEqual(with_effects.meta["val"][1].dtype, torch.int32)
        self.assertEqual(out.meta["val"].dtype, torch.int32)
        self.assertEqual(tensor_dtypes(inner_graph), [torch.int32] * 2)

        updater = FakeTensorUpdater(graph)
        view.args = (x, torch.float32)
        with V.set_fake_mode(fake_mode):
            updater.incremental_update()

        self.assertEqual(view.meta["val"].dtype, torch.float32)
        self.assertEqual(with_effects.meta["val"][1].dtype, torch.float32)
        self.assertEqual(out.meta["val"].dtype, torch.float32)
        self.assertEqual(tensor_dtypes(inner_graph), [torch.float32] * 2)

    def test_subgraph_hop_retraces_changed_outer_operands(self):
        from torch._higher_order_ops.flex_attention import (
            flex_attention as flex_attention_hop,
        )
        from torch.nn.attention.flex_attention import _create_empty_block_mask

        def score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return score

        def fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            block_mask = _create_empty_block_mask(q, k)
            out, _, _ = flex_attention_hop(
                q,
                k,
                v,
                score_mod,
                block_mask.as_tuple(),
                1.0,
                {},
            )
            return out

        graph = make_fx(fn, tracing_mode="fake")(
            torch.randn(2, 2, 4, 4),
            torch.randn(2, 2, 4, 4),
            torch.randn(2, 2, 4, 4),
        )
        updater = FakeTensorUpdater(graph)
        fake_mode = self._get_faketensormode(graph)
        query_node = next(
            n for n in graph.graph.nodes if n.op == "placeholder" and n.name == "q_1"
        )
        key_node = next(
            n for n in graph.graph.nodes if n.op == "placeholder" and n.name == "k_1"
        )
        value_node = next(
            n for n in graph.graph.nodes if n.op == "placeholder" and n.name == "v_1"
        )
        flex_node = next(
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.higher_order.flex_attention
        )
        old_stride = flex_node.meta["val"][0].stride()

        with graph.graph.inserting_after(query_node):
            cast_query_node = graph.graph.call_function(
                torch.ops.aten.to.dtype, (query_node, torch.float16)
            )
        with graph.graph.inserting_after(cast_query_node):
            cloned_query_node = graph.graph.call_function(
                aten.clone.default,
                (cast_query_node,),
                {"memory_format": torch.channels_last},
            )
        with graph.graph.inserting_after(key_node):
            cast_key_node = graph.graph.call_function(
                torch.ops.aten.to.dtype, (key_node, torch.float16)
            )
        with graph.graph.inserting_after(value_node):
            cast_value_node = graph.graph.call_function(
                torch.ops.aten.to.dtype, (value_node, torch.float16)
            )
        flex_node.replace_input_with(query_node, cloned_query_node)
        flex_node.replace_input_with(key_node, cast_key_node)
        flex_node.replace_input_with(value_node, cast_value_node)

        with V.set_fake_mode(fake_mode):
            updater.incremental_update()
            is_valid, args, kwargs = get_fake_args_kwargs(flex_node, graph)
            self.assertTrue(is_valid)
            expected_fake = flex_node.target(*args, **kwargs)

        self.assertNotEqual(flex_node.meta["val"][0].stride(), old_stride)
        self.assertEqual(flex_node.meta["val"][0].dtype, torch.float16)
        self.assertEqual(flex_node.meta["val"][0].stride(), expected_fake[0].stride())

    def test_incremental_update_noop(self):
        def true_fn(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def false_fn(x: torch.Tensor) -> torch.Tensor:
            return x - 1

        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

        graph = make_fx(fn, tracing_mode="fake")(torch.randn(4))
        updater = FakeTensorUpdater(graph)

        with V.set_fake_mode(self._get_faketensormode(graph)):
            self.assertEqual(updater.incremental_update(), 0)
            self.assertEqual(updater.incremental_update(), 0)

    def test_get_fake_args_kwargs_missing_meta_is_invalid(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        x.meta["val"] = torch.empty(4, device="meta")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, y))

        is_valid, args, kwargs = get_fake_args_kwargs(add)

        self.assertFalse(is_valid)
        self.assertIs(args[1], y)
        self.assertEqual(kwargs, {})

    def test_incremental_update_retries_invalid_node(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        z = graph.placeholder("z")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, z))
        graph.output((add, mul))
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            x.meta["val"] = torch.empty(4)
            y_fake = torch.empty(4)
            z_fake = torch.empty(4)

        updater = FakeTensorUpdater(graph_module)
        with V.set_fake_mode(fake_mode):
            y.meta["val"] = y_fake
            self.assertEqual(updater.incremental_update(), 1)
            self.assertIn("val", add.meta)
            self.assertNotIn("val", mul.meta)
            z.meta["val"] = z_fake
            self.assertEqual(updater.incremental_update(), 1)
            self.assertEqual(updater.incremental_update(), 0)

        self.assertIn("val", mul.meta)

    def test_get_fake_args_kwargs_tensor_get_attr_without_meta_is_invalid(self):
        class Root(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buf", torch.ones(4))

        root = Root()
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        buf = graph.get_attr("buf")
        add = graph.call_function(torch.ops.aten.add.Tensor, (x, buf))
        graph.output(add)
        graph_module = torch.fx.GraphModule(root, graph)
        x.meta["val"] = torch.empty(4, device="meta")

        is_valid, args, kwargs = get_fake_args_kwargs(add, graph_module)

        self.assertFalse(is_valid)
        self.assertIs(args[1], buf)
        self.assertEqual(kwargs, {})

    def test_get_fake_args_kwargs_tensor_container_get_attr_is_invalid(self):
        class Root(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.values = (torch.ones(4),)

        root = Root()
        graph = torch.fx.Graph()
        values = graph.get_attr("values")
        getitem = graph.call_function(operator.getitem, (values, 0))
        graph.output(getitem)
        graph_module = torch.fx.GraphModule(root, graph)

        is_valid, args, kwargs = get_fake_args_kwargs(getitem, graph_module)

        self.assertFalse(is_valid)
        self.assertIs(args[0], values)
        self.assertEqual(kwargs, {})

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

    def test_fake_tensor_same_recursion(self):
        l = [1, 2, 3]
        l.append(l)
        m = [4, 5, 6, l]
        # If recursion is broken, we'll get a recursion error here.
        self.assertTrue(_is_fake_tensor_same(l, l, {}))
        self.assertFalse(_is_fake_tensor_same(l, m, {}))


if __name__ == "__main__":
    run_tests()
