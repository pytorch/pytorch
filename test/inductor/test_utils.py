# Owner(s): ["module: inductor"]

import importlib.util
import unittest
from collections.abc import Callable

from sympy import I, Max, Min, Symbol, sympify

import torch
from torch._inductor.fx_utils import count_flops_fx, countable_fx, FakeTensorUpdater
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


class TestFakeTensorUpdater(TestCase):
    @staticmethod
    def _make_inductor_lowering_function(
        *,
        output_metadata_ignores_input_storage: bool = False,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def lowering_fn(x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("lowering_fn should not run under FakeTensorUpdater")

        lowering_fn._inductor_lowering_function = True  # type: ignore[attr-defined]
        lowering_fn._inductor_lowering_output_metadata_ignores_input_storage = (  # type: ignore[attr-defined]
            output_metadata_ignores_input_storage
        )
        return lowering_fn

    @classmethod
    def _build_graph_with_inductor_lowering_node(
        cls,
    ) -> tuple[
        torch.fx.GraphModule,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
    ]:
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(cls._make_inductor_lowering_function(), (neg,))
        graph.output(lowered)
        return torch.fx.GraphModule({}, graph), x, y, neg, lowered

    def test_unchanged_inductor_lowering_node_is_ignored(self):
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 0)
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_changed_node_back_to_previous_hash_updates_metadata(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm.graph)
            neg.args = (y,)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()
            self.assertEqual(num_updated, 1)
            self.assertEqual(tuple(neg.meta["val"].shape), (4, 5))

            neg.args = (x,)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))

    def test_new_inductor_lowering_node_with_metadata_is_ignored(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm.graph)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            lowered.meta["val"] = neg.meta["val"]
            output.args = (lowered,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 0)
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_marked_inductor_lowering_node_ignores_storage_only_dependency_change(
        self,
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (neg,),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            neg.args = (y,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_marked_inductor_lowering_node_ignores_storage_only_kwarg_change(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (),
            {"other": neg},
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            neg.args = (y,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_unmarked_inductor_lowering_node_rejects_storage_only_dependency_change(
        self,
    ):
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_marked_inductor_lowering_node_rejects_dtype_dependency_change(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (neg,),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3, dtype=torch.float32))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3, dtype=torch.float64))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_new_inductor_lowering_node_with_changed_dependency_raises(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm.graph)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            lowered.meta["val"] = neg.meta["val"]
            output.args = (lowered,)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

        self.assertEqual(tuple(neg.meta["val"].shape), (4, 5))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_new_inductor_lowering_node_without_metadata_raises(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm.graph)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            output.args = (lowered,)

            with self.assertRaisesRegex(RuntimeError, "already carry fake metadata"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_changed_inductor_lowering_node_raises_before_stale_metadata(self):
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm.graph)
            with gm.graph.inserting_before(lowered):
                neg_replacement = gm.graph.call_function(aten.neg.default, (y,))
            lowered.args = (neg_replacement,)

            with self.assertRaisesRegex(
                RuntimeError,
                "_inductor_lowering_function nodes",
            ):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

        self.assertEqual(tuple(neg_replacement.meta["val"].shape), (4, 5))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))


if __name__ == "__main__":
    run_tests()
