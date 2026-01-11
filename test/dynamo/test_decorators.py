# Owner(s): ["module: dynamo"]
import functools
import operator
import os
import re
import unittest.mock as mock
from unittest.mock import patch

import torch
import torch._dynamo.testing
from torch._dynamo.exc import IncorrectUsage, Unsupported
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfWindows,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase


def my_custom_function(x):
    return x + 1


class DecoratorTests(PytreeRegisteringTestCase):
    def test_disallow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.disallow_in_graph(torch.sub)
        fn(torch.randn(10))
        torch._dynamo.allow_in_graph(torch.sub)

        # check for graph break on sub
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_disable_for_custom_op(self):
        import torch.library
        from torch.library import Library

        foo = Library("foo", "DEF")  # noqa: TOR901
        try:
            foo.define("custom(Tensor self) -> Tensor")

            # Dynamic shape data dependent operator. For static shape compilation, Dynamo
            # should graph break on it. But, the meta kernel is not implemented properly.
            @torch.library.impl(foo, "custom", "CPU")
            def foo_cpu(x):
                return x.nonzero()

            # Disallow does not work because of extra python frames with torch.library python API
            orig_custom = torch.ops.foo.custom
            try:
                torch.ops.foo.custom = torch._dynamo.disable(torch.ops.foo.custom)

                def fn(x):
                    a = torch.nn.functional.relu(x)
                    b = torch.ops.foo.custom(a)
                    c = torch.cos(b)
                    return c

                x = torch.randint(2, (100,))
                ref = fn(x)

                cnts = torch._dynamo.testing.CompileCounter()
                opt_fn = torch.compile(fn, backend=cnts)
                res = opt_fn(x)
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(ref, res)
            finally:
                torch.ops.foo.custom = orig_custom
        finally:
            foo._destroy()

    def test_disable_ignores_outer_wraps(self):
        def orig_inner():
            pass

        def inner():
            pass

        inner._torchdynamo_orig_callable = orig_inner

        @functools.wraps(inner)
        def wrapper():
            raise AssertionError("wrapper called")

        # This behavior is not ideal, but supporting it would add overhead
        # to callsites of eval_frame.innermost_fn. A warning would also be very noisy.
        torch._dynamo.disable(fn=wrapper, recursive=True)

    def test_disable_nn_modules_forward_hook(self):
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        # Disable my monkeypatching
        model.layer0 = torch._dynamo.disable(model.layer0)

        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_model = torch.compile(model, backend=cnts)
        opt_model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_disable_nn_module_with_class_decorator(self):
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch._dynamo.disable
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        @torch.compile(backend=cnts)
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        def hook(module, args):
            inp = args[0].sigmoid()
            return (inp,)

        model = SimpleModel()
        model.layer0.register_forward_pre_hook(hook)

        model(torch.randn(4))

        # check for no graph break
        self.assertEqual(cnts.frame_count, 2)

        gm0 = cnts.graphs[0]
        # Check that the first graph has sin node, and no sigmoid
        self.assertTrue(any(node.target is torch.sin for node in gm0.graph.nodes))
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm0.graph.nodes)
        )

        gm1 = cnts.graphs[1]
        # Check that the first graph does not have sigmoid. sigmoid is used in
        # both hook and disabled module.
        self.assertTrue(
            all(node.target is not torch.sigmoid for node in gm1.graph.nodes)
        )

    def test_allow_in_graph(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        torch._dynamo.allow_in_graph(my_custom_function)
        fn(torch.randn(10))
        torch._dynamo.disallow_in_graph(my_custom_function)

        # check for no graph break
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_allow_in_graph_no_id_reuse(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def do_allow_in_graph(x):
            return x + 1

        torch._dynamo.allow_in_graph(do_allow_in_graph)
        del do_allow_in_graph

        # `id(dont_allow_in_graph)` would likely match `id(do_allow_in_graph)`
        # We want to make sure Dynamo always trace through
        # `dont_allow_in_graph`, by checking for the explicit graph break.
        def dont_allow_in_graph(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend=cnts)
        def fn(a):
            x = torch.add(a, 1)
            x = torch.add(x, 1)
            x = dont_allow_in_graph(x)
            x = torch.add(x, 1)
            x = torch.add(x, 1)
            return x

        fn(torch.randn(10))

        # Check for graph break
        self.assertEqual(cnts.frame_count, 3)

    def test_incorrect_usage_disallow_in_graph(self):
        with self.assertRaises(IncorrectUsage):

            @torch._dynamo.disallow_in_graph
            def fn1(x):
                return x.cos()

    def test_nonstrict_trace_tensor_args(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x, y, z):
            torch._dynamo.graph_break()
            return x * y + z

        def fn(x, y):
            t0 = x + 1
            t1 = trace_me(x, y, t0)
            t2 = t1 + y
            return t0 * t2

        x, y = torch.randn(10), torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            t0 = trace_me(x, d)
            return t0 + 1

        x = torch.randn(10)
        d = {"a": 2}
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d)
        res = opt_fn(x, d)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_newly_constructed_dict_with_side_effects(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x):
            d = {}
            d["a"] = 2
            t0 = trace_me(x, d)
            return t0 + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_dict_with_side_effects(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x, d):
            torch._dynamo.graph_break()
            return x * d["a"]

        def fn(x, d):
            d["a"] = x + 1
            t0 = trace_me(x, d)
            return t0 + 2

        x = torch.randn(10)
        d0 = {"a": 0}
        d1 = dict(d0)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, d0)
        res = opt_fn(x, d1)
        self.assertEqual(ref, res)
        self.assertEqual(d0, d1)

    def test_nonstrict_trace_pre_existing_custom_class(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            res = trace_me(p)
            return res, p.x, p.y

        p = Point(torch.ones(10), torch.ones(1))
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p)
        res = opt_fn(p)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_custom_class_with_side_effects(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(p):
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        p1 = Point(torch.ones(10), torch.ones(1))
        p2 = Point(torch.ones(10), torch.ones(1))
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(p1)
        res = opt_fn(p2)
        self.assertEqual(ref, res)
        self.assertEqual(p1.x, p2.x)
        self.assertEqual(p1.y, p2.y)

    def test_nonstrict_trace_newly_constructed_custom_class_with_side_effects(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        def fn(x, y):
            p = Point(x, y)
            p.x = p.x + 1
            p.y = p.y + 2
            res = trace_me(p)
            return res, p.x, p.y

        x, y = torch.ones(10), torch.ones(1)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_nested_custom_class(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        self.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
            serialized_type_name=f"{Point.__module__}.{Point.__qualname__}",
        )

        def trace_point(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        x, y = torch.ones(10), torch.ones(1)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_pre_existing_register_constant_type_guard(self):
        class State:
            def __init__(self, n):
                self.n = n

            def get_num(self):
                torch._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

            def __repr__(self):
                return f"State({self.n})"

            def __fx_repr__(self):
                return f"State({self.n})", {"State": State}

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `torch.compile`.
        torch._library.opaque_object.register_opaque_type(State, typ="value")

        @torch._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")

        @torch.compile(fullgraph=True, backend=cnts)
        def fn(x, s):
            res = trace_me(x, s)
            return res

        x = torch.ones(10)
        # Make sure recompilation didn't happen.
        self.assertEqual(cnts.frame_count, 0)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)
        fn(x, State(42))
        self.assertEqual(cnts.frame_count, 1)

        # Make sure recompilation did happen.
        fn(x, State(41))
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_int_and_float_output(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return len(x.shape), 0.42

        def fn(x):
            n1, n2 = trace_me(x)
            return x * n1 + n2

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_tuple_and_sym_int_output(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 1, x.size(0)

        def fn(x):
            t0, n = trace_me(x)
            return t0 * n

        x = torch.randn(10)
        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function(self):
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = torch._dynamo.nonstrict_trace(trace_me)(x)
            return res + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_inside_compiled_function_kwarg(self):
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 42

        def fn(x):
            res = torch._dynamo.nonstrict_trace(traceable_fn=trace_me)(x)
            return res + 1

        x = torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_on_method(self):
        class Num:
            def __init__(self, n):
                self.n = n

            @torch._dynamo.nonstrict_trace
            def trace_me(self, t):
                torch._dynamo.graph_break()
                return t + self.n

        self.register_pytree_node(
            Num,
            lambda num: ((num.n,), ()),
            lambda n, _: Num(n[0]),
            serialized_type_name=f"{Num.__module__}.{Num.__qualname__}",
        )

        def fn(x, n):
            num = Num(n)
            return num.trace_me(x)

        x, n = torch.randn(10), 42
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, n)
        res = opt_fn(x, n)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_captured_external_tensor(self):
        cst = torch.ones(1)

        @torch._dynamo.nonstrict_trace
        def trace_me(x, y):
            torch._dynamo.graph_break()
            return x * y + cst

        def fn(x, y):
            return trace_me(x, y)

        x, y = torch.randn(10), torch.randn(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nonstrict_trace_no_action_at_a_distance(self):
        def trace_me(x):
            torch._dynamo.graph_break()
            return x + 42

        # No effect on traceability of `trace_me`
        torch._dynamo.nonstrict_trace(trace_me)

        def fn(x):
            res = trace_me(x)
            return res + 1

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        opt_fn = torch.compile(fn, backend=cnts)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        # There should be 1 graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_nonstrict_trace_inside_compiled_function_error(self):
        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            def trace_me(x, y):
                torch._dynamo.graph_break()
                return x * y

            res = torch._dynamo.nonstrict_trace(trace_me)(x, y)
            return res + 1

        try:
            fn(torch.ones(10), torch.ones(1))
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            msg = "Applying `nonstrict_trace` to function <trace_me>; however, `nonstrict_trace` currently requires the function to be defined outside `torch.compile` region."  # NOQA: B950
            self.assertIn(msg, str(e))

    def test_nonstrict_trace_custom_class_error(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @torch._dynamo.nonstrict_trace
        def trace_me(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(p):
            res = trace_me(p)
            return res + 1

        try:
            p = Point(torch.ones(10), torch.ones(1))
            fn(p)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_nested_custom_class_error(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        def trace_point(p):
            torch._dynamo.graph_break()
            return p.x * p.y

        @torch._dynamo.nonstrict_trace
        def trace_point_tensor(pt):
            torch._dynamo.graph_break()
            return pt.t + trace_point(pt.p)

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        try:
            fn(torch.ones(10), torch.ones(1))
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn("Invalid input type for nonstrict_trace-ed function", str(e))

    def test_nonstrict_trace_custom_class_output_error(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        @torch._dynamo.nonstrict_trace
        def trace_me(x):
            torch._dynamo.graph_break()
            return Point(x, x + 1)

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            p = trace_me(x)
            return p.x * p.y

        try:
            x = torch.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Unsupported output type for nonstrict_trace-ed function", str(e)
            )

    def test_nonstrict_newly_constructed_trace_register_constant_type_error(self):
        class State:
            def __init__(self, n):
                self.n = n

            def get_num(self):
                torch._dynamo.graph_break()
                return self.n

            def __eq__(self, other):
                return isinstance(other, State) and self.n == other.n

            def __hash__(self):
                return hash(self.n)

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `torch.compile`.
        torch._library.opaque_object.register_opaque_type(State, typ="reference")

        @torch._dynamo.nonstrict_trace
        def trace_me(x, s):
            return x * s.get_num()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x):
            s = State(10)
            res = trace_me(x, s)
            return res

        try:
            x = torch.ones(10)
            fn(x)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "An opaque object was created in the middle of the program.",
                str(e),
            )

    def test_nonstrict_trace_object_in_context_error(self):
        class Point:
            x: torch.Tensor
            y: torch.Tensor

            def __init__(self, x, y):
                self.x = x
                self.y = y

        class PointTensor:
            p: Point
            t: torch.Tensor

            def __init__(self, p, t):
                self.p = p
                self.t = t

        self.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.t,), pt.p),
            lambda ts, p: PointTensor(p, ts[0]),
            serialized_type_name=f"{PointTensor.__module__}.{PointTensor.__qualname__}",
        )

        @torch._dynamo.nonstrict_trace
        def trace_me(pt):
            torch._dynamo.graph_break()
            return pt.t + pt.p.x * pt.p.y

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_me(pt)
            return res

        try:
            x, y = torch.ones(10), torch.ones(1)
            fn(x, y)
            self.assertFalse(True)  # must raise error before this
        except torch._dynamo.exc.Unsupported as e:
            self.assertIn(
                "Invalid use of pytree_flatten with nonstrict_trace-ed function", str(e)
            )

    def test_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            torch._dynamo.graph_break()
            x = torch.cos(x)
            x = torch.cos(x)
            return x

        fn(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_skip_frame(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            torch._dynamo.skip_frame()
            return x + 1

        inp = torch.ones(3, 3)
        self.assertEqual(fn(inp), inp + 2)
        self.assertEqual(cnts.frame_count, 0)

        @torch.compile(backend=cnts)
        def gn(x):
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 1
            torch._dynamo.skip_frame()
            return x + 1

        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 1)

    def test_step_unsupported(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1 + 2
            torch._dynamo.step_unsupported()
            return x + 4

        inp = torch.ones(3)
        self.assertEqual(fn(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_step_unsupported_empty_checkpoint(self):
        @torch.compile(backend="eager")
        def fn(x):
            torch._dynamo.step_unsupported()
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp), inp + 1)

    @skipIfWindows(
        msg="TODO: (xuhancn), confirm if torch.compiler.disable work on Windows."
    )
    def test_disable_recursive_false(self):
        def fn2(x):
            return x + 1

        @torch._dynamo.disable(recursive=False)
        def fn1(x):
            if torch.compiler.is_compiling():
                raise RuntimeError("bad")
            x = x.sigmoid()
            return fn2(x.cos())

        def fn(x):
            return fn1(x.tan())

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4))
        self.assertEqual(cnts.frame_count, 2)

        # test that applying disable nonrecursive doesn't modify the original function
        def fn3(x):
            if torch.compiler.is_compiling():
                return x - 1
            return fn2(x) + 2

        @torch.compile(backend=cnts)
        def outer(f, x):
            return f(x)

        inp = torch.ones(3)
        fn3_disabled = torch._dynamo.disable(fn3, recursive=False)

        torch._dynamo.reset()

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        torch._dynamo.reset()

        cnts.clear()
        res = outer(fn3_disabled, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp + 3)

        cnts.clear()
        res = outer(fn3, inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

        # directly compiling a disabled function should result in a compile
        torch._dynamo.reset()
        cnts.clear()
        res = torch.compile(fn3_disabled, backend=cnts)(inp)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, inp - 1)

    def test_disable_recursive_false_weird(self):
        from torch._dynamo.types import FrameAction, FrameExecStrategy

        # test the case where the next invocation of the function is
        # manually skipped
        def fn(x):
            if torch.compiler.is_compiling():
                return x - 1
            return x + 1

        fn_disabled = torch._dynamo.disable(fn, recursive=False)

        torch._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
        )

        @torch.compile(backend="eager")
        def outer(fn, x):
            return fn(x)

        inp = torch.ones(3)
        self.assertEqual(outer(fn_disabled, inp), inp + 1)

        torch._dynamo.eval_frame.set_code_exec_strategy(
            fn.__code__, FrameExecStrategy(FrameAction.DEFAULT, FrameAction.DEFAULT)
        )

        self.assertEqual(torch.compile(fn, backend="eager")(inp), inp - 1)

    def test_substitute_in_graph(self):
        counters.clear()

        # NB: Choose another C function for test when we support operator.indexOf
        #     out of the box
        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 1)

        torch._dynamo.reset()
        counters.clear()

        with self.assertRaisesRegex(TypeError, "Signature mismatch"):

            @torch._dynamo.substitute_in_graph(operator.indexOf)
            def _(sequence, x):
                for i, item in enumerate(sequence):
                    if item is x or item == x:
                        return i
                raise ValueError("sequence.index(x): x not in sequence")

        @torch._dynamo.substitute_in_graph(operator.indexOf)
        def polyfill(a, b):
            for i, item in enumerate(a):
                if item is b or item == b:
                    return i
            raise ValueError("sequence.index(x): x not in sequence")

        cnts = torch._dynamo.testing.CompileCounter()
        fn = operator.indexOf
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

        torch._dynamo.reset()
        counters.clear()

        cnts = torch._dynamo.testing.CompileCounter()
        fn = polyfill
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        out = fn([1, 2, 3, 4, 5], 3)
        opt_out = opt_fn([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, opt_out)
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(len(counters["graph_break"]), 0)

    @patch.object(torch._dynamo.config, "suppress_errors", True)
    def test_nested_disable_decorator(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.disable()
        def fn1(x):
            return torch.sin(x) * 10

        @torch.compile(backend=cnts)
        def fn2(x):
            x = x + 1
            x = x + 1
            x = fn1(x)  # graph break
            x = x + 1
            x = x + 1
            return x

        @torch.compile(backend=cnts, fullgraph=True)
        def fn3(x):
            return fn2(x)

        fn2(torch.randn(4, 5))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with self.assertRaisesRegex(
            Unsupported, r"Skip calling `torch.compiler.disable\(\)`d function"
        ):
            fn3(torch.randn(4, 5))

    def test_disable_optimize(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, disable=True)
        def f1(x):
            return x + 1

        f1(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        @torch.compile(backend=cnt, disable=True)
        def f2(x):
            return x + 1

        f2(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

        with patch.dict(os.environ, {"TORCHDYNAMO_DISABLE": "1"}):

            @torch.compile(backend=cnt)
            def f3(x):
                return x + 1

            f3(torch.ones(6))
        self.assertEqual(cnt.frame_count, 0)

    def test_torch_guards_stack_frame_register_inlining_disable(self):
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.a = y

            @torch._dynamo.disable
            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.a)

        e = encoder(2.0)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(e, backend="eager")(x)

        self.assertEqual(len(seen_frames), 0)

    def test_torch_guards_stack_frame_register_inlining_partially_disable(self):
        y = torch.nn.Parameter(torch.tensor([0.25, 0.25]))
        x = torch.tensor([0.5, 0.5])

        class encoder(torch.nn.Module):
            def __init__(self, y):
                super().__init__()
                self.register_parameter("param", y)

            @torch._dynamo.disable
            def helper_disabled(self, x, y):
                return x.sin() * y.cos()

            def helper(self, x, y):
                return x * y

            def forward(self, a, *args):
                x = a + a
                return self.helper(x, self.param) + self.helper_disabled(x, self.param)

        e = encoder(y)

        cnt = torch._dynamo.testing.CompileCounter()
        torch.compile(e, backend=cnt)(x)

        # first frame is before disable, second frame is after disable
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 3)

    def _test_mark_static_address(self, guarded):
        # This test verifies that dynamo properly marks inputs as static
        # when using the mark_static_address API.
        # For both inline_inbuilt_nn_modules True and False, we expect the
        # tensor to be present in the buffers attribute of the graph.

        compiles_with_buffers = 0
        compiles = 0

        def debug_compiler(gm, _):
            nonlocal compiles_with_buffers
            nonlocal compiles
            compiles_with_buffers += len(gm._buffers) > 0
            compiles += 1
            return gm

        @torch.compile(backend=debug_compiler)
        def fn(x):
            return x + 1

        inp = torch.ones(2)

        torch._dynamo.mark_static_address(inp, guard=guarded)

        fn(inp)
        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        inp2 = torch.ones(2)

        # if guarded, should trigger another recompile
        # since it was not marked static, compiles with buffers
        # should not be incremented
        fn(inp2)

        if guarded:
            self.assertEqual(compiles_with_buffers, 1)

        self.assertEqual(compiles, 2 if guarded else 1)

    def test_mark_static_address_guarded(self):
        with torch._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=True)

        self._test_mark_static_address(guarded=True)

    def test_mark_static_address_unguarded(self):
        with torch._dynamo.config.patch("inline_inbuilt_nn_modules", True):
            self._test_mark_static_address(guarded=False)

        self._test_mark_static_address(guarded=False)

    def test_class_methods(self):
        class A:
            @classmethod
            def my_class_method(cls, arg1):
                return cls, arg1

            @staticmethod
            def my_static_method(arg1):
                return None, arg1

            def my_regular_method(self, arg1):
                return self, arg1

        class B(A):
            def my_class_method(self, arg1):
                return super().my_class_method(arg1)

            def my_static_method(self, arg1):
                return super().my_static_method(arg1)

        class C(A):
            @classmethod
            def my_class_method(cls, arg1):
                return super().my_class_method(arg1)

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(a, b, c):
            # We want a function that does not graph break but
            # does generate custom bytecode
            v1 = a.my_class_method(1)
            v2 = A.my_class_method(2)
            v3 = a.my_static_method(3)
            v4 = A.my_static_method(4)
            v5 = a.my_regular_method(5)
            v6 = b.my_class_method(6)
            v7 = b.my_static_method(7)
            v8 = c.my_class_method(8)
            v9 = C.my_class_method(9)
            torch.rand(2)
            return v1, v2, v3, v4, v5, v6, v7, v8, v9

        a, b, c = A(), B(), C()
        v1, v2, v3, v4, v5, _, v7, v8, v9 = fn(a, b, c)

        self.assertEqual(v1, (A, 1))
        self.assertEqual(v2, (A, 2))
        self.assertEqual(v3, (None, 3))
        self.assertEqual(v4, (None, 4))
        self.assertEqual(v5, (a, 5))
        # TODO fix me: we do not resolve classmethods properly
        # from a regular method
        # self.assertEqual(v6, (B, 6))
        self.assertEqual(v7, (None, 7))
        self.assertEqual(v8, (C, 8))
        self.assertEqual(v9, (C, 9))

        self.assertEqual(cnt.frame_count, 1)

    def test_assume_constant_result_on_user_defined_fn(self):
        @torch._dynamo.assume_constant_result
        def const_fn(n, s):
            return torch.full([n], s)

        def fn(B):
            B = const_fn(B.size(0), 13)
            X = B * 2
            return X.tolist()

        B_list = [8] * 32

        B = torch.tensor(B_list, dtype=torch.int32)
        torch._dynamo.decorators.mark_static(B, 0)

        with torch._dynamo.config.patch(
            capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
        ):
            self.assertEqual(
                fn(B),
                torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B),
            )

    def test_assume_constant_result_on_computation_with_graph_input(self):
        @torch._dynamo.assume_constant_result
        def check(y):
            return y[0].item() == 1

        def fn(x, y):
            if check(y):
                return x + 2
            else:
                return x + 1

        y = torch.tensor([1])
        x = torch.tensor(1)

        self.assertEqual(fn(x, y), torch.compile(fn)(x, y))

    def test_set_stance_aot_eager_then_compile(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with torch.compiler.set_stance("aot_eager_then_compile"):
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})
            fn(4, torch.randn(4), {0: torch.randn(4)})

        # Would have been 4 without stance
        self.assertEqual(cnts.op_count, 2)

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_nn_module(self):
        @torch._dynamo.mark_static
        class Mock(torch.nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c

            def forward(self, x):
                return x * self.c

        cnts = torch._dynamo.testing.CompileCounter()
        mod1 = Mock(10)
        mod2 = Mock(20)
        mod3 = Mock(30)
        opt_mod1 = torch.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = torch.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = torch.compile(mod3, backend=cnts, fullgraph=True)

        x = torch.randn(4, 4)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because self.c would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    def test_set_stance_eager_then_compile(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            return x * y * z[0]

        with torch.compiler.set_stance("eager_then_compile"):
            fn(1, torch.randn(1), {0: torch.randn(1)})
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})

        self.assertEqual(cnts.frame_count, 1)

    def test_set_stance_eager_then_compile_with_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x, y, z):
            y = torch.sin(y)
            torch._dynamo.graph_break()
            y = torch.cos(y)
            return x * y * z[0]

        with torch.compiler.set_stance("eager_then_compile"):
            fn(1, torch.randn(1), {0: torch.randn(1)})
            fn(2, torch.randn(2), {0: torch.randn(2)})
            fn(3, torch.randn(3), {0: torch.randn(3)})

        # frame count 2 since we added a graph break
        self.assertEqual(cnts.frame_count, 2)

    def test_set_stance_force_eager(self):
        @torch.compile(backend="eager")
        def a(x):
            if torch._dynamo.is_compiling():
                return x + 1
            return x + 2

        @torch.compiler.set_stance("force_eager")
        def b(x):
            return a(x)

        def c(x):
            out0 = a(x)
            with torch.compiler.set_stance("force_eager"):
                out1 = a(x)
            return out0, out1, a(x)

        inp = torch.ones(3)
        # test that decorating b has no overall side effect
        self.assertEqual(a(inp), inp + 1)

        self.assertEqual(b(inp), inp + 2)
        self.assertEqual(c(inp), (inp + 1, inp + 2, inp + 1))

        torch.compiler.set_stance("force_eager")
        self.assertEqual(a(inp), inp + 2)
        torch.compiler.set_stance("default")
        self.assertEqual(a(inp), inp + 1)

    def test_set_stance_eager_on_recompile(self):
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("eager_on_recompile"):
            out2 = a(inp, 1)
            out3 = a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)
        self.assertEqual(out3, inp + 4)

    def test_set_stance_fail_on_recompile(self):
        @torch.compile(backend="eager", dynamic=False)
        def a(x, n):
            if torch._dynamo.is_compiling():
                return x + n + 1
            return x + n + 2

        inp = torch.ones(3)
        out1 = a(inp, 1)
        with torch.compiler.set_stance("fail_on_recompile"):
            out2 = a(inp, 1)
            with self.assertRaisesRegex(RuntimeError, "fail_on_recompile"):
                a(inp, 2)

        self.assertEqual(out1, inp + 2)
        self.assertEqual(out2, inp + 2)

    def test_fail_on_recompile_shows_guard_details(self):
        @torch.compile(backend="eager", dynamic=False)
        def f(x):
            return x + 1

        f(torch.ones(4))
        f(torch.ones(5))

        def post_munge(s):
            return re.sub(r"line number: \d+", "line number: N", s)

        with torch.compiler.set_stance("fail_on_recompile"):
            f(torch.ones(4))
            self.assertExpectedInlineMunged(
                RuntimeError,
                lambda: f(torch.ones(7)),
                """\
Detected recompile when torch.compile stance is 'fail_on_recompile'. filename: 'test_decorators.py', function name: 'f', line number: N
    triggered by the following guard failure(s):
    - 0/0: tensor 'x' size mismatch at index 0. expected 4, actual 7
    - 0/1: tensor 'x' size mismatch at index 0. expected 5, actual 7""",  # noqa: B950
                post_munge=post_munge,
            )

    def test_set_stance_fail_on_recompile_with_disable(self):
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))
        # should not raise error
        with torch.compiler.set_stance("fail_on_recompile"):
            f(torch.randn(3, 3))

    def test_set_stance_forbid_in_graph(self):
        @torch.compiler.set_stance("force_eager")
        def a(x):
            return x + 1

        @torch.compile(backend="eager")
        def b(x):
            return a(x)

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            b(torch.ones(3))

        @torch.compile(backend="eager")
        def c(x):
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            c(torch.ones(3))

        @torch.compile(backend="eager")
        @torch.compiler.set_stance("force_eager")
        def d(x):
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            d(torch.ones(3))

        @torch.compile(backend="eager")
        def e(x):
            with torch._dynamo.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            e(torch.ones(3))

        @torch.compile(backend="eager")
        def f(x):
            torch._dynamo.eval_frame._set_stance("force_eager")
            return x + 1

        with self.assertRaisesRegex(
            AssertionError, "Attempt to trace forbidden callable"
        ):
            f(torch.ones(3))

        @torch.compile(backend="eager")
        def g(x):
            torch._dynamo.skip_frame()
            # NOTE: torch._dynamo.is_compiling() will get traced
            # and return true. torch.compiler.is_compiling() is skipped
            # and will return false.
            if torch.compiler.is_compiling():
                raise RuntimeError("Expect this frame to be skipped")
            # should not be traced, but eval frame callback is still set
            with torch.compiler.set_stance("force_eager"):
                return x + 1

        with self.assertRaisesRegex(RuntimeError, "set_stance in a torch.compile"):
            g(torch.ones(3))

    def test_set_stance_force_backend(self):
        @torch.compile
        def a(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compiler.set_stance("default", force_backend=cnts)
        def b(x):
            return a(x)

        b(torch.ones(3))

        self.assertEqual(cnts.frame_count, 1)

        @torch.compiler.set_stance("default", force_backend="eager")
        def c(x):
            return a(x)

        # just make sure this doesn't crash
        c(torch.ones(3))

        with self.assertRaisesRegex(RuntimeError, "force_backend"):

            @torch.compiler.set_stance("force_eager", force_backend="eager")
            def d(x):
                pass

    def test_set_stance_force_backend_with_disable(self):
        @torch.compiler.disable
        def inner(x):
            return x

        @torch.compile(backend="eager")
        def f(x):
            return inner(x)

        f(torch.randn(3, 3))

        def fail_backend(gm, ex):
            raise RuntimeError("fail!")

        # should not raise error
        with torch.compiler.set_stance("default", force_backend=fail_backend):
            f(torch.randn(3, 3))

    # also tests a lot of torch._dynamo.patch_dynamo_config functionality
    def test_dont_skip_tracing(self):
        from torch._dynamo.test_dont_skip_tracing_functions import f1, f3, f4, f5, f6

        cnts = torch._dynamo.testing.CompileCounter()

        # make sure test_dont_skip_tracing_functions is actually skipped by trace rules
        torch.compile(f1, backend=cnts)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 0)

        f1_unskip = torch._dynamo.dont_skip_tracing(f1)

        # basic test
        def g1(x):
            return f1_unskip(x)

        cnts.clear()
        torch.compile(g1, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is traceable
        def g2(x):
            return torch._dynamo.dont_skip_tracing(f1)(x)

        cnts.clear()
        torch.compile(g2, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to non-skipped function
        @torch._dynamo.dont_skip_tracing
        def g3(x):
            return f1(x)

        cnts.clear()
        torch.compile(g3, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test that dont_skip_tracing is recursive, applied to skipped function
        f3_unskip = torch._dynamo.dont_skip_tracing(f3)
        cnts.clear()
        torch.compile(f3_unskip, backend=cnts, fullgraph=True)(torch.randn(3))
        self.assertEqual(cnts.frame_count, 1)

        # test dont_skip_tracing with graph breaks
        inp = torch.ones(3)
        res = torch.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 6)

        @torch.compile(backend=cnts)
        def g4(x):
            x = f5(x, 1)
            x = torch._dynamo.dont_skip_tracing(f6)(x)
            x = f5(x, 8)
            return x

        res = g4(inp)
        self.assertEqual(res, inp + 6)

        # test nested dont_skip_tracing
        # this also happens to test if a previously skipped frame (f4)
        # can actually be compiled if called as a top-level function (in the case of a graph break)
        # TODO the reset is necessary for now since attempting to trace f4 previously
        # resulted in an unconditional skip
        torch._dynamo.reset()
        f4_unskip = torch._dynamo.dont_skip_tracing(f4)
        res = torch.compile(f4_unskip, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test dont_skip_tracing that is activated outside torch.compile
        f4_unskip2 = torch._dynamo.dont_skip_tracing(torch.compile(f4, backend=cnts))
        res = f4_unskip2(inp)
        self.assertEqual(res, inp + 15)

        # test context manager from inside
        @torch.compile(backend=cnts)
        def g5(x):
            x = f5(x, 1)
            with torch._dynamo.dont_skip_tracing():
                x = f5(x, 2)
                torch._dynamo.graph_break()
                x = f5(x, 4)
            x = f5(x, 8)
            return x

        res = g5(inp)
        self.assertEqual(res, inp + 6)

        # test context manager from outside
        with torch._dynamo.dont_skip_tracing():
            res = torch.compile(f4, backend=cnts)(inp)
        self.assertEqual(res, inp + 15)

        # test skipped function from different dont_skip_tracing regions
        @torch.compile(backend=cnts)
        def g6(x):
            fn1 = f5
            with torch._dynamo.dont_skip_tracing():
                fn2 = f5
                x = fn1(x, 1)
            x = fn2(x, 2)
            return x

        res = g6(inp)
        self.assertEqual(res, inp + 1)

    def test_patch_dynamo_config_errors(self):
        @torch.compile(backend="eager")
        def f1(x):
            with torch._dynamo.patch_dynamo_config(nonexistent=False):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f1(torch.randn(3))

        @torch.compile(backend="eager")
        def f2(x):
            with torch._dynamo.patch_dynamo_config("verbose", {"a": 1}):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "patch_dynamo_config does not support .* with non-safe-constant"
        ):
            f2(torch.randn(3))

        @torch.compile(backend="eager")
        def f3(x):
            with torch._dynamo.patch_dynamo_config({"recompile_limit": 1}):
                return x + 1

        with self.assertRaisesRegex(Exception, "patch_dynamo_config does not support"):
            f3(torch.randn(3))

        @torch.compile(backend="eager")
        def f4(x):
            with torch._dynamo.patch_dynamo_config(verbose=object()):
                return x + 1

        with self.assertRaisesRegex(
            Exception, "Cannot convert patch_dynamo_config args/kwargs to constants."
        ):
            f4(torch.randn(3))

    def test_error_on_graph_break(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f1(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 2

        inp = torch.ones(3)
        self.assertEqual(f1(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

        @torch.compile(backend=cnts)
        def f2(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f2(inp)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f3(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
                x = x + 2
                torch._dynamo.graph_break()
            return x + 4

        cnts.clear()
        self.assertEqual(f3(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner_f4(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f4(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.skip_frame()
                return inner_f4(x)

        cnts.clear()
        self.assertEqual(f4(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

    def test_error_on_graph_break_nested(self):
        # error_on_graph_break in a nested frame
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner_f5(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f5(x):
            x = x + 1
            return inner_f5(x)

        inp = torch.ones(3)
        self.assertEqual(f5(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner_f6(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f6(x):
            x = x + 1
            return inner_f6(x)

        cnts.clear()
        self.assertEqual(f6(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 3)

        def inner_f7(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f7(x):
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(Unsupported):
            f7(inp)

    def test_error_on_graph_break_nested_with_skip(self):
        # error_on_graph_break in a nested frame with a skipped frame in between
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner2_f8(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        def inner1_f8(x):
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.skip_frame()
            return inner2_f8(x)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f8(x):
            x = x + 1
            return inner1_f8(x)

        inp = torch.ones(3)
        self.assertEqual(f8(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner2_f9(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.disable(recursive=False)
        def inner1_f9(x):
            return inner2_f9(x)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f9(x):
            x = x + 1
            return inner1_f9(x)

        with self.assertRaises(Unsupported):
            f9(inp)

        # test export with error_on_graph_break(False) still errors

    def test_error_on_graph_break_export(self):
        @torch._dynamo.error_on_graph_break(False)
        def inner(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        def f(x):
            x = x + 1
            return inner(x)

        with self.assertRaises(Unsupported):
            torch._dynamo.export(f)(torch.ones(3))

    def test_error_on_graph_break_nested_deep(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def inner1_f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def inner2_f1(x):
            return inner1_f1(x)

        def inner3_f1(x):
            with torch._dynamo.error_on_graph_break(False):
                return inner2_f1(x)

        def inner4_f1(x):
            return inner3_f1(x)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f1(x):
            x = x + 4
            return inner4_f1(x)

        inp = torch.ones(3)
        self.assertEqual(f1(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 4)

        def inner1_f2(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def inner2_f2(x):
            return inner1_f2(x)

        def inner3_f2(x):
            with torch._dynamo.error_on_graph_break(True):
                return inner2_f2(x)

        def inner4_f2(x):
            return inner3_f2(x)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f2(x):
            x = x + 4
            return inner4_f2(x)

        with self.assertRaises(Unsupported):
            f2(inp)

    def test_error_on_graph_break_error(self):
        @torch.compile(backend="eager")
        def f1():
            with torch._dynamo.error_on_graph_break(foo="bar"):
                pass

        @torch.compile(backend="eager")
        def f2():
            with torch._dynamo.error_on_graph_break():
                pass

        @torch.compile(backend="eager")
        def f3():
            with torch._dynamo.error_on_graph_break("foo"):
                pass

        with self.assertRaises(Exception):
            f1()
        with self.assertRaises(Exception):
            f2()
        with self.assertRaises(Exception):
            f3()

    def test_nested_compile_error_on_graph_break(self):
        inp = torch.ones(3)

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def inner_f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend="eager")
        def f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            f1(inp)

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend="eager")
        def inner_f2(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def f2(x):
            return inner_f2(x)

        self.assertEqual(f2(inp), inp + 3)

    def test_error_on_graph_break_fullgraph(self):
        # Test that error_on_graph_break=False cannot override fullgraph=True
        inp = torch.ones(3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            x = x + 1
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 2

        with self.assertRaises(Unsupported):
            f(inp)

    def test_error_on_graph_break_empty_graph(self):
        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="eager")
        def f():
            return 1

        self.assertEqual(f(), 1)

    def test_error_on_graph_break_nonempty_checkpoint(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            x = x + 1
            x = x + 1
            x = x + 1
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 1

        with self.assertRaises(Unsupported):
            fn(torch.ones(3))

        self.assertEqual(cnts.frame_count, 0)

    def test_nested_compile_fullgraph(self):
        # Test that fullgraph=True cannot be toggled back by fullgraph=False
        inp = torch.ones(3)

        @torch.compile(backend="eager", fullgraph=True)
        def inner_f1(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend="eager", fullgraph=False)
        def outer_f1(x):
            return inner_f1(x)

        with self.assertRaises(Unsupported):
            outer_f1(inp)

        @torch.compile(backend="eager", fullgraph=False)
        def inner_f2(x):
            torch._dynamo.graph_break()
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def outer_f2(x):
            return inner_f2(x)

        with self.assertRaises(Unsupported):
            outer_f2(inp)

    def test_disable_recursive_flags(self):
        class SimpleLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                return self.layer0(torch.sigmoid(inp))

        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer0 = SimpleLinear()
                self.layer1 = torch.nn.Linear(4, 4)

            def forward(self, inp):
                z = self.layer0(torch.sin(inp))
                return self.layer1(z)

        for recursive_flag in [True, False]:
            model = SimpleModel()
            other_model = SimpleModel()

            model.forward = torch._dynamo.disable(
                model.forward,
                recursive=recursive_flag,
            )
            self.assertEqual(
                torch._dynamo.is_dynamo_disable_recursive(model.forward),
                recursive_flag,
            )

            other_model = torch._dynamo.disable(other_model, recursive=recursive_flag)
            self.assertEqual(
                torch._dynamo.is_dynamo_disable_recursive(
                    other_model.forward
                    if isinstance(other_model, torch.nn.Module)
                    else other_model
                ),
                recursive_flag,
            )

            # check the model is compilable
            torch.compile(model)
            torch.compile(other_model)

    def test_dynamo_disable_annotations(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("buffer", torch.rand(2, 2))

            @torch._dynamo.disable()
            def f1(self, x) -> torch.Tensor:
                return x + self.buffer + 1

            @torch._dynamo.disable()
            def f2(self, x) -> torch.Tensor:
                return x + self.buffer + 2

            def forward(self, x) -> torch.Tensor:
                return self.f1(x) + self.f2(x)

        model = SimpleModel()
        inp = torch.rand(2, 2)
        with torch.fx.traceback.preserve_node_meta():
            exported_model = torch.export.export(model, (inp,))
        graph = exported_model.graph_module.graph
        found_f1 = False
        found_f2 = False
        for node in graph.nodes:
            if "custom" in node.meta:
                if "_torchdynamo_disable_method" in node.meta["custom"]:
                    if node.meta["custom"]["_torchdynamo_disable_method"] == "f1":
                        found_f1 = True
                    elif node.meta["custom"]["_torchdynamo_disable_method"] == "f2":
                        found_f2 = True
        self.assertTrue(found_f1)
        self.assertTrue(found_f2)
        model.forward = torch._dynamo.disable(model.forward, recursive=False)
        with self.assertRaises(RuntimeError):
            exported_model = torch.export.export(model, (inp,))

    def _assert_models_equal(
        self,
        model_expected,
        model_test,
        x_expected,
        x_test,
    ):
        """
        Helper to compare two models' forward outputs and gradients.

        Args:
            model_expected: The reference model (typically eager)
            model_test: The model to test (typically compiled)
            x_expected: Input tensor for expected model
            x_test: Input tensor for test model (should be cloned from x_expected)
        """
        # Check forward results match
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        # Check gradients match
        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

        # Check parameter gradients match
        expected_grads = {
            name: param.grad for name, param in model_expected.named_parameters()
        }
        test_grads = {name: param.grad for name, param in model_test.named_parameters()}

        self.assertEqual(set(expected_grads.keys()), set(test_grads.keys()))
        for name in expected_grads:
            if expected_grads[name] is not None:
                self.assertEqual(
                    expected_grads[name],
                    test_grads[name],
                    msg=f"Gradient mismatch for parameter {name}",
                )

    def _test_leaf_function_helper(self, mod_class, args_fn, loss_fn):
        import torch.utils._pytree as pytree
        from torch._dynamo.testing import AotEagerAndRecordGraphs, EagerAndRecordGraphs

        mod_eager = mod_class()
        mod_compile_eager = mod_class()
        mod_compile_eager.load_state_dict(dict(mod_eager.state_dict()))
        mod_compile_aot = mod_class()
        mod_compile_aot.load_state_dict(dict(mod_eager.state_dict()))

        args = args_fn()
        args_clone = pytree.tree_map(
            lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
        )
        args_clone2 = pytree.tree_map(
            lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
        )

        out_eager = mod_eager(*args)
        loss_fn(out_eager).backward()

        eager_backend = EagerAndRecordGraphs()
        out_compile_eager = torch.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )(*args_clone)
        loss_fn(out_compile_eager).backward()

        backend = AotEagerAndRecordGraphs()
        out_compile_aot = torch.compile(
            mod_compile_aot, backend=backend, fullgraph=True
        )(*args_clone2)
        loss_fn(out_compile_aot).backward()

        self.assertEqual(out_eager, out_compile_eager)
        self.assertEqual(out_eager, out_compile_aot)

        for (name_eager, param_eager), (_, param_compile_eager), (
            _,
            param_compile_aot,
        ) in zip(
            mod_eager.named_parameters(),
            mod_compile_eager.named_parameters(),
            mod_compile_aot.named_parameters(),
        ):
            self.assertEqual(
                param_eager.grad,
                param_compile_eager.grad,
                msg=f"Gradient mismatch for {name_eager} between eager and compile_eager",
            )
            self.assertEqual(
                param_eager.grad,
                param_compile_aot.grad,
                msg=f"Gradient mismatch for {name_eager} between eager and compile_aot",
            )

        pytree.tree_map(
            lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
            if isinstance(x, torch.Tensor) and x.requires_grad
            else None,
            args,
            args_clone,
        )
        pytree.tree_map(
            lambda x, compile_x: self.assertEqual(x.grad, compile_x.grad)
            if isinstance(x, torch.Tensor) and x.requires_grad
            else None,
            args,
            args_clone2,
        )
        return (
            normalize_gm(eager_backend.graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
        )

    def test_leaf_function_simple(self):
        from torch._dynamo.decorators import leaf_function

        class NonTracable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return (self.linear(x),)
                else:
                    return (self.linear(x) + x,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            NonTracable, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_args_0_: "f32[3, 3]", L_fn_modules_linear_parameters_weight_: "f32[3, 3]", L_fn_modules_linear_parameters_bias_: "f32[3]"):
        l_args_0_ = L_args_0_
        l_fn_modules_linear_parameters_weight_ = L_fn_modules_linear_parameters_weight_
        l_fn_modules_linear_parameters_bias_ = L_fn_modules_linear_parameters_bias_

        real_fn : torch.utils._pytree.TreeSpec = self.real_fn
        fake_fn : torch.utils._pytree.TreeSpec = self.fake_fn
        forward_input_spec : torch.utils._pytree.TreeSpec = self.forward_input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, forward_input_spec, 0, l_fn_modules_linear_parameters_weight_, l_fn_modules_linear_parameters_bias_, l_args_0_);  real_fn = fake_fn = forward_input_spec = l_fn_modules_linear_parameters_weight_ = l_fn_modules_linear_parameters_bias_ = l_args_0_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, None, 0, primals_2, primals_3, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, None, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3, 3]" = invoke_leaf_function_1[3];  invoke_leaf_function_1 = None
        return (getitem_4, getitem_2, getitem_3)
""",  # noqa: B950
        )

    def test_leaf_function_in_inner_module(self):
        from torch._dynamo.decorators import leaf_function

        class NonTracable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return (self.linear(x),)
                else:
                    return (self.linear(x) + x,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class Simple(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.non_tracable = NonTracable()

            def forward(self, x):
                return self.non_tracable(self.linear(x))

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(outs: tuple[torch.Tensor, ...]) -> torch.Tensor:
            out = 0
            for o in outs:
                out += o.sum()
            return out

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            Simple, args_fn, loss_fn
        )

    def test_leaf_function_non_forward_method(self):
        """Test leaf_function decorator on a non-forward method of an nn.Module."""
        from torch._dynamo.decorators import leaf_function

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def custom_method(self, x):
                if x.sum() > 0:
                    return (self.linear(x),)
                return (self.linear(x) + x,)

            @custom_method.fake_impl
            def custom_method(self, x):
                return (self.linear(x),)

            def forward(self, x):
                return self.custom_method(x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            MyModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch.utils._pytree.TreeSpec = self.real_fn
        fake_fn : torch.utils._pytree.TreeSpec = self.fake_fn
        custom_method_input_spec : torch.utils._pytree.TreeSpec = self.custom_method_input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, custom_method_input_spec, 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = custom_method_input_spec = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, None, 0, primals_2, primals_3, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, None, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3, 3]" = invoke_leaf_function_1[3];  invoke_leaf_function_1 = None
        return (getitem_4, getitem_2, getitem_3)
""",  # noqa: B950
        )

    def test_leaf_function_with_logging(self):
        """Test annotated method containing logging/print statements."""
        from torch._dynamo.decorators import leaf_function

        class LoggingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                print("Processing input")
                return (self.linear(x),)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            # Called 3 times: eager, compile_eager, and compile_aot
            self.assertEqual(mock_print.call_count, 3)

    def test_leaf_function_global_and_closure_read_only(self):
        """Test leaf_function reading global variables (tensors and non-tensors) and closures."""
        from torch._dynamo.decorators import leaf_function

        GLOBAL_SCALE = 2.0
        GLOBAL_TENSOR = torch.tensor([1.0, 2.0, 3.0])

        class ClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = torch.nn.Parameter(torch.ones(3))

            @leaf_function
            def forward(self, x):
                out = self.linear(x) * GLOBAL_SCALE * self.scale
                out = out + GLOBAL_TENSOR + self.offset
                return (out,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x) + self.offset,)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            ClosureModule, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_args_0_: "f32[3, 3]", L_fn_parameters_offset_: "f32[3]", L_fn_modules_linear_parameters_weight_: "f32[3, 3]", L_fn_modules_linear_parameters_bias_: "f32[3]"):
        l_args_0_ = L_args_0_
        l_fn_parameters_offset_ = L_fn_parameters_offset_
        l_fn_modules_linear_parameters_weight_ = L_fn_modules_linear_parameters_weight_
        l_fn_modules_linear_parameters_bias_ = L_fn_modules_linear_parameters_bias_

        real_fn : torch.utils._pytree.TreeSpec = self.real_fn
        fake_fn : torch.utils._pytree.TreeSpec = self.fake_fn
        forward_input_spec : torch.utils._pytree.TreeSpec = self.forward_input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, forward_input_spec, 0, l_fn_parameters_offset_, l_fn_modules_linear_parameters_weight_, l_fn_modules_linear_parameters_bias_, l_args_0_);  real_fn = fake_fn = forward_input_spec = l_fn_parameters_offset_ = l_fn_modules_linear_parameters_weight_ = l_fn_modules_linear_parameters_bias_ = l_args_0_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, None, 0, primals_2, primals_3, primals_4, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_4 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, None, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3, 3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3]" = invoke_leaf_function_1[3]
        getitem_5: "f32[3, 3]" = invoke_leaf_function_1[4];  invoke_leaf_function_1 = None
        return (getitem_5, getitem_2, getitem_3, getitem_4)
""",  # noqa: B950
        )
        # self.assertEqual(GLOBAL_TENSOR, torch.tensor([4.0, 5.0, 6.0]))

    def test_leaf_function_pytree_inputs(self):
        """Test leaf_function with pytree (dict) inputs."""
        from torch._dynamo.decorators import leaf_function

        class PytreeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, inputs):
                if inputs["x"].sum() > 0:
                    return (self.linear(inputs["x"]), inputs["y"] + 1)
                return (self.linear(inputs["x"]) + inputs["y"], inputs["y"] - 1)

            @forward.fake_impl
            def forward(self, inputs):
                return (self.linear(inputs["x"]), inputs["y"])

        def args_fn():
            return (
                {
                    "x": torch.randn(3, 3, requires_grad=True),
                    "y": torch.randn(3, 3, requires_grad=True),
                },
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PytreeModule, args_fn, loss_fn)

    def test_leaf_function_nested_annotations(self):
        """Test nested annotations where an annotated method calls another annotated module."""
        from torch._dynamo.decorators import leaf_function

        class InnerLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                # Simple non-traceable logic without data-dependent control flow
                y = self.linear(x)
                return (y + x,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class OuterLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerLeaf()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                # The inner module's forward is also a leaf_function
                z = self.linear(x)
                return self.inner(z + x)

            @forward.fake_impl
            def forward(self, x):
                return self.inner(self.linear(x))

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            OuterLeaf, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_args_0_: "f32[3, 3]", L_fn_modules_inner_modules_linear_parameters_weight_: "f32[3, 3]", L_fn_modules_inner_modules_linear_parameters_bias_: "f32[3]", L_fn_modules_linear_parameters_weight_: "f32[3, 3]", L_fn_modules_linear_parameters_bias_: "f32[3]"):
        l_args_0_ = L_args_0_
        l_fn_modules_inner_modules_linear_parameters_weight_ = L_fn_modules_inner_modules_linear_parameters_weight_
        l_fn_modules_inner_modules_linear_parameters_bias_ = L_fn_modules_inner_modules_linear_parameters_bias_
        l_fn_modules_linear_parameters_weight_ = L_fn_modules_linear_parameters_weight_
        l_fn_modules_linear_parameters_bias_ = L_fn_modules_linear_parameters_bias_

        real_fn : torch.utils._pytree.TreeSpec = self.real_fn
        fake_fn : torch.utils._pytree.TreeSpec = self.fake_fn
        forward_input_spec : torch.utils._pytree.TreeSpec = self.forward_input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, forward_input_spec, 0, l_fn_modules_inner_modules_linear_parameters_weight_, l_fn_modules_inner_modules_linear_parameters_bias_, l_fn_modules_linear_parameters_weight_, l_fn_modules_linear_parameters_bias_, l_args_0_);  real_fn = fake_fn = forward_input_spec = l_fn_modules_inner_modules_linear_parameters_weight_ = l_fn_modules_inner_modules_linear_parameters_bias_ = l_fn_modules_linear_parameters_weight_ = l_fn_modules_linear_parameters_bias_ = l_args_0_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, None, 0, primals_2, primals_3, primals_4, primals_5, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_4 = primals_5 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, None, tangents_1);  _tree_spec_constant2 = _tree_spec_constant3 = tangents_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_1[1]
        getitem_3: "f32[3]" = invoke_leaf_function_1[2]
        getitem_4: "f32[3, 3]" = invoke_leaf_function_1[3]
        getitem_5: "f32[3]" = invoke_leaf_function_1[4]
        getitem_6: "f32[3, 3]" = invoke_leaf_function_1[5];  invoke_leaf_function_1 = None
        return (getitem_6, getitem_2, getitem_3, getitem_4, getitem_5)
""",  # noqa: B950
        )

    def test_leaf_function_sequential_module_list(self):
        """Test ModuleList where each module has leaf_function annotation."""
        from torch._dynamo.decorators import leaf_function

        class LeafModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return (self.linear(x),)
                return (self.linear(x) + x,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class SequentialLeafModules(torch.nn.Module):
            def __init__(self, n_modules=3):
                super().__init__()
                self.modules_list = torch.nn.ModuleList(
                    [LeafModule() for _ in range(n_modules)]
                )

            def forward(self, x):
                for mod in self.modules_list:
                    x = mod(x)[0]
                return (x,)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        dynamo_graph_str, fw_graph_str, bw_graph_str = self._test_leaf_function_helper(
            SequentialLeafModules, args_fn, loss_fn
        )
        self.assertExpectedInline(
            dynamo_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_modules_list_modules_0_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_modules_list_modules_0_modules_linear_parameters_bias_: "f32[3]", L_self_modules_modules_list_modules_1_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_modules_list_modules_1_modules_linear_parameters_bias_: "f32[3]", L_self_modules_modules_list_modules_2_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_modules_list_modules_2_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_modules_list_modules_0_modules_linear_parameters_weight_ = L_self_modules_modules_list_modules_0_modules_linear_parameters_weight_
        l_self_modules_modules_list_modules_0_modules_linear_parameters_bias_ = L_self_modules_modules_list_modules_0_modules_linear_parameters_bias_
        l_self_modules_modules_list_modules_1_modules_linear_parameters_weight_ = L_self_modules_modules_list_modules_1_modules_linear_parameters_weight_
        l_self_modules_modules_list_modules_1_modules_linear_parameters_bias_ = L_self_modules_modules_list_modules_1_modules_linear_parameters_bias_
        l_self_modules_modules_list_modules_2_modules_linear_parameters_weight_ = L_self_modules_modules_list_modules_2_modules_linear_parameters_weight_
        l_self_modules_modules_list_modules_2_modules_linear_parameters_bias_ = L_self_modules_modules_list_modules_2_modules_linear_parameters_bias_

        real_fn : torch.utils._pytree.TreeSpec = self.real_fn
        fake_fn : torch.utils._pytree.TreeSpec = self.fake_fn
        forward_input_spec : torch.utils._pytree.TreeSpec = self.forward_input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, forward_input_spec, 0, l_self_modules_modules_list_modules_0_modules_linear_parameters_weight_, l_self_modules_modules_list_modules_0_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = forward_input_spec = l_self_modules_modules_list_modules_0_modules_linear_parameters_weight_ = l_self_modules_modules_list_modules_0_modules_linear_parameters_bias_ = l_x_ = None
        x: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        real_fn_0 : torch.utils._pytree.TreeSpec = self.real_fn_0
        fake_fn_0 : torch.utils._pytree.TreeSpec = self.fake_fn_0
        forward_input_spec_0 : torch.utils._pytree.TreeSpec = self.forward_input_spec_0
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(real_fn_0, fake_fn_0, forward_input_spec_0, 1, l_self_modules_modules_list_modules_1_modules_linear_parameters_weight_, l_self_modules_modules_list_modules_1_modules_linear_parameters_bias_, x);  real_fn_0 = fake_fn_0 = forward_input_spec_0 = l_self_modules_modules_list_modules_1_modules_linear_parameters_weight_ = l_self_modules_modules_list_modules_1_modules_linear_parameters_bias_ = x = None
        x_1: "f32[3, 3]" = invoke_leaf_function_1[0];  invoke_leaf_function_1 = None
        real_fn_1 : torch.utils._pytree.TreeSpec = self.real_fn_1
        fake_fn_1 : torch.utils._pytree.TreeSpec = self.fake_fn_1
        forward_input_spec_1 : torch.utils._pytree.TreeSpec = self.forward_input_spec_1
        invoke_leaf_function_2 = torch.ops.higher_order.invoke_leaf_function(real_fn_1, fake_fn_1, forward_input_spec_1, 2, l_self_modules_modules_list_modules_2_modules_linear_parameters_weight_, l_self_modules_modules_list_modules_2_modules_linear_parameters_bias_, x_1);  real_fn_1 = fake_fn_1 = forward_input_spec_1 = l_self_modules_modules_list_modules_2_modules_linear_parameters_weight_ = l_self_modules_modules_list_modules_2_modules_linear_parameters_bias_ = x_1 = None
        x_2: "f32[3, 3]" = invoke_leaf_function_2[0];  invoke_leaf_function_2 = None
        return (x_2,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 3]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]", primals_6: "f32[3, 3]", primals_7: "f32[3]"):
        _tree_spec_constant0 = self._tree_spec_constant0
        _tree_spec_constant1 = self._tree_spec_constant1
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant0, _tree_spec_constant1, None, 0, primals_2, primals_3, primals_1);  _tree_spec_constant0 = _tree_spec_constant1 = primals_2 = primals_3 = primals_1 = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        _tree_spec_constant2 = self._tree_spec_constant2
        _tree_spec_constant3 = self._tree_spec_constant3
        invoke_leaf_function_1 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant2, _tree_spec_constant3, None, 1, primals_4, primals_5, getitem);  _tree_spec_constant2 = _tree_spec_constant3 = primals_4 = primals_5 = getitem = None
        getitem_1: "f32[3, 3]" = invoke_leaf_function_1[0];  invoke_leaf_function_1 = None
        _tree_spec_constant4 = self._tree_spec_constant4
        _tree_spec_constant5 = self._tree_spec_constant5
        invoke_leaf_function_2 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant4, _tree_spec_constant5, None, 2, primals_6, primals_7, getitem_1);  _tree_spec_constant4 = _tree_spec_constant5 = primals_6 = primals_7 = getitem_1 = None
        getitem_2: "f32[3, 3]" = invoke_leaf_function_2[0];  invoke_leaf_function_2 = None
        return (getitem_2,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]"):
        _tree_spec_constant6 = self._tree_spec_constant6
        _tree_spec_constant7 = self._tree_spec_constant7
        invoke_leaf_function_3 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant6, _tree_spec_constant7, None, tangents_1);  _tree_spec_constant6 = _tree_spec_constant7 = tangents_1 = None
        getitem_4: "f32[3, 3]" = invoke_leaf_function_3[1]
        getitem_5: "f32[3]" = invoke_leaf_function_3[2]
        getitem_6: "f32[3, 3]" = invoke_leaf_function_3[3];  invoke_leaf_function_3 = None
        _tree_spec_constant8 = self._tree_spec_constant8
        _tree_spec_constant9 = self._tree_spec_constant9
        invoke_leaf_function_4 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant8, _tree_spec_constant9, None, getitem_6);  _tree_spec_constant8 = _tree_spec_constant9 = getitem_6 = None
        getitem_8: "f32[3, 3]" = invoke_leaf_function_4[1]
        getitem_9: "f32[3]" = invoke_leaf_function_4[2]
        getitem_10: "f32[3, 3]" = invoke_leaf_function_4[3];  invoke_leaf_function_4 = None
        _tree_spec_constant10 = self._tree_spec_constant10
        _tree_spec_constant11 = self._tree_spec_constant11
        invoke_leaf_function_5 = torch.ops.higher_order.invoke_leaf_function(_tree_spec_constant10, _tree_spec_constant11, None, getitem_10);  _tree_spec_constant10 = _tree_spec_constant11 = getitem_10 = None
        getitem_12: "f32[3, 3]" = invoke_leaf_function_5[1]
        getitem_13: "f32[3]" = invoke_leaf_function_5[2]
        getitem_14: "f32[3, 3]" = invoke_leaf_function_5[3];  invoke_leaf_function_5 = None
        return (getitem_14, getitem_12, getitem_13, getitem_8, getitem_9, getitem_4, getitem_5)
""",  # noqa: B950
        )

    def test_leaf_function_data_dependent_nonzero(self):
        """Test leaf_function with nonzero() returning the indices tensor, wrapped by an outer module."""
        from torch._dynamo.decorators import leaf_function

        class NonzeroModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                out = self.linear(x)
                nonzero_indices = (out > 0).nonzero()
                return (out, nonzero_indices)

            @forward.fake_impl
            def forward(self, x):
                out = self.linear(x)
                return out, (out > 0).nonzero()

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = torch.nn.Linear(3, 3)
                self.nonzero_module = NonzeroModule()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                # Pre-process x before passing to NonzeroModule
                x = self.pre_linear(x)
                x = torch.relu(x)
                out, nonzero_indices = self.nonzero_module(x)
                # Use the nonzero indices to index into the output
                num_nonzero = nonzero_indices.shape[0]
                # Scale the output and add a value based on count of nonzero elements
                scaled_out = out * self.scale + num_nonzero
                return scaled_out, nonzero_indices

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(OuterModule, args_fn, loss_fn)

    def test_leaf_function_data_dependent_item(self):
        from torch._dynamo.decorators import leaf_function

        class ItemModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                out = self.linear(x)
                scalar_value = out.sum().item()
                return (out, scalar_value)

            @forward.fake_impl
            def forward(self, x):
                out = self.linear(x)
                return (out, out.sum().item())

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(ItemModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_multiple_compiled_submodules(self, backend):
        from torch._dynamo.decorators import leaf_function

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return (self.linear(x),)
                else:
                    return (self.linear(x) + x,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class CompiledSubmodule1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = torch.nn.Linear(4, 4)
                self.leaf = LeafModule(4, 4)

            def forward(self, x):
                x = self.pre_linear(x)
                x = torch.relu(x)
                out = self.leaf(x)[0]
                return out

        class CompiledSubmodule2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = LeafModule(4, 4)
                self.post_linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                out = self.leaf(x)[0]
                out = self.post_linear(out)
                return torch.sigmoid(out)

        class CompiledSubmodule3(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)

            def forward(self, x):
                out1 = self.leaf1(x)[0]
                out2 = self.leaf2(x)[0]
                return out1 + out2

        class TopLevelModule(torch.nn.Module):
            def __init__(self, compile_submodules=False):
                super().__init__()
                self.submodule1 = CompiledSubmodule1()
                self.submodule2 = CompiledSubmodule2()
                self.submodule3 = CompiledSubmodule3()
                self.final_linear = torch.nn.Linear(4, 4)
                self.compile_submodules = compile_submodules

            def forward(self, x):
                if self.compile_submodules:
                    out1 = torch.compile(self.submodule1, backend=backend)(x)
                    out2 = torch.compile(self.submodule2, backend=backend)(out1)
                    out3 = torch.compile(self.submodule3, backend=backend)(out2)
                else:
                    out1 = self.submodule1(x)
                    out2 = self.submodule2(out1)
                    out3 = self.submodule3(out2)
                final = self.final_linear(out3)
                return final

        # Create eager and compiled versions with identical weights
        model_eager = TopLevelModule(compile_submodules=False)
        model_compiled = TopLevelModule(compile_submodules=True)
        model_compiled.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_compiled = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(
            model_eager,
            model_compiled,
            x,
            x_compiled,
        )

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("do_compile", [False, True])
    def test_leaf_function_with_graph_breaks(self, backend, do_compile):
        """
        Test that leaf_function works correctly even when there are graph breaks
        between multiple leaf function calls. This tests the graph_bytecode_inputs
        infrastructure which must correctly restore nn.Module references across
        graph breaks.
        """
        from torch._dynamo.decorators import leaf_function

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            @leaf_function
            def forward(self, x):
                # Data-dependent behavior to ensure this is truly opaque
                if x.sum() > 0:
                    return (self.linear(x),)
                else:
                    return (self.linear(x) + 1,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class TopLevelModule(torch.nn.Module):
            def __init__(self, do_compile=False, backend="eager"):
                super().__init__()
                self.leaf1 = LeafModule(4, 4)
                self.leaf2 = LeafModule(4, 4)
                self.leaf3 = LeafModule(4, 4)
                self.final_linear = torch.nn.Linear(4, 4)
                self.do_compile = do_compile
                self.backend = backend

            def _forward(self, x):
                # First leaf function call
                out1 = self.leaf1(x)[0]

                # Manual graph break
                torch._dynamo.graph_break()

                # Second leaf function call after graph break
                out2 = self.leaf2(out1)[0]

                # Another manual graph break
                torch._dynamo.graph_break()

                # Third leaf function call after another graph break
                out3 = self.leaf3(out2)[0]

                # Final processing
                result = self.final_linear(out3)
                return result

            def forward(self, x):
                if self.do_compile:
                    return torch.compile(
                        self._forward, backend=self.backend, fullgraph=False
                    )(x)
                else:
                    return self._forward(x)

        # Create eager and compiled versions with identical weights
        model_eager = TopLevelModule(do_compile=False)
        model_test = TopLevelModule(do_compile=do_compile, backend=backend)
        model_test.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_test = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(model_eager, model_test, x, x_test)

    def test_leaf_function_with_module_input(self):
        from torch._dynamo.decorators import leaf_function

        class HelperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @leaf_function
            def forward(self, helper_mod, x):
                if x.sum() > 0:
                    return (helper_mod(x),)
                else:
                    return (helper_mod(x) + x,)

            @forward.fake_impl
            def forward(self, helper_mod, x):
                return (helper_mod(x),)

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = MainModule()
                self.helper = HelperModule()

            def forward(self, x):
                return self.main(self.helper, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_in_pytree(self):
        from torch._dynamo.decorators import leaf_function

        class HelperModule(torch.nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = scale

            def forward(self, x):
                return self.linear(x) * self.scale

        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @leaf_function
            def forward(self, modules_dict, x):
                if x.sum() > 0:
                    return (modules_dict["first"](x) + modules_dict["second"](x),)
                else:
                    return (modules_dict["first"](x) - modules_dict["second"](x),)

            @forward.fake_impl
            def forward(self, modules_dict, x):
                return (modules_dict["first"](x) + modules_dict["second"](x),)

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = MainModule()
                self.helper1 = HelperModule(scale=1.0)
                self.helper2 = HelperModule(scale=0.5)

            def forward(self, x):
                modules_dict = {"first": self.helper1, "second": self.helper2}
                return self.main(modules_dict, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_as_kwarg(self):
        from torch._dynamo.decorators import leaf_function

        class HelperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @leaf_function
            def forward(self, x, helper_mod=None):
                if x.sum() > 0:
                    return (helper_mod(x),)
                else:
                    return (helper_mod(x) + x,)

            @forward.fake_impl
            def forward(self, x, helper_mod=None):
                return (helper_mod(x),)

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = MainModule()
                self.helper = HelperModule()

            def forward(self, x):
                return self.main(x, helper_mod=self.helper)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_no_fake_fn(self):
        from torch._dynamo.decorators import leaf_function

        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                return (self.linear(x),)

            # No fake_impl - uses forward itself as fake_impl

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(SimpleModule, args_fn, loss_fn)

    def test_leaf_function_no_fake_fn_data_dependent_shape(self):
        from torch._dynamo.decorators import leaf_function

        class NonzeroModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @leaf_function
            def forward(self, x):
                return (x.nonzero(),)

            # No fake_impl - uses forward itself as fake_impl

        mod = NonzeroModule()
        x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

        result = mod(x)
        self.assertEqual(result[0].shape[0], 2)

        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        result_compiled = compiled_mod(x)
        self.assertEqual(result[0], result_compiled[0])

    def test_leaf_function_no_fake_fn_with_constant_tensor_closure(self):
        from torch._dynamo.decorators import leaf_function

        constant_weight = torch.randn(3, 3)

        class ConstantClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @leaf_function
            def forward(self, x):
                return (x @ constant_weight,)

            # No fake_impl - uses forward itself as fake_impl

        mod = ConstantClosureModule()
        x = torch.randn(3, 3, requires_grad=True)

        result = mod(x)
        expected = x @ constant_weight
        self.assertEqual(result[0], expected)

        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        result_compiled = compiled_mod(x)
        self.assertEqual(result[0], result_compiled[0])

    def test_leaf_function_validation_shape_mismatch(self):
        from torch._dynamo.decorators import leaf_function

        class MismatchedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                return (self.linear(x),)

            @forward.fake_impl
            def forward(self, x):
                return (torch.zeros(x.shape[0], 6),)

        mod = MismatchedModule()
        x = torch.randn(3, 3)

        compiled_mod = torch.compile(mod, backend="eager")
        with self.assertRaisesRegex(RuntimeError, "Shape mismatch"):
            compiled_mod(x)

    def test_leaf_function_validation_dtype_mismatch(self):
        from torch._dynamo.decorators import leaf_function

        class DtypeMismatchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                return (self.linear(x),)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x).double(),)

        mod = DtypeMismatchModule()
        x = torch.randn(3, 3)

        compiled_mod = torch.compile(mod, backend="eager")
        with self.assertRaisesRegex(RuntimeError, "Dtype mismatch"):
            compiled_mod(x)

    def test_leaf_function_validation_structure_mismatch(self):
        from torch._dynamo.decorators import leaf_function

        class StructureMismatchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                return (self.linear(x),)

            @forward.fake_impl
            def forward(self, x):
                return self.linear(x)

        mod = StructureMismatchModule()
        x = torch.randn(3, 3)

        compiled_mod = torch.compile(mod, backend="eager")
        with self.assertRaises((RuntimeError, AssertionError)):
            compiled_mod(x)

    def test_leaf_function_validation_disabled(self):
        import torch._dynamo.config as config
        from torch._dynamo.decorators import leaf_function

        class MismatchedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                return (self.linear(x),)

            @forward.fake_impl
            def forward(self, x):
                return (torch.zeros(x.shape[0], 6),)

        mod = MismatchedModule()
        x = torch.randn(3, 3)

        # With validation disabled, shape mismatch should not raise
        with config.patch(validate_leaf_function_outputs=False):
            compiled_mod = torch.compile(mod, backend="eager")
            result = compiled_mod(x)
            self.assertEqual(result[0].shape, (3, 3))

    def test_leaf_function_setter_pattern(self):
        """Test the new setter-pattern leaf_function decorator."""
        from torch._dynamo.decorators import leaf_function

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:  # data-dependent control flow
                    return (self.linear(x),)
                return (self.linear(x) + 1,)

            @forward.fake_impl
            def forward(self, x):
                return (self.linear(x),)

        class TopLevelMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf_mod = TestModule()

            def forward(self, x):
                return self.leaf_mod(x)

        mod = TopLevelMod()
        opt_mod = torch.compile(mod, fullgraph=True, backend="eager")
        x = torch.randn(10, 10)
        result = opt_mod(x)
        # Should compile successfully with fake_impl
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10, 10))

    def test_leaf_function_no_fake_impl(self):
        """Test leaf_function without fake_impl setter - uses forward itself as fake."""
        from torch._dynamo.decorators import leaf_function

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            @leaf_function
            def forward(self, x):
                # No data-dependent control flow, so forward itself can be fake_impl
                return (self.linear(x),)

            # No @forward.fake_impl - uses forward itself

        class TopLevelMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf_mod = TestModule()

            def forward(self, x):
                return self.leaf_mod(x)

        mod = TopLevelMod()
        opt_mod = torch.compile(mod, fullgraph=True, backend="eager")
        x = torch.randn(10, 10)
        result = opt_mod(x)
        # Should compile successfully without explicit fake_impl
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (10, 10))

        # Also verify eager execution works
        eager_result = mod(x)
        self.assertEqual(result[0], eager_result[0])

    def test_leaf_function_dict_output(self):
        from torch._dynamo.decorators import leaf_function

        class DictOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return {"a": self.linear1(x), "b": self.linear2(x)}
                else:
                    return {"a": self.linear1(x) + 1, "b": self.linear2(x) + 1}

            @forward.fake_impl
            def forward(self, x):
                return {"a": self.linear1(x), "b": self.linear2(x)}

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out["a"].sum() + out["b"].sum()

        self._test_leaf_function_helper(DictOutputModule, args_fn, loss_fn)

    def test_leaf_function_nested_output(self):
        from torch._dynamo.decorators import leaf_function

        class NestedOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            @leaf_function
            def forward(self, x):
                if x.sum() > 0:
                    return {
                        "out": (self.linear1(x), self.linear2(x)),
                        "extra": self.linear3(x),
                    }
                else:
                    return {
                        "out": (self.linear1(x) + 1, self.linear2(x) + 1),
                        "extra": self.linear3(x) + 1,
                    }

            @forward.fake_impl
            def forward(self, x):
                return {
                    "out": (self.linear1(x), self.linear2(x)),
                    "extra": self.linear3(x),
                }

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out["out"][0].sum() + out["out"][1].sum() + out["extra"].sum()

        self._test_leaf_function_helper(NestedOutputModule, args_fn, loss_fn)


instantiate_parametrized_tests(DecoratorTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
