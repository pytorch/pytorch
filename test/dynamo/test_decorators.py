# Owner(s): ["module: dynamo"]
import copy
import functools
import operator
import os
import re
import unittest.mock as mock
from unittest.mock import patch

import torch
import torch._dynamo.config as config
import torch._dynamo.testing
from torch._dynamo.decorators import leaf_function
from torch._dynamo.exc import Unsupported
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
        with self.assertRaisesRegex(RuntimeError, "disallow_in_graph is expected"):

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
        class State(torch._opaque_base.OpaqueBase):
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
        class State(torch._opaque_base.OpaqueBase):
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

    def test_nonstrict_trace_nn_module_dict_input(self):
        @torch._dynamo.nonstrict_trace
        def trace_me(x, modules):
            torch._dynamo.graph_break()
            return modules["a"](x) + modules["b"](x)

        def fn(x, modules):
            return trace_me(x, modules) + 1

        linear_a = torch.nn.Linear(4, 4)
        linear_b = torch.nn.Linear(4, 4)
        modules = {"a": linear_a, "b": linear_b}
        x = torch.randn(4, 4)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        ref = fn(x, modules)
        res = opt_fn(x, modules)
        self.assertEqual(ref, res)

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

    def test_fullgraph_eval_frame_override(self):
        # NOTE it is NOT enough to just call a torch.compile'd function in a compiled
        # function returned by the backend - this is because we apply disable(recursive=True)
        # to compiled functions and if we call a directly torch.compile'd function, that
        # "overrides" the disable(recursive=True) - i.e. this behavior is intentional.

        # Instead, we will patch symbolic_convert.InstructionTranslator.codegen_return_with_pops to
        # append a bunch of additional bytecode that will run a function that is not disabled.
        global inner

        y = torch.ones(3)

        def inner():
            nonlocal y
            y += 1

        from torch._dynamo.bytecode_transformation import (
            create_call_function,
            create_instruction,
            Instruction,
        )
        from torch._dynamo.symbolic_convert import InstructionTranslatorBase

        old_codegen_return = InstructionTranslatorBase.codegen_return_with_pops

        def codegen_return_with_pops(self, *args) -> list[Instruction]:
            insts = old_codegen_return(*args)
            if not insts[-1].opname.startswith("RETURN"):
                raise AssertionError(
                    f"Expected RETURN instruction, got {insts[-1].opname}"
                )
            # to prevent infinite recursion
            if self.f_code.co_name != "inner":
                insts[-1:-1] = [
                    create_instruction("LOAD_GLOBAL", argval="inner"),
                    *create_call_function(0, True),
                    create_instruction("POP_TOP"),
                ]
            return insts

        def fn(x):
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()

        with mock.patch(
            "torch._dynamo.symbolic_convert.InstructionTranslatorBase.codegen_return_with_pops",
            codegen_return_with_pops,
        ):
            # fullgraph=False will result in inner being traced!
            opt_fn_1 = torch.compile(fn, backend=cnts, fullgraph=False)

            # inner compiled
            opt_fn_1(torch.ones(3))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(y, torch.ones(3) + 1)

            torch._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # NOTE do not fully reset dynamo - to ensure eval frame override is applied for cache hits
            opt_fn_2 = torch.compile(fn, backend=cnts, fullgraph=True)

            with torch._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=False
            ):
                # fullgraph=True will result in inner being skipped!
                opt_fn_2(torch.ones(3))
                self.assertEqual(cnts.frame_count, 0)
                self.assertEqual(y, torch.ones(3) + 2)

            with torch._dynamo.config.patch(
                error_on_dynamo_callback_in_fullgraph_compiled_code=True
            ):
                # fullgraph=True will result in error when attempting to compile inner
                with self.assertRaisesRegex(
                    RuntimeError, "Dynamo: expected not to compile nested code"
                ):
                    opt_fn_2(torch.ones(3))

            torch._dynamo.eval_frame.reset_code(inner.__code__)
            cnts.clear()
            # if we run fullgraph=False again, inner is compiled again (because we reset_code)
            opt_fn_1(torch.ones(3))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(y, torch.ones(3) + 3)

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

        self.assertEqual(fn(x, y), torch.compile(fn, backend="eager")(x, y))

    def test_justknobs_check(self):
        def fn(x, y):
            if torch._utils_internal.justknobs_check("test", True):
                return x + y
            else:
                return x - y

        x = torch.randn(2, 2, device="cpu")
        y = torch.randn(2, 2, device="cpu")
        eager_out = fn(x, y)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        compiled_out = compiled_fn(x, y)
        self.assertEqual(eager_out, compiled_out)

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
            torch.compile(model, backend="eager")
            torch.compile(other_model, backend="eager")

    def test_disable_class_and_instance_method(self):
        # Test that decorating a method at class definition time and then
        # re-decorating the instance method works correctly. This tests the
        # fix in innermost_fn that stops unwrapping when hitting a bound method.
        from torch._dynamo.eval_frame import innermost_fn

        class Foo:
            def run(self, a, b, c):
                return self.work(a, b, c)

            @torch._dynamo.disable
            def work(self, a, b, c):
                return a + b - c

        foo = Foo()
        # Re-decorate the instance method
        foo.work = torch._dynamo.disable(foo.work)

        a = torch.randint(0, 10, (10,))
        b = torch.randint(0, 10, (10,))
        c = torch.randint(0, 10, (10,))

        # Should work without error - self should be correctly bound
        result = foo.run(a, b, c)
        self.assertEqual(result, a + b - c)

        # Also test nested disable on instance methods
        foo2 = Foo()
        foo2.work = torch._dynamo.disable(torch._dynamo.disable(foo2.work))
        result2 = foo2.run(a, b, c)
        self.assertEqual(result2, a + b - c)

        # Test innermost_fn shortcut behavior for unbound methods
        # disable(disable(Foo.method)) should unwrap to the original function
        class Bar:
            def method(self, x):
                return x + 1

        bar = Bar()
        bound_method = bar.method

        original_method = Bar.method
        disabled_once = torch._dynamo.disable(Bar.method)
        disabled_twice = torch._dynamo.disable(disabled_once)
        # innermost_fn should find the original unbound method
        self.assertIs(innermost_fn(disabled_twice), original_method)
        self.assertIs(innermost_fn(disabled_once), original_method)

        # Test innermost_fn shortcut behavior for bound methods
        # disable(disable(obj.method)) should stop at the bound method
        # innermost_fn should return the bound method itself, not unwrap it
        self.assertIs(innermost_fn(bound_method), bound_method)
        # Wrapping a bound method should also preserve the binding
        disabled_bound = torch._dynamo.disable(bound_method)
        self.assertIs(innermost_fn(disabled_bound), bound_method)

    def test_disable_functools_wraps(self):
        # Test that functools.wraps copying _torchdynamo_orig_callable doesn't
        # cause innermost_fn to bypass the outer wrapper. This tests the fix
        # using _torchdynamo_wrapper_id to verify the attribute was set by our
        # decorator, not copied by functools.wraps.
        from torch._dynamo.eval_frame import innermost_fn

        @torch._dynamo.disable
        def inner_fn(x):
            return x + 1

        # Outer wrapper uses functools.wraps which copies _torchdynamo_orig_callable
        @functools.wraps(inner_fn)
        def outer_wrapper(x):
            return inner_fn(x) * 2

        # innermost_fn should NOT follow the copied _torchdynamo_orig_callable
        # because _torchdynamo_wrapper_id won't match
        self.assertIs(innermost_fn(outer_wrapper), outer_wrapper)

        # Applying disable to outer_wrapper should wrap outer_wrapper, not inner_fn
        disabled_outer = torch._dynamo.disable(outer_wrapper)

        x = torch.tensor([1.0, 2.0, 3.0])
        expected = outer_wrapper(x)  # (x+1)*2 = [4, 6, 8]
        actual = disabled_outer(x)
        self.assertEqual(expected, actual)

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
        out_expected = model_expected(x_expected)
        out_test = model_test(x_test)
        self.assertEqual(out_expected, out_test)

        loss_expected = out_expected.sum()
        loss_test = out_test.sum()
        loss_expected.backward()
        loss_test.backward()
        self.assertEqual(x_expected.grad, x_test.grad)

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

        eager_backend = EagerAndRecordGraphs()
        compiled_eager = torch.compile(
            mod_compile_eager, backend=eager_backend, fullgraph=True
        )

        backend = AotEagerAndRecordGraphs()
        compiled_aot = torch.compile(mod_compile_aot, backend=backend, fullgraph=True)

        for _ in range(2):
            mod_eager.zero_grad()
            mod_compile_eager.zero_grad()
            mod_compile_aot.zero_grad()

            args = args_fn()
            args_clone = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )
            args_clone2 = pytree.tree_map(
                lambda x: x.clone().detach().requires_grad_(x.requires_grad), args
            )

            out_eager = mod_eager(*args)
            loss_fn(out_eager).backward()

            out_compile_eager = compiled_eager(*args_clone)
            loss_fn(out_compile_eager).backward()

            out_compile_aot = compiled_aot(*args_clone2)
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
        @leaf_function
        def non_tracable_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @non_tracable_forward.register_fake
        def non_tracable_forward_fake(mod, x):
            return (mod.linear(x),)

        class NonTracable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return non_tracable_forward(self, x)

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
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_2, requires_grad_indices = (1, 2, 3));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4];  with_effects_1 = None
        return (getitem_6, getitem_4, getitem_5, getitem_2)
""",  # noqa: B950
        )

    def test_leaf_function_with_logging(self):
        @leaf_function
        def logging_forward(mod, x):
            print("Processing input")
            return (mod.linear(x),)

        @logging_forward.register_fake
        def logging_forward_fake(mod, x):
            return (mod.linear(x),)

        class LoggingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return logging_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print") as mock_print:
            self._test_leaf_function_helper(LoggingModule, args_fn, loss_fn)
            mock_print.assert_any_call("Processing input")
            # Called 6 times: eager, compile_eager, and compile_aot, 2 iterations each
            self.assertEqual(mock_print.call_count, 6)

    def test_leaf_function_dynamic_autograd_module_config(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        @leaf_function
        def configurable_scale(mod, x):
            # Branch based on module config, not input
            if mod.use_double_scale:
                return (mod.linear(x) * 2,)
            else:
                return (mod.linear(x) * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(mod, x):
            return (mod.linear(x),)

        class ConfigurableModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.use_double_scale = True  # Config attribute

            def forward(self, x):
                return configurable_scale(self, x)

        mod_eager = ConfigurableModule()
        mod_compiled = ConfigurableModule()
        mod_compiled.load_state_dict(dict(mod_eager.state_dict()))

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(mod_compiled, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)

        mod_eager.use_double_scale = True
        mod_compiled.use_double_scale = True

        x1 = x_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)

        out_eager_1 = mod_eager(x1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)

        mod_eager.zero_grad()
        mod_compiled.zero_grad()

        mod_eager.use_double_scale = False
        mod_compiled.use_double_scale = False

        x2 = x_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)

        out_eager_2 = mod_eager(x2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)

        # Same inputs but different config -> different gradients
        # This proves leaf_function builds autograd dynamically (not burned in at trace time)
        self.assertNotEqual(x1.grad, x2.grad)

        # Verify only ONE compilation happened (no recompilation when changing config)
        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_dynamic_autograd_closure(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        config = {"use_double_scale": True}

        @leaf_function
        def configurable_scale(x, y):
            # Branch based on closure variable, not input
            if config["use_double_scale"]:
                return (x @ y * 2,)
            else:
                return (x @ y * 3,)

        @configurable_scale.register_fake
        def configurable_scale_fake(x, y):
            return (x @ y,)

        def fn(x, y):
            return configurable_scale(x, y)

        counter = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=True)

        x_value = torch.randn(3, 3)
        y_value = torch.randn(3, 3)

        config["use_double_scale"] = True

        x1 = x_value.clone().requires_grad_(True)
        y1 = y_value.clone().requires_grad_(True)
        x1_clone = x_value.clone().requires_grad_(True)
        y1_clone = y_value.clone().requires_grad_(True)

        out_eager_1 = fn(x1, y1)
        out_eager_1[0].sum().backward()

        out_compiled_1 = compiled_fn(x1_clone, y1_clone)
        out_compiled_1[0].sum().backward()

        self.assertEqual(out_eager_1, out_compiled_1)
        self.assertEqual(x1.grad, x1_clone.grad)
        self.assertEqual(y1.grad, y1_clone.grad)

        config["use_double_scale"] = False

        x2 = x_value.clone().requires_grad_(True)
        y2 = y_value.clone().requires_grad_(True)
        x2_clone = x_value.clone().requires_grad_(True)
        y2_clone = y_value.clone().requires_grad_(True)

        out_eager_2 = fn(x2, y2)
        out_eager_2[0].sum().backward()

        out_compiled_2 = compiled_fn(x2_clone, y2_clone)
        out_compiled_2[0].sum().backward()

        self.assertEqual(out_eager_2, out_compiled_2)
        self.assertEqual(x2.grad, x2_clone.grad)
        self.assertEqual(y2.grad, y2_clone.grad)

        # Same inputs but different closure -> different gradients
        # This proves leaf_function builds autograd dynamically (not burned in at trace time)
        self.assertNotEqual(x1.grad, x2.grad)
        self.assertNotEqual(y1.grad, y2.grad)

        # Verify only ONE compilation happened (no recompilation when changing closure)
        self.assertEqual(counter.frame_count, 1)

    def test_leaf_function_closure_constants_without_grad(self):
        closure_scale = 2.0
        closure_tensor = torch.tensor([1.0, 2.0, 3.0])

        @leaf_function
        def closure_forward(mod, x):
            out = mod.linear(x) * closure_scale * mod.scale
            out = out + closure_tensor + mod.offset
            return (out,)

        @closure_forward.register_fake
        def closure_forward_fake(mod, x):
            return (mod.linear(x) + mod.offset,)

        class ClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = 3.0
                self.offset = torch.nn.Parameter(torch.ones(3))

            def forward(self, x):
                return closure_forward(self, x)

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
    def forward(self, L_x_: "f32[3, 3]", L_self_parameters_offset_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_parameters_offset_ = L_self_parameters_offset_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_parameters_offset_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_parameters_offset_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3]", primals_4: "f32[3, 3]", primals_5: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_5, primals_2, requires_grad_indices = (1, 2, 3, 4));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_5 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3]" = with_effects_1[2]
        getitem_5: "f32[3, 3]" = with_effects_1[3]
        getitem_6: "f32[3]" = with_effects_1[4]
        getitem_7: "f32[3, 3]" = with_effects_1[5];  with_effects_1 = None
        return (getitem_7, getitem_4, getitem_5, getitem_6, getitem_2)
""",  # noqa: B950
        )

    def test_leaf_function_pytree_inputs(self):
        @leaf_function
        def pytree_forward(mod, inputs):
            if inputs["x"].sum() > 0:
                return (mod.linear(inputs["x"]), inputs["y"] + 1)
            return (mod.linear(inputs["x"]) + inputs["y"], inputs["y"] - 1)

        @pytree_forward.register_fake
        def pytree_forward_fake(mod, inputs):
            return (mod.linear(inputs["x"]), inputs["y"])

        class PytreeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, inputs):
                return pytree_forward(self, inputs)

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
        @leaf_function
        def inner_leaf_forward(mod, x):
            y = mod.linear(x)
            return (y + x,)

        @inner_leaf_forward.register_fake
        def inner_leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class InnerLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return inner_leaf_forward(self, x)

        @leaf_function
        def outer_leaf_forward(mod, x):
            z = mod.linear(x)
            return mod.inner(z + x)

        @outer_leaf_forward.register_fake
        def outer_leaf_forward_fake(mod, x):
            return mod.inner(mod.linear(x))

        class OuterLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerLeaf()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return outer_leaf_forward(self, x)

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
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_inner_modules_linear_parameters_bias_: "f32[3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_inner_modules_linear_parameters_weight_ = L_self_modules_inner_modules_linear_parameters_weight_
        l_self_modules_inner_modules_linear_parameters_bias_ = L_self_modules_inner_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', 0, l_self_modules_inner_modules_linear_parameters_weight_, l_self_modules_inner_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_, l_x_);  real_fn = fake_fn = input_spec = l_self_modules_inner_modules_linear_parameters_weight_ = l_self_modules_inner_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = l_x_ = None
        getitem: "f32[3, 3]" = invoke_leaf_function[0];  invoke_leaf_function = None
        return (getitem,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]", primals_5: "f32[3, 3]", primals_6: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', 0, primals_3, primals_4, primals_5, primals_6, primals_2, requires_grad_indices = (1, 2, 3, 4, 5));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = primals_3 = primals_4 = primals_5 = primals_6 = primals_2 = None

        getitem: "f32[0]" = with_effects[0]
        getitem_1: "f32[3, 3]" = with_effects[1];  with_effects = None
        return (getitem, getitem_1)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph_str,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 3]", tangents_token: "f32[0]"):
        _opaque_obj2 = self._opaque_obj2
        _opaque_obj3 = self._opaque_obj3
        _tree_spec_constant1 = self._tree_spec_constant1
        with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.higher_order.invoke_leaf_function, _opaque_obj2, _opaque_obj3, _tree_spec_constant1, '', tangents_1, requires_grad_indices = ());  tangents_token = _opaque_obj2 = _opaque_obj3 = _tree_spec_constant1 = tangents_1 = None
        getitem_2: "f32[0]" = with_effects_1[0]
        getitem_4: "f32[3, 3]" = with_effects_1[2]
        getitem_5: "f32[3]" = with_effects_1[3]
        getitem_6: "f32[3, 3]" = with_effects_1[4]
        getitem_7: "f32[3]" = with_effects_1[5]
        getitem_8: "f32[3, 3]" = with_effects_1[6];  with_effects_1 = None
        return (getitem_8, getitem_4, getitem_5, getitem_6, getitem_7, getitem_2)
""",  # noqa: B950
        )

    def test_leaf_function_data_dependent_nonzero(self):
        @leaf_function
        def nonzero_forward(mod, x):
            out = mod.linear(x)
            nonzero_indices = (out > 0).nonzero()
            return (out, nonzero_indices)

        @nonzero_forward.register_fake
        def nonzero_forward_fake(mod, x):
            out = mod.linear(x)
            return out, (out > 0).nonzero()

        class NonzeroModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return nonzero_forward(self, x)

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_linear = torch.nn.Linear(3, 3)
                self.nonzero_module = NonzeroModule()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                x = self.pre_linear(x)
                x = torch.relu(x)
                out, nonzero_indices = self.nonzero_module(x)
                num_nonzero = nonzero_indices.shape[0]
                scaled_out = out * self.scale + num_nonzero
                return scaled_out, nonzero_indices

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(OuterModule, args_fn, loss_fn)

    def test_leaf_function_data_dependent_item(self):
        @leaf_function
        def item_forward(mod, x):
            out = mod.linear(x)
            scalar_value = out.sum().item()
            return (out, scalar_value)

        @item_forward.register_fake
        def item_forward_fake(mod, x):
            out = mod.linear(x)
            return (out, out.sum().item())

        class ItemModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return item_forward(self, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(ItemModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_multiple_compiled_submodules(self, backend):
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + x,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

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
        @leaf_function
        def leaf_forward(mod, x):
            if x.sum() > 0:
                return (mod.linear(x),)
            else:
                return (mod.linear(x) + 1,)

        @leaf_forward.register_fake
        def leaf_forward_fake(mod, x):
            return (mod.linear(x),)

        class LeafModule(torch.nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)

            def forward(self, x):
                return leaf_forward(self, x)

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
                out1 = self.leaf1(x)[0]
                torch._dynamo.graph_break()
                out2 = self.leaf2(out1)[0]
                torch._dynamo.graph_break()
                out3 = self.leaf3(out2)[0]
                result = self.final_linear(out3)
                return result

            def forward(self, x):
                if self.do_compile:
                    return torch.compile(
                        self._forward, backend=self.backend, fullgraph=False
                    )(x)
                else:
                    return self._forward(x)

        model_eager = TopLevelModule(do_compile=False)
        model_test = TopLevelModule(do_compile=do_compile, backend=backend)
        model_test.load_state_dict(model_eager.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        x_test = x.clone().detach().requires_grad_(True)

        self._assert_models_equal(model_eager, model_test, x, x_test)

    def test_leaf_function_with_module_in_pytree(self):
        @leaf_function
        def main_forward(modules_dict, x):
            if x.sum() > 0:
                return (modules_dict["first"](x) + modules_dict["second"](x),)
            else:
                return (modules_dict["first"](x) - modules_dict["second"](x),)

        @main_forward.register_fake
        def main_forward_fake(modules_dict, x):
            return (modules_dict["first"](x) + modules_dict["second"](x),)

        class HelperModule(torch.nn.Module):
            def __init__(self, scale=1.0):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.scale = scale

            def forward(self, x):
                return self.linear(x) * self.scale

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper1 = HelperModule(scale=1.0)
                self.helper2 = HelperModule(scale=0.5)

            def forward(self, x):
                modules_dict = {"first": self.helper1, "second": self.helper2}
                return main_forward(modules_dict, x)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_with_module_as_kwarg(self):
        @leaf_function
        def main_forward(x, helper_mod=None):
            if x.sum() > 0:
                return (helper_mod(x),)
            else:
                return (helper_mod(x) + x,)

        @main_forward.register_fake
        def main_forward_fake(x, helper_mod=None):
            return (helper_mod(x),)

        class HelperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        class WrapperModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.helper = HelperModule()

            def forward(self, x):
                return main_forward(x, helper_mod=self.helper)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        self._test_leaf_function_helper(WrapperModule, args_fn, loss_fn)

    def test_leaf_function_missing_fake_impl_error(self):
        @leaf_function
        def no_fake_impl_forward(mod, x):
            return (mod.linear(x),)

        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return no_fake_impl_forward(self, x)

        mod = SimpleModule()
        x = torch.randn(3, 3)

        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            mod(x)

        compiled_mod = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Exception, "requires a fake implementation"):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_constant_tensor_closure_error(self, backend):
        constant_weight = torch.randn(3, 3)

        @leaf_function
        def constant_closure_forward(x):
            return (x @ constant_weight,)

        @constant_closure_forward.register_fake
        def constant_closure_forward_fake(x):
            return (x @ constant_weight,)

        class ConstantClosureModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return constant_closure_forward(x)

        mod = ConstantClosureModule()
        x = torch.randn(3, 3, requires_grad=True)

        result = mod(x)
        expected = x @ constant_weight
        self.assertEqual(result[0], expected)

        compiled_mod = torch.compile(mod, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_error(self, backend):
        @leaf_function
        def mutate_input(x):
            x.add_(1)
            return (x,)

        @mutate_input.register_fake
        def mutate_input_fake(x):
            x.add_(1)
            return (x,)

        def fn(x):
            return mutate_input(x)

        x = torch.randn(3, 3)

        x_eager = x.clone()
        with self.assertRaisesRegex(RuntimeError, "Undeclared in-place mutation"):
            fn(x_eager)

        x = torch.randn(3, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with self.assertRaisesRegex(RuntimeError, "leaf Variable that requires grad"):
            compiled_fn(x.clone().requires_grad_(True))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_validation_dtype_mismatch(self, backend):
        @leaf_function
        def dtype_mismatch_forward(mod, x):
            return (mod.linear(x),)

        @dtype_mismatch_forward.register_fake
        def dtype_mismatch_forward_fake(mod, x):
            return (mod.linear(x).double(),)

        class DtypeMismatchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return dtype_mismatch_forward(self, x)

        mod = DtypeMismatchModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=True):
            compiled_mod = torch.compile(mod, backend=backend)
            with self.assertRaisesRegex(RuntimeError, "Dtype mismatch"):
                compiled_mod(x)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("validate_outputs", [True, False])
    def test_leaf_function_validation_shape_mismatch(self, backend, validate_outputs):
        @leaf_function
        def mismatched_forward(mod, x):
            return (mod.linear(x),)

        @mismatched_forward.register_fake
        def mismatched_forward_fake(mod, x):
            return (torch.zeros(x.shape[0], 6),)

        class MismatchedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return mismatched_forward(self, x)

        mod = MismatchedModule()
        x = torch.randn(3, 3)

        with config.patch(leaf_function_validate_outputs=validate_outputs):
            compiled_mod = torch.compile(mod, backend=backend)
            if validate_outputs:
                with self.assertRaises((RuntimeError, AssertionError)):
                    compiled_mod(x)
            else:
                result = compiled_mod(x)
                self.assertEqual(result[0].shape, (3, 3))

    def test_leaf_function_no_module_inputs(self):
        @leaf_function
        def my_custom_fn(inputs: dict[str, torch.Tensor], scale: float, offset: int):
            x = inputs["x"]
            y = inputs["y"]
            if x.sum() > 0:
                return (x * scale + y + offset, x.sum() + y.sum())
            return (x * scale - y + offset, x.sum() - y.sum())

        @my_custom_fn.register_fake
        def my_custom_fn_fake(
            inputs: dict[str, torch.Tensor], scale: float, offset: int
        ):
            x = inputs["x"]
            y = inputs["y"]
            return (x * scale + y + offset, x.sum() + y.sum())

        class NoModuleInputsModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2.0
                self.offset = 1

            def forward(self, x, y):
                inputs = {"x": x, "y": y}
                return my_custom_fn(inputs, self.scale, self.offset)

        def args_fn():
            return (
                torch.randn(3, 3, requires_grad=True),
                torch.randn(3, 3, requires_grad=True),
            )

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(NoModuleInputsModule, args_fn, loss_fn)

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_multiple_tensors(
        self, backend, check_escaped_gradients
    ):
        weight1 = torch.randn(3, 3, requires_grad=True)
        weight2 = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_multiple_closures(x):
            return (x @ weight1 + x @ weight2,)

        @uses_multiple_closures.register_fake
        def uses_multiple_closures_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_multiple_closures(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x)
            else:
                result = compiled_fn(x)
                self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_input_no_grad(
        self, backend, check_escaped_gradients
    ):
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            result = compiled_fn(x)
            self.assertEqual(result[0].shape, (2, 3))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("check_escaped_gradients", [True, False])
    def test_leaf_function_escaped_gradient_mixed_inputs(
        self, backend, check_escaped_gradients
    ):
        base1 = torch.randn(3, 3, requires_grad=True)
        base2 = torch.randn(3, 4, requires_grad=True)
        closure_weight1 = base1 * 2
        closure_weight2 = base2 * 3

        @leaf_function
        def mixed_inputs(x, y):
            out1 = x @ closure_weight1 + y
            out2 = x @ closure_weight2
            return (out1, out2)

        @mixed_inputs.register_fake
        def mixed_inputs_fake(x, y):
            return (torch.empty(x.shape[0], 3), torch.empty(x.shape[0], 4))

        def fn(x, y):
            return mixed_inputs(x, y)

        x = torch.randn(2, 3, requires_grad=True)
        y = torch.randn(2, 3, requires_grad=False)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(
            leaf_function_check_escaped_gradients=check_escaped_gradients
        ):
            if check_escaped_gradients:
                with self.assertRaisesRegex(RuntimeError, "2 tensor"):
                    compiled_fn(x, y)
            else:
                result = compiled_fn(x, y)
                self.assertEqual(result[0].shape, (2, 3))
                self.assertEqual(result[1].shape, (2, 4))

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_error_message_contains_tensor_info(
        self, backend
    ):
        closure_weight = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 5),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 4, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with config.patch(leaf_function_check_escaped_gradients=True):
            with self.assertRaisesRegex(RuntimeError, r"shape=\[4, 5\].*dtype="):
                compiled_fn(x)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_escaped_gradient_actually_lost(self, backend):
        closure_weight = torch.randn(3, 3, requires_grad=True)

        @leaf_function
        def uses_closure(x):
            return (x @ closure_weight,)

        @uses_closure.register_fake
        def uses_closure_fake(x):
            return (torch.empty(x.shape[0], 3),)

        def fn(x):
            return uses_closure(x)

        x = torch.randn(2, 3, requires_grad=True)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        loss = result[0].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNone(closure_weight.grad)

    def test_leaf_function_and_nonstrict_trace_mutually_exclusive(self):
        from torch._dynamo.decorators import leaf_function, nonstrict_trace

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @leaf_function
            @nonstrict_trace
            def bad_fn1(x):
                return (x,)

        with self.assertRaisesRegex(
            ValueError,
            "cannot be both marked as @leaf_function and @nonstrict_trace",
        ):

            @nonstrict_trace
            @leaf_function
            def bad_fn2(x):
                return (x,)

    def test_leaf_function_no_return_value(self):
        printed = []

        @leaf_function
        def fn_no_return(x):
            print("processing")

        @fn_no_return.register_fake
        def fn_no_return_fake(x):
            pass

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                fn_no_return(x)
                return (self.linear(x),)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum()

        with patch("builtins.print", lambda *args, **kwargs: printed.append(args)):
            eager_graph, fw_graph, bw_graph = self._test_leaf_function_helper(
                Mod, args_fn, loss_fn
            )
        self.assertTrue(any("processing" in p for p in printed))
        self.assertExpectedInline(
            eager_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_self_modules_linear_parameters_weight_: "f32[3, 3]", L_self_modules_linear_parameters_bias_: "f32[3]"):
        l_x_ = L_x_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_

        real_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.real_fn
        fake_fn : torch._higher_order_ops.invoke_leaf_function._LeafCallable = self.fake_fn
        input_spec : torch.utils._pytree.TreeSpec = self.input_spec
        invoke_leaf_function = torch.ops.higher_order.invoke_leaf_function(real_fn, fake_fn, input_spec, '', l_x_);  real_fn = fake_fn = input_spec = invoke_leaf_function = None

        linear: "f32[3, 3]" = torch._C._nn.linear(l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  l_x_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        return (linear,)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            fw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[0]", primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", primals_4: "f32[3]"):
        _opaque_obj0 = self._opaque_obj0
        _opaque_obj1 = self._opaque_obj1
        _tree_spec_constant0 = self._tree_spec_constant0
        with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.higher_order.invoke_leaf_function, _opaque_obj0, _opaque_obj1, _tree_spec_constant0, '', primals_2, requires_grad_indices = (0,));  primals_1 = _opaque_obj0 = _opaque_obj1 = _tree_spec_constant0 = None

        getitem: "f32[0]" = with_effects[0];  with_effects = None

        t: "f32[3, 3]" = torch.ops.aten.t.default(primals_3)
        addmm: "f32[3, 3]" = torch.ops.aten.addmm.default(primals_4, primals_2, t);  primals_4 = t = None
        return (getitem, addmm, primals_2, primals_3)
""",  # noqa: B950
        )
        self.assertExpectedInline(
            bw_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[3, 3]", primals_3: "f32[3, 3]", tangents_1: "f32[3, 3]"):
        t: "f32[3, 3]" = torch.ops.aten.t.default(primals_3);  primals_3 = None
        t_1: "f32[3, 3]" = torch.ops.aten.t.default(t);  t = None
        mm: "f32[3, 3]" = torch.ops.aten.mm.default(tangents_1, t_1);  t_1 = None
        t_2: "f32[3, 3]" = torch.ops.aten.t.default(tangents_1)
        mm_1: "f32[3, 3]" = torch.ops.aten.mm.default(t_2, primals_2);  t_2 = primals_2 = None
        t_3: "f32[3, 3]" = torch.ops.aten.t.default(mm_1);  mm_1 = None
        sum_1: "f32[1, 3]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[3]" = torch.ops.aten.view.default(sum_1, [3]);  sum_1 = None
        t_4: "f32[3, 3]" = torch.ops.aten.t.default(t_3);  t_3 = None
        return (mm, t_4, view)
""",  # noqa: B950
        )

    def test_leaf_function_output_structure_mismatch(self):
        @leaf_function
        def mismatched_fn(x):
            return {"a": x, "b": x * 2}

        @mismatched_fn.register_fake
        def mismatched_fn_fake(x):
            return (x, x * 2)

        def fn(x):
            return mismatched_fn(x)

        x = torch.randn(3, 3)
        with self.assertRaisesRegex(AssertionError, "output structure mismatch"):
            torch.compile(fn, backend="eager")(x)

    def test_leaf_function_nested_output(self):
        @leaf_function
        def nested_output_fn(linear1, linear2, linear3, x):
            if x.sum() > 0:
                return {
                    "out": (linear1(x), linear2(x)),
                    "extra": linear3(x),
                    "count": 42,
                }
            else:
                return {
                    "out": (linear1(x) + 1, linear2(x) + 1),
                    "extra": linear3(x) + 1,
                    "count": 42,
                }

        @nested_output_fn.register_fake
        def nested_output_fn_fake(linear1, linear2, linear3, x):
            return {
                "out": (linear1(x), linear2(x)),
                "extra": linear3(x),
                "count": 42,
            }

        class NestedOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                result = nested_output_fn(self.linear1, self.linear2, self.linear3, x)
                return (
                    result["out"][0] * result["count"]
                    + result["out"][1]
                    + result["extra"]
                )

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out.sum()

        self._test_leaf_function_helper(NestedOutputModule, args_fn, loss_fn)

    def test_leaf_function_custom_pytree_output(self):
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

        @leaf_function
        def point_fn(linear1, linear2, x):
            return (Point(linear1(x), linear2(x)), 0.5)

        @point_fn.register_fake
        def point_fn_fake(linear1, linear2, x):
            return (Point(linear1(x), linear2(x)), 0.5)

        class PointModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)

            def forward(self, x):
                p, scale = point_fn(self.linear1, self.linear2, x)
                return (p.x * scale, p.y * scale)

        def args_fn():
            return (torch.randn(3, 3, requires_grad=True),)

        def loss_fn(out):
            return out[0].sum() + out[1].sum()

        self._test_leaf_function_helper(PointModule, args_fn, loss_fn)

    def test_leaf_function_fake_requires_grad_ignored(self):
        @leaf_function
        def my_fn(x):
            return (x * 2,)

        @my_fn.register_fake
        def my_fn_fake(x):
            return (torch.empty_like(x).requires_grad_(False),)

        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return my_fn(x)

        x = torch.randn(3, 3, requires_grad=True)
        out = fn(x)

        self.assertTrue(out[0].requires_grad)
        out[0].sum().backward()
        self.assertIsNotNone(x.grad)

        graph = backend.graphs[0]
        for node in graph.graph.nodes:
            if node.op == "call_function" and "invoke_leaf_function" in str(
                node.target
            ):
                example_value = node.meta.get("example_value")
                self.assertIsNotNone(example_value)
                self.assertTrue(example_value[0].requires_grad)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_non_grad(self, backend):
        @leaf_function(mutates_args={"buf"})
        def mutate_buffer(x, buf):
            buf.add_(1)
            return (x + buf,)

        @mutate_buffer.register_fake
        def mutate_buffer_fake(x, buf):
            buf.add_(1)
            return (x + buf,)

        def fn(x, buf):
            return mutate_buffer(x, buf)

        x = torch.randn(3, 3)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x + buf + 1
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_mixed(self, backend):
        @leaf_function(mutates_args={"buf"})
        def mixed_fn(x, buf):
            buf.mul_(2)
            return (x * buf,)

        @mixed_fn.register_fake
        def mixed_fn_fake(x, buf):
            buf.mul_(2)
            return (x * buf,)

        def fn(x, buf):
            return mixed_fn(x, buf)

        x = torch.randn(3, 3, requires_grad=True)
        buf = torch.randn(3, 3)

        buf_eager = buf.clone()
        result_eager = fn(x, buf_eager)
        expected = x * (buf * 2)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(buf_eager, buf * 2)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        buf_compiled = buf.clone()
        result_compiled = compiled_fn(x, buf_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(buf_compiled, buf * 2)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_module_buffer(self, backend):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return update_stats(self, x)

        @leaf_function(mutates_args={"model.running_mean"})
        def update_stats(model, x):
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        @update_stats.register_fake
        def update_stats_fake(model, x):
            model.running_mean.add_(x.mean(dim=0))
            return (model.linear(x),)

        mod = MyModule()
        x = torch.randn(4, 3)

        mod_eager = copy.deepcopy(mod)
        result_eager = mod_eager(x)
        expected_mean = torch.zeros(3) + x.mean(dim=0)
        self.assertEqual(mod_eager.running_mean, expected_mean)

        mod_compiled = copy.deepcopy(mod)
        compiled_mod = torch.compile(mod_compiled, backend=backend, fullgraph=True)
        result_compiled = compiled_mod(x)
        self.assertEqual(result_compiled, result_eager)
        self.assertEqual(mod_compiled.running_mean, expected_mean)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree(self, backend):
        @leaf_function(mutates_args={"buffers"})
        def update_buffers(x, buffers):
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        @update_buffers.register_fake
        def update_buffers_fake(x, buffers):
            for buf in buffers:
                buf.add_(1)
            return (x + sum(buffers),)

        def fn(x, buffers):
            return update_buffers(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + (bufs[1] + 1)
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1] + 1)

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1] + 1)

    @parametrize("backend", ["eager", "aot_eager"])
    def test_leaf_function_input_mutation_pytree_fine_grained(self, backend):
        @leaf_function(mutates_args={"buffers[0]"})
        def update_first(x, buffers):
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        @update_first.register_fake
        def update_first_fake(x, buffers):
            buffers[0].add_(1)
            return (x + buffers[0] + buffers[1],)

        def fn(x, buffers):
            return update_first(x, buffers)

        x = torch.randn(3, 3)
        bufs = [torch.randn(3, 3), torch.randn(3, 3)]

        bufs_eager = [b.clone() for b in bufs]
        result_eager = fn(x, bufs_eager)
        expected = x + (bufs[0] + 1) + bufs[1]
        self.assertEqual(result_eager[0], expected)
        self.assertEqual(bufs_eager[0], bufs[0] + 1)
        self.assertEqual(bufs_eager[1], bufs[1])

        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
        bufs_compiled = [b.clone() for b in bufs]
        result_compiled = compiled_fn(x, bufs_compiled)
        self.assertEqual(result_compiled[0], expected)
        self.assertEqual(bufs_compiled[0], bufs[0] + 1)
        self.assertEqual(bufs_compiled[1], bufs[1])

    def test_leaf_function_mutates_args_invalid_parameter(self):
        with self.assertRaisesRegex(ValueError, "refers to parameter 'buf'"):

            @leaf_function(mutates_args={"buf"})
            def bad_fn(x, buffers):
                buffers.add_(1)
                return (x + buffers,)

        with self.assertRaisesRegex(ValueError, "refers to parameter 'mdl'"):

            @leaf_function(mutates_args={"mdl.running_mean"})
            def bad_fn2(x, model):
                model.running_mean.add_(1)
                return (x,)

    def test_leaf_function_mutates_args_non_leaf_expression(self):
        @leaf_function(mutates_args={"model"})
        def bad_fn(x, model):
            model.running_mean.add_(1)
            return (x,)

        @bad_fn.register_fake
        def bad_fn_fake(x, model):
            model.running_mean.add_(1)
            return (x,)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(3))

            def forward(self, x):
                return bad_fn(x, self)

        mod = MyModule()
        x = torch.randn(3)
        compiled_fn = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, "resolved to a non-leaf value"
        ):
            compiled_fn(x)


instantiate_parametrized_tests(DecoratorTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
