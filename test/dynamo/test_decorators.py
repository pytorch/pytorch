# Owner(s): ["module: dynamo"]
import functools
import operator
import os
import re
import unittest.mock as mock
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import IncorrectUsage, Unsupported
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import skipIfWindows


def my_custom_function(x):
    return x + 1


class DecoratorTests(torch._dynamo.test_case.TestCase):
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
        foo.define("custom(Tensor self) -> Tensor")

        # Dynamic shape data dependent operator. For static shape compilation, Dynamo
        # should graph break on it. But, the meta kernel is not implemented properly.
        @torch.library.impl(foo, "custom", "CPU")
        def foo_cpu(x):
            return x.nonzero()

        # Disallow does not work because of extra python frames with torch.library python API
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

        torch.utils._pytree.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
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

        torch.utils._pytree.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
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

        torch.utils._pytree.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
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

        torch.utils._pytree.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
        )

        torch.utils._pytree.register_pytree_node(
            Point,
            lambda p: ((p.x, p.y), ()),
            lambda xy, _: Point(xy[0], xy[1]),
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

        # Assume `State` is implemented in C, and the author didn't bother to
        # provide a pytree decomposition for it, and its instances are safe to
        # treat as a constant by `torch.compile`.
        torch.utils._pytree.register_constant(State)

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

        torch.utils._pytree.register_pytree_node(
            Num,
            lambda num: ((num.n,), ()),
            lambda n, _: Num(n[0]),
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

        torch.utils._pytree.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.p, pt.t), ()),
            lambda pt, _: PointTensor(pt[0], pt[1]),
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
        torch.utils._pytree.register_constant(State)

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
                "Input marked with `pytree.register_constant` constructed in the `torch.compile` region",
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

        torch.utils._pytree.register_pytree_node(
            PointTensor,
            lambda pt: ((pt.t,), pt.p),
            lambda ts, p: PointTensor(p, ts[0]),
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

        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        self.assertEqual(
            fn(B), torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)(B)
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
