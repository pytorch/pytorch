# Owner(s): ["module: dynamo", "module: higher order operators"]
from dataclasses import dataclass

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.flat_apply import (
    flat_apply,
    func_to_graphable,
    is_graphable,
    to_graphable,
)


def distance(a, b, norm):
    if norm.typ == "l2":
        return torch.sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    elif norm.typ == "l1":
        return (a.x - b.x).abs() + (a.y - b.y).abs()


@dataclass(frozen=True)
class Norm:
    typ: str


pytree.register_constant(Norm)


@dataclass
class Point:
    x: torch.Tensor
    y: torch.Tensor


pytree.register_dataclass(Point)


class FlatApplyTests(torch._dynamo.test_case.TestCase):
    def test_simple(self):
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))
        norm = Norm("l2")

        args = (a, b)
        kwargs = {"norm": norm}

        empty_list, func_spec = func_to_graphable(distance)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, distance(*args, **kwargs))

    def test_non_tensor_output(self):
        tensor = torch.tensor

        a = Point(tensor(0.0), tensor(0.0))
        b = Point(tensor(3.0), tensor(4.0))

        args = (a, b)
        kwargs = {}

        def f(a, b):
            return [a.x + 1, (b.x + 2, [a.y + 3, 4.0], "5"), 6 + b.y]

        empty_list, func_spec = func_to_graphable(f)
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, f(*args, **kwargs))

    def test_nonstrict_trace_dynamo_graph(self):
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

        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            p = Point(x, y)
            t = x + y
            pt = PointTensor(p, t)
            res = trace_point_tensor(pt)
            return res

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10]", L_y_: "f32[10]"):
        l_x_ = L_x_
        l_y_ = L_y_

        t: "f32[10]" = l_x_ + l_y_

        trace_point_tensor_spec : torch.utils.pytree.python.PyTreeSpec = self.trace_point_tensor_spec
        trace_point_tensor_input_spec : torch.utils.pytree.python.PyTreeSpec = self.trace_point_tensor_input_spec
        res: "f32[10]" = torch.ops.higher_order.flat_apply(trace_point_tensor_spec, trace_point_tensor_input_spec, l_x_, l_y_, t);  trace_point_tensor_spec = trace_point_tensor_input_spec = l_x_ = l_y_ = t = None
        return (res,)
""",  # NOQA: B950
        )

    def test_nonstrict_trace_captured_tensor_post_aot_graph(self):
        cst = torch.ones(1)

        @torch._dynamo.nonstrict_trace
        def trace_me(x, y):
            torch._dynamo.graph_break()
            return x * y + cst

        backend = AotEagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            return trace_me(x, y)

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[10]", arg1_1: "f32[10]"):
        mul: "f32[10]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        _tensor_constant0 = self._tensor_constant0
        add: "f32[10]" = torch.ops.aten.add.Tensor(mul, _tensor_constant0);  mul = _tensor_constant0 = None
        return (add,)
""",  # NOQA: B950
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
