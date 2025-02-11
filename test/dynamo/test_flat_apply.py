# Owner(s): ["module: dynamo", "module: higher order operators"]
from dataclasses import dataclass

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
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


@dataclass
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

    def test_mark_traceable_dynamo_graph(self):
        @torch._dynamo.mark_traceable
        def func(x, y, z):
            torch._dynamo.graph_break()
            return x * y + z

        backend = EagerAndRecordGraphs()

        @torch.compile(fullgraph=True, backend=backend)
        def fn(x, y):
            t0 = x + 1
            t1 = func(x, y, t0)
            t2 = t1 + y
            return t2

        fn(torch.randn(10), torch.randn(10))
        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[10]", L_y_: "f32[10]"):
        l_x_ = L_x_
        l_y_ = L_y_

        t0: "f32[10]" = l_x_ + 1

        func_0 = self.func_0
        func_in_spec_0 = self.func_in_spec_0
        t1: "f32[10]" = torch.ops.higher_order.flat_apply(func_0, func_in_spec_0, l_x_, l_y_, t0);  func_0 = func_in_spec_0 = l_x_ = t0 = None

        t2: "f32[10]" = t1 + l_y_;  t1 = l_y_ = None
        return (t2,)
""",  # NOQA: B950
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
