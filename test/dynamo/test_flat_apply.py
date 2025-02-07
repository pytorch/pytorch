# Owner(s): ["module: dynamo", "module: higher order operators"]
from dataclasses import dataclass

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
