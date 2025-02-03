# Owner(s): ["module: dynamo", "module: higher order operators"]
from dataclasses import dataclass

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree


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
        from torch._higher_order_ops.flat_apply import (
            ConstantFunction,
            flat_apply,
            is_graphable,
            to_graphable,
        )

        empty_list, func_spec = pytree.tree_flatten(ConstantFunction(distance))
        self.assertEqual(empty_list, [])

        flat_args, in_spec = to_graphable((args, kwargs))

        for arg in flat_args:
            self.assertTrue(is_graphable(arg))

        # Test flat_apply returns same thing as original function
        result = flat_apply(func_spec, in_spec, *flat_args)
        self.assertEqual(result, distance(*args, **kwargs))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
