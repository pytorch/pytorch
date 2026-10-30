# Owner(s): ["module: dynamo"]
"""Tests for BuiltinTypeVariable: builtin type objects such as dict, list and bytearray."""

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class BuiltinTypeVariableTests(torch._dynamo.test_case.TestCase):
    @parametrize("ty", [dict, list, bytearray], name_fn=lambda ty: ty.__name__)
    def test_reconstruct(self, ty):
        # type(ty()) has no source, so returning it calls BuiltinTypeVariable.reconstruct
        def fn(x, ty):
            return type(ty()), x + 1

        x = torch.randn(3)
        res_ty, res_x = torch.compile(fn, backend="eager", fullgraph=True)(x, ty)
        self.assertIs(res_ty, ty)
        self.assertEqual(res_x, x + 1)

    @parametrize("ty", [dict, list, bytearray], name_fn=lambda ty: ty.__name__)
    def test_reconstruct_across_graph_break(self, ty):
        def fn(x, ty):
            res_ty = type(ty())
            torch._dynamo.graph_break()
            return res_ty, x + 1

        x = torch.randn(3)
        res_ty, res_x = torch.compile(fn, backend="eager")(x, ty)
        self.assertIs(res_ty, ty)
        self.assertEqual(res_x, x + 1)

    @parametrize("ty", [dict, list, bytearray], name_fn=lambda ty: ty.__name__)
    def test_repr_str_of_type(self, ty):
        def fn(ty):
            return repr(ty), str(ty)

        res = torch.compile(fn, backend="eager", fullgraph=True)(ty)
        self.assertEqual(res, fn(ty))

    @parametrize("ty", [dict, list, bytearray], name_fn=lambda ty: ty.__name__)
    def test_type_members(self, ty):
        def fn(ty):
            return (
                ty.__mro__,
                ty.__bases__,
                ty.__base__,
                ty.__flags__,
                ty.__basicsize__,
                ty.__itemsize__,
            )

        res = torch.compile(fn, backend="eager", fullgraph=True)(ty)
        self.assertEqual(res, fn(ty))

    @parametrize(
        "ty_val",
        [(dict, {1: 2}), (list, [1, 2]), (bytearray, bytearray(b"ab"))],
        name_fn=lambda ty_val: ty_val[0].__name__,
    )
    def test_unbound_repr_str(self, ty_val):
        ty, val = ty_val

        def fn(ty, val):
            return ty.__repr__(val), ty.__str__(val)

        res = torch.compile(fn, backend="eager", fullgraph=True)(ty, val)
        self.assertEqual(res, fn(ty, val))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
