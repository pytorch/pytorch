# Owner(s): ["module: dynamo"]
"""Tests for the per-type tp_methods / tp_getset / tp_members of ConstantVariable."""

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


def _name(*parts):
    return "_".join(str(p).replace(".", "_") for p in parts)


@instantiate_parametrized_tests
class ConstantVariableTableTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "val,attr",
        [
            (3, "real"),
            (3, "imag"),
            (3, "numerator"),
            (3, "denominator"),
            (True, "numerator"),
            (2.5, "real"),
            (2.5, "imag"),
            (3j, "real"),
            (3j, "imag"),
        ],
        name_fn=lambda val, attr: _name(type(val).__name__, attr),
    )
    def test_getset_and_member_attrs(self, val, attr):
        def fn(v):
            return getattr(v, attr)

        res = torch.compile(fn, backend="eager", fullgraph=True)(val)
        self.assertEqual(res, fn(val))
        self.assertIs(type(res), type(fn(val)))

    @parametrize(
        "val,attr",
        [(3, "upper"), ("abc", "bit_length"), (3j, "numerator"), (2.5, "numerator")],
        name_fn=lambda val, attr: _name(type(val).__name__, attr),
    )
    def test_attr_of_other_constant_type_is_missing(self, val, attr):
        def fn(v):
            return hasattr(v, attr)

        res = torch.compile(fn, backend="eager", fullgraph=True)(val)
        self.assertFalse(res)
        self.assertEqual(res, fn(val))

    @parametrize(
        "meth,val",
        [
            (int.bit_length, 255),
            (int.bit_length, True),
            (str.upper, "abc"),
            (float.is_integer, 2.0),
            (complex.conjugate, 3j),
            (bytes.hex, b"ab"),
        ],
        name_fn=lambda meth, val: _name(meth.__qualname__, type(val).__name__),
    )
    def test_unbound_method_on_constant(self, meth, val):
        def fn():
            return meth(val)

        res = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(res, fn())

    def test_unbound_method_result_list_is_mutable(self):
        def fn():
            parts = str.split("a b")
            parts.append("c")
            return parts

        res = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(res, fn())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
