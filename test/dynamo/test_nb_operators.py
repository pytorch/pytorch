# Owner(s): ["module: dynamo"]

"""Tests for | and or operators in PyTorch Dynamo."""

import collections
import functools

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class UserDefinedDict(dict):
    pass


class UserDefinedClassWithOr:
    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(self.value | other.value)
        return UserDefinedClassWithOr(self.value | other)

    def __ror__(self, other):
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(other.value | self.value)
        return UserDefinedClassWithOr(other | self.value)

    def __eq__(self, other):
        if isinstance(other, UserDefinedClassWithOr):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"UserDefinedClassWithOr({self.value})"


class LeftOrClass:
    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, LeftOrClass):
            return LeftOrClass(self.value | other.value)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LeftOrClass):
            return LeftOrClass(other.value | self.value)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, LeftOrClass) and self.value == other.value

    def __repr__(self):
        return f"LeftOrClass({self.value})"


class RightOrClass:
    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, RightOrClass):
            return RightOrClass(self.value + "|" + other.value)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LeftOrClass):
            return f"LeftOrClass({other.value})|RightOrClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, RightOrClass) and self.value == other.value

    def __repr__(self):
        return f"RightOrClass({self.value})"


class MyDict(dict):
    def __or__(self, other):
        return NotImplemented


class MySet(set):
    def __or__(self, other):
        return NotImplemented


class NonDict:
    def __or__(self, other):
        return "wrong result"


class _IntSubWithOr(int):
    def __or__(self, other):
        return "_IntSubWithOr.__or__"

    def __ror__(self, other):
        return "_IntSubWithOr.__ror__"


class _BaseWithOr:
    def __or__(self, other):
        return "_BaseWithOr.__or__"

    def __ror__(self, other):
        return "_BaseWithOr.__ror__"


class _SubWithOr(_BaseWithOr):
    def __or__(self, other):
        return "_SubWithOr.__or__"

    def __ror__(self, other):
        return "_SubWithOr.__ror__"


class _InheritedSub(_BaseWithOr):
    pass


def _make_dict(d):
    return dict(d)


def _make_defaultdict(d):
    return collections.defaultdict(int, d)


def _make_ordered(d):
    return collections.OrderedDict(d.items())


def _make_userdict(d):
    return UserDefinedDict(d)


_DICT_COMBOS = {
    "dict_dict": (_make_dict, _make_dict),
    "dict_defaultdict": (_make_dict, _make_defaultdict),
    "defaultdict_dict": (_make_defaultdict, _make_dict),
    "defaultdict_defaultdict": (_make_defaultdict, _make_defaultdict),
    "dict_ordereddict": (_make_dict, _make_ordered),
    "ordereddict_dict": (_make_ordered, _make_dict),
    "ordereddict_ordereddict": (_make_ordered, _make_ordered),
    "dict_userdict": (_make_dict, _make_userdict),
    "userdict_dict": (_make_userdict, _make_dict),
    "userdict_userdict": (_make_userdict, _make_userdict),
}

_INPLACE_COMBOS = {
    "dict": (dict, {"a": 1, "b": 2}, {"b": 20, "c": 3}, {"a": 1, "b": 20, "c": 3}),
    "set": (set, {1, 2}, {2, 3}, {1, 2, 3}),
    "defaultdict": (
        functools.partial(collections.defaultdict, int),
        {"a": 1, "b": 2},
        {"b": 20, "c": 3},
        {"a": 1, "b": 20, "c": 3},
    ),
}


class TestNbOr(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Logical or ---

    @make_dynamo_test
    def test_or_with_booleans(self):
        self.assertEqual(True or True, True)  # noqa: SIM222
        self.assertEqual(True or False, True)  # noqa: SIM222
        self.assertEqual(False or True, True)  # noqa: SIM222
        self.assertEqual(False or False, False)  # noqa: RUF100

    @make_dynamo_test
    def test_or_with_integers(self):
        self.assertEqual(0 or 5, 5)  # noqa: SIM222
        self.assertEqual(5 or 0, 5)  # noqa: SIM222
        self.assertEqual(3 or 7, 3)  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_strings(self):
        self.assertEqual("" or "hello", "hello")  # noqa: SIM222
        self.assertEqual("hello" or "world", "hello")  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_containers(self):
        self.assertEqual([] or [1, 2], [1, 2])  # noqa: SIM222
        self.assertEqual([1, 2] or [], [1, 2])  # noqa: SIM222
        self.assertEqual(None or 5, 5)  # noqa: SIM222

    @make_dynamo_test
    def test_or_short_circuit(self):
        x = 5
        result = x or (1 / 0)
        self.assertEqual(result, 5)

    @make_dynamo_test
    def test_or_chained(self):
        self.assertEqual(0 or 0 or 3 or 4, 3)  # noqa: SIM222
        self.assertEqual(False or False or True, True)  # noqa: SIM222

    # --- Bitwise | integers ---

    @make_dynamo_test
    def test_bitwise_or_integers(self):
        self.assertEqual(5 | 3, 7)  # 101 | 011 = 111
        self.assertEqual(12 | 10, 14)  # 1100 | 1010 = 1110
        self.assertEqual(5 | 0, 5)
        self.assertEqual(256 | 128, 384)

    @make_dynamo_test
    def test_bitwise_or_negative_integers(self):
        self.assertEqual(-1 | 0, -1)
        self.assertEqual(5 | -1, -1)
        self.assertEqual(-2 | -3, -1)

    @make_dynamo_test
    def test_bitwise_or_integers_chained(self):
        self.assertEqual(1 | 2 | 4 | 8, 15)

    # --- Bitwise | booleans ---

    @make_dynamo_test
    def test_bitwise_or_bools(self):
        self.assertEqual(True | True, True)
        self.assertEqual(True | False, True)
        self.assertEqual(False | False, False)

    @make_dynamo_test
    def test_bitwise_or_int_and_bool(self):
        self.assertEqual(5 | True, 5)
        self.assertEqual(0 | True, 1)

    # --- Bitwise | set ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ({1, 2}, {2, 3}, {1, 2, 3}),
            ({1}, {2}, {1, 2}),
            ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}),
        ],
    )
    @make_dynamo_test
    def test_set_union(self, operand1, operand2, expected):
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_set_union_with_empty(self):
        self.assertEqual({1, 2} | set(), {1, 2})
        self.assertEqual(set() | {1, 2}, {1, 2})

    @make_dynamo_test
    def test_set_union_empty(self):
        self.assertEqual(set() | set(), set())

    @make_dynamo_test
    def test_set_union_chained(self):
        self.assertEqual({1} | {2} | {3}, {1, 2, 3})
        self.assertEqual({1, 2} | {2, 3} | {3, 4}, {1, 2, 3, 4})

    # --- Bitwise | frozenset ---

    @parametrize(
        "operand1,operand2,expected",
        [
            (frozenset({1, 2}), frozenset({2, 3}), frozenset({1, 2, 3})),
            (frozenset({1}), frozenset({2}), frozenset({1, 2})),
            (frozenset({1, 2, 3}), frozenset({1, 2, 3}), frozenset({1, 2, 3})),
        ],
    )
    @make_dynamo_test
    def test_frozenset_union(self, operand1, operand2, expected):
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_frozenset_union_with_empty(self):
        self.assertEqual(frozenset({1, 2}) | frozenset(), frozenset({1, 2}))
        self.assertEqual(frozenset() | frozenset({1, 2}), frozenset({1, 2}))

    @make_dynamo_test
    def test_frozenset_union_empty(self):
        self.assertEqual(frozenset() | frozenset(), frozenset())

    @make_dynamo_test
    def test_frozenset_union_chained(self):
        self.assertEqual(
            frozenset({1}) | frozenset({2}) | frozenset({3}),
            frozenset({1, 2, 3}),
        )

    # --- Bitwise | dict combinations ---

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_operation(self, combo):
        left_fn, right_fn = _DICT_COMBOS[combo]
        left = left_fn({"a": 1, "b": 2})
        right = right_fn({"b": 20, "c": 3})
        self.assertEqual(left | right, {"a": 1, "b": 20, "c": 3})

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_empty(self, combo):
        left_fn, right_fn = _DICT_COMBOS[combo]
        self.assertEqual(left_fn({"a": 1}) | right_fn({}), {"a": 1})

    @parametrize("combo", list(_DICT_COMBOS.keys()))
    @make_dynamo_test
    def test_dict_or_chained(self, combo):
        left_fn, right_fn = _DICT_COMBOS[combo]
        d1 = left_fn({"a": 1})
        d2 = right_fn({"b": 2})
        d3 = left_fn({"c": 3})
        self.assertEqual(d1 | d2 | d3, {"a": 1, "b": 2, "c": 3})

    # --- Inplace |= ---

    @parametrize("combo", list(_INPLACE_COMBOS.keys()))
    @make_dynamo_test
    def test_inplace_or(self, combo):
        container_type, data1, data2, expected = _INPLACE_COMBOS[combo]
        left = container_type(data1)
        right = container_type(data2)
        left |= right
        self.assertEqual(left, expected)

    # --- Reversed or (__ror__) ---

    @make_dynamo_test
    def test_reversed_or_with_integer(self):
        obj = UserDefinedClassWithOr(3)
        result = 5 | obj
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_with_user_defined_object(self):
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        result = obj1 | obj2
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_chained(self):
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        result = 0 | obj1 | obj2 | obj3
        self.assertEqual(result.value, 7)

    @make_dynamo_test
    def test_dict_ror_valid(self):
        d = {"a": 1}
        myd = MyDict({"a": 2})
        result = myd | d
        self.assertEqual(result, {"a": 1})

    @make_dynamo_test
    def test_set_ror_valid(self):
        s = {1, 2}
        mys = MySet({2, 3})
        result = mys | s
        self.assertEqual(result, {1, 2, 3})

    @make_dynamo_test
    def test_non_dict_or_invalid(self):
        d = {"a": 1}
        non_dict = NonDict()
        result = d.__ror__(non_dict)
        self.assertIs(result, NotImplemented)

    # --- Unsupported types ---

    @make_dynamo_test
    def test_list_or_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            [1, 2] | [3, 4]

    @make_dynamo_test
    def test_tuple_or_tuple_raises_type_error(self):
        with self.assertRaises(TypeError):
            (1, 2) | (3, 4)

    @make_dynamo_test
    def test_empty_list_or_list_raises_type_error(self):
        with self.assertRaises(TypeError):
            [] | [1, 2]

    @make_dynamo_test
    def test_empty_tuple_or_tuple_raises_type_error(self):
        with self.assertRaises(TypeError):
            () | (1, 2)

    # --- User-defined __or__ ---

    @make_dynamo_test
    def test_user_defined_or_basic(self):
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        self.assertEqual(obj1 | obj2, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_with_integer(self):
        obj = UserDefinedClassWithOr(5)
        self.assertEqual(obj | 3, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_zero(self):
        obj = UserDefinedClassWithOr(5)
        self.assertEqual(obj | 0, UserDefinedClassWithOr(5))

    @make_dynamo_test
    def test_user_defined_or_chained(self):
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        self.assertEqual(obj1 | obj2 | obj3, UserDefinedClassWithOr(7))

    # --- Cross-type user-defined or ---

    @make_dynamo_test
    def test_left_or_left_uses_or(self):
        a = LeftOrClass(5)
        b = LeftOrClass(3)
        self.assertEqual(a | b, LeftOrClass(7))

    @make_dynamo_test
    def test_right_or_right_uses_or(self):
        a = RightOrClass("a")
        b = RightOrClass("b")
        self.assertEqual(a | b, RightOrClass("a|b"))

    @make_dynamo_test
    def test_left_or_right_falls_back_to_ror(self):
        a = LeftOrClass(42)
        b = RightOrClass("x")
        self.assertEqual(a | b, "LeftOrClass(42)|RightOrClass(x)")

    @make_dynamo_test
    def test_right_or_left_raises(self):
        a = RightOrClass("x")
        b = LeftOrClass(42)
        with self.assertRaises(TypeError):
            a | b

    # --- Tensors ---

    @make_dynamo_test
    def test_tensor_to_bool(self):
        t_nonzero = torch.tensor(1)
        t_zero = torch.tensor(0)
        self.assertTrue(bool(t_nonzero))
        self.assertFalse(bool(t_zero))

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        self.assertEqual(_IntSubWithOr(1) | 1, "_IntSubWithOr.__or__")
        self.assertEqual(1 | _IntSubWithOr(1), "_IntSubWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        self.assertEqual(_BaseWithOr() | 1, "_BaseWithOr.__or__")
        self.assertEqual(1 | _BaseWithOr(), "_BaseWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        self.assertEqual(_SubWithOr() | _BaseWithOr(), "_SubWithOr.__or__")
        self.assertEqual(_BaseWithOr() | _SubWithOr(), "_SubWithOr.__ror__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        self.assertIs(_InheritedSub.__ror__, _BaseWithOr.__ror__)
        self.assertEqual(_InheritedSub() | 1, "_BaseWithOr.__or__")
        self.assertEqual(1 | _InheritedSub(), "_BaseWithOr.__ror__")
        self.assertEqual(_InheritedSub() | _BaseWithOr(), "_BaseWithOr.__or__")
        self.assertEqual(_BaseWithOr() | _InheritedSub(), "_BaseWithOr.__or__")


instantiate_parametrized_tests(TestNbOr)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
