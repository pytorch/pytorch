# Owner(s): ["module: dynamo"]

"""Tests for nb_or (|, or) and nb_subtract (-) operators in PyTorch Dynamo."""

import collections
import functools

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet


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

    # --- Bitwise | OrderedSet ---

    @make_dynamo_test
    def test_orderedset_union_basic(self):
        operand1 = OrderedSet([1, 2])
        operand2 = OrderedSet([2, 3])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_single(self):
        operand1 = OrderedSet([1])
        operand2 = OrderedSet([2])
        expected = OrderedSet([1, 2])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_duplicate(self):
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([1, 2, 3])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 | operand2, expected)

    @make_dynamo_test
    def test_orderedset_union_with_empty(self):
        self.assertEqual(OrderedSet([1, 2]) | OrderedSet(), OrderedSet([1, 2]))
        self.assertEqual(OrderedSet() | OrderedSet([1, 2]), OrderedSet([1, 2]))

    @make_dynamo_test
    def test_orderedset_union_empty(self):
        self.assertEqual(OrderedSet() | OrderedSet(), OrderedSet())

    @make_dynamo_test
    def test_orderedset_union_chained(self):
        self.assertEqual(
            OrderedSet([1]) | OrderedSet([2]) | OrderedSet([3]),
            OrderedSet([1, 2, 3]),
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


# --- Helper classes for sub tests ---

# Rat class from CPython test_binop.py for testing tp_as_number.tp_subtract


def _gcd(a, b):
    """Greatest common divisor using Euclid's algorithm."""
    while a:
        a, b = b % a, a
    return b


def _isint(x):
    """Test whether an object is an instance of int."""
    return isinstance(x, int)


def _isnum(x):
    """Test whether an object is an instance of a built-in numeric type."""
    for T in int, float, complex:
        if isinstance(x, T):
            return 1
    return 0


def _isRat(x):
    """Test whether an object is an instance of the Rat class."""
    return isinstance(x, Rat)


class Rat:
    """Rational number implemented as a normalized pair of ints."""

    __slots__ = ["__num", "__den"]

    def __init__(self, num=0, den=1):
        """Constructor: Rat([num[, den]]).

        The arguments must be ints, and default to (0, 1)."""
        if not _isint(num):
            raise TypeError("Rat numerator must be int ({num})")
        if not _isint(den):
            raise TypeError("Rat denominator must be int ({den})")
        # But the zero is always on
        if den == 0:
            raise ZeroDivisionError("zero denominator")
        g = _gcd(den, num)
        self.__num = int(num // g)
        self.__den = int(den // g)

    def _get_num(self):
        """Accessor function for read-only 'num' attribute of Rat."""
        return self.__num

    num = property(_get_num, None)

    def _get_den(self):
        """Accessor function for read-only 'den' attribute of Rat."""
        return self.__den

    den = property(_get_den, None)

    def __repr__(self):
        """Convert a Rat to a string resembling a Rat constructor call."""
        return f"Rat({self.__num}, {self.__den})"

    def __str__(self):
        """Convert a Rat to a string resembling a decimal numeric value."""
        return str(float(self))

    def __float__(self):
        """Convert a Rat to a float."""
        return self.__num * 1.0 / self.__den

    def __int__(self):
        """Convert a Rat to an int; self.den must be 1."""
        if self.__den == 1:
            try:
                return int(self.__num)
            except OverflowError:
                raise OverflowError(
                    f"{repr(self)} too large to convert to int"
                ) from None
        raise ValueError(f"can't convert {repr(self)} to int")

    def __add__(self, other):
        """Add two Rats, or a Rat and a number."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                self.__num * other.__den + other.__num * self.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return float(self) + other
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        """Subtract two Rats, or a Rat and a number."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                self.__num * other.__den - other.__num * self.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return float(self) - other
        return NotImplemented

    def __rsub__(self, other):
        """Subtract two Rats, or a Rat and a number (reversed args)."""
        if _isint(other):
            other = Rat(other)
        if _isRat(other):
            return Rat(
                other.__num * self.__den - self.__num * other.__den,
                self.__den * other.__den,
            )
        if _isnum(other):
            return other - float(self)
        return NotImplemented

    def __mul__(self, other):
        """Multiply two Rats, or a Rat and a number."""
        if _isRat(other):
            return Rat(self.__num * other.__num, self.__den * other.__den)
        if _isint(other):
            return Rat(self.__num * other, self.__den)
        if _isnum(other):
            return float(self) * other
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Divide two Rats, or a Rat and a number."""
        if _isRat(other):
            return Rat(self.__num * other.__den, self.__den * other.__num)
        if _isint(other):
            return Rat(self.__num, self.__den * other)
        if _isnum(other):
            return float(self) / other
        return NotImplemented

    def __rtruediv__(self, other):
        """Divide two Rats, or a Rat and a number (reversed args)."""
        if _isRat(other):
            return Rat(other.__num * self.__den, other.__den * self.__num)
        if _isint(other):
            return Rat(other * self.__den, self.__num)
        if _isnum(other):
            return other / float(self)
        return NotImplemented

    def __floordiv__(self, other):
        """Divide two Rats, returning the floored result."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        x = self / other
        return x.__num // x.__den

    def __rfloordiv__(self, other):
        """Divide two Rats, returning the floored result (reversed args)."""
        x = other / self
        return x.__num // x.__den

    def __divmod__(self, other):
        """Divide two Rats, returning quotient and remainder."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        x = self // other
        return (x, self - other * x)

    def __rdivmod__(self, other):
        """Divide two Rats, returning quotient and remainder (reversed args)."""
        if _isint(other):
            other = Rat(other)
        elif not _isRat(other):
            return NotImplemented
        return divmod(other, self)

    def __mod__(self, other):
        """Take one Rat modulo another."""
        return divmod(self, other)[1]

    def __rmod__(self, other):
        """Take one Rat modulo another (reversed args)."""
        return divmod(other, self)[1]

    def __eq__(self, other):
        """Compare two Rats for equality."""
        if _isint(other):
            return self.__den == 1 and self.__num == other
        if _isRat(other):
            return self.__num == other.__num and self.__den == other.__den
        if _isnum(other):
            return float(self) == other
        return NotImplemented


class UserDefinedClassWithSub:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        if isinstance(other, UserDefinedClassWithSub):
            return UserDefinedClassWithSub(self.value - other.value)
        return UserDefinedClassWithSub(self.value - other)

    def __rsub__(self, other):
        if isinstance(other, UserDefinedClassWithSub):
            return UserDefinedClassWithSub(other.value - self.value)
        return UserDefinedClassWithSub(other - self.value)

    def __eq__(self, other):
        if isinstance(other, UserDefinedClassWithSub):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"UserDefinedClassWithSub({self.value})"


class LeftSubClass:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        if isinstance(other, LeftSubClass):
            return LeftSubClass(self.value - other.value)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, LeftSubClass):
            return LeftSubClass(other.value - self.value)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, LeftSubClass) and self.value == other.value

    def __repr__(self):
        return f"LeftSubClass({self.value})"


class RightSubClass:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        if isinstance(other, RightSubClass):
            return RightSubClass(self.value - other.value)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, LeftSubClass):
            return f"LeftSubClass({other.value})-RightSubClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, RightSubClass) and self.value == other.value

    def __repr__(self):
        return f"RightSubClass({self.value})"


class _IntSubWithSub(int):
    def __sub__(self, other):
        return "_IntSubWithSub.__sub__"

    def __rsub__(self, other):
        return "_IntSubWithSub.__rsub__"


class _BaseWithSub:
    def __sub__(self, other):
        return "_BaseWithSub.__sub__"

    def __rsub__(self, other):
        return "_BaseWithSub.__rsub__"


class _SubWithSub(_BaseWithSub):
    def __sub__(self, other):
        return "_SubWithSub.__sub__"

    def __rsub__(self, other):
        return "_SubWithSub.__rsub__"


class _InheritedSubSub(_BaseWithSub):
    pass


# --- Helper classes for NotImplemented tests ---


class _SubNotImplemented:
    """Class where __sub__ returns NotImplemented"""

    def __sub__(self, other):
        return NotImplemented


class _RSubNotImplemented:
    """Class where __sub__ exists but __rsub__ returns NotImplemented"""

    def __sub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented


class _SubReturnsMarker:
    """Class where __sub__ returns a marker string"""

    def __sub__(self, other):
        return NotImplemented


class _RSubReturnsMarker:
    """Class where __rsub__ returns a marker string"""

    def __rsub__(self, other):
        return "_RSubReturnsMarker.__rsub__ called"


class TestNbSub(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Arithmetic sub ---

    @make_dynamo_test
    def test_sub_integers(self):
        self.assertEqual(10 - 3, 7)
        self.assertEqual(0 - 5, -5)
        self.assertEqual(5 - 5, 0)

    @make_dynamo_test
    def test_sub_floats(self):
        self.assertAlmostEqual(1.5 - 0.5, 1.0)
        self.assertAlmostEqual(3.14 - 1.14, 2.0)

    @make_dynamo_test
    def test_sub_negative(self):
        self.assertEqual(-3 - (-5), 2)
        self.assertEqual(-3 - 5, -8)

    @make_dynamo_test
    def test_sub_chained(self):
        self.assertEqual(10 - 3 - 2 - 1, 4)

    @make_dynamo_test
    def test_sub_zero(self):
        self.assertEqual(5 - 0, 5)
        self.assertEqual(0 - 0, 0)

    # --- Sub booleans ---

    @make_dynamo_test
    def test_sub_bools(self):
        self.assertEqual(True - True, 0)
        self.assertEqual(True - False, 1)
        self.assertEqual(False - False, 0)

    @make_dynamo_test
    def test_sub_int_and_bool(self):
        self.assertEqual(5 - True, 4)
        self.assertEqual(0 - True, -1)

    @make_dynamo_test
    def test_sub_int_mixed(self):
        self.assertEqual(100 - 1, 99)
        self.assertEqual(1 - 100, -99)
        self.assertEqual(-10 - (-10), 0)
        x = 42
        self.assertEqual(x - 2, 40)

    @make_dynamo_test
    def test_sub_float_mixed(self):
        self.assertAlmostEqual(0.1 - 0.1, 0.0)
        self.assertAlmostEqual(1.0 - 0.3, 0.7)
        self.assertAlmostEqual(-1.5 - (-2.5), 1.0)
        x = 3.14
        self.assertAlmostEqual(x - 1.0, 2.14)

    @make_dynamo_test
    def test_sub_complex(self):
        self.assertEqual((3 + 4j) - (1 + 2j), 2 + 2j)
        self.assertEqual((1 + 0j) - (0 + 1j), 1 - 1j)
        self.assertEqual((0 + 0j) - (0 + 0j), 0j)
        self.assertEqual((-1 - 1j) - (-1 - 1j), 0j)

    @make_dynamo_test
    def test_sub_complex_with_real(self):
        self.assertEqual((5 + 3j) - 2, 3 + 3j)
        self.assertEqual((5 + 3j) - 2.0, 3 + 3j)
        self.assertEqual(10 - (3 + 4j), 7 - 4j)
        self.assertEqual(1.5 - (0.5 + 1j), 1 - 1j)

    # --- Set difference ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ({1, 2, 3}, {2, 3}, {1}),
            ({1, 2}, {2}, {1}),
            ({1, 2, 3}, {4, 5}, {1, 2, 3}),
        ],
    )
    @make_dynamo_test
    def test_set_difference(self, operand1, operand2, expected):
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_set_difference_with_empty(self):
        self.assertEqual({1, 2} - set(), {1, 2})
        self.assertEqual(set() - {1, 2}, set())

    @make_dynamo_test
    def test_set_difference_empty(self):
        self.assertEqual(set() - set(), set())

    @make_dynamo_test
    def test_set_difference_chained(self):
        self.assertEqual({1, 2, 3, 4} - {2} - {3}, {1, 4})

    # --- Frozenset difference ---

    @parametrize(
        "operand1,operand2,expected",
        [
            (frozenset({1, 2, 3}), frozenset({2, 3}), frozenset({1})),
            (frozenset({1, 2}), frozenset({2}), frozenset({1})),
            (frozenset({1, 2, 3}), frozenset({4, 5}), frozenset({1, 2, 3})),
        ],
    )
    @make_dynamo_test
    def test_frozenset_difference(self, operand1, operand2, expected):
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_frozenset_difference_with_empty(self):
        self.assertEqual(frozenset({1, 2}) - frozenset(), frozenset({1, 2}))
        self.assertEqual(frozenset() - frozenset({1, 2}), frozenset())

    # --- Dict view difference ---

    @make_dynamo_test
    def test_dict_keys_difference(self):
        self.assertEqual({"a": 1, "b": 2}.keys() - {"b"}, {"a"})

    @make_dynamo_test
    def test_dict_items_difference(self):
        left = {"a": 1, "b": 2}.items()
        right = {("b", 2)}
        self.assertEqual(left - right, {("a", 1)})

    def test_dict_values_difference_unsupported(self):
        def fn():
            {"a": 1}.values() - {"b": 2}.values()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn()

    # --- OrderedSet difference ---

    @make_dynamo_test
    def test_orderedset_difference_basic(self):
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([2, 3])
        expected = OrderedSet([1])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_single(self):
        operand1 = OrderedSet([1, 2])
        operand2 = OrderedSet([2])
        expected = OrderedSet([1])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_no_overlap(self):
        operand1 = OrderedSet([1, 2, 3])
        operand2 = OrderedSet([4, 5])
        expected = OrderedSet([1, 2, 3])
        self.assertEqual(operand1 - operand2, expected)

    @make_dynamo_test
    def test_orderedset_difference_with_empty(self):
        self.assertEqual(OrderedSet([1, 2]) - OrderedSet(), OrderedSet([1, 2]))
        self.assertEqual(OrderedSet() - OrderedSet([1, 2]), OrderedSet())

    @make_dynamo_test
    def test_orderedset_difference_empty(self):
        self.assertEqual(OrderedSet() - OrderedSet(), OrderedSet())

    @make_dynamo_test
    def test_orderedset_difference_chained(self):
        self.assertEqual(
            OrderedSet([1, 2, 3, 4]) - OrderedSet([2]) - OrderedSet([3]),
            OrderedSet([1, 4]),
        )

    # --- Inplace -= ---

    @make_dynamo_test
    def test_inplace_sub_integers(self):
        x = 10
        x -= 3
        self.assertEqual(x, 7)

    @make_dynamo_test
    def test_inplace_sub_set(self):
        s = {1, 2, 3}
        s -= {2, 3}
        self.assertEqual(s, {1})

    @make_dynamo_test
    def test_inplace_sub_frozenset(self):
        f = frozenset({1, 2, 3})
        f -= frozenset({2, 3})
        self.assertEqual(f, frozenset({1}))

    @make_dynamo_test
    def test_inplace_sub_set_with_empty(self):
        s = {1, 2, 3}
        s -= set()
        self.assertEqual(s, {1, 2, 3})

    @make_dynamo_test
    def test_inplace_sub_frozenset_with_empty(self):
        f = frozenset({1, 2, 3})
        f -= frozenset()
        self.assertEqual(f, frozenset({1, 2, 3}))

    @make_dynamo_test
    def test_inplace_sub_orderedset_mutates_alias(self):
        s = OrderedSet([1, 2, 3])
        alias = s
        s -= OrderedSet([2])
        self.assertEqual(s, OrderedSet([1, 3]))
        self.assertEqual(alias, OrderedSet([1, 3]))

    # --- Reversed sub (__rsub__) ---

    @make_dynamo_test
    def test_reversed_sub_with_integer(self):
        obj = UserDefinedClassWithSub(3)
        result = 10 - obj
        self.assertEqual(result, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_reversed_sub_with_user_defined_object(self):
        obj1 = UserDefinedClassWithSub(5)
        obj2 = UserDefinedClassWithSub(3)
        result = obj1 - obj2
        self.assertEqual(result, UserDefinedClassWithSub(2))

    @make_dynamo_test
    def test_reversed_sub_chained(self):
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        obj3 = UserDefinedClassWithSub(2)
        result = obj1 - obj2 - obj3
        self.assertEqual(result, UserDefinedClassWithSub(5))

    # --- User-defined __sub__ ---

    @make_dynamo_test
    def test_user_defined_sub_basic(self):
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        self.assertEqual(obj1 - obj2, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_user_defined_sub_with_integer(self):
        obj = UserDefinedClassWithSub(10)
        self.assertEqual(obj - 3, UserDefinedClassWithSub(7))

    @make_dynamo_test
    def test_user_defined_sub_zero(self):
        obj = UserDefinedClassWithSub(5)
        self.assertEqual(obj - 0, UserDefinedClassWithSub(5))

    @make_dynamo_test
    def test_user_defined_sub_chained(self):
        obj1 = UserDefinedClassWithSub(10)
        obj2 = UserDefinedClassWithSub(3)
        obj3 = UserDefinedClassWithSub(2)
        self.assertEqual(obj1 - obj2 - obj3, UserDefinedClassWithSub(5))

    # --- Cross-type user-defined sub ---

    @make_dynamo_test
    def test_left_sub_left_uses_sub(self):
        a = LeftSubClass(10)
        b = LeftSubClass(3)
        self.assertEqual(a - b, LeftSubClass(7))

    @make_dynamo_test
    def test_right_sub_right_uses_sub(self):
        a = RightSubClass(10)
        b = RightSubClass(3)
        self.assertEqual(a - b, RightSubClass(7))

    @make_dynamo_test
    def test_left_sub_right_falls_back_to_rsub(self):
        a = LeftSubClass(10)
        b = RightSubClass(3)
        self.assertEqual(a - b, "LeftSubClass(10)-RightSubClass(3)")

    @make_dynamo_test
    def test_right_sub_left_raises(self):
        a = RightSubClass(10)
        b = LeftSubClass(3)
        with self.assertRaises(TypeError):
            a - b

    # --- Subclass right-op dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        self.assertEqual(_IntSubWithSub(1) - 1, "_IntSubWithSub.__sub__")
        self.assertEqual(1 - _IntSubWithSub(1), "_IntSubWithSub.__rsub__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        self.assertEqual(_BaseWithSub() - 1, "_BaseWithSub.__sub__")
        self.assertEqual(1 - _BaseWithSub(), "_BaseWithSub.__rsub__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        self.assertEqual(_SubWithSub() - _BaseWithSub(), "_SubWithSub.__sub__")
        self.assertEqual(_BaseWithSub() - _SubWithSub(), "_SubWithSub.__rsub__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        self.assertIs(_InheritedSubSub.__rsub__, _BaseWithSub.__rsub__)
        self.assertEqual(_InheritedSubSub() - 1, "_BaseWithSub.__sub__")
        self.assertEqual(1 - _InheritedSubSub(), "_BaseWithSub.__rsub__")
        self.assertEqual(_InheritedSubSub() - _BaseWithSub(), "_BaseWithSub.__sub__")
        self.assertEqual(_BaseWithSub() - _InheritedSubSub(), "_BaseWithSub.__sub__")

    # --- Rat class tests from CPython test_binop.py ---

    @make_dynamo_test
    def test_rat_sub(self):
        self.assertEqual(Rat(7, 2) - Rat(7, 5), Rat(21, 10))
        self.assertEqual(Rat(7, 5) - 1, Rat(2, 5))
        self.assertEqual(1 - Rat(3, 5), Rat(2, 5))
        self.assertAlmostEqual(Rat(3, 2) - 1.0, 0.5)
        self.assertAlmostEqual(1.0 - Rat(1, 2), 0.5)

    @make_dynamo_test
    def test_rat_rsub(self):
        # Test reversed subtraction with Rat objects
        obj = Rat(3)
        result = 10 - obj
        self.assertEqual(result, Rat(7))

    @make_dynamo_test
    def test_rat_sub_with_different_denominators(self):
        # More complex rational subtraction
        a = Rat(5, 6)
        b = Rat(1, 3)
        result = a - b
        self.assertEqual(result, Rat(1, 2))

    @make_dynamo_test
    def test_rat_sub_negative_result(self):
        # Subtraction resulting in negative
        a = Rat(1, 4)
        b = Rat(1, 2)
        result = a - b
        self.assertEqual(result, Rat(-1, 4))

    # --- NotImplemented handling from test_descr.py ---

    @make_dynamo_test
    def test_sub_not_implemented_returns_type_error(self):
        # When __sub__ returns NotImplemented, should raise TypeError
        a = _SubNotImplemented()
        with self.assertRaises(TypeError):
            a - a

    @make_dynamo_test
    def test_rsub_not_implemented_returns_type_error(self):
        # When __rsub__ returns NotImplemented, should raise TypeError
        b = _RSubNotImplemented()
        with self.assertRaises(TypeError):
            b - b

    @make_dynamo_test
    def test_sub_mixed_not_implemented_fallback(self):
        # When left __sub__ returns NotImplemented, try right __rsub__
        a = _SubReturnsMarker()
        b = _RSubReturnsMarker()
        result = a - b
        self.assertEqual(result, "_RSubReturnsMarker.__rsub__ called")

    def test_sub_float_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = 2.5 - s
            b = s - 1.25
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_sub_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = 3 - s
            b = s - 7
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_sub_bool_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = True - s
            b = s - False
            return x - a - b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    # --- OrderedSet reversed subtraction (__rsub__) ---

    @make_dynamo_test
    def test_orderedset_rsub_basic(self):
        # Test reversed subtraction: other_set - ordered_set
        os = OrderedSet([2, 3])
        s = {1, 2, 3}
        result = s - os
        self.assertEqual(result, {1})

    @make_dynamo_test
    def test_orderedset_rsub_with_set(self):
        # Test that set - OrderedSet works correctly
        os = OrderedSet([1, 2])
        s = {1, 2, 3}
        result = s - os
        self.assertEqual(result, {3})

    @make_dynamo_test
    def test_orderedset_rsub_both_orderedsets(self):
        # Test OrderedSet - OrderedSet (forward subtraction)
        os1 = OrderedSet([1, 2, 3])
        os2 = OrderedSet([2, 3])
        result = os1 - os2
        self.assertEqual(result, OrderedSet([1]))


class _AddNotImplemented:
    """Class where __add__ returns NotImplemented"""

    def __add__(self, other):
        return NotImplemented


class _RAddNotImplemented:
    """Class where __add__ exists but __radd__ returns NotImplemented"""

    def __add__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented


class _BaseWithAdd:
    def __add__(self, other):
        return "_BaseWithAdd.__add__"

    def __radd__(self, other):
        return "_BaseWithAdd.__radd__"


class _SubWithAdd(_BaseWithAdd):
    def __add__(self, other):
        return "_SubWithAdd.__add__"


class _InheritedSubAdd(_BaseWithAdd):
    pass


class TestNbAdd(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # --- Arithmetic add ---

    @make_dynamo_test
    def test_add_integers(self):
        self.assertEqual(10 + 3, 13)
        self.assertEqual(0 + 5, 5)
        self.assertEqual(5 + 5, 10)

    @make_dynamo_test
    def test_add_floats(self):
        self.assertAlmostEqual(1.5 + 0.5, 2.0)
        self.assertAlmostEqual(3.14 + 1.14, 4.28)

    @make_dynamo_test
    def test_add_negative(self):
        self.assertEqual(-3 + (-5), -8)
        self.assertEqual(-3 + 5, 2)

    @make_dynamo_test
    def test_add_chained(self):
        self.assertEqual(10 + 3 + 2 + 1, 16)

    @make_dynamo_test
    def test_add_zero(self):
        self.assertEqual(5 + 0, 5)
        self.assertEqual(0 + 0, 0)

    # --- Add booleans ---

    @make_dynamo_test
    def test_add_bools(self):
        self.assertEqual(True + True, 2)
        self.assertEqual(True + False, 1)
        self.assertEqual(False + False, 0)

    @make_dynamo_test
    def test_add_int_and_bool(self):
        self.assertEqual(5 + True, 6)
        self.assertEqual(0 + True, 1)

    # --- String concatenation ---

    @make_dynamo_test
    def test_add_strings(self):
        self.assertEqual("hello" + " " + "world", "hello world")
        self.assertEqual("a" + "b", "ab")
        self.assertEqual("" + "test", "test")

    # --- List concatenation ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ([1, 2], [3, 4], [1, 2, 3, 4]),
            ([], [1, 2], [1, 2]),
            ([1, 2], [], [1, 2]),
        ],
    )
    @make_dynamo_test
    def test_add_lists(self, operand1, operand2, expected):
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_add_lists_empty(self):
        self.assertEqual([] + [], [])

    @make_dynamo_test
    def test_add_lists_chained(self):
        self.assertEqual([1] + [2] + [3], [1, 2, 3])
        self.assertEqual([1, 2] + [3] + [4, 5], [1, 2, 3, 4, 5])

    # --- Tuple concatenation ---

    @parametrize(
        "operand1,operand2,expected",
        [
            ((1, 2), (3, 4), (1, 2, 3, 4)),
            ((), (1, 2), (1, 2)),
            ((1, 2), (), (1, 2)),
        ],
    )
    @make_dynamo_test
    def test_add_tuples(self, operand1, operand2, expected):
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_add_tuples_empty(self):
        self.assertEqual(() + (), ())

    @make_dynamo_test
    def test_add_tuples_chained(self):
        self.assertEqual((1,) + (2,) + (3,), (1, 2, 3))

    # --- Inplace +=  ---

    @make_dynamo_test
    def test_inplace_add_integers(self):
        x = 5
        x += 3
        self.assertEqual(x, 8)

    @make_dynamo_test
    def test_inplace_add_floats(self):
        x = 1.5
        x += 2.5
        self.assertAlmostEqual(x, 4.0)

    @make_dynamo_test
    def test_inplace_add_strings(self):
        x = "hello"
        x += " world"
        self.assertEqual(x, "hello world")

    @make_dynamo_test
    def test_inplace_add_lists(self):
        x = [1, 2]
        x += [3, 4]
        self.assertEqual(x, [1, 2, 3, 4])

    # --- Reversed add (__radd__) ---

    @make_dynamo_test
    def test_reversed_add_with_integer(self):
        obj = _BaseWithAdd()
        result = 5 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_radd_priority(self):
        # Test that __radd__ is called when left side doesn't implement __add__
        obj = _BaseWithAdd()
        result = 5 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    # --- Subclass method resolution order ---

    @make_dynamo_test
    def test_subclass_add(self):
        obj = _SubWithAdd()
        result = obj + 1
        self.assertEqual(result, "_SubWithAdd.__add__")

    @make_dynamo_test
    def test_subclass_radd(self):
        obj = _SubWithAdd()
        result = 1 + obj
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_subclass_and_base_add(self):
        sub = _SubWithAdd()
        base = _BaseWithAdd()
        result = sub + base
        self.assertEqual(result, "_SubWithAdd.__add__")

    @make_dynamo_test
    def test_subclass_priority_in_reverse(self):
        self.assertIs(_SubWithAdd.__radd__, _BaseWithAdd.__radd__)
        self.assertEqual(_SubWithAdd() + 1, "_SubWithAdd.__add__")
        self.assertEqual(1 + _SubWithAdd(), "_BaseWithAdd.__radd__")
        self.assertEqual(_SubWithAdd() + _BaseWithAdd(), "_SubWithAdd.__add__")
        self.assertEqual(_BaseWithAdd() + _SubWithAdd(), "_BaseWithAdd.__add__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        self.assertIs(_InheritedSubAdd.__radd__, _BaseWithAdd.__radd__)
        self.assertEqual(_InheritedSubAdd() + 1, "_BaseWithAdd.__add__")
        self.assertEqual(1 + _InheritedSubAdd(), "_BaseWithAdd.__radd__")
        self.assertEqual(_InheritedSubAdd() + _BaseWithAdd(), "_BaseWithAdd.__add__")
        self.assertEqual(_BaseWithAdd() + _InheritedSubAdd(), "_BaseWithAdd.__add__")

    # --- Rat class tests from CPython test_binop.py ---

    @make_dynamo_test
    def test_rat_add(self):
        # Tests from CPython test_binop.py RatTestCase.test_add
        self.assertEqual(Rat(2, 3) + Rat(1, 3), 1)
        self.assertEqual(Rat(2, 3) + 1, Rat(5, 3))
        self.assertEqual(1 + Rat(2, 3), Rat(5, 3))
        self.assertAlmostEqual(1.0 + Rat(1, 2), 1.5)
        self.assertAlmostEqual(Rat(1, 2) + 1.0, 1.5)

    @make_dynamo_test
    def test_rat_add_with_rat(self):
        # Addition of two rationals
        self.assertEqual(Rat(7, 2) + Rat(7, 5), Rat(49, 10))

    @make_dynamo_test
    def test_rat_add_rat_and_integer(self):
        # Addition of rational and integer
        self.assertEqual(Rat(7, 5) + 1, Rat(12, 5))

    @make_dynamo_test
    def test_rat_add_integer_and_rat(self):
        # Addition of integer and rational (tests __radd__)
        self.assertEqual(1 + Rat(3, 5), Rat(8, 5))

    @make_dynamo_test
    def test_rat_add_rat_and_float(self):
        # Addition of rational and float
        self.assertAlmostEqual(Rat(3, 2) + 1.0, 2.5)

    @make_dynamo_test
    def test_rat_add_float_and_rat(self):
        # Addition of float and rational (tests __radd__)
        self.assertAlmostEqual(1.0 + Rat(3, 2), 2.5)

    @make_dynamo_test
    def test_rat_radd(self):
        # Test reversed addition with Rat objects
        obj = Rat(3)
        result = 10 + obj
        self.assertEqual(result, Rat(13))

    @make_dynamo_test
    def test_rat_add_with_different_denominators(self):
        # More complex rational addition
        a = Rat(5, 6)
        b = Rat(1, 3)
        result = a + b
        self.assertEqual(result, Rat(7, 6))

    @make_dynamo_test
    def test_rat_add_result_simplification(self):
        # Addition resulting in simplifiable fraction
        a = Rat(1, 4)
        b = Rat(1, 4)
        result = a + b
        self.assertEqual(result, Rat(1, 2))

    # --- NotImplemented handling from test_descr.py ---

    @make_dynamo_test
    def test_add_not_implemented_left(self):
        # When left operand returns NotImplemented, should try right's __radd__
        obj = _AddNotImplemented()
        # Left returns NotImplemented, right (int) doesn't have __radd__, raises TypeError
        with self.assertRaises(TypeError):
            obj + 5

    @make_dynamo_test
    def test_add_not_implemented_with_custom_radd(self):
        # When left returns NotImplemented, should use right's __radd__ if available
        obj = _AddNotImplemented()
        custom = _BaseWithAdd()
        # Left (__add__) returns NotImplemented, right (__radd__) is called
        result = obj + custom
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    @make_dynamo_test
    def test_radd_not_implemented_uses_left(self):
        # When left has __add__ that works, it should be called even if right's __radd__ returns NotImplemented
        obj = _RAddNotImplemented()
        custom = _BaseWithAdd()
        result = custom + obj
        # Left (__add__) is called and returns successfully, right is not tried
        self.assertEqual(result, "_BaseWithAdd.__add__")

    @make_dynamo_test
    def test_radd_not_implemented_right_used(self):
        # When both __add__ and __radd__ return NotImplemented, should raise TypeError
        obj1 = _RAddNotImplemented()
        obj2 = _RAddNotImplemented()
        with self.assertRaises(TypeError):
            obj1 + obj2

    @make_dynamo_test
    def test_add_not_implemented_reverse_with_integer(self):
        # Right operand with NotImplemented when left is int
        obj = _AddNotImplemented()
        with self.assertRaises(TypeError):
            5 + obj

    @make_dynamo_test
    def test_add_not_implemented_with_int_and_custom(self):
        # _AddNotImplemented returns NotImplemented, then _BaseWithAdd.__radd__ is called
        obj_not_impl = _AddNotImplemented()
        obj_impl = _BaseWithAdd()
        # obj_not_impl.__add__ returns NotImplemented, tries obj_impl.__radd__
        result = obj_not_impl + obj_impl
        self.assertEqual(result, "_BaseWithAdd.__radd__")

    def test_add_float_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = 2.5 + s
            b = s + 1.25
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = 3 + s
            b = s + 7
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_add_bool_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            a = True + s
            b = s + False
            return x + a + b

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


instantiate_parametrized_tests(TestNbOr)
instantiate_parametrized_tests(TestNbSub)
instantiate_parametrized_tests(TestNbAdd)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
