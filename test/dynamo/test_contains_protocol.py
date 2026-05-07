# Owner(s): ["module: dynamo"]

"""
Tests for the `in` operator and __contains__ protocol in PyTorch Dynamo.

Tests cover:
- sq_contains protocol: list, tuple, str, range, set, frozenset
- mp_contains protocol: dict, dict.keys()
- Fallback iteration: objects with __iter__ but no __contains__
- Sequence-protocol fallback: objects with __getitem__ but no __contains__/__iter__
- operator.contains() — exercises the builtin.py call_contains path
- User-defined classes with __contains__
- User-defined classes without __contains__ (iteration fallback)
- Error handling: unhashable types, non-iterable objects, __contains__ raises,
  non-bool return values, str TypeError, mid-iteration raise
"""

import operator
import types

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


# ---------------------------------------------------------------------------
# Helper classes defined at module level (not inside compiled functions)
# ---------------------------------------------------------------------------


class WithContains:
    """Has explicit __contains__."""

    def __init__(self, data):
        self.data = data

    def __contains__(self, item):
        return item in self.data


class WithIterNoContains:
    """Has __iter__ but no __contains__ — forces iteration fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class WithGetitemNoContains:
    """Sequence protocol via __getitem__, no __contains__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]


class WithContainsAndIter:
    """Has both __contains__ and __iter__; __contains__ should be preferred."""

    def __init__(self, data):
        self.data = data
        self.iter_calls = 0

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.data)


class ListIterWrapper:
    """List wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled so membership tests differ from base list
        return iter([x * 2 for x in self.data])


class ListGetitemWrapper:
    """List wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # shifts each value up by 100
        return self.data[idx] + 100


class DictIterWrapper:
    """Dict wrapper with custom __iter__ (over values) and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data.values())


class TupleIterWrapper:
    """Tuple wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled
        return iter([x * 2 for x in self.data])


class TupleGetitemWrapper:
    """Tuple wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # shifts each value up by 100
        return self.data[idx] + 100


class SetIterWrapper:
    """Set wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled
        return iter([x * 2 for x in self.data])


class DictGetitemWrapper:
    """Dict wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data
        self.sorted_keys = sorted(data.keys())

    def __getitem__(self, idx):
        # access values by index via sorted keys
        return self.data[self.sorted_keys[idx]]


class ListSubclassCustomContains(list):
    """Subclass of list that overrides __contains__."""

    def __contains__(self, item):
        return item in [x * 2 for x in super().__iter__()]


class DictSubclassCustomContains(dict):
    """dict subclass whose __contains__ checks values instead of keys."""

    def __contains__(self, item):
        return item in self.values()


class SetSubclassCustomContains(set):
    """set subclass whose __contains__ negates the base class result."""

    def __contains__(self, item):
        return not super().__contains__(item)


class ContainsRaisesTypeError:
    """__contains__ unconditionally raises TypeError."""

    def __contains__(self, item):
        raise TypeError("bad operand")


class ContainsReturnsTruthy:
    """__contains__ returns a non-bool truthy value."""

    def __contains__(self, item):
        return 42


class ContainsReturnsFalsy:
    """__contains__ returns a non-bool falsy value."""

    def __contains__(self, item):
        return 0


class NoIterNoContains:
    """Has neither __iter__ nor __contains__ — triggers TypeError on `in`."""


class RaisesDuringIter:
    """Iterator that raises ValueError partway through."""

    def __iter__(self):
        yield 1
        yield 2
        raise ValueError("mid-iteration error")


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class _ContainsBase:
    """Base test class for __contains__ protocol with parameterized types."""

    thetype = None  # Override in subclass
    data = [1, 2, 3]
    empty = []
    item = 2
    missing_item = 4
    has_contains = True  # Override in subclass if type doesn't implement __contains__
    has_iter = True  # Override in subclass if type doesn't implement __iter__
    has_getitem = True  # Override in subclass if type doesn't implement __getitem__

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def init(self, thetype, data):
        return thetype(data)

    @make_dynamo_test
    def test_contains(self):
        # Basic membership tests for the main type under test
        seq = self.init(self.thetype, self.data)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item in seq)

    @make_dynamo_test
    def test_contains_negation(self):
        # Test `not in` operator
        seq = self.init(self.thetype, self.data)
        self.assertFalse(self.item not in seq)
        self.assertTrue(self.missing_item not in seq)

    @make_dynamo_test
    def test_contains_operator_module(self):
        # Test operator.contains()
        seq = self.init(self.thetype, self.data)
        self.assertTrue(operator.contains(seq, self.item))
        self.assertFalse(operator.contains(seq, self.missing_item))

    @make_dynamo_test
    def test_contains_empty(self):
        # Test on empty container
        seq = self.init(self.thetype, self.empty)
        self.assertFalse(self.item in seq)
        self.assertTrue(self.missing_item not in seq)

    @make_dynamo_test
    def test_has_contains_method(self):
        # Verify whether the type has __contains__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__contains__")
        self.assertEqual(
            has_method,
            self.has_contains,
            f"{self.thetype.__name__} __contains__ presence mismatch",
        )

    @make_dynamo_test
    def test_has_iter_method(self):
        # Verify whether the type has __iter__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__iter__")
        self.assertEqual(
            has_method,
            self.has_iter,
            f"{self.thetype.__name__} __iter__ presence mismatch",
        )

    @make_dynamo_test
    def test_has_getitem_method(self):
        # Verify whether the type has __getitem__ method as expected
        seq = self.init(self.thetype, self.data)
        has_method = hasattr(seq, "__getitem__")
        self.assertEqual(
            has_method,
            self.has_getitem,
            f"{self.thetype.__name__} __getitem__ presence mismatch",
        )


class ListContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = list


class TupleContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = tuple


class StrContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = str
    data = "abc"
    item = "b"
    missing_item = "d"


class DictContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = dict
    data = {"a": 1, "b": 2, "c": 3}
    item = "b"
    missing_item = "d"
    empty = {}


class MappingProxyContainsTest(DictContainsTest):
    thetype = types.MappingProxyType


class SetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = set
    has_getitem = False


class FrozensetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = frozenset
    has_getitem = False


class WithContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithContains
    has_iter = False
    has_getitem = False


class WithIterNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithIterNoContains
    has_contains = False
    has_getitem = False


class WithGetitemNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithGetitemNoContains
    has_contains = False
    has_iter = False


class WithContainsAndIterTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithContainsAndIter
    has_getitem = False


class ListIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 1
    has_contains = False
    has_getitem = False


class ListGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class DictIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictIterWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will iterate over values
    missing_item = "a"
    has_contains = False
    has_getitem = False


class TupleIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = TupleIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 10
    has_contains = False
    has_getitem = False


class TupleGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = TupleGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class SetIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = SetIterWrapper
    data = {1, 2, 3}
    item = 4  # Will be doubled by __iter__
    missing_item = 10
    has_contains = False
    has_getitem = False


class DictGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictGetitemWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will be retrieved by __getitem__
    missing_item = 10
    has_contains = False
    has_iter = False


class ListSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ checks doubled values
    missing_item = 10


class DictSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictSubclassCustomContains
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Custom __contains__ checks values not keys
    missing_item = 10


class SetSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = SetSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ negates base class result
    missing_item = 1
    has_getitem = False

    @make_dynamo_test
    def test_contains_empty(self):
        # Test on empty container
        seq = self.thetype(self.empty)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item not in seq)


class RangeContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = range
    data = (0, 10, 2)
    item = 2
    missing_item = 10
    empty = (0, 0, 1)

    def init(self, thetype, data):
        return thetype(*data)


class ContainsNonBoolReturnTest(torch._dynamo.test_case.TestCase):
    """Test __contains__ that returns non-bool truthy/falsy values."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains_truthy(self):
        seq = ContainsReturnsTruthy()
        # Non-bool truthy return values should be coerced to True
        result1 = 2 in seq
        result2 = None in seq
        return result1 and result2

    @make_dynamo_test
    def test_contains_falsy(self):
        seq = ContainsReturnsFalsy()
        # Non-bool falsy return values should be coerced to False
        result1 = 2 in seq
        result2 = None in seq
        return not result1 and not result2


class NoIterNoContainsTest(torch._dynamo.test_case.TestCase):
    """Tests for objects with neither __iter__ nor __contains__"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_no_iter_no_contains_raises_typeerror(self):
        # Should raise TypeError when neither __iter__ nor __contains__ exists
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 in obj

    @make_dynamo_test
    def test_no_iter_no_contains_not_in_raises_typeerror(self):
        # Should raise TypeError for `not in` operator as well
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 not in obj


class RaisesDuringIterTest(torch._dynamo.test_case.TestCase):
    """Tests for iterators that raise exceptions during iteration"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_raises_during_iter_found_before_error(self):
        # Item 1 found before error occurs
        obj = RaisesDuringIter()
        result = 1 in obj
        self.assertTrue(result)

    @make_dynamo_test
    def test_raises_during_iter_not_found_raises_error(self):
        # Item not found, error occurs during iteration
        obj = RaisesDuringIter()
        with self.assertRaises(ValueError) as cm:
            _ = 3 in obj
        self.assertEqual(str(cm.exception), "mid-iteration error")


class ContainsRaisesTypeErrorTest(torch._dynamo.test_case.TestCase):
    """Tests for __contains__ that raises TypeError"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains_raises_typeerror(self):
        # __contains__ raises TypeError
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 in obj
        self.assertEqual(str(cm.exception), "bad operand")

    @make_dynamo_test
    def test_contains_raises_typeerror_not_in(self):
        # __contains__ raises TypeError for `not in` operator
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 not in obj
        self.assertEqual(str(cm.exception), "bad operand")


class RangeContainsMiscTest(torch._dynamo.test_case.TestCase):
    """Specific tests for range __contains__ protocol"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_range_basic(self):
        # Basic range containment
        seq = range(5)
        self.assertTrue(2 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_with_start_stop(self):
        # Test range with start and stop
        seq = range(5, 15)
        self.assertTrue(10 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(15 in seq)

    @make_dynamo_test
    def test_range_with_step(self):
        # Test range with step
        seq = range(0, 10, 2)
        self.assertTrue(4 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_negative_step(self):
        # Test range with negative step
        seq = range(10, 0, -1)
        self.assertTrue(5 in seq)
        self.assertFalse(0 in seq)
        self.assertFalse(11 in seq)

    @make_dynamo_test
    def test_range_empty(self):
        # Test empty range
        seq = range(5, 5)
        self.assertFalse(5 in seq)
        self.assertFalse(4 in seq)

    @make_dynamo_test
    def test_range_single_element(self):
        # Test range with single element
        seq = range(5, 6)
        self.assertTrue(5 in seq)
        self.assertFalse(4 in seq)
        self.assertFalse(6 in seq)

    @make_dynamo_test
    def test_range_negative_numbers(self):
        # Test range with negative numbers
        seq = range(-5, 5)
        self.assertTrue(-3 in seq)
        self.assertTrue(0 in seq)
        self.assertFalse(-6 in seq)
        self.assertFalse(5 in seq)

    @make_dynamo_test
    def test_range_negation(self):
        # Test `not in` operator
        seq = range(5)
        self.assertFalse(2 not in seq)
        self.assertTrue(10 not in seq)

    @make_dynamo_test
    def test_range_operator_module(self):
        # Test operator.contains()
        seq = range(10)
        self.assertTrue(operator.contains(seq, 5))
        self.assertFalse(operator.contains(seq, 15))


class SetInSetTest(torch._dynamo.test_case.TestCase):
    """
    CPython's set.__contains__ converts an unhashable set key to frozenset and
    retries, so `{1, 2} in {frozenset({1, 2})}` returns True even though set is
    not hashable.  Ref: Objects/setobject.c::set_contains.
    """

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_set_in_frozenset_set_found(self):
        s1 = {1, 2}
        s2 = {frozenset({1, 2}), frozenset({3, 4})}
        self.assertTrue(s1 in s2)

    @make_dynamo_test
    def test_set_in_frozenset_set_not_found(self):
        s1 = {5, 6}
        s2 = {frozenset({1, 2}), frozenset({3, 4})}
        self.assertFalse(s1 in s2)

    @make_dynamo_test
    def test_empty_set_in_frozenset_set(self):
        s2 = {frozenset(), frozenset({1})}
        self.assertTrue(set() in s2)

    @make_dynamo_test
    def test_set_not_in_operator(self):
        s1 = {1, 2}
        s2 = {frozenset({1, 2})}
        self.assertFalse(s1 not in s2)

    @make_dynamo_test
    def test_set_missing_not_in_operator(self):
        s1 = {9}
        s2 = {frozenset({1, 2})}
        self.assertTrue(s1 not in s2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
