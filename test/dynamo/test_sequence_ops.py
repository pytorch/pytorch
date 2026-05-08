# Owner(s): ["module: dynamo"]

"""Tests for sequence protocol operations (sq_*) in PyTorch Dynamo."""

import collections
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


# --- User-defined sequence classes ---


class UserDefinedList(list):
    """User-defined list subclass."""


class UserDefinedTuple(tuple):
    """User-defined tuple subclass."""

    __slots__ = []


class UserDefinedDeque(collections.deque):
    """User-defined deque subclass."""


class UserDefinedSequence:
    """User-defined sequence class with __add__ and __iadd__."""

    def __init__(self, items):
        self.items = list(items)

    def __add__(self, other):
        if isinstance(other, UserDefinedSequence):
            return UserDefinedSequence(self.items + other.items)
        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, UserDefinedSequence):
            self.items.extend(other.items)
            return self
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, UserDefinedSequence):
            return self.items == other.items
        return False

    def __repr__(self):
        return f"UserDefinedSequence({self.items})"


class TestSqConcat(torch._dynamo.test_case.TestCase):
    """Tests for sq_concat (+) and sq_inplace_concat (+=) operators for sequences."""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

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
    def test_list_concat(self, operand1, operand2, expected):
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_list_concat_empty(self):
        self.assertEqual([] + [], [])

    @make_dynamo_test
    def test_list_concat_chained(self):
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
    def test_tuple_concat(self, operand1, operand2, expected):
        self.assertEqual(operand1 + operand2, expected)

    @make_dynamo_test
    def test_tuple_concat_empty(self):
        self.assertEqual(() + (), ())

    @make_dynamo_test
    def test_tuple_concat_chained(self):
        self.assertEqual((1,) + (2,) + (3,), (1, 2, 3))

    # --- String concatenation (via ConstantVariable) ---

    @make_dynamo_test
    def test_string_concat(self):
        self.assertEqual("hello" + " " + "world", "hello world")
        self.assertEqual("a" + "b", "ab")
        self.assertEqual("" + "test", "test")

    @make_dynamo_test
    def test_string_concat_empty(self):
        self.assertEqual("" + "", "")

    # --- Inplace list concatenation (+=) ---

    @make_dynamo_test
    def test_list_inplace_concat(self):
        x = [1, 2]
        x += [3, 4]
        self.assertEqual(x, [1, 2, 3, 4])

    @make_dynamo_test
    def test_list_inplace_concat_empty(self):
        x = []
        x += [1, 2]
        self.assertEqual(x, [1, 2])

    @make_dynamo_test
    def test_list_inplace_concat_to_empty(self):
        x = [1, 2]
        x += []
        self.assertEqual(x, [1, 2])

    # --- Type mismatch errors ---

    @make_dynamo_test
    def test_list_concat_with_tuple_raises(self):
        with self.assertRaises(TypeError):
            [1, 2] + (3, 4)

    @make_dynamo_test
    def test_tuple_concat_with_list_raises(self):
        with self.assertRaises(TypeError):
            (1, 2) + [3, 4]

    @make_dynamo_test
    def test_list_concat_with_string_raises(self):
        with self.assertRaises(TypeError):
            [1, 2] + "string"

    @make_dynamo_test
    def test_list_concat_with_none_raises(self):
        with self.assertRaises(TypeError):
            [1, 2] + None

    # --- User-defined list subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_list_concat(self):
        a = UserDefinedList([1, 2])
        b = UserDefinedList([3, 4])
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_user_defined_list_concat_with_list(self):
        a = UserDefinedList([1, 2])
        b = [3, 4]
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    # --- User-defined tuple subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_tuple_concat(self):
        a = UserDefinedTuple([1, 2])
        b = UserDefinedTuple([3, 4])
        result = a + b
        self.assertEqual(result, (1, 2, 3, 4))

    @make_dynamo_test
    def test_user_defined_tuple_concat_with_tuple(self):
        a = UserDefinedTuple([1, 2])
        b = (3, 4)
        result = a + b
        self.assertEqual(result, (1, 2, 3, 4))

    # --- User-defined deque subclass concatenation ---

    @unittest.expectedFailure
    @make_dynamo_test
    def test_user_defined_deque_concat(self):
        a = UserDefinedDeque([1, 2])
        b = UserDefinedDeque([3, 4])
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @unittest.expectedFailure
    @make_dynamo_test
    def test_user_defined_deque_inplace_concat(self):
        a = UserDefinedDeque([1, 2])
        a += UserDefinedDeque([3, 4])
        self.assertEqual(list(a), [1, 2, 3, 4])

    # --- Deque concatenation ---

    @make_dynamo_test
    def test_deque_concat(self):
        d1 = collections.deque([1, 2])
        d2 = collections.deque([3, 4])
        result = d1 + d2
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_deque_concat_empty(self):
        d1 = collections.deque([])
        d2 = collections.deque([1, 2])
        result = d1 + d2
        self.assertEqual(list(result), [1, 2])

    @make_dynamo_test
    def test_deque_concat_with_maxlen(self):
        d1 = collections.deque([1, 2], maxlen=3)
        d2 = collections.deque([3, 4])
        result = d1 + d2
        # Result respects left operand's maxlen of 3
        self.assertEqual(list(result), [2, 3, 4])

    # --- Inplace deque concatenation (+=) ---

    @make_dynamo_test
    def test_deque_inplace_concat(self):
        d = collections.deque([1, 2])
        d += collections.deque([3, 4])
        self.assertEqual(list(d), [1, 2, 3, 4])

    @make_dynamo_test
    def test_deque_inplace_concat_with_maxlen(self):
        d = collections.deque([1, 2], maxlen=3)
        d += collections.deque([3, 4])
        # Result respects maxlen of 3
        self.assertEqual(list(d), [2, 3, 4])

    # --- torch.Size concatenation ---

    @make_dynamo_test
    def test_torch_size_concat(self):
        s1 = torch.Size([1, 2, 3])
        s2 = torch.Size([4, 5])
        result = s1 + s2
        self.assertEqual(result, torch.Size([1, 2, 3, 4, 5]))
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_empty(self):
        s1 = torch.Size([])
        s2 = torch.Size([1, 2])
        result = s1 + s2
        self.assertEqual(result, torch.Size([1, 2]))

    @make_dynamo_test
    def test_torch_size_inplace_concat(self):
        s = torch.Size([1, 2])
        s += torch.Size([3, 4])
        self.assertEqual(s, torch.Size([1, 2, 3, 4]))
        self.assertIsInstance(s, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_with_tuple(self):
        # torch.Size + tuple works since torch.Size subclasses tuple
        s = torch.Size([1, 2])
        result = s + (3, 4)
        self.assertEqual(list(result), [1, 2, 3, 4])
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_with_user_defined_tuple(self):
        # torch.Size + UserDefinedTuple should work
        s = torch.Size([1, 2])
        ud_tuple = UserDefinedTuple([3, 4])
        result = s + ud_tuple
        self.assertEqual(list(result), [1, 2, 3, 4])
        self.assertIsInstance(result, torch.Size)

    @make_dynamo_test
    def test_torch_size_concat_multiple_tuples(self):
        # torch.Size + regular tuple + UserDefinedTuple
        s = torch.Size([1, 2])
        result = s + (3, 4) + UserDefinedTuple([5, 6])
        self.assertEqual(list(result), [1, 2, 3, 4, 5, 6])
        self.assertIsInstance(result, torch.Size)

    # --- User-defined sequence class ---

    @make_dynamo_test
    def test_user_defined_sequence_concat(self):
        a = UserDefinedSequence([1, 2])
        b = UserDefinedSequence([3, 4])
        result = a + b
        self.assertEqual(result, UserDefinedSequence([1, 2, 3, 4]))

    @make_dynamo_test
    def test_user_defined_sequence_inplace_concat(self):
        a = UserDefinedSequence([1, 2])
        a += UserDefinedSequence([3, 4])
        self.assertEqual(a, UserDefinedSequence([1, 2, 3, 4]))

    @make_dynamo_test
    def test_user_defined_sequence_empty(self):
        a = UserDefinedSequence([])
        b = UserDefinedSequence([1, 2])
        result = a + b
        self.assertEqual(result, UserDefinedSequence([1, 2]))

    @make_dynamo_test
    def test_user_defined_sequence_chained(self):
        a = UserDefinedSequence([1])
        b = UserDefinedSequence([2])
        c = UserDefinedSequence([3])
        result = a + b + c
        self.assertEqual(result, UserDefinedSequence([1, 2, 3]))


instantiate_parametrized_tests(TestSqConcat)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
