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


# ---------------------------------------------------------------------------
# sq_ass_item / mp_ass_subscript on sequences (__setitem__)
# ---------------------------------------------------------------------------


class _WithSetitem:
    """Sequence-shaped class with explicit __setitem__ (int keys)."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


class _MutableSequence:
    """Custom sequence with __setitem__ via sq_ass_item."""

    def __init__(self, data):
        self.data = list(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)


class _ListSubclassSetitem(list):
    def __setitem__(self, key, value):
        super().__setitem__(key, value + 1000)


class _ListSubclassNoOverride(list):
    pass


class _ListSubclassExtra(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.write_count = 0

    def __setitem__(self, key, value):
        self.write_count += 1
        super().__setitem__(key, value)


class _ListSubclassSlice(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            super().__setitem__(key, [v + 100 for v in value])
        else:
            super().__setitem__(key, value)


# (label, factory) — factory builds an instance from a list.
_SEQUENCE_TYPES = [
    ("list", list),
    ("deque", collections.deque),
    ("WithSetitem", _WithSetitem),
    ("MutableSequence", _MutableSequence),
]


class TestSqAssItem(torch._dynamo.test_case.TestCase):
    """All sequence __setitem__ tests in one class."""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    # -- parameterized over sequence container type --

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_basic(self, label, factory):
        c = factory([1, 2, 3])
        c[0] = 100
        self.assertEqual(c[0], 100)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_multiple(self, label, factory):
        c = factory([1, 2, 3])
        c[0] = 100
        c[1] = 200
        self.assertEqual(c[0], 100)
        self.assertEqual(c[1], 200)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_replaces_value(self, label, factory):
        c = factory([1, 2, 3])
        c[0] = 999
        self.assertEqual(c[0], 999)

    @parametrize("label,factory", _SEQUENCE_TYPES)
    @make_dynamo_test
    def test_setitem_negative_index(self, label, factory):
        c = factory([1, 2, 3])
        c[-1] = 777
        self.assertEqual(c[-1], 777)

    # -- list slice (mp_ass_subscript_impl slice branch) --

    @make_dynamo_test
    def test_slice_basic(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:3] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 4, 5])

    @make_dynamo_test
    def test_slice_step(self):
        lst = [1, 2, 3, 4, 5]
        lst[::2] = [10, 30, 50]
        self.assertEqual(lst, [10, 2, 30, 4, 50])

    @make_dynamo_test
    def test_slice_extending(self):
        lst = [1, 2, 3]
        lst[10:] = [4, 5]
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    @make_dynamo_test
    def test_slice_shrinking(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:4] = [99]
        self.assertEqual(lst, [1, 99, 5])

    @make_dynamo_test
    def test_slice_empty_insert(self):
        lst = [1, 2, 3]
        lst[1:1] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 2, 3])

    @make_dynamo_test
    def test_slice_full_replace(self):
        lst = [1, 2, 3]
        lst[:] = [10, 20, 30, 40]
        self.assertEqual(lst, [10, 20, 30, 40])

    @make_dynamo_test
    def test_slice_negative(self):
        lst = [1, 2, 3, 4, 5]
        lst[-2:] = [99, 100]
        self.assertEqual(lst, [1, 2, 3, 99, 100])

    # -- nb_index path (bool key) --

    @make_dynamo_test
    def test_nb_index_bool_key(self):
        lst = [1, 2, 3]
        lst[True] = 99
        self.assertEqual(lst[1], 99)

    # -- SymNode key --

    @make_dynamo_test
    def test_symnode_list_key(self):
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 1] = 99
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_symnode_list_key_zero(self):
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 5] = 99
        self.assertEqual(lst[0], 99)

    @make_dynamo_test
    def test_symnode_list_key_then_read(self):
        t = torch.randn(5)
        lst = [10, 20, 30, 40, 50]
        idx = t.shape[0] - 3
        lst[idx] = 999
        self.assertEqual(lst[idx], 999)
        self.assertEqual(lst[2], 999)

    @make_dynamo_test
    def test_symnode_slice_stop(self):
        t = torch.randn(3)
        lst = [1, 2, 3, 4, 5]
        lst[1 : t.shape[0]] = [88, 99]
        self.assertEqual(lst, [1, 88, 99, 4, 5])

    @make_dynamo_test
    def test_symnode_slice_start_stop(self):
        t = torch.randn(4)
        lst = [1, 2, 3, 4, 5]
        lst[t.shape[0] - 3 : t.shape[0]] = [77, 88, 99]
        self.assertEqual(lst, [1, 77, 88, 99, 5])

    @make_dynamo_test
    def test_symnode_deque_key(self):
        t = torch.randn(5)
        d = collections.deque([0, 0, 0, 0, 0])
        d[t.shape[0] - 1] = 99
        self.assertEqual(d[4], 99)

    @make_dynamo_test
    def test_symnode_mutable_sequence_key(self):
        t = torch.randn(5)
        seq = _MutableSequence([0, 0, 0, 0, 0])
        seq[t.shape[0] - 2] = 99
        self.assertEqual(seq[3], 99)

    # -- list subclass --

    @make_dynamo_test
    def test_subclass_list_inherited_setitem(self):
        lst = _ListSubclassNoOverride([1, 2, 3])
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(list(lst), [99, 2, 3])

    @make_dynamo_test
    def test_subclass_list_inherited_slice(self):
        lst = _ListSubclassNoOverride([1, 2, 3, 4, 5])
        lst[1:3] = [88, 99]
        self.assertEqual(list(lst), [1, 88, 99, 4, 5])

    @make_dynamo_test
    def test_subclass_list_extra_state(self):
        lst = _ListSubclassExtra([0, 0, 0])
        lst[0] = 5
        lst[1] = 6
        self.assertEqual(lst[0], 5)
        self.assertEqual(lst[1], 6)
        self.assertEqual(lst.write_count, 2)

    @make_dynamo_test
    def test_subclass_list_slice_override(self):
        lst = _ListSubclassSlice([1, 2, 3, 4, 5])
        lst[1:3] = [10, 20]
        self.assertEqual(list(lst), [1, 110, 120, 4, 5])

    @make_dynamo_test
    def test_subclass_list_negative_index(self):
        lst = _ListSubclassNoOverride([1, 2, 3])
        lst[-1] = 99
        self.assertEqual(lst[-1], 99)

    @make_dynamo_test
    def test_subclass_list_replace_existing(self):
        lst = _ListSubclassExtra([10, 20, 30])
        lst[1] = 99
        self.assertEqual(lst[1], 99)
        self.assertEqual(lst.write_count, 1)

    @make_dynamo_test
    def test_subclass_list_overriding(self):
        lst = _ListSubclassSetitem([0, 0, 0])
        lst[0] = 5
        self.assertEqual(lst[0], 1005)

    # -- mutation visibility --

    def test_mutation_outer_list_persists(self):
        outer = [1, 2, 3]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            outer[0] = 99
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, [99, 2, 3])

    @make_dynamo_test
    def test_mutation_setitem_then_read(self):
        lst = [1, 2, 3]
        lst[0] = 99
        v = lst[0]
        self.assertEqual(v, 99)

    @make_dynamo_test
    def test_mutation_list_of_dicts(self):
        lst = [{"a": 1}, {"b": 2}]
        lst[0]["a"] = 100
        self.assertEqual(lst[0]["a"], 100)

    @make_dynamo_test
    def test_mutation_multiple_sequential(self):
        lst = [0, 0, 0, 0, 0]
        lst[0] = 1
        lst[1] = 2
        lst[2] = 3
        lst[3] = 4
        lst[4] = 5
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    # -- misc index handling --

    @make_dynamo_test
    def test_list_int_var_index(self):
        lst = [1, 2, 3]
        idx = 1
        lst[idx] = 42
        self.assertEqual(lst[1], 42)

    @make_dynamo_test
    def test_list_negative_index_middle(self):
        lst = [1, 2, 3, 4, 5]
        lst[-2] = 88
        self.assertEqual(lst[-2], 88)
        self.assertEqual(lst[3], 88)

    @make_dynamo_test
    def test_list_negative_index_last(self):
        lst = [1, 2, 3, 4, 5]
        lst[-1] = 99
        self.assertEqual(lst[-1], 99)
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_mutable_sequence_negative_index(self):
        seq = _MutableSequence([1, 2, 3, 4, 5])
        seq[-1] = 99
        self.assertEqual(seq[-1], 99)
        self.assertEqual(seq[4], 99)

    # -- errors --

    @make_dynamo_test
    def test_error_list_out_of_bounds(self):
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[10] = 100

    @make_dynamo_test
    def test_error_list_negative_oob(self):
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[-100] = 1

    @make_dynamo_test
    def test_error_list_string_key(self):
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst["a"] = 1

    @make_dynamo_test
    def test_error_list_float_key(self):
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst[2.5] = 1

    @make_dynamo_test
    def test_error_tuple_setitem(self):
        t = (1, 2, 3)
        with self.assertRaises(TypeError):
            t[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_error_str_setitem(self):
        s = "abc"
        with self.assertRaises(TypeError):
            s[0] = "x"  # type: ignore[index]

    @make_dynamo_test
    def test_error_bytes_setitem(self):
        b = b"abc"
        with self.assertRaises(TypeError):
            b[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_error_frozenset_setitem(self):
        fs = frozenset([1, 2, 3])
        with self.assertRaises(TypeError):
            fs[0] = 1  # type: ignore[index]


instantiate_parametrized_tests(TestSqAssItem)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
