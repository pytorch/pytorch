# Owner(s): ["module: dynamo"]

"""Tests for sequence protocol operations (sq_*) in PyTorch Dynamo."""

import collections

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

    # --- User-defined tuple subclass construction ---

    @make_dynamo_test
    def test_user_defined_tuple_construct_empty(self):
        # tuple.__new__(cls) with no iterable arg builds an empty tuple.
        a = UserDefinedTuple()
        self.assertEqual(a, ())
        self.assertEqual(len(a), 0)
        self.assertIs(type(a), UserDefinedTuple)

    @make_dynamo_test
    def test_user_defined_tuple_construct_from_iterables(self):
        self.assertEqual(UserDefinedTuple([]), ())
        self.assertEqual(UserDefinedTuple(set()), ())
        self.assertEqual(UserDefinedTuple([1, 2, 3]), (1, 2, 3))
        self.assertEqual(UserDefinedTuple(obj for obj in [1, 2, 3]), (1, 2, 3))
        nested = tuple(UserDefinedTuple([obj]) for obj in [1, 2, 3])
        self.assertEqual(nested, ((1,), (2,), (3,)))

    # --- User-defined deque subclass concatenation ---

    @make_dynamo_test
    def test_user_defined_deque_concat(self):
        a = UserDefinedDeque([1, 2])
        b = UserDefinedDeque([3, 4])
        result = a + b
        self.assertEqual(list(result), [1, 2, 3, 4])

    @make_dynamo_test
    def test_user_defined_deque_maxlen(self):
        d = UserDefinedDeque([1, 2, 3], maxlen=2)
        self.assertEqual(list(d), [2, 3])
        self.assertEqual(d.maxlen, 2)

    def test_user_defined_deque_graph_input(self):
        # A deque subclass passed in as a graph input is built by builder.py
        # (sourced path), and in-place mutations are replayed onto the original.
        def fn(d):
            d.append(99)
            return len(d)

        for maxlen in (None, 3):
            d = UserDefinedDeque([1, 2, 3], maxlen=maxlen)
            ref = UserDefinedDeque([1, 2, 3], maxlen=maxlen)
            ref_ret = fn(ref)
            ret = torch.compile(fn, backend="eager", fullgraph=True)(d)
            self.assertEqual(ret, ref_ret)
            self.assertEqual(list(d), list(ref))
            self.assertEqual(d.maxlen, ref.maxlen)

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

    # --- Inplace deque repeat (*=) ---

    @make_dynamo_test
    def test_deque_inplace_repeat(self):
        d = collections.deque([1, 2])
        d *= 3
        self.assertEqual(list(d), [1, 2, 1, 2, 1, 2])

    @make_dynamo_test
    def test_deque_inplace_repeat_with_maxlen(self):
        d = collections.deque([1, 2], maxlen=3)
        d *= 3
        # A bounded deque keeps the last maxlen items after repeating.
        self.assertEqual(list(d), [2, 1, 2])

    @make_dynamo_test
    def test_deque_inplace_repeat_maxlen_zero(self):
        d = collections.deque([1, 2], maxlen=0)
        d *= 3
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_inplace_repeat_zero(self):
        d = collections.deque([1, 2, 3])
        d *= 0
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_inplace_repeat_negative(self):
        d = collections.deque([1, 2, 3])
        d *= -1
        self.assertEqual(list(d), [])

    # --- list re-init (list.__init__) ---

    @make_dynamo_test
    def test_list_reinit_clears_and_extends(self):
        x = list(range(-5, 0))
        x.__init__(range(3))
        self.assertEqual(x, [0, 1, 2])

    @make_dynamo_test
    def test_list_reinit_empty(self):
        x = [1, 2, 3]
        x.__init__()
        self.assertEqual(x, [])

    @make_dynamo_test
    def test_list_reinit_self_referential(self):
        # CPython clears before extending, so a.__init__(a) yields [].
        x = [1, 2, 3]
        x.__init__(x)
        self.assertEqual(x, [])

    @make_dynamo_test
    def test_list_reinit_non_iterable_clears_then_raises(self):
        # clear happens before extend, so the TypeError leaves the list empty.
        x = [1, 2, 3]
        with self.assertRaises(TypeError):
            x.__init__(5)
        self.assertEqual(x, [])

    # --- deque re-init (deque.__init__) ---

    @make_dynamo_test
    def test_deque_reinit_clears_and_extends(self):
        d = collections.deque(range(-5, 0))
        d.__init__(range(3))
        self.assertEqual(list(d), [0, 1, 2])

    @make_dynamo_test
    def test_deque_reinit_resets_maxlen(self):
        d = collections.deque([1, 2], maxlen=2)
        d.__init__(range(4))
        # No maxlen passed -> maxlen reset to None, all items kept
        self.assertEqual(d.maxlen, None)
        self.assertEqual(list(d), [0, 1, 2, 3])

    @make_dynamo_test
    def test_deque_reinit_with_maxlen(self):
        d = collections.deque([1, 2])
        d.__init__(range(5), 3)
        self.assertEqual(d.maxlen, 3)
        self.assertEqual(list(d), [2, 3, 4])

    @make_dynamo_test
    def test_deque_reinit_empty(self):
        d = collections.deque([1, 2, 3])
        d.__init__()
        self.assertEqual(list(d), [])

    @make_dynamo_test
    def test_deque_reinit_negative_maxlen(self):
        d = collections.deque([1, 2])
        with self.assertRaises(ValueError):
            d.__init__(range(3), -1)

    @make_dynamo_test
    def test_deque_reinit_float_maxlen(self):
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.__init__(maxlen=2.0)

    @make_dynamo_test
    def test_deque_reinit_str_maxlen(self):
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.__init__(maxlen="3")

    @make_dynamo_test
    def test_deque_ctor_float_maxlen(self):
        with self.assertRaises(TypeError):
            collections.deque([1, 2], maxlen=2.0)

    @make_dynamo_test
    def test_deque_reinit_overflow_maxlen(self):
        d = collections.deque([1, 2])
        with self.assertRaises(OverflowError):
            d.__init__(maxlen=2**63)

    @make_dynamo_test
    def test_deque_ctor_bool_maxlen(self):
        # bool is an int subclass; CPython accepts it.
        d = collections.deque(range(5), maxlen=True)
        self.assertEqual(list(d), [4])

    # --- deque copy (deque.copy / copy.copy / __copy__) ---

    @make_dynamo_test
    def test_deque_copy_method(self):
        d = collections.deque([1, 2, 3])
        e = d.copy()
        self.assertEqual(list(e), [1, 2, 3])
        self.assertIsNone(e.maxlen)
        d.append(4)
        self.assertEqual(list(e), [1, 2, 3])

    @make_dynamo_test
    def test_deque_copy_preserves_maxlen(self):
        d = collections.deque([1, 2, 3], maxlen=3)
        e = d.copy()
        self.assertEqual(list(e), [1, 2, 3])
        self.assertEqual(e.maxlen, 3)

    @make_dynamo_test
    def test_deque_copy_module(self):
        import copy

        d = collections.deque([1, 2], maxlen=5)
        e = copy.copy(d)
        self.assertEqual(list(e), [1, 2])
        self.assertEqual(e.maxlen, 5)

    @make_dynamo_test
    def test_deque_copy_shares_elements(self):
        mut = [10]
        d = collections.deque([mut])
        e = d.copy()
        mut[0] = 11
        self.assertEqual(list(d), list(e))

    @make_dynamo_test
    def test_deque_copy_too_many_args(self):
        d = collections.deque([1, 2])
        with self.assertRaises(TypeError):
            d.copy(1)

    # --- deque iterator mutation detection ---

    @make_dynamo_test
    def test_deque_iter_mutated_pop_raises(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d.pop()
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_mutated_append_raises(self):
        d = collections.deque()
        it = iter(d)
        d.append(10)
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_reversed_iter_mutated_raises(self):
        d = collections.deque("abc")
        it = reversed(d)
        d.appendleft("z")
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_unmutated_ok(self):
        d = collections.deque("abc")
        self.assertEqual(list(iter(d)), ["a", "b", "c"])
        self.assertEqual(list(reversed(d)), ["c", "b", "a"])

    @make_dynamo_test
    def test_deque_reverse_iterator_type(self):
        klass = type(reversed(collections.deque()))
        self.assertEqual(list(klass(collections.deque("abcd"))), list(reversed("abcd")))

    @make_dynamo_test
    def test_deque_forward_iterator_type(self):
        klass = type(iter(collections.deque()))
        self.assertEqual(list(klass(collections.deque("abcd"))), list("abcd"))

    @make_dynamo_test
    def test_deque_reverse_iterator_bad_arg(self):
        klass = type(reversed(collections.deque()))
        with self.assertRaises(TypeError):
            klass([1, 2, 3])

    # Eager parity: only mutations that bump CPython's deque->state invalidate
    # an active iterator. reverse() and setitem do NOT bump; delitem and a
    # non-empty extend do.

    @make_dynamo_test
    def test_deque_iter_reverse_does_not_raise(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d.reverse()
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_setitem_does_not_raise(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d[0] = "Z"
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_delitem_raises(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        del d[0]
        with self.assertRaises(RuntimeError):
            next(it)

    @make_dynamo_test
    def test_deque_iter_extend_empty_does_not_raise(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extend([])
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_extendleft_empty_does_not_raise(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extendleft([])
        next(it)  # must not raise

    @make_dynamo_test
    def test_deque_iter_extend_nonempty_raises(self):
        d = collections.deque("abcdefg")
        it = iter(d)
        d.extend([1])
        with self.assertRaises(RuntimeError):
            next(it)

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
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

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
    def test_subclass_list_inherited(self):
        class L(list):
            pass

        lst = L([1, 2, 3])
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(list(lst), [99, 2, 3])

        # slice
        lst2 = L([1, 2, 3, 4, 5])
        lst2[1:3] = [88, 99]
        self.assertEqual(list(lst2), [1, 88, 99, 4, 5])

        # negative
        lst3 = L([1, 2, 3])
        lst3[-1] = 99
        self.assertEqual(lst3[-1], 99)

    @make_dynamo_test
    def test_subclass_list_extra_state(self):
        class L(list):
            def __init__(self, *args):
                super().__init__(*args)
                self.write_count = 0

            def __setitem__(self, key, value):
                self.write_count += 1
                super().__setitem__(key, value)

        lst = L([0, 0, 0])
        lst[0] = 5
        lst[1] = 6
        self.assertEqual(lst[0], 5)
        self.assertEqual(lst[1], 6)
        self.assertEqual(lst.write_count, 2)

        # replace existing
        lst2 = L([10, 20, 30])
        lst2[1] = 99
        self.assertEqual(lst2[1], 99)
        self.assertEqual(lst2.write_count, 1)

    @make_dynamo_test
    def test_subclass_list_slice_override(self):
        class L(list):
            def __setitem__(self, key, value):
                if isinstance(key, slice):
                    super().__setitem__(key, [v + 100 for v in value])
                else:
                    super().__setitem__(key, value)

        lst = L([1, 2, 3, 4, 5])
        lst[1:3] = [10, 20]
        self.assertEqual(list(lst), [1, 110, 120, 4, 5])

    @make_dynamo_test
    def test_subclass_list_overriding(self):
        class L(list):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1000)

        lst = L([0, 0, 0])
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

    # -- slice assignment errors --

    @make_dynamo_test
    def test_error_slice_step_zero(self):
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "slice step cannot be zero"):
            lst[::0] = [10, 20]

    @make_dynamo_test
    def test_error_extended_slice_size_mismatch(self):
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(
            ValueError, "attempt to assign sequence of size .* to extended slice"
        ):
            lst[::2] = [10, 20]

    @make_dynamo_test
    def test_error_slice_non_iterable_simple(self):
        lst = [1, 2, 3]
        # CPython 3.12.5+ reports "must assign iterable to extended slice" even
        # for simple slices; earlier versions say "can only assign an iterable".
        # See https://github.com/pytorch/pytorch/issues/187774.
        with self.assertRaisesRegex(
            TypeError,
            "must assign iterable to extended slice|can only assign an iterable",
        ):
            lst[1:2] = 99  # type: ignore[assignment]

    @make_dynamo_test
    def test_error_slice_non_iterable_extended(self):
        lst = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(
            TypeError, "must assign iterable to extended slice"
        ):
            lst[::2] = 99  # type: ignore[assignment]

    # -- slice assignment: alternative RHS types --

    @make_dynamo_test
    def test_slice_assign_tuple_rhs(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:3] = (10, 20)
        self.assertEqual(lst, [1, 10, 20, 4, 5])

    @make_dynamo_test
    def test_slice_assign_string_rhs(self):
        lst = [1, 2, 3]
        lst[1:2] = "ab"
        self.assertEqual(lst, [1, "a", "b", 3])

    @make_dynamo_test
    def test_slice_assign_negative_step(self):
        lst = [1, 2, 3, 4, 5]
        lst[::-1] = [50, 40, 30, 20, 10]
        self.assertEqual(lst, [10, 20, 30, 40, 50])

    @make_dynamo_test
    def test_slice_assign_empty_to_simple_slice(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:4] = []
        self.assertEqual(lst, [1, 5])

    def test_slice_mutation_visible_outside(self):
        # Outer list (captured by closure) mutated via slice assignment inside
        # a compiled function. Verifies side_effects.mutation is registered on
        # the slice path of mp_ass_subscript_impl.
        outer = [1, 2, 3, 4, 5]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            outer[1:3] = [10, 20]
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, [1, 10, 20, 4, 5])

    # -- __delitem__ (NULL value via mp_ass_subscript / sq_ass_item) --
    # Only list/deque among the parametrized types support __delitem__.

    @parametrize(
        "label,factory",
        [("list", list), ("deque", collections.deque)],
    )
    @make_dynamo_test
    def test_delitem_basic(self, label, factory):
        c = factory([1, 2, 3, 4, 5])
        del c[2]
        self.assertEqual(list(c), [1, 2, 4, 5])

    @parametrize(
        "label,factory",
        [("list", list), ("deque", collections.deque)],
    )
    @make_dynamo_test
    def test_delitem_negative(self, label, factory):
        c = factory([1, 2, 3, 4, 5])
        del c[-1]
        self.assertEqual(list(c), [1, 2, 3, 4])

    @make_dynamo_test
    def test_delitem_list_first(self):
        lst = [1, 2, 3]
        del lst[0]
        self.assertEqual(lst, [2, 3])

    @make_dynamo_test
    def test_delitem_list_slice(self):
        lst = [1, 2, 3, 4, 5]
        del lst[1:3]
        self.assertEqual(lst, [1, 4, 5])

    @make_dynamo_test
    def test_delitem_list_slice_step(self):
        lst = [1, 2, 3, 4, 5]
        del lst[::2]
        self.assertEqual(lst, [2, 4])

    @make_dynamo_test
    def test_delitem_list_full_slice(self):
        lst = [1, 2, 3]
        del lst[:]
        self.assertEqual(lst, [])

    @make_dynamo_test
    def test_delitem_list_multiple(self):
        lst = [1, 2, 3, 4, 5]
        del lst[0]
        del lst[0]
        self.assertEqual(lst, [3, 4, 5])

    @make_dynamo_test
    def test_delitem_symnode_key(self):
        t = torch.randn(5)
        lst = [10, 20, 30, 40, 50]
        del lst[t.shape[0] - 1]
        self.assertEqual(lst, [10, 20, 30, 40])

    @make_dynamo_test
    def test_delitem_error_oob(self):
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            del lst[10]

    @make_dynamo_test
    def test_delitem_error_string_key(self):
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            del lst["a"]

    @make_dynamo_test
    def test_delitem_error_tuple(self):
        t = (1, 2, 3)
        with self.assertRaises(TypeError):
            del t[0]  # type: ignore[attr-defined]

    @make_dynamo_test
    def test_delitem_error_str(self):
        s = "abc"
        with self.assertRaises(TypeError):
            del s[0]  # type: ignore[attr-defined]

    @make_dynamo_test
    def test_delitem_subclass_list_inherited(self):
        class L(list):
            pass

        lst = L([1, 2, 3, 4])
        del lst[1]
        self.assertEqual(list(lst), [1, 3, 4])

    @make_dynamo_test
    def test_delitem_subclass_list_override(self):
        class L(list):
            def __init__(self, *args):
                super().__init__(*args)
                self.del_count = 0

            def __delitem__(self, key):
                self.del_count += 1
                super().__delitem__(key)

        lst = L([1, 2, 3])
        del lst[0]
        del lst[0]
        self.assertEqual(list(lst), [3])
        self.assertEqual(lst.del_count, 2)


instantiate_parametrized_tests(TestSqAssItem)


class _IndexObj:
    def __init__(self, n):
        self.n = int(n)

    def __index__(self):
        return self.n


class TestRangeUserIndex(torch._dynamo.test_case.TestCase):
    # range() and range subscript apply __index__ (PyNumber_Index) to their
    # arguments and slice members.

    def test_range_args_with_index(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return list(range(_IndexObj(2), _IndexObj(6), _IndexObj(2)))

        self.assertEqual(fn(), [2, 4])

    def test_range_slice_with_index(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return range(10)[: _IndexObj(5)]

        self.assertEqual(fn(), range(5))

    def test_range_index_raises(self):
        class IX:
            def __index__(self):
                raise RuntimeError("boom")

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            try:
                range(IX())
                return "no_error"
            except RuntimeError:
                return "runtime_error"

        self.assertEqual(fn(), "runtime_error")

    def test_range_index_non_int(self):
        class IN:
            def __index__(self):
                return "not a number"  # noqa: PLE0305

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            try:
                range(IN())
                return "no_error"
            except TypeError:
                return "type_error"

        self.assertEqual(fn(), "type_error")


class TestRangeIteratorSetstate(torch._dynamo.test_case.TestCase):
    # range_iterator.__setstate__(k) sets the iterator index, clamped to
    # [0, len], mirroring CPython rangeiter_setstate.

    def test_setstate_resumes_from_index(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = iter(range(10, 20, 2))
            it.__setstate__(2)
            return list(it)

        self.assertEqual(fn(), [14, 16, 18])

    def test_setstate_on_reversed(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = reversed(range(10, 20, 2))
            it.__setstate__(3)
            return list(it)

        self.assertEqual(fn(), [12, 10])

    def test_setstate_negative_clamps_to_zero(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = iter(range(5))
            it.__setstate__(-100)
            return list(it)

        self.assertEqual(fn(), [0, 1, 2, 3, 4])

    def test_setstate_overshoot_clamps_to_len(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = iter(range(5))
            it.__setstate__(100)
            return list(it)

        self.assertEqual(fn(), [])

    def test_setstate_after_partial_consume(self):
        # __setstate__ is RELATIVE to the current advanced position: after two
        # next() calls (index 2), __setstate__(7) advances 7 more -> index 9,
        # so only [9] remains. Matches eager CPython.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = iter(range(10))
            next(it)
            next(it)
            it.__setstate__(7)
            return list(it)

        self.assertEqual(fn(), [9])

    def test_length_hint_tracks_index(self):
        # After one next() (index 1), __setstate__(4) advances 4 -> index 5,
        # so 5 items remain. Matches eager CPython.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            it = iter(range(10))
            next(it)
            it.__setstate__(4)
            return it.__length_hint__()

        self.assertEqual(fn(), 5)


class TestRangeDynamicBounds(torch._dynamo.test_case.TestCase):
    # With assume_static_by_default=False a captured range object's
    # start/stop/step are wrapped as symbolic ints; range math must specialize
    # them instead of assuming a plain python constant.
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_index_symbolic_bounds(self):
        keys = range(10)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return keys[0]

        self.assertEqual(fn(), 0)

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_slice_symbolic_bounds(self):
        keys = range(1, 10, 2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return keys[1:3]

        self.assertEqual(fn(), range(1, 10, 2)[1:3])

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_range_iter_symbolic_bounds(self):
        keys = range(5)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return list(keys)

        self.assertEqual(fn(), [0, 1, 2, 3, 4])


class TestRangeContains(torch._dynamo.test_case.TestCase):
    # range.__contains__ uses the arithmetic fast path only for exact int/bool
    # operands; everything else falls back to an __eq__ linear scan, matching
    # CPython range_contains / _PySequence_IterSearch.
    def test_non_int_members(self):
        class AlwaysEq:
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        class IntSubclassEq(int):
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return (
                1.0 in range(3),
                True in range(3),
                (1 + 0j) in range(3),
                AlwaysEq() in range(3),
                IntSubclassEq(11) in range(10),
                5 in range(3),
                2.5 in range(3),
            )

        self.assertEqual(fn(), (True, True, True, True, True, False, False))

    def test_negative_step_and_strided(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return (
                -1 in range(0, -20, -1),
                1.0 in range(0, -20, -1),
                2 in range(0, 101, 2),
                1 in range(0, 101, 2),
                2.0 in range(0, 101, 2),
                100.0 in range(0, 101, 2),
            )

        self.assertEqual(fn(), (True, False, True, False, True, True))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
