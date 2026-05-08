# Owner(s): ["module: dynamo"]

"""
Tests for the `__setitem__` operator and item assignment protocol in PyTorch Dynamo.

Tests cover:
- mp_ass_subscript protocol: list, dict, custom mutable mappings
- sq_ass_item protocol: custom sequences with __len__
- Negative index handling for sequences
- Error handling: non-subscriptable objects, wrong key types, out of bounds
- User-defined classes with __setitem__
"""

import collections

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


# ---------------------------------------------------------------------------
# Helper classes defined at module level (not inside compiled functions)
# ---------------------------------------------------------------------------


class WithSetitem:
    """Has explicit __setitem__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


class WithSetitemViaMapping:
    """Custom mapping with __setitem__ via mp_ass_subscript."""

    def __init__(self, data=None):
        self.data = data or {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class MutableSequence:
    """Custom sequence with __setitem__ via sq_ass_item."""

    def __init__(self, data):
        self.data = list(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)


class NoSetitem:
    """Object without __setitem__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class _SetitemBase:
    """Base test class for __setitem__ protocol with parameterized types."""

    thetype = None  # Override in subclass
    data = [1, 2, 3]
    empty = []
    index = 0
    new_value = 100
    supports_new_keys = False  # Override if type supports adding new keys

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
    def test_setitem_basic(self):
        # Basic item assignment
        container = self.init(self.thetype, self.data.copy())
        container[self.index] = self.new_value
        # Verify that the assignment was successful
        self.assertEqual(container[self.index], self.new_value)

    @make_dynamo_test
    def test_setitem_multiple(self):
        # Multiple assignments
        container = self.init(self.thetype, self.data.copy())
        container[0] = 100
        container[1] = 200
        self.assertEqual(container[0], 100)
        self.assertEqual(container[1], 200)

    @make_dynamo_test
    def test_setitem_replaces_value(self):
        # Assignment replaces existing value
        container = self.init(self.thetype, self.data.copy())
        container[self.index] = 999
        self.assertEqual(container[self.index], 999)

    @make_dynamo_test
    def test_setitem_negative_index(self):
        # Negative index assignment (if type supports __len__)
        if hasattr(self.thetype, "__len__"):
            container = self.init(self.thetype, self.data.copy())
            container[-1] = 777
            # Verify assignment worked (exact comparison depends on type)
            self.assertEqual(container[-1], 777)

    @make_dynamo_test
    def test_setitem_new_key(self):
        # Adding new key (only for types that support it)
        if self.supports_new_keys:
            container = self.init(self.thetype, self.empty.copy())
            container["new"] = 42
            self.assertEqual(container["new"], 42)


class ListSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = list
    supports_new_keys = False


class DictSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = dict
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True


class WithSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = WithSetitem
    supports_new_keys = False


class WithSetitemViaMappingTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = WithSetitemViaMapping
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True


class MutableSequenceTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = MutableSequence
    supports_new_keys = False


class DequeSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = collections.deque
    supports_new_keys = False


class OrderedDictSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = collections.OrderedDict
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True


class DefaultDictSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = collections.defaultdict
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True

    def init(self, thetype, data):
        return collections.defaultdict(int, data)


class CounterSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = collections.Counter
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True


class DictSubclassSetitem(dict):
    pass


class DictSubclassSetitemTest(_SetitemBase, torch._dynamo.test_case.TestCase):
    thetype = DictSubclassSetitem
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    index = "b"
    new_value = 100
    supports_new_keys = True


class ListSliceSetitemTest(torch._dynamo.test_case.TestCase):
    """Slice assignment on list — exercises ListVariable.mp_ass_subscript_impl slice branch."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_basic_slice(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:3] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 4, 5])

    @make_dynamo_test
    def test_step_slice(self):
        lst = [1, 2, 3, 4, 5]
        lst[::2] = [10, 30, 50]
        self.assertEqual(lst, [10, 2, 30, 4, 50])

    @make_dynamo_test
    def test_extending_slice(self):
        lst = [1, 2, 3]
        lst[10:] = [4, 5]
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    @make_dynamo_test
    def test_shrinking_slice(self):
        lst = [1, 2, 3, 4, 5]
        lst[1:4] = [99]
        self.assertEqual(lst, [1, 99, 5])

    @make_dynamo_test
    def test_empty_slice_insert(self):
        lst = [1, 2, 3]
        lst[1:1] = [10, 20]
        self.assertEqual(lst, [1, 10, 20, 2, 3])

    @make_dynamo_test
    def test_full_slice_replace(self):
        lst = [1, 2, 3]
        lst[:] = [10, 20, 30, 40]
        self.assertEqual(lst, [10, 20, 30, 40])

    @make_dynamo_test
    def test_negative_slice(self):
        lst = [1, 2, 3, 4, 5]
        lst[-2:] = [99, 100]
        self.assertEqual(lst, [1, 2, 3, 99, 100])


class NbIndexSetitemTest(torch._dynamo.test_case.TestCase):
    """Custom __index__ as setitem key — exercises nb_index_impl in mp_ass_subscript_impl."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_bool_key(self):
        lst = [1, 2, 3]
        lst[True] = 99
        self.assertEqual(lst[1], 99)


class SymNodeSetitemTest(torch._dynamo.test_case.TestCase):
    """Tensor-shape derived index — exercises SymNodeVariable path in sq_ass_item_impl."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_symnode_key(self):
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 1] = 99
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_list_symnode_key_zero(self):
        t = torch.randn(5)
        lst = [0, 0, 0, 0, 0]
        lst[t.shape[0] - 5] = 99
        self.assertEqual(lst[0], 99)

    @make_dynamo_test
    def test_list_symnode_key_then_read(self):
        t = torch.randn(5)
        lst = [10, 20, 30, 40, 50]
        idx = t.shape[0] - 3
        lst[idx] = 999
        self.assertEqual(lst[idx], 999)
        self.assertEqual(lst[2], 999)

    @make_dynamo_test
    def test_list_slice_symnode_stop(self):
        t = torch.randn(3)
        lst = [1, 2, 3, 4, 5]
        lst[1 : t.shape[0]] = [88, 99]
        self.assertEqual(lst, [1, 88, 99, 4, 5])

    @make_dynamo_test
    def test_list_slice_symnode_start_stop(self):
        t = torch.randn(4)
        lst = [1, 2, 3, 4, 5]
        lst[t.shape[0] - 3 : t.shape[0]] = [77, 88, 99]
        self.assertEqual(lst, [1, 77, 88, 99, 5])

    @make_dynamo_test
    def test_deque_symnode_key(self):
        t = torch.randn(5)
        d = collections.deque([0, 0, 0, 0, 0])
        d[t.shape[0] - 1] = 99
        self.assertEqual(d[4], 99)

    @make_dynamo_test
    def test_mutable_sequence_symnode_key(self):
        t = torch.randn(5)
        seq = MutableSequence([0, 0, 0, 0, 0])
        seq[t.shape[0] - 2] = 99
        self.assertEqual(seq[3], 99)

    @make_dynamo_test
    def test_dict_symnode_value(self):
        t = torch.randn(5)
        d = {}
        d["len"] = t.shape[0]
        self.assertEqual(d["len"], 5)


class MutationVisibilityTest(torch._dynamo.test_case.TestCase):
    """Verify setitem side effects persist post-compile and are visible within graph."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def test_outer_list_mutation_persists(self):
        outer = [1, 2, 3]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            outer[0] = 99
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, [99, 2, 3])

    def test_outer_dict_mutation_persists(self):
        outer = {"a": 1}

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            outer["a"] = 99
            outer["b"] = 100
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, {"a": 99, "b": 100})

    @make_dynamo_test
    def test_setitem_then_read(self):
        lst = [1, 2, 3]
        lst[0] = 99
        v = lst[0]
        self.assertEqual(v, 99)


class ErrorHandlingSetitemTest(torch._dynamo.test_case.TestCase):
    """Tests for error cases in __setitem__."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_out_of_bounds(self):
        # Out of bounds assignment should raise IndexError
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[10] = 100

    @make_dynamo_test
    def test_no_setitem_raises_typeerror(self):
        # Object without __setitem__ should raise TypeError
        obj = NoSetitem([1, 2, 3])
        with self.assertRaises(TypeError):
            obj[0] = 100

    @make_dynamo_test
    def test_dict_various_key_types(self):
        # Dict with various key types
        d = {}
        d["key1"] = "value1"
        d[42] = "value42"
        self.assertEqual(d["key1"], "value1")
        self.assertEqual(d[42], "value42")

    @make_dynamo_test
    def test_list_with_int_conversion(self):
        # Test that indices are converted to int properly
        lst = [1, 2, 3]
        idx = 1
        lst[idx] = 42
        self.assertEqual(lst[1], 42)

    @make_dynamo_test
    def test_dict_with_various_types(self):
        # Dict with various value types
        d = {}
        d["int"] = 42
        d["str"] = "hello"
        d["list"] = [1, 2, 3]
        d["dict"] = {"nested": True}
        self.assertEqual(d["int"], 42)
        self.assertEqual(d["str"], "hello")
        self.assertEqual(d["list"], [1, 2, 3])
        self.assertEqual(d["dict"], {"nested": True})

    @make_dynamo_test
    def test_nested_setitem(self):
        # Nested structure modification
        d = {"list": [1, 2, 3]}
        lst = d["list"]
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(d["list"][0], 99)

    @make_dynamo_test
    def test_list_of_dicts_setitem(self):
        # List of dicts, modifying dict values
        lst = [{"a": 1}, {"b": 2}]
        lst[0]["a"] = 100
        self.assertEqual(lst[0]["a"], 100)

    @make_dynamo_test
    def test_multiple_sequential_setitems(self):
        # Multiple sequential assignments
        lst = [0, 0, 0, 0, 0]
        lst[0] = 1
        lst[1] = 2
        lst[2] = 3
        lst[3] = 4
        lst[4] = 5
        self.assertEqual(lst, [1, 2, 3, 4, 5])

    @make_dynamo_test
    def test_negative_index_list(self):
        # Negative index on list
        lst = [1, 2, 3, 4, 5]
        lst[-1] = 99
        self.assertEqual(lst[-1], 99)
        self.assertEqual(lst[4], 99)

    @make_dynamo_test
    def test_negative_index_middle(self):
        # Negative index in the middle
        lst = [1, 2, 3, 4, 5]
        lst[-2] = 88
        self.assertEqual(lst[-2], 88)
        self.assertEqual(lst[3], 88)

    @make_dynamo_test
    def test_mutable_sequence_negative_index(self):
        # Negative index on custom mutable sequence
        seq = MutableSequence([1, 2, 3, 4, 5])
        seq[-1] = 99
        self.assertEqual(seq[-1], 99)
        self.assertEqual(seq[4], 99)

    @make_dynamo_test
    def test_list_string_key_typeerror(self):
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst["a"] = 1

    @make_dynamo_test
    def test_list_float_key_typeerror(self):
        lst = [1, 2, 3]
        with self.assertRaises(TypeError):
            lst[2.5] = 1

    @make_dynamo_test
    def test_list_negative_oob(self):
        lst = [1, 2, 3]
        with self.assertRaises(IndexError):
            lst[-100] = 1

    @make_dynamo_test
    def test_tuple_setitem_typeerror(self):
        t = (1, 2, 3)
        with self.assertRaises(TypeError):
            t[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_str_setitem_typeerror(self):
        s = "abc"
        with self.assertRaises(TypeError):
            s[0] = "x"  # type: ignore[index]

    @make_dynamo_test
    def test_bytes_setitem_typeerror(self):
        b = b"abc"
        with self.assertRaises(TypeError):
            b[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_frozenset_setitem_typeerror(self):
        fs = frozenset([1, 2, 3])
        with self.assertRaises(TypeError):
            fs[0] = 1  # type: ignore[index]

    @make_dynamo_test
    def test_dict_unhashable_key_typeerror(self):
        d = {}
        with self.assertRaises(TypeError):
            d[[1, 2]] = 1  # type: ignore[index]


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
