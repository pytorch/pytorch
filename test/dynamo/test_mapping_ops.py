# Owner(s): ["module: dynamo"]

"""Tests for mapping protocol operations (mp_*) in PyTorch Dynamo.

Covers mp_ass_subscript (__setitem__) on builtin mappings, user-defined
mappings, and dict subclasses.
"""

import collections

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


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


class _DictSubclassParam(dict):
    pass


def _make_defaultdict(data):
    return collections.defaultdict(int, data)


# (label, factory) — factory builds an instance from a dict.
_MAPPING_TYPES = [
    ("dict", dict),
    ("OrderedDict", collections.OrderedDict),
    ("Counter", collections.Counter),
    ("defaultdict", _make_defaultdict),
    ("DictSubclass", _DictSubclassParam),
    ("WithSetitemViaMapping", WithSetitemViaMapping),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMpAssSubscript(torch._dynamo.test_case.TestCase):
    """All mapping __setitem__ tests in one class."""

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

    # -- parameterized over mapping container type --

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_basic(self, label, factory):
        c = factory({"a": 1, "b": 2, "c": 3})
        c["b"] = 100
        self.assertEqual(c["b"], 100)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_multiple(self, label, factory):
        c = factory({"a": 1, "b": 2, "c": 3})
        c["a"] = 100
        c["b"] = 200
        self.assertEqual(c["a"], 100)
        self.assertEqual(c["b"], 200)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_replaces_value(self, label, factory):
        c = factory({"a": 1, "b": 2, "c": 3})
        c["b"] = 999
        self.assertEqual(c["b"], 999)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_new_key(self, label, factory):
        c = factory({})
        c["new"] = 42
        self.assertEqual(c["new"], 42)

    # -- dict subclass --

    @make_dynamo_test
    def test_subclass_dict_inherited_setitem(self):
        class D(dict):
            pass

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

    @make_dynamo_test
    def test_subclass_dict_with_missing(self):
        class D(dict):
            def __missing__(self, key):
                return -1

        d = D()
        d["a"] = 5
        self.assertEqual(d["a"], 5)
        self.assertEqual(d["unknown"], -1)

    @make_dynamo_test
    def test_subclass_dict_track_writes(self):
        class D(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.writes = []

            def __setitem__(self, key, value):
                self.writes.append((key, value))
                super().__setitem__(key, value)

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d.writes, [("a", 1), ("b", 2)])
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

        # also exercise replace-existing
        d2 = D({"a": 1})
        d2["a"] = 99
        self.assertEqual(d2["a"], 99)
        self.assertEqual(d2.writes, [("a", 99)])

    @make_dynamo_test
    def test_subclass_dict_overriding(self):
        class D(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1000)

        d = D()
        d["a"] = 5
        self.assertEqual(d["a"], 1005)

    @make_dynamo_test
    def test_subclass_ordereddict(self):
        class D(collections.OrderedDict):
            pass

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(list(d.keys()), ["a", "b"])
        self.assertEqual(d["a"], 1)

    # -- key / value variation --

    @make_dynamo_test
    def test_dict_various_key_types(self):
        d = {}
        d["key1"] = "value1"
        d[42] = "value42"
        self.assertEqual(d["key1"], "value1")
        self.assertEqual(d[42], "value42")

    @make_dynamo_test
    def test_dict_various_value_types(self):
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
    def test_symnode_dict_value(self):
        t = torch.randn(5)
        d = {}
        d["len"] = t.shape[0]
        self.assertEqual(d["len"], 5)

    # -- mutation visibility --

    def test_mutation_outer_dict_persists(self):
        outer = {"a": 1}

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            outer["a"] = 99
            outer["b"] = 100
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, {"a": 99, "b": 100})

    @make_dynamo_test
    def test_mutation_nested_dict(self):
        d = {"list": [1, 2, 3]}
        lst = d["list"]
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(d["list"][0], 99)

    # -- errors --

    @make_dynamo_test
    def test_error_dict_unhashable_key(self):
        d = {}
        with self.assertRaises(TypeError):
            d[[1, 2]] = 1  # type: ignore[index]

    # -- __delitem__ (NULL value via mp_ass_subscript) --
    # WithSetitemViaMapping has no __delitem__; restrict to dict-like types.

    @parametrize(
        "label,factory",
        [
            ("dict", dict),
            ("OrderedDict", collections.OrderedDict),
            ("Counter", collections.Counter),
            ("defaultdict", _make_defaultdict),
            ("DictSubclass", _DictSubclassParam),
        ],
    )
    @make_dynamo_test
    def test_delitem_basic(self, label, factory):
        c = factory({"a": 1, "b": 2, "c": 3})
        del c["b"]
        self.assertNotIn("b", c)
        self.assertEqual(c["a"], 1)
        self.assertEqual(c["c"], 3)

    @parametrize(
        "label,factory",
        [
            ("dict", dict),
            ("OrderedDict", collections.OrderedDict),
            ("Counter", collections.Counter),
            ("defaultdict", _make_defaultdict),
            ("DictSubclass", _DictSubclassParam),
        ],
    )
    @make_dynamo_test
    def test_delitem_then_set(self, label, factory):
        c = factory({"a": 1, "b": 2})
        del c["a"]
        c["a"] = 99
        self.assertEqual(c["a"], 99)

    @make_dynamo_test
    def test_delitem_dict_missing_key(self):
        d = {"a": 1}
        with self.assertRaises(KeyError):
            del d["missing"]

    @make_dynamo_test
    def test_delitem_dict_all_keys(self):
        d = {"a": 1, "b": 2, "c": 3}
        del d["a"]
        del d["b"]
        del d["c"]
        self.assertEqual(len(d), 0)

    @make_dynamo_test
    def test_delitem_dict_unhashable_key(self):
        d = {"a": 1}
        with self.assertRaises(TypeError):
            del d[[1, 2]]  # type: ignore[index]

    @make_dynamo_test
    def test_delitem_subclass_dict_inherited(self):
        class D(dict):
            pass

        d = D({"a": 1, "b": 2})
        del d["a"]
        self.assertEqual(dict(d), {"b": 2})

    @make_dynamo_test
    def test_delitem_subclass_dict_override(self):
        class D(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.deleted = []

            def __delitem__(self, key):
                self.deleted.append(key)
                super().__delitem__(key)

        d = D({"a": 1, "b": 2, "c": 3})
        del d["a"]
        del d["c"]
        self.assertEqual(dict(d), {"b": 2})
        self.assertEqual(d.deleted, ["a", "c"])

    def test_delitem_outer_dict_persists(self):
        outer = {"a": 1, "b": 2}

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            del outer["a"]
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, {"b": 2})


instantiate_parametrized_tests(TestMpAssSubscript)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
