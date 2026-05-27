# Owner(s): ["module: dynamo"]
"""Tests for hash_impl: unified hash() / __hash__ protocol in Dynamo."""

import sys

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TpHashTests(torch._dynamo.test_case.TestCase):
    def _assert_hash_equals(self, value):
        """Assert Dynamo's hash(value) matches eager Python's hash(value)."""
        expected = hash(value)

        def fn(_):
            return hash(value)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def _assert_hash_raises(self, make_value, *, expect_msg):
        """Assert hash(make_value()) raises TypeError with expect_msg under Dynamo."""

        def fn(_):
            try:
                return hash(make_value())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn(expect_msg, result)

    # --- Constants (ConstantVariable) ---

    def test_hash_int(self):
        self._assert_hash_equals(42)

    def test_hash_str(self):
        self._assert_hash_equals("hello")

    def test_hash_bool(self):
        self._assert_hash_equals(True)

    def test_hash_none(self):
        self._assert_hash_equals(None)

    def test_hash_float(self):
        self._assert_hash_equals(3.14)

    # --- Tuples (TupleVariable) ---

    def test_hash_tuple(self):
        self._assert_hash_equals((1, 2, 3))

    def test_hash_tuple_strings(self):
        self._assert_hash_equals(("hello", "world"))

    def test_hash_tuple_mixed(self):
        self._assert_hash_equals((1, "hello", 3.14, None, True))

    def test_hash_empty_tuple(self):
        self._assert_hash_equals(())

    def test_hash_nested_tuple(self):
        self._assert_hash_equals(((1, 2), (3, 4)))

    def test_hash_tuple_with_tensor(self):
        """Tuple containing a tensor exercises the _RawHash fallback path
        (not the constant fast-path), and the hash is consistent."""
        t = torch.tensor([1.0])

        def fn(x):
            tup = (1, "hello", t)
            return hash(tup), hash(tup)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    def test_hash_tuple_with_tensor_as_dict_key(self):
        """Tuple containing a tensor can be used as a dict key."""
        t = torch.tensor([1.0])

        def fn(x):
            tup = (1, "hello", t)
            d = {tup: 99}
            return d[tup]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 99)

    def test_hash_torch_size(self):
        self._assert_hash_equals(torch.Size([1, 2, 3]))

    def test_hash_torch_size_from_shape(self):
        """hash(x.shape) where shape items are symbolic, not plain constants."""

        def fn(x):
            s = x.shape
            d = {}
            d[s] = 42
            return d[s]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3, 4))
        self.assertEqual(result, 42)

    # --- Frozenset (FrozensetVariable) ---

    def test_hash_frozenset(self):
        self._assert_hash_equals(frozenset({1, 2, 3}))

    def test_hash_empty_frozenset(self):
        self._assert_hash_equals(frozenset())

    def test_hash_frozenset_with_tensor(self):
        """Frozenset containing a tensor exercises the _RawHash fallback."""
        t = torch.tensor(1.0)

        def fn(x):
            fs = frozenset({1, 2, t})
            return hash(fs), hash(fs)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    # --- Range (RangeVariable) ---

    def test_hash_range(self):
        self._assert_hash_equals(range(10))

    def test_hash_range_with_step(self):
        self._assert_hash_equals(range(0, 10, 2))

    def test_hash_empty_range(self):
        self._assert_hash_equals(range(0))

    def test_hash_single_element_range(self):
        self._assert_hash_equals(range(1))

    # --- Slice (SliceVariable) ---

    def test_hash_slice(self):
        """SliceVariable: CPython slicehash (hashable in 3.12+)."""
        if sys.version_info >= (3, 12):
            self._assert_hash_equals(slice(1, 10, 2))
        else:
            self._assert_hash_raises(
                lambda: slice(1, 10, 2), expect_msg="unhashable type: 'slice'"
            )

    # --- Unhashable types raise TypeError ---

    def test_hash_list_raises(self):
        self._assert_hash_raises(lambda: [1, 2], expect_msg="unhashable type: 'list'")

    def test_hash_dict_raises(self):
        self._assert_hash_raises(lambda: {1: 2}, expect_msg="unhashable type: 'dict'")

    def test_hash_set_raises(self):
        self._assert_hash_raises(lambda: {1, 2}, expect_msg="unhashable type: 'set'")

    def test_hash_deque_raises(self):
        from collections import deque

        self._assert_hash_raises(
            lambda: deque([1, 2]), expect_msg="unhashable type: 'deque'"
        )

    def test_hash_mappingproxy_raises(self):
        import types

        self._assert_hash_raises(
            lambda: types.MappingProxyType({1: 2}),
            expect_msg="unhashable type: 'dict'",
        )

    def test_hash_tuple_with_list_raises(self):
        self._assert_hash_raises(
            lambda: ([1, 2], 3), expect_msg="unhashable type: 'list'"
        )

    def test_hash_tuple_with_dict_raises(self):
        self._assert_hash_raises(
            lambda: (1, {2: 3}), expect_msg="unhashable type: 'dict'"
        )

    # --- Dict views ---

    def test_hash_dict_keys_raises(self):
        self._assert_hash_raises(
            lambda: {1: 2}.keys(), expect_msg="unhashable type: 'dict_keys'"
        )

    def test_hash_dict_items_raises(self):
        self._assert_hash_raises(
            lambda: {1: 2}.items(), expect_msg="unhashable type: 'dict_items'"
        )

    def test_hash_dict_values(self):
        """dict_values IS hashable (identity hash) — unlike dict_keys/dict_items,
        it does not define __eq__, so CPython does not set __hash__ = None.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6630
        """
        d = {1: 2, 3: 4}
        self._assert_hash_equals(d.values())

    # --- Tensor (TensorVariable) ---

    def test_hash_tensor_as_dict_key(self):
        """hash(tensor) works for dict key usage."""
        t = torch.tensor([1.0, 2.0])

        def fn(x):
            d = {t: 42}
            return d[t]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_tensor_consistent(self):
        """hash(tensor) is consistent across calls within a compiled function."""
        t = torch.tensor([1.0, 2.0])

        def fn(x):
            return hash(t), hash(t)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    def test_hash_tensor_return_graph_breaks(self):
        """Returning hash(tensor) graph-breaks — tensor hash is fake."""
        from torch._dynamo.exc import Unsupported

        t = torch.tensor([1.0, 2.0])

        def fn(x):
            return hash(t)

        with self.assertRaisesRegex(Unsupported, "FakeIdVariable"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))

    # --- SymNodeVariable ---

    def test_hash_symbolic_int(self):
        """hash() on a symbolic int specializes the value."""

        def fn(x):
            s = x.shape[0]
            d = {s: 99}
            return d[s]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 99)

    # --- Functions ---

    def test_hash_function(self):
        def my_func():
            pass

        def fn(x):
            d = {my_func: 42}
            return d[my_func]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_lambda(self):
        my_lambda = lambda: None  # noqa: E731

        def fn(x):
            d = {my_lambda: 42}
            return d[my_lambda]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    # --- User-defined classes ---

    def test_hash_user_object_default(self):
        """Class without __hash__ override uses identity hash."""

        class MyClass:
            pass

        self._assert_hash_equals(MyClass())

    def test_hash_user_class_custom_hash(self):
        """Class with custom __hash__ is traced into, no graph break."""

        class MyObj:
            def __init__(self, x):
                self.x = x

            def __hash__(self):
                return self.x * 31

        self._assert_hash_equals(MyObj(7))

    def test_hash_user_class_custom_hash_as_dict_key(self):
        """Custom __hash__ works for dict key usage."""

        class MyKey:
            def __init__(self, v):
                self.v = v

            def __hash__(self):
                return hash(self.v)

            def __eq__(self, other):
                return isinstance(other, MyKey) and self.v == other.v

        key = MyKey(42)

        def fn(x):
            d = {key: 99}
            return d[key]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 99)

    def test_hash_user_class_unhashable(self):
        """Class with __hash__ = None raises TypeError."""

        class Unhashable:
            __hash__ = None

        obj = Unhashable()
        self._assert_hash_raises(
            lambda: obj, expect_msg="unhashable type: 'Unhashable'"
        )

    def test_hash_user_class_eq_without_hash(self):
        """Class with __eq__ but no __hash__ is unhashable (CPython sets __hash__=None)."""

        class EqOnly:
            def __eq__(self, other):
                return True

        obj = EqOnly()
        self._assert_hash_raises(lambda: obj, expect_msg="unhashable type: 'EqOnly'")

    def test_hash_object_created_in_compile_region(self):
        """hash() on an object created inside torch.compile uses VT identity."""

        class MyObj:
            pass

        def fn(x):
            obj = MyObj()
            d = {obj: 42}
            return d[obj]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_custom_hash_returns_fake_id(self):
        """Custom __hash__ returning id() of sourceless object: no graph break."""

        class HashByInnerID:
            def __init__(self):
                self.data = []

            def __hash__(self):
                return id(self.data)

        def fn(x):
            a = HashByInnerID()
            b = HashByInnerID()
            d = {a: 1, b: 2}
            return d[a] + d[b]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 3)

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_hash_custom_hash_returns_symnode(self):
        """Custom __hash__ returning a dynamic shape: specializes, no graph break."""

        class ShapeHash:
            def __init__(self, t):
                self.t = t

            def __hash__(self):
                return self.t.shape[0]

        def fn(x):
            obj = ShapeHash(x)
            d = {obj: 42}
            return d[obj]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 42)

    def test_hash_custom_hash_non_int_raises(self):
        """Custom __hash__ returning a non-int raises TypeError."""

        class BadHash:
            def __hash__(self):
                return "not an int"  # noqa: PLE0309

        obj = BadHash()
        self._assert_hash_raises(
            lambda: obj,
            expect_msg="__hash__ method should return an integer",
        )

    def test_hash_custom_hash_c_tp_hash(self):
        """User class inheriting C tp_hash (datetime_hash): no graph break."""
        import datetime

        class DateHash(datetime.datetime):
            pass

        dt = DateHash(2024, 1, 1)
        self._assert_hash_equals(dt)

    def test_hash_allow_c_hash_api(self):
        """allow_c_hash registers a C extension type's hash as safe."""
        import sqlite3

        from torch._dynamo.exc import Unsupported

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a, b)")
        conn.execute("INSERT INTO t VALUES (1, 2)")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM t").fetchone()

        # Before registration: graph-breaks
        with self.assertRaisesRegex(Unsupported, "Untraceable C tp_hash"):
            torch.compile(lambda _: hash(row), backend="eager", fullgraph=True)(
                torch.tensor(0)
            )

        torch._dynamo.reset()

        # After registration: no graph break
        torch._dynamo.allow_c_hash(sqlite3.Row)
        self._assert_hash_equals(row)

    def test_hash_non_constant_getattr_graph_breaks(self):
        """Hashing a non-constant GetAttrVariable graph-breaks instead of crashing.

        Regression test for test_public_api_surface: v.__name__ in seen where
        the GetAttrVariable wraps a dict with non-constant values (ModuleSpec).
        """

        def fn(x):
            seen = set()
            seen.add(torch.fx.__name__)
            for v in torch.fx.__dict__.values():
                if hasattr(v, "__name__") and v.__name__ in seen:
                    pass
            return x

        result = torch.compile(fn, backend="eager")(torch.tensor(0))
        self.assertEqual(result, torch.tensor(0))

    # --- functools.partial ---

    def test_hash_partial_as_dict_key(self):
        """Same partial used as dict key works."""
        from functools import partial

        def gn(x, y):
            return x + y

        def fn(x):
            p = partial(gn, x=1)
            d = {p: 42}
            return d[p]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_partial_preexisting_as_dict_key(self):
        """Pre-existing partial works as dict key."""
        from functools import partial

        def gn(x, y):
            return x + y

        p = partial(gn, x=1)

        def fn(x):
            d = {p: 42}
            return d[p]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_partial_identity_based(self):
        """Two equivalent partials are different keys, matching CPython."""
        from functools import partial

        def gn(x, y):
            return x + y

        def fn(x):
            p1 = partial(gn, x=1)
            p2 = partial(gn, x=1)
            d = {p1: 42}
            try:
                return d[p2]
            except KeyError:
                return -1

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -1)

    # --- Frozen dataclass ---

    def test_hash_frozen_dataclass(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            x: int
            y: int

        p = Point(1, 2)
        self._assert_hash_equals(p)

    # --- Weakref ---

    def test_hash_weakref(self):
        import weakref

        class MyObj:
            pass

        obj = MyObj()
        self._assert_hash_equals(weakref.ref(obj))

    # --- Bound method / user method ---

    def test_hash_bound_method(self):
        """GetAttrVariable wrapping a bound method — has wrapper_hash/meth_hash."""
        s = "hello"
        expected = hash(s.upper)

        def fn(_):
            return hash(s.upper)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_user_method(self):
        """UserMethodVariable: CPython method_hash."""

        class MyClass:
            def my_method(self):
                pass

        obj = MyClass()
        expected = hash(obj.my_method)

        def fn(_):
            return hash(obj.my_method)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    # --- Builtin types ---

    def test_hash_type(self):
        self._assert_hash_equals(int)

    def test_hash_builtin(self):
        self._assert_hash_equals(len)

    def test_hash_list_type(self):
        """hash(list) exercises the as_python_constant fallback in base hash_impl."""
        self._assert_hash_equals(list)

    # --- C tp_hash types ---

    def test_hash_datetime(self):
        import datetime

        self._assert_hash_equals(datetime.datetime(2024, 1, 1))

    def test_hash_date(self):
        import datetime

        self._assert_hash_equals(datetime.date(2024, 1, 1))

    def test_hash_timedelta(self):
        import datetime

        self._assert_hash_equals(datetime.timedelta(days=7))

    def test_hash_timezone(self):
        import datetime

        self._assert_hash_equals(datetime.timezone.utc)

    def test_hash_decimal(self):
        import decimal

        self._assert_hash_equals(decimal.Decimal("3.14"))

    def test_hash_re_pattern(self):
        import re

        self._assert_hash_equals(re.compile(r"\d+"))

    def test_hash_method_wrapper(self):
        """method-wrapper has wrapper_hash (C tp_hash), handled via as_python_constant."""
        mw = "hello".__hash__
        self._assert_hash_equals(mw)

    # --- TypingVariable ---

    def test_hash_typing_union(self):
        """TypingVariable: custom hash on typing special forms."""
        import typing

        u = typing.Union[int, str]  # noqa: UP007
        self._assert_hash_equals(u)

    # --- BaseTorchVariable ---

    def test_hash_torch_function(self):
        """BaseTorchVariable: hash of torch ops."""
        self._assert_hash_equals(torch.sin)

    # --- DistributedVariable ---

    def test_hash_distributed_group_member(self):
        """DistributedVariable (WorldMetaClassVariable): hash of GroupMember."""
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed not available")
        import torch.distributed as dist

        self._assert_hash_equals(dist.GroupMember)

    # --- OpaqueObjectClassVariable ---

    def test_hash_opaque_value_type(self):
        """OpaqueObjectClassVariable: hash of registered opaque value type."""
        from torch._library.opaque_object import register_opaque_type

        class _HashTestOpaque:
            def __init__(self, x):
                self.x = x

            def __eq__(self, other):
                return isinstance(other, _HashTestOpaque) and self.x == other.x

            def __hash__(self):
                return hash(self.x)

            def __fx_repr__(self):
                return (f"_HashTestOpaque({self.x})", {})

        register_opaque_type(_HashTestOpaque, typ="value")
        self._assert_hash_equals(_HashTestOpaque(42))

    # --- Dunder __hash__ and consistency ---

    def test_dunder_hash_int(self):
        expected = (42).__hash__()

        def fn(x):
            return (42).__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_dunder_hash_str(self):
        expected = "hello".__hash__()

        def fn(x):
            return "hello".__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_consistency(self):
        expected = hash(42)

        def fn(x):
            return (42).__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_consistency_tuple(self):
        t = (1, 2, 3)
        expected = hash(t)

        def fn(x):
            return t.__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    # --- FakeIdVariable propagation through containers ---

    def test_hash_tuple_with_sourceless_object_graph_breaks_on_return(self):
        """hash(tuple) containing a sourceless object can't be returned."""

        class MyObj:
            pass

        def fn(x):
            obj = MyObj()
            return hash((1, obj))

        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnt)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertEqual(cnt.frame_count, 0)

    def test_hash_tuple_with_sourceless_object_works_as_dict_key(self):
        """hash(tuple) containing a sourceless object works for dict keys."""

        class MyObj:
            pass

        def fn(x):
            obj = MyObj()
            tup = (1, obj)
            d = {tup: 42}
            return d[tup]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_tuple_with_tensor_graph_breaks_on_return(self):
        """hash(tuple) containing a sourceless tensor can't be returned."""

        def fn(x):
            t = torch.tensor(1.0)
            return hash((1, "hello", t))

        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnt)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertEqual(cnt.frame_count, 0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
