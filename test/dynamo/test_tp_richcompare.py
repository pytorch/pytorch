# Owner(s): ["module: dynamo"]
"""Tests for richcompare_impl: unified comparison protocol in Dynamo."""

import operator

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing


class TpRichcompareTests(torch._dynamo.test_case.TestCase):
    def _assert_cmp_equals(self, a, b, op, *, expect_type_error=None):
        """Assert Dynamo's op(a, b) matches eager Python's op(a, b).

        If expect_type_error is set, also asserts that the eager behavior
        matches the expectation (True = must raise, False = must not raise).
        """
        try:
            expected = op(a, b)
        except TypeError as eager_exc:
            if expect_type_error is not None:
                self.assertTrue(
                    expect_type_error,
                    f"Expected {op.__name__}({a!r}, {b!r}) to succeed but "
                    f"eager raised TypeError: {eager_exc}",
                )

            def fn(_, _op=op, _a=a, _b=b):
                try:
                    return _op(_a, _b)
                except TypeError as e:
                    return str(e)

            result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
            self.assertIn("not supported between", result)
            return

        if expect_type_error is not None:
            self.assertFalse(
                expect_type_error,
                f"Expected {op.__name__}({a!r}, {b!r}) to raise TypeError "
                f"but eager returned {expected!r}",
            )

        def fn(_):
            return op(a, b)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    _ALL_OPS = (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )
    _ORDERING_OPS = frozenset({operator.lt, operator.le, operator.gt, operator.ge})
    _LE_GE = frozenset({operator.le, operator.ge})

    def _assert_all_cmp_equals(self, a, b, *, error_ops=frozenset()):
        """Assert all 6 comparison ops match eager for (a, b) and (b, a).

        error_ops: set of operator functions expected to raise TypeError.
        Use _ORDERING_OPS for types that support eq/ne but not ordering.
        """
        for op in self._ALL_OPS:
            with self.subTest(op=op.__name__, order="a,b"):
                self._assert_cmp_equals(a, b, op, expect_type_error=(op in error_ops))
            torch._dynamo.reset()
            with self.subTest(op=op.__name__, order="b,a"):
                self._assert_cmp_equals(b, a, op, expect_type_error=(op in error_ops))
            torch._dynamo.reset()

    def _assert_sourceless_cmp_equals(
        self, make_a, make_b, op, *, expect_type_error=None
    ):
        """Like _assert_cmp_equals but for objects created inside the compile region."""
        try:
            expected = op(make_a(), make_b())
        except TypeError:
            if expect_type_error is not None:
                self.assertTrue(expect_type_error)

            def fn(_, _op=op, _make_a=make_a, _make_b=make_b):
                try:
                    return _op(_make_a(), _make_b())
                except TypeError as e:
                    return str(e)

            result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
            self.assertIn("not supported between", result)
            return

        if expect_type_error is not None:
            self.assertFalse(expect_type_error)

        def fn(_, _op=op, _make_a=make_a, _make_b=make_b):
            return _op(_make_a(), _make_b())

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def _assert_all_sourceless_cmp_equals(
        self, make_a, make_b, *, error_ops=frozenset()
    ):
        """Like _assert_all_cmp_equals but for objects created inside the compile region."""
        for op in self._ALL_OPS:
            with self.subTest(op=op.__name__):
                self._assert_sourceless_cmp_equals(
                    make_a, make_b, op, expect_type_error=(op in error_ops)
                )
            torch._dynamo.reset()

    # =====================================================================
    # Constants (ConstantVariable)
    # =====================================================================

    def test_int_cmp_equal(self):
        self._assert_all_cmp_equals(42, 42)

    def test_int_cmp_diff(self):
        self._assert_all_cmp_equals(1, 2)

    def test_str_cmp_equal(self):
        self._assert_all_cmp_equals("hello", "hello")

    def test_str_cmp_diff(self):
        self._assert_all_cmp_equals("hello", "world")

    def test_float_cmp_equal(self):
        self._assert_all_cmp_equals(1.5, 1.5)

    def test_float_cmp_diff(self):
        self._assert_all_cmp_equals(1.5, 2.5)

    def test_bool_cmp_equal(self):
        self._assert_all_cmp_equals(True, True)

    def test_bool_cmp_diff(self):
        self._assert_all_cmp_equals(True, False)

    def test_mixed_numeric_cmp_equal(self):
        self._assert_all_cmp_equals(1, 1.0)

    def test_mixed_numeric_cmp_diff(self):
        self._assert_all_cmp_equals(1, 2.5)

    # =====================================================================
    # Cross-type comparison (identity fallback)
    # =====================================================================

    def test_cross_type_list_tuple(self):
        self._assert_all_cmp_equals([1, 2], (1, 2), error_ops=self._ORDERING_OPS)

    def test_cross_type_int_str(self):
        self._assert_all_cmp_equals(1, "1", error_ops=self._ORDERING_OPS)

    # =====================================================================
    # List / Tuple comparison (BaseListVariable)
    # =====================================================================

    def test_list_cmp(self):
        self._assert_all_cmp_equals([1, 2], [1, 3])

    def test_list_cmp_equal(self):
        self._assert_all_cmp_equals([1, 2, 3], [1, 2, 3])

    def test_list_cmp_length_mismatch(self):
        self._assert_all_cmp_equals([1, 2], [1, 2, 3])

    def test_list_cmp_empty(self):
        self._assert_all_cmp_equals([], [])

    def test_tuple_cmp(self):
        self._assert_all_cmp_equals((1, 2), (1, 3))

    def test_tuple_cmp_equal(self):
        self._assert_all_cmp_equals((1, 2, 3), (1, 2, 3))

    # =====================================================================
    # Dict comparison (ConstDictVariable)
    # =====================================================================

    def test_dict_cmp_equal(self):
        self._assert_all_cmp_equals(
            {1: 2, 3: 4}, {1: 2, 3: 4}, error_ops=self._ORDERING_OPS
        )

    def test_dict_cmp_diff(self):
        self._assert_all_cmp_equals({1: 2}, {3: 4}, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Set comparison (SetVariable)
    # =====================================================================

    def test_set_cmp_subset(self):
        self._assert_all_cmp_equals({1, 2}, {1, 2, 3})

    def test_set_cmp_equal(self):
        self._assert_all_cmp_equals({1, 2, 3}, {3, 2, 1})

    def test_set_cmp_disjoint(self):
        self._assert_all_cmp_equals({1, 2}, {3, 4})

    def test_frozenset_cmp_equal(self):
        self._assert_all_cmp_equals(frozenset({1, 2}), frozenset({2, 1}))

    def test_frozenset_cmp_diff(self):
        self._assert_all_cmp_equals(frozenset({1, 2}), frozenset({1, 2, 3}))

    def test_set_cross_type(self):
        self._assert_all_cmp_equals({1, 2}, [1, 2], error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Range comparison (RangeVariable)
    # =====================================================================

    def test_range_cmp_equal(self):
        self._assert_all_cmp_equals(range(10), range(10), error_ops=self._ORDERING_OPS)

    def test_range_cmp_diff(self):
        self._assert_all_cmp_equals(range(10), range(5), error_ops=self._ORDERING_OPS)

    def test_range_cmp_with_step(self):
        self._assert_all_cmp_equals(
            range(0, 10, 2), range(0, 10, 2), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Slice comparison (SliceVariable)
    # =====================================================================

    def test_slice_cmp_equal(self):
        self._assert_all_cmp_equals(slice(1, 10), slice(1, 10))

    def test_slice_cmp_diff(self):
        self._assert_all_cmp_equals(slice(1, 10), slice(2, 10))

    def test_slice_cmp_with_step(self):
        self._assert_all_cmp_equals(slice(1, 10, 2), slice(1, 10, 3))

    # =====================================================================
    # Tensor comparison (TensorVariable)
    # =====================================================================

    def test_tensor_cmp(self):
        self._assert_all_cmp_equals(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 3]))

    def test_tensor_cmp_equal(self):
        self._assert_all_cmp_equals(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))

    # =====================================================================
    # User-defined comparison methods
    # =====================================================================

    def test_user_defined_eq_and_lt(self):
        """User type with __eq__ and __lt__: gt works via reflected lt, le/ge raise."""

        class MyObj:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

            def __lt__(self, other):
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val < other.val

        self._assert_all_cmp_equals(MyObj(1), MyObj(2), error_ops=self._LE_GE)
        self._assert_all_cmp_equals(MyObj(42), MyObj(42), error_ops=self._LE_GE)

    def test_user_defined_eq_only(self):
        """Class with __eq__ but no ordering — ordering raises TypeError."""

        class MyObj:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_cmp_equals(MyObj(1), MyObj(2), error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(MyObj(42), MyObj(42), error_ops=self._ORDERING_OPS)

    def test_user_defined_no_cmp(self):
        """Type with no comparison methods — identity eq/ne, ordering raises."""

        class MyObj:
            pass

        self._assert_all_cmp_equals(MyObj(), MyObj(), error_ops=self._ORDERING_OPS)

    def test_user_defined_cross_type(self):
        """UDOV vs constant and vs different UDOV — __eq__ returns NotImplemented,
        falls back to identity (False); ordering raises TypeError."""

        class Foo:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if not isinstance(other, Foo):
                    return NotImplemented
                return self.val == other.val

        class Bar:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if not isinstance(other, Bar):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_cmp_equals(Foo(1), 42, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(Foo(1), Bar(1), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Subclass priority
    # =====================================================================

    def test_subclass_priority(self):
        """B(A) overrides __eq__ — A() == B() calls B.__eq__ first."""

        class A:
            def __eq__(self, other):
                return "a_eq"

        class B(A):
            def __eq__(self, other):
                return "b_eq"

        self._assert_all_cmp_equals(B(), A(), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Identity fallback
    # =====================================================================

    def test_identity_cmp(self):
        """object.__eq__/__ne__: identity-based for same and different objects."""

        class MyObj:
            pass

        a = MyObj()
        self._assert_all_cmp_equals(a, a, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(a, MyObj(), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Function / module comparison (identity-based)
    # =====================================================================

    def test_function_cmp(self):
        """User functions use identity comparison."""

        def f():
            pass

        def g():
            pass

        self._assert_all_cmp_equals(f, f, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(f, g, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Exception comparison (identity-based)
    # =====================================================================

    def test_exception_cmp(self):
        self._assert_all_cmp_equals(
            ValueError("a"), ValueError("a"), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Consistency: operator.eq matches __eq__
    # =====================================================================

    def test_consistency_int(self):
        """operator.eq(a, b) matches a.__eq__(b) for ints."""

        def fn(_):
            a, b = 1, 2
            return operator.eq(a, b), a.__eq__(b)

        r1, r2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(r1, r2)

    def test_consistency_list(self):
        def fn(_):
            a, b = [1, 2], [1, 2]
            return a == b, a.__eq__(b)

        r1, r2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(r1, r2)

    # =====================================================================
    # Graph-break tests
    # =====================================================================

    def _make_graph_break_class(self, *ops):
        """Create a class where the given comparison ops contain a graph break."""

        class Obj:
            def __init__(self, val):
                self.val = val

        for op in ops:

            def method(self, other, _op=op):
                torch._dynamo.graph_break()
                if not isinstance(other, Obj):
                    return NotImplemented
                return getattr(operator, _op)(self.val, other.val)

            setattr(Obj, f"__{op}__", method)
        return Obj

    def test_graph_break_in_eq_via_ne(self):
        """ne traces into __eq__ (via generic_richcompare) which graph-breaks."""
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(0) != Obj(0)

        _, ne = fn(torch.ones(1))
        self.assertFalse(ne)

    def test_graph_break_in_eq_via_eq(self):
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(1) == Obj(1)

        _, eq = fn(torch.ones(1))
        self.assertTrue(eq)

    def test_graph_break_in_eq_not_equal(self):
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(1) == Obj(2)

        _, eq = fn(torch.ones(1))
        self.assertFalse(eq)

    def test_graph_break_in_lt(self):
        Obj = self._make_graph_break_class("lt")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(1) < Obj(2)

        _, lt = fn(torch.ones(1))
        self.assertTrue(lt)

    def test_graph_break_in_lt_false(self):
        Obj = self._make_graph_break_class("lt")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(3) < Obj(2)

        _, lt = fn(torch.ones(1))
        self.assertFalse(lt)

    def test_graph_break_in_ne(self):
        Obj = self._make_graph_break_class("ne")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(1) != Obj(2)

        _, ne = fn(torch.ones(1))
        self.assertTrue(ne)

    def test_graph_break_in_gt(self):
        Obj = self._make_graph_break_class("gt")

        @torch.compile(backend="eager")
        def fn(x):
            return x + 1, Obj(3) > Obj(1)

        _, gt = fn(torch.ones(1))
        self.assertTrue(gt)

    def test_graph_break_in_multiple_ops(self):
        """Class where both __eq__ and __lt__ graph-break."""
        Obj = self._make_graph_break_class("eq", "lt")

        @torch.compile(backend="eager")
        def fn(x):
            a, b = Obj(1), Obj(2)
            return x + 1, a == b, a < b

        _, eq, lt = fn(torch.ones(1))
        self.assertFalse(eq)
        self.assertTrue(lt)

    # =====================================================================
    # Dunder method dispatch
    # =====================================================================

    def test_dunder_eq_int(self):
        def fn(_):
            return (42).__eq__(42)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    def test_dunder_ne_str(self):
        def fn(_):
            return "hello".__ne__("world")

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    def test_dunder_lt_list(self):
        def fn(_):
            return [1, 2].__lt__([1, 3])

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    # =====================================================================
    # C extension type comparison (datetime, decimal)
    # =====================================================================

    def test_datetime_cmp_equal(self):
        import datetime

        self._assert_all_cmp_equals(
            datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
        )

    def test_datetime_cmp_diff(self):
        import datetime

        self._assert_all_cmp_equals(
            datetime.date(2024, 1, 1), datetime.date(2024, 6, 1)
        )

    def test_decimal_cmp_equal(self):
        import decimal

        self._assert_all_cmp_equals(decimal.Decimal("1.5"), decimal.Decimal("1.5"))

    def test_decimal_cmp_diff(self):
        import decimal

        self._assert_all_cmp_equals(decimal.Decimal("1.0"), decimal.Decimal("2.0"))

    # =====================================================================
    # Enum comparison
    # =====================================================================

    def test_enum_cmp(self):
        from enum import Enum

        class Color(Enum):
            RED = 1
            BLUE = 2

        self._assert_all_cmp_equals(Color.RED, Color.RED, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(Color.RED, Color.BLUE, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # dict_keys / dict_items comparison (set-like)
    # =====================================================================

    def test_dict_keys_cmp(self):
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: {2: "x", 1: "y"}.keys(),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a"}.keys(),
            lambda: {1: "x", 2: "y"}.keys(),
        )

    # =====================================================================
    # NoneType
    # =====================================================================

    def test_none_cmp(self):
        self._assert_all_cmp_equals(None, None, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Comparison with None (common pattern: obj == None / obj is None)
    # =====================================================================

    def test_obj_eq_none(self):
        self._assert_all_cmp_equals(42, None, error_ops=self._ORDERING_OPS)

    def test_list_eq_none(self):
        self._assert_all_cmp_equals([1, 2], None, error_ops=self._ORDERING_OPS)

    def test_str_eq_none(self):
        self._assert_all_cmp_equals("hello", None, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Sourceless objects (created inside compile region)
    # =====================================================================

    def test_sourceless_udov_eq(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_sourceless_cmp_equals(
            lambda: MyObj(1), lambda: MyObj(1), error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: MyObj(1), lambda: MyObj(2), error_ops=self._ORDERING_OPS
        )

    def test_sourceless_list_eq(self):
        self._assert_all_sourceless_cmp_equals(lambda: [1, 2], lambda: [1, 2])
        self._assert_all_sourceless_cmp_equals(lambda: [1, 2], lambda: [3, 4])

    # =====================================================================
    # NaN comparison
    # =====================================================================

    def test_nan_cmp_self(self):
        nan = float("nan")
        self._assert_all_cmp_equals(nan, nan)

    def test_nan_cmp_value(self):
        nan = float("nan")
        self._assert_all_cmp_equals(nan, 1.0)

    # =====================================================================
    # torch.Size comparison (tuple subclass, common in shape ops)
    # =====================================================================

    def test_torch_size_cmp_equal(self):
        self._assert_all_cmp_equals(torch.Size([1, 2, 3]), torch.Size([1, 2, 3]))

    def test_torch_size_cmp_diff(self):
        self._assert_all_cmp_equals(torch.Size([1, 2]), torch.Size([1, 3]))

    # =====================================================================
    # set vs frozenset cross-comparison
    # =====================================================================

    def test_set_frozenset_cmp(self):
        self._assert_all_cmp_equals({1, 2, 3}, frozenset({1, 2, 3}))

    def test_set_frozenset_cmp_diff(self):
        self._assert_all_cmp_equals({1, 2}, frozenset({1, 2, 3}))

    # =====================================================================
    # Nested container comparison
    # =====================================================================

    def test_nested_list_cmp(self):
        self._assert_all_cmp_equals([1, [2, 3]], [1, [2, 3]])

    def test_nested_list_cmp_diff(self):
        self._assert_all_cmp_equals([1, [2, 3]], [1, [2, 4]])

    def test_nested_tuple_cmp(self):
        self._assert_all_cmp_equals((1, (2, 3)), (1, (2, 3)))

    # =====================================================================
    # dict_items comparison (set-like)
    # =====================================================================

    def test_dict_items_cmp(self):
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.items(),
            lambda: {1: "a", 2: "b"}.items(),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a"}.items(),
            lambda: {1: "b"}.items(),
        )

    # =====================================================================
    # bytes comparison
    # =====================================================================

    def test_bytes_cmp(self):
        self._assert_all_cmp_equals(b"hello", b"world")

    def test_bytes_cmp_equal(self):
        self._assert_all_cmp_equals(b"hello", b"hello")

    # =====================================================================
    # complex comparison (eq/ne only, ordering raises)
    # =====================================================================

    def test_complex_cmp(self):
        self._assert_all_cmp_equals(1 + 2j, 1 + 2j, error_ops=self._ORDERING_OPS)

    def test_complex_cmp_diff(self):
        self._assert_all_cmp_equals(1 + 2j, 3 + 4j, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # re.Pattern comparison (C extension type)
    # =====================================================================

    def test_re_pattern_cmp_equal(self):
        import re

        self._assert_all_cmp_equals(
            re.compile(r"\d+"), re.compile(r"\d+"), error_ops=self._ORDERING_OPS
        )

    def test_re_pattern_cmp_diff(self):
        import re

        self._assert_all_cmp_equals(
            re.compile(r"\d+"), re.compile(r"\w+"), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # allow_c_slot API for C extension type comparison
    # =====================================================================

    def test_allow_c_slot_api(self):
        """allow_c_slot registers a C type's comparison, then it works."""
        import sqlite3

        from torch._dynamo.exc import Unsupported

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a, b)")
        conn.execute("INSERT INTO t VALUES (1, 2)")
        conn.row_factory = sqlite3.Row
        row1 = conn.execute("SELECT * FROM t").fetchone()
        row2 = conn.execute("SELECT * FROM t").fetchone()

        # Before registration: graph-breaks
        with self.assertRaisesRegex(Unsupported, "Untraceable C tp_richcompare"):
            torch.compile(lambda _: row1 == row2, backend="eager", fullgraph=True)(
                torch.tensor(0)
            )

        torch._dynamo.reset()

        # After registration: works
        torch._dynamo.allow_c_slot(sqlite3.Row)
        self._assert_cmp_equals(row1, row1, operator.eq)
        self._assert_cmp_equals(row1, row2, operator.eq)

    # =====================================================================
    # __eq__ returning NotImplemented visible to user code
    # =====================================================================

    def test_eq_returns_not_implemented(self):
        """a.__eq__(b) can return NotImplemented when types are incompatible."""

        class MyObj:
            def __eq__(self, other):
                if not isinstance(other, MyObj):
                    return NotImplemented
                return True

        obj = MyObj()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(_):
            result = obj.__eq__(42)
            return result is NotImplemented

        self.assertTrue(fn(torch.tensor(0)))

    # =====================================================================
    # FakeIdVariable comparison (id() on sourceless objects)
    # =====================================================================

    def test_fake_id_comparison_graph_breaks(self):
        """FakeIdVariable comparisons are not guaranteed to be sound (the
        compile-time fake id may not match runtime values), so we
        unconditionally graph-break.
        See test_id.py for additional FakeId tests.
        """

        @torch.compile(backend="eager")
        def fn(x):
            lst = [1, 2, 3]
            if id(lst) == 42:
                return x + 1
            return x + 2

        result = fn(torch.tensor(0.0))
        expected = torch.tensor(2.0)
        self.assertEqual(result, expected)

    # =====================================================================
    # Comparison driving control flow (guard behavior)
    # =====================================================================

    def test_comparison_guard(self):
        """Comparison with a constant drives control flow (installs guard)."""

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, val):
            if val == 0:
                return x + 1
            return x + 2

        self.assertEqual(fn(torch.tensor(0.0), 0), torch.tensor(1.0))
        self.assertEqual(fn(torch.tensor(0.0), 1), torch.tensor(2.0))

    # =====================================================================
    # Sourceless containers (created inside compile region)
    # =====================================================================

    def test_sourceless_dict_eq(self):
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: 2}, lambda: {1: 2}, error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: 2}, lambda: {3: 4}, error_ops=self._ORDERING_OPS
        )

    def test_sourceless_set_eq(self):
        self._assert_all_sourceless_cmp_equals(lambda: {1, 2}, lambda: {2, 1})
        self._assert_all_sourceless_cmp_equals(lambda: {1, 2}, lambda: {3, 4})

    def test_sourceless_tuple_eq(self):
        self._assert_all_sourceless_cmp_equals(lambda: (1, 2), lambda: (1, 2))
        self._assert_all_sourceless_cmp_equals(lambda: (1, 2), lambda: (1, 3))

    def test_sourceless_range_eq(self):
        self._assert_all_sourceless_cmp_equals(
            lambda: range(10), lambda: range(10), error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: range(10), lambda: range(5), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Builtin function comparison (BaseBuiltinVariable)
    # =====================================================================

    def test_builtin_cmp(self):
        self._assert_all_cmp_equals(len, len, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(len, int, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Module comparison (PythonModuleVariable)
    # =====================================================================

    def test_module_cmp(self):
        import math

        @torch.compile(backend="eager")
        def fn(x):
            eq = torch == torch
            ne = torch != math
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # torch op comparison (BaseTorchVariable)
    # =====================================================================

    def test_torch_op_cmp(self):
        self._assert_all_cmp_equals(torch.sin, torch.sin, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(torch.sin, torch.cos, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # TypingVariable comparison
    # =====================================================================

    def test_typing_cmp(self):
        import typing

        List = typing.List  # noqa: UP006
        Dict = typing.Dict  # noqa: UP006

        @torch.compile(backend="eager")
        def fn(x):
            eq = List == List
            ne = List != Dict
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # MethodWrapperVariable comparison
    # =====================================================================

    def test_method_wrapper_cmp(self):
        self._assert_all_cmp_equals(
            [].__add__, [].__add__, error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # functools.partial comparison (identity-based)
    # =====================================================================

    def test_partial_cmp(self):
        import functools

        p1 = functools.partial(int, base=2)
        p2 = functools.partial(int, base=2)

        @torch.compile(backend="eager")
        def fn(x):
            eq = p1 == p1
            ne = p1 != p2
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # NN module comparison (identity-based)
    # =====================================================================

    def test_nn_module_cmp(self):
        m1 = torch.nn.Linear(3, 3)
        m2 = torch.nn.Linear(3, 3)
        self._assert_all_cmp_equals(m1, m1, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(m1, m2, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # SymNode comparison
    # =====================================================================

    def test_symnode_cmp(self):
        def fn(x):
            s = x.shape[0]
            return s == s, s != s, s < 100, s >= 1

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertTrue(result[3])

    # =====================================================================
    # OrderedDict comparison (UserDefinedDictVariable)
    # =====================================================================

    def test_ordered_dict_cmp(self):
        """OrderedDict structural equality (order-sensitive)."""
        from collections import OrderedDict

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedDict(a=1, b=2),
            lambda: OrderedDict(a=1, b=2),
            error_ops=self._ORDERING_OPS,
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedDict(a=1, b=2),
            lambda: OrderedDict(b=2, a=1),
            error_ops=self._ORDERING_OPS,
        )

    # =====================================================================
    # namedtuple comparison (UserDefinedTupleVariable)
    # =====================================================================

    def test_namedtuple_cmp(self):
        """namedtuple structural equality."""
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        self._assert_all_sourceless_cmp_equals(lambda: Point(1, 2), lambda: Point(1, 2))
        self._assert_all_sourceless_cmp_equals(lambda: Point(1, 2), lambda: Point(3, 4))

    # =====================================================================
    # MappingProxy comparison (identity-based)
    # =====================================================================

    def test_mappingproxy_cmp(self):
        from types import MappingProxyType

        d = {1: 2}
        mp = MappingProxyType(d)
        self._assert_all_cmp_equals(mp, mp, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # dict_values comparison (identity-based, no tp_richcompare)
    # =====================================================================

    def test_dict_values_cmp(self):
        """dict_values uses identity comparison (no tp_richcompare)."""

        @torch.compile(backend="eager", fullgraph=True)
        def fn(_):
            d = {1: "a", 2: "b"}
            v = d.values()
            return v == v

        self.assertTrue(fn(torch.tensor(0)))

    # =====================================================================
    # Sourceless torch.Size
    # =====================================================================

    def test_sourceless_torch_size(self):
        self._assert_all_sourceless_cmp_equals(
            lambda: torch.Size([1, 2, 3]),
            lambda: torch.Size([1, 2, 3]),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: torch.Size([1, 2]),
            lambda: torch.Size([1, 3]),
        )

    # =====================================================================
    # Pybind11 enum comparison (DispatchKey, TransformType)
    # =====================================================================

    def test_pybind11_enum_cmp(self):
        """Pybind11 enums with C-level __eq__: constant-folded via allowlist."""
        dk_cpu = torch.DispatchKey.CPU
        dk_cuda = torch.DispatchKey.CUDA
        self._assert_all_cmp_equals(dk_cpu, dk_cpu, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(dk_cpu, dk_cuda, error_ops=self._ORDERING_OPS)

    def test_pybind11_enum_non_singleton(self):
        """Pybind11 enum values returned from C++ may not be the cached singleton."""
        ks = torch._C.DispatchKeySet(torch.DispatchKey.CPU)
        dk = ks.highestPriorityTypeId()
        self._assert_all_cmp_equals(
            dk, torch.DispatchKey.CPU, error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # OrderedSet comparison
    # =====================================================================

    def test_ordered_set_cmp_equal(self):
        """OrderedSet vs OrderedSet: structural equality via SetVariable."""
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2, 3]),
            lambda: OrderedSet([3, 2, 1]),
        )

    def test_ordered_set_cmp_subset(self):
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2]),
            lambda: OrderedSet([1, 2, 3]),
        )

    def test_ordered_set_vs_set(self):
        """OrderedSet vs plain set: cross-type accepted by SetVariable."""
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2, 3]),
            lambda: {3, 2, 1},
        )

    # =====================================================================
    # UserDefinedSetVariable cross-type comparison
    # =====================================================================

    def test_user_defined_set_cmp(self):
        """class MySet(set) vs MySet: unwraps to SetVariable."""

        class MySet(set):
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2, 3}),
            lambda: MySet({3, 2, 1}),
        )

    def test_user_defined_set_vs_plain_set(self):
        """class MySet(set) vs set: cross-type comparison."""

        class MySet(set):
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2, 3}),
            lambda: {3, 2, 1},
        )

    def test_user_defined_set_vs_frozenset(self):
        class MySet(set):
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2}),
            lambda: frozenset({1, 2, 3}),
        )

    # =====================================================================
    # IS_OP (is / is not)
    # =====================================================================

    def test_is_op(self):
        """IS_OP dispatches directly, not through COMPARE_OP's graph-break wrapper."""
        a = [1, 2, 3]

        def fn(x, obj):
            return obj is None, obj is not None, x is x

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0), a)
        self.assertEqual(result, (False, True, True))

    # =====================================================================
    # GetAttrVariable resolution
    # =====================================================================

    def test_getattr_bound_method_cmp(self):
        """tensor.data_ptr == captured_data_ptr after graph break."""
        t = torch.randn(5, 5)
        data_ptr = t.data_ptr

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(copy, dp):
            return copy.data_ptr == dp

        self.assertTrue(fn(t, data_ptr))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
