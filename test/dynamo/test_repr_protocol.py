# Owner(s): ["module: dynamo"]

"""Tests for tp_repr slot implementation in Dynamo.

Tests that repr() and __repr__() are dispatched through the unified
repr_impl / generic_repr path, matching CPython's PyObject_Repr semantics.
"""

import collections
import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, TestCase


class ReprProtocolTests(TestCase):
    def test_repr_constant_int(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = 42
            return x + len(repr(val))

        x = torch.randn(2)
        ref = x + len(repr(42))
        self.assertEqual(fn(x), ref)

    def test_repr_constant_string(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = "hello"
            return x + len(repr(val))

        x = torch.randn(2)
        ref = x + len(repr("hello"))
        self.assertEqual(fn(x), ref)

    def test_repr_constant_none(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = None
            return x + len(repr(val))

        x = torch.randn(2)
        ref = x + len(repr(None))
        self.assertEqual(fn(x), ref)

    def test_repr_constant_bool(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            val = True
            return x + len(repr(val))

        x = torch.randn(2)
        ref = x + len(repr(True))
        self.assertEqual(fn(x), ref)

    def test_repr_user_defined_object_custom_repr(self):
        class Config:
            def __repr__(self):
                return "Config()"

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, config):
            return x * len(repr(config))

        config = Config()
        x = torch.randn(2, 2)
        ref = x * len(repr(config))
        self.assertEqual(fn(x, config), ref)

    def test_repr_user_defined_object_default_repr(self):
        class Obj:
            pass

        obj = Obj()

        @torch.compile(backend="eager")
        def fn(x, o):
            r = repr(o)
            return x + len(r)

        x = torch.randn(2)
        fn(x, obj)

    def test_repr_exception_no_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError
            except ValueError as e:
                return t.sin(), repr(e)

        t = torch.randn(2)
        y, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(r, "ValueError()")

    def test_repr_exception_single_arg(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError("test error")
            except ValueError as e:
                return t.sin(), repr(e)

        t = torch.randn(2)
        y, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(r, "ValueError('test error')")

    def test_repr_exception_multi_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            try:
                raise ValueError("hello", 42)
            except ValueError as e:
                return t.sin(), repr(e)

        t = torch.randn(2)
        y, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(r, "ValueError('hello', 42)")

    def test_repr_range(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            r = range(1, 10, 2)
            return x + len(repr(r))

        x = torch.randn(2)
        ref = x + len(repr(range(1, 10, 2)))
        self.assertEqual(fn(x), ref)

    def test_repr_dict(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            d = {"a": 1, "b": 2}
            return x + len(repr(d))

        x = torch.randn(2)
        fn(x)

    def test_repr_dict_view_keys(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            d = {"a": 1, "b": 2}
            keys = d.keys()
            r = repr(keys)
            return x + len(r)

        x = torch.randn(2)
        fn(x)

    def test_repr_user_defined_class(self):
        class MyClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            r = repr(MyClass)
            return x + len(r)

        x = torch.randn(2)
        fn(x)

    def test_repr_dunder_method(self):
        """Test that obj.__repr__() routes through the same path as repr(obj)."""

        class Config:
            def __repr__(self):
                return "Config(x=1)"

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, config):
            return x * len(config.__repr__())

        config = Config()
        x = torch.randn(2, 2)
        ref = x * len(config.__repr__())
        self.assertEqual(fn(x, config), ref)

    def test_repr_defaultdict(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            d = collections.defaultdict(list)
            d["a"].append(1)
            return x + len(repr(d))

        x = torch.randn(2)
        fn(x)

    def test_repr_returns_string(self):
        """CPython raises TypeError if __repr__ returns non-string."""

        class BadRepr:
            def __repr__(self):
                return 42  # type: ignore[return-value]

        obj = BadRepr()
        # CPython itself would raise TypeError here
        with self.assertRaises(TypeError):
            repr(obj)


if __name__ == "__main__":
    run_tests()
