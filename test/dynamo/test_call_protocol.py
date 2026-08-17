# Owner(s): ["module: dynamo"]
"""Tests for tp_call: callable() (PyCallable_Check) and the __call__ slot."""

import functools
import operator
import types

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


_NOT_CALLABLE_OBJECT = object()
_A_MODULE = types.ModuleType("mod")


class _NoCall:
    def meth(self):
        pass


class _WithCall:
    def __call__(self, x):
        return x + 1


class _WithCallChild(_WithCall):
    pass


class _WithUntraceableCall:
    # __call__ is a C builtin with no traceable Python body; the type has a
    # tp_call slot (callable) but Dynamo cannot trace the call -> graph break.
    __call__ = operator.add


class TpCallTests(TestCase):
    @make_dynamo_test
    def test_callable_builtin(self):
        assert callable(len)  # noqa: S101

    @make_dynamo_test
    def test_callable_str_false(self):
        assert not callable("a")  # noqa: S101

    @make_dynamo_test
    def test_callable_lambda(self):
        assert callable(lambda x, y: x + y)  # noqa: S101

    @make_dynamo_test
    def test_callable_type(self):
        # type objects are callable via their metaclass's tp_call
        assert callable(int)  # noqa: S101
        assert callable(_NoCall)  # noqa: S101

    @make_dynamo_test
    def test_callable_bound_method(self):
        c = _NoCall()
        assert callable(c.meth)  # noqa: S101

    @make_dynamo_test
    def test_callable_instance_without_call(self):
        # __call__ is looked up on the class slot, not the instance
        assert not callable(_NoCall())  # noqa: S101

    @make_dynamo_test
    def test_callable_instance_with_call(self):
        assert callable(_WithCall())  # noqa: S101

    @make_dynamo_test
    def test_callable_subclass_instance(self):
        assert callable(_WithCallChild())  # noqa: S101

    def test_callable_user_object(self):
        # callable() on a sourced user object inside the compiled region
        def fn(x, obj, nc):
            return callable(obj), callable(nc)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            fn(torch.randn(2), _WithCall(), _NoCall()),
            compiled(torch.randn(2), _WithCall(), _NoCall()),
        )

    def test_call_via_call_function(self):
        obj = _WithCall()

        def fn(x):
            return obj(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_dunder_call(self):
        obj = _WithCall()

        def fn(x):
            return obj.__call__(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_dunder_call_matches_call(self):
        obj = _WithCall()

        def fn(x):
            return obj(x) + obj.__call__(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    @make_dynamo_test
    def test_object_not_callable(self):
        # PyObject_Call on a type with no tp_call slot raises TypeError.
        try:
            _NOT_CALLABLE_OBJECT()
        except TypeError as e:
            assert str(e) == "'object' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    @make_dynamo_test
    def test_int_not_callable(self):
        n = 5
        try:
            n()
        except TypeError as e:
            assert str(e) == "'int' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    @make_dynamo_test
    def test_module_not_callable(self):
        try:
            _A_MODULE()
        except TypeError as e:
            assert str(e) == "'module' object is not callable"  # noqa: S101
        else:
            raise AssertionError("expected TypeError")

    def test_untraceable_dunder_call_graph_breaks(self):
        obj = _WithUntraceableCall()

        def fn(x):
            return obj(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            Unsupported, "call to a callable object with no traceable __call__"
        ):
            compiled(torch.randn(2))

    def test_polyfilled_function_dunder_call(self):
        # PolyfilledFunctionVariable.__call__ must route back to call_function
        # instead of resolving the method-wrapper as a polyfill handler.
        def fn(x):
            return functools.reduce.__call__(operator.add, [x, x, x])

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))

    def test_functools_partial_call(self):
        def g(a, b):
            return a + b

        p = functools.partial(g, 10)

        def fn(x):
            return p(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2)
        self.assertEqual(fn(x), compiled(x))


if __name__ == "__main__":
    run_tests()
