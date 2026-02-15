# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_functools.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

# ======= END DYNAMO PATCH =======


import abc
import builtins
import collections
import collections.abc
import copy
from itertools import permutations
import pickle
from random import choice
import sys
from test import support
import threading
import time
import typing
import unittest
import unittest.mock
import weakref
import gc
from weakref import proxy
import contextlib
from inspect import Signature

from test.support import import_helper
from test.support import threading_helper

import functools

py_functools = import_helper.import_fresh_module('functools',
                                                 blocked=['_functools'])
c_functools = import_helper.import_fresh_module('functools',
                                                fresh=['_functools'])

decimal = import_helper.import_fresh_module('decimal', fresh=['_decimal'])


@contextlib.contextmanager
def replaced_module(name, replacement):
    original_module = sys.modules[name]
    sys.modules[name] = replacement
    try:
        yield
    finally:
        sys.modules[name] = original_module

def capture(*args, **kw):
    """capture all positional and keyword arguments"""
    return args, kw


def signature(part):
    """ return the signature of a partial object """
    return (part.func, part.args, part.keywords, part.__dict__)

class MyTuple(tuple):
    pass

class BadTuple(tuple):
    def __add__(self, other):
        return list(self) + list(other)

class MyDict(dict):
    pass


class _TestPartial:

    def test_basic_examples(self):
        p = self.partial(capture, 1, 2, a=10, b=20)
        self.assertTrue(callable(p))
        self.assertEqual(p(3, 4, b=30, c=40),
                         ((1, 2, 3, 4), dict(a=10, b=30, c=40)))
        p = self.partial(map, lambda x: x*10)
        self.assertEqual(list(p([1,2,3,4])), [10, 20, 30, 40])

    def test_attributes(self):
        p = self.partial(capture, 1, 2, a=10, b=20)
        # attributes should be readable
        self.assertEqual(p.func, capture)
        self.assertEqual(p.args, (1, 2))
        self.assertEqual(p.keywords, dict(a=10, b=20))

    def test_argument_checking(self):
        self.assertRaises(TypeError, self.partial)     # need at least a func arg
        try:
            self.partial(2)()
        except TypeError:
            pass
        else:
            self.fail('First arg not checked for callability')

    def test_protection_of_callers_dict_argument(self):
        # a caller's dictionary should not be altered by partial
        def func(a=10, b=20):
            return a
        d = {'a':3}
        p = self.partial(func, a=5)
        self.assertEqual(p(**d), 3)
        self.assertEqual(d, {'a':3})
        p(b=7)
        self.assertEqual(d, {'a':3})

    def test_kwargs_copy(self):
        # Issue #29532: Altering a kwarg dictionary passed to a constructor
        # should not affect a partial object after creation
        d = {'a': 3}
        p = self.partial(capture, **d)
        self.assertEqual(p(), ((), {'a': 3}))
        d['a'] = 5
        self.assertEqual(p(), ((), {'a': 3}))

    def test_arg_combinations(self):
        # exercise special code paths for zero args in either partial
        # object or the caller
        p = self.partial(capture)
        self.assertEqual(p(), ((), {}))
        self.assertEqual(p(1,2), ((1,2), {}))
        p = self.partial(capture, 1, 2)
        self.assertEqual(p(), ((1,2), {}))
        self.assertEqual(p(3,4), ((1,2,3,4), {}))

    def test_kw_combinations(self):
        # exercise special code paths for no keyword args in
        # either the partial object or the caller
        p = self.partial(capture)
        self.assertEqual(p.keywords, {})
        self.assertEqual(p(), ((), {}))
        self.assertEqual(p(a=1), ((), {'a':1}))
        p = self.partial(capture, a=1)
        self.assertEqual(p.keywords, {'a':1})
        self.assertEqual(p(), ((), {'a':1}))
        self.assertEqual(p(b=2), ((), {'a':1, 'b':2}))
        # keyword args in the call override those in the partial object
        self.assertEqual(p(a=3, b=2), ((), {'a':3, 'b':2}))

    def test_positional(self):
        # make sure positional arguments are captured correctly
        for args in [(), (0,), (0,1), (0,1,2), (0,1,2,3)]:
            p = self.partial(capture, *args)
            expected = args + ('x',)
            got, empty = p('x')
            self.assertTrue(expected == got and empty == {})

    def test_keyword(self):
        # make sure keyword arguments are captured correctly
        for a in ['a', 0, None, 3.5]:
            p = self.partial(capture, a=a)
            expected = {'a':a,'x':None}
            empty, got = p(x=None)
            self.assertTrue(expected == got and empty == ())

    def test_no_side_effects(self):
        # make sure there are no side effects that affect subsequent calls
        p = self.partial(capture, 0, a=1)
        args1, kw1 = p(1, b=2)
        self.assertTrue(args1 == (0,1) and kw1 == {'a':1,'b':2})
        args2, kw2 = p()
        self.assertTrue(args2 == (0,) and kw2 == {'a':1})

    def test_error_propagation(self):
        def f(x, y):
            x / y
        self.assertRaises(ZeroDivisionError, self.partial(f, 1, 0))
        self.assertRaises(ZeroDivisionError, self.partial(f, 1), 0)
        self.assertRaises(ZeroDivisionError, self.partial(f), 1, 0)
        self.assertRaises(ZeroDivisionError, self.partial(f, y=0), 1)

    def test_weakref(self):
        f = self.partial(int, base=16)
        p = proxy(f)
        self.assertEqual(f.func, p.func)
        f = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, 'func')

    def test_with_bound_and_unbound_methods(self):
        data = list(map(str, range(10)))
        join = self.partial(str.join, '')
        self.assertEqual(join(data), '0123456789')
        join = self.partial(''.join)
        self.assertEqual(join(data), '0123456789')

    def test_nested_optimization(self):
        partial = self.partial
        inner = partial(signature, 'asdf')
        nested = partial(inner, bar=True)
        flat = partial(signature, 'asdf', bar=True)
        self.assertEqual(signature(nested), signature(flat))

    def test_nested_optimization_bug(self):
        partial = self.partial
        with torch._dynamo.error_on_graph_break(False):
            class Builder:
                def __call__(self, tag, *children, **attrib):
                    return (tag, children, attrib)

                def __getattr__(self, tag):
                    return partial(self, tag)

        B = Builder()
        m = B.m
        assert m(1, 2, a=2) == ('m', (1, 2), dict(a=2))

    def test_nested_partial_with_attribute(self):
        # see issue 25137
        partial = self.partial

        def foo(bar):
            return bar

        p = partial(foo, 'first')
        p2 = partial(p, 'second')
        p2.new_attr = 'spam'
        self.assertEqual(p2.new_attr, 'spam')

    def test_repr(self):
        args = (object(), object())
        args_repr = ', '.join(repr(a) for a in args)
        kwargs = {'a': object(), 'b': object()}
        kwargs_reprs = ['a={a!r}, b={b!r}'.format_map(kwargs),
                        'b={b!r}, a={a!r}'.format_map(kwargs)]
        name = f"{self.partial.__module__}.{self.partial.__qualname__}"

        f = self.partial(capture)
        self.assertEqual(f'{name}({capture!r})', repr(f))

        f = self.partial(capture, *args)
        self.assertEqual(f'{name}({capture!r}, {args_repr})', repr(f))

        f = self.partial(capture, **kwargs)
        self.assertIn(repr(f),
                      [f'{name}({capture!r}, {kwargs_repr})'
                       for kwargs_repr in kwargs_reprs])

        f = self.partial(capture, *args, **kwargs)
        self.assertIn(repr(f),
                      [f'{name}({capture!r}, {args_repr}, {kwargs_repr})'
                       for kwargs_repr in kwargs_reprs])

    def test_recursive_repr(self):
        name = f"{self.partial.__module__}.{self.partial.__qualname__}"

        f = self.partial(capture)
        f.__setstate__((f, (), {}, {}))
        try:
            self.assertEqual(repr(f), '%s(...)' % (name,))
        finally:
            f.__setstate__((capture, (), {}, {}))

        f = self.partial(capture)
        f.__setstate__((capture, (f,), {}, {}))
        try:
            self.assertEqual(repr(f), '%s(%r, ...)' % (name, capture,))
        finally:
            f.__setstate__((capture, (), {}, {}))

        f = self.partial(capture)
        f.__setstate__((capture, (), {'a': f}, {}))
        try:
            self.assertEqual(repr(f), '%s(%r, a=...)' % (name, capture,))
        finally:
            f.__setstate__((capture, (), {}, {}))

    def test_pickle(self):
        with replaced_module('functools', self.module):
            f = self.partial(signature, ['asdf'], bar=[True])
            f.attr = []
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                f_copy = pickle.loads(pickle.dumps(f, proto))
                self.assertEqual(signature(f_copy), signature(f))

    def test_copy(self):
        f = self.partial(signature, ['asdf'], bar=[True])
        f.attr = []
        f_copy = copy.copy(f)
        self.assertEqual(signature(f_copy), signature(f))
        self.assertIs(f_copy.attr, f.attr)
        self.assertIs(f_copy.args, f.args)
        self.assertIs(f_copy.keywords, f.keywords)

    def test_deepcopy(self):
        f = self.partial(signature, ['asdf'], bar=[True])
        f.attr = []
        f_copy = copy.deepcopy(f)
        self.assertEqual(signature(f_copy), signature(f))
        self.assertIsNot(f_copy.attr, f.attr)
        self.assertIsNot(f_copy.args, f.args)
        self.assertIsNot(f_copy.args[0], f.args[0])
        self.assertIsNot(f_copy.keywords, f.keywords)
        self.assertIsNot(f_copy.keywords['bar'], f.keywords['bar'])

    def test_setstate(self):
        f = self.partial(signature)
        f.__setstate__((capture, (1,), dict(a=10), dict(attr=[])))

        self.assertEqual(signature(f),
                         (capture, (1,), dict(a=10), dict(attr=[])))
        self.assertEqual(f(2, b=20), ((1, 2), {'a': 10, 'b': 20}))

        f.__setstate__((capture, (1,), dict(a=10), None))

        self.assertEqual(signature(f), (capture, (1,), dict(a=10), {}))
        self.assertEqual(f(2, b=20), ((1, 2), {'a': 10, 'b': 20}))

        f.__setstate__((capture, (1,), None, None))
        #self.assertEqual(signature(f), (capture, (1,), {}, {}))
        self.assertEqual(f(2, b=20), ((1, 2), {'b': 20}))
        self.assertEqual(f(2), ((1, 2), {}))
        self.assertEqual(f(), ((1,), {}))

        f.__setstate__((capture, (), {}, None))
        self.assertEqual(signature(f), (capture, (), {}, {}))
        self.assertEqual(f(2, b=20), ((2,), {'b': 20}))
        self.assertEqual(f(2), ((2,), {}))
        self.assertEqual(f(), ((), {}))

    def test_setstate_errors(self):
        f = self.partial(signature)
        self.assertRaises(TypeError, f.__setstate__, (capture, (), {}))
        self.assertRaises(TypeError, f.__setstate__, (capture, (), {}, {}, None))
        self.assertRaises(TypeError, f.__setstate__, [capture, (), {}, None])
        self.assertRaises(TypeError, f.__setstate__, (None, (), {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, None, {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, [], {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, (), [], None))

    def test_setstate_subclasses(self):
        f = self.partial(signature)
        f.__setstate__((capture, MyTuple((1,)), MyDict(a=10), None))
        s = signature(f)
        self.assertEqual(s, (capture, (1,), dict(a=10), {}))
        self.assertIs(type(s[1]), tuple)
        self.assertIs(type(s[2]), dict)
        r = f()
        self.assertEqual(r, ((1,), {'a': 10}))
        self.assertIs(type(r[0]), tuple)
        self.assertIs(type(r[1]), dict)

        f.__setstate__((capture, BadTuple((1,)), {}, None))
        s = signature(f)
        self.assertEqual(s, (capture, (1,), {}, {}))
        self.assertIs(type(s[1]), tuple)
        r = f(2)
        self.assertEqual(r, ((1, 2), {}))
        self.assertIs(type(r[0]), tuple)

    def test_recursive_pickle(self):
        with replaced_module('functools', self.module):
            f = self.partial(capture)
            f.__setstate__((f, (), {}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    # gh-117008: Small limit since pickle uses C stack memory
                    with support.infinite_recursion(100):
                        with self.assertRaises(RecursionError):
                            pickle.dumps(f, proto)
            finally:
                f.__setstate__((capture, (), {}, {}))

            f = self.partial(capture)
            f.__setstate__((capture, (f,), {}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    try:
                        self.assertIs(f_copy.args[0], f_copy)
                    finally:
                        f_copy.__setstate__((capture, (), {}, {}))
            finally:
                f.__setstate__((capture, (), {}, {}))

            f = self.partial(capture)
            f.__setstate__((capture, (), {'a': f}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    try:
                        self.assertIs(f_copy.keywords['a'], f_copy)
                    finally:
                        f_copy.__setstate__((capture, (), {}, {}))
            finally:
                f.__setstate__((capture, (), {}, {}))

    # Issue 6083: Reference counting bug
    def test_setstate_refcount(self):
        with torch._dynamo.error_on_graph_break(False):
            class BadSequence:
                def __len__(self):
                    return 4
                def __getitem__(self, key):
                    if key == 0:
                        return max
                    elif key == 1:
                        return tuple(range(1000000))
                    elif key in (2, 3):
                        return {}
                    raise IndexError

        f = self.partial(object)
        self.assertRaises(TypeError, f.__setstate__, BadSequence())

    def test_partial_as_method(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                meth = self.partial(capture, 1, a=2)
                cmeth = classmethod(self.partial(capture, 1, a=2))
                smeth = staticmethod(self.partial(capture, 1, a=2))

        a = A()
        self.assertEqual(A.meth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        self.assertEqual(A.cmeth(3, b=4), ((1, A, 3), {'a': 2, 'b': 4}))
        self.assertEqual(A.smeth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        with self.assertWarns(FutureWarning) as w:
            self.assertEqual(a.meth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        self.assertEqual(w.filename, __file__)
        self.assertEqual(a.cmeth(3, b=4), ((1, A, 3), {'a': 2, 'b': 4}))
        self.assertEqual(a.smeth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))

    def test_partial_genericalias(self):
        alias = self.partial[int]
        self.assertIs(alias.__origin__, self.partial)
        self.assertEqual(alias.__args__, (int,))
        self.assertEqual(alias.__parameters__, ())


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestPartialC(_TestPartial, CPythonTestCase):
    if c_functools:
        module = c_functools
        partial = c_functools.partial

    def test_attributes_unwritable(self):
        # attributes should not be writable
        p = self.partial(capture, 1, 2, a=10, b=20)
        self.assertRaises(AttributeError, setattr, p, 'func', map)
        self.assertRaises(AttributeError, setattr, p, 'args', (1, 2))
        self.assertRaises(AttributeError, setattr, p, 'keywords', dict(a=1, b=2))

        p = self.partial(hex)
        try:
            del p.__dict__
        except TypeError:
            pass
        else:
            self.fail('partial object allowed __dict__ to be deleted')

    def test_manually_adding_non_string_keyword(self):
        p = self.partial(capture)
        # Adding a non-string/unicode keyword to partial kwargs
        p.keywords[1234] = 'value'
        r = repr(p)
        self.assertIn('1234', r)
        self.assertIn("'value'", r)
        with self.assertRaises(TypeError):
            p()

    def test_keystr_replaces_value(self):
        p = self.partial(capture)

        with torch._dynamo.error_on_graph_break(False):
            class MutatesYourDict(object):
                def __str__(self):
                    p.keywords[self] = ['sth2']
                    return 'astr'

        # Replacing the value during key formatting should keep the original
        # value alive (at least long enough).
        p.keywords[MutatesYourDict()] = ['sth']
        r = repr(p)
        self.assertIn('astr', r)
        self.assertIn("['sth']", r)


class TestPartialPy(_TestPartial, CPythonTestCase):
    module = py_functools
    partial = py_functools.partial


if c_functools:
    class CPartialSubclass(c_functools.partial):
        pass

class PyPartialSubclass(py_functools.partial):
    pass

@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestPartialCSubclass(TestPartialC):
    if c_functools:
        partial = CPartialSubclass

    # partial subclasses are not optimized for nested calls
    test_nested_optimization = None

class TestPartialPySubclass(TestPartialPy):
    partial = PyPartialSubclass

class TestPartialMethod(CPythonTestCase):

    class A(object):
        nothing = functools.partialmethod(capture)
        positional = functools.partialmethod(capture, 1)
        keywords = functools.partialmethod(capture, a=2)
        both = functools.partialmethod(capture, 3, b=4)
        spec_keywords = functools.partialmethod(capture, self=1, func=2)

        nested = functools.partialmethod(positional, 5)

        over_partial = functools.partialmethod(functools.partial(capture, c=6), 7)

        static = functools.partialmethod(staticmethod(capture), 8)
        cls = functools.partialmethod(classmethod(capture), d=9)

    a = A()

    def test_arg_combinations(self):
        self.assertEqual(self.a.nothing(), ((self.a,), {}))
        self.assertEqual(self.a.nothing(5), ((self.a, 5), {}))
        self.assertEqual(self.a.nothing(c=6), ((self.a,), {'c': 6}))
        self.assertEqual(self.a.nothing(5, c=6), ((self.a, 5), {'c': 6}))

        self.assertEqual(self.a.positional(), ((self.a, 1), {}))
        self.assertEqual(self.a.positional(5), ((self.a, 1, 5), {}))
        self.assertEqual(self.a.positional(c=6), ((self.a, 1), {'c': 6}))
        self.assertEqual(self.a.positional(5, c=6), ((self.a, 1, 5), {'c': 6}))

        self.assertEqual(self.a.keywords(), ((self.a,), {'a': 2}))
        self.assertEqual(self.a.keywords(5), ((self.a, 5), {'a': 2}))
        self.assertEqual(self.a.keywords(c=6), ((self.a,), {'a': 2, 'c': 6}))
        self.assertEqual(self.a.keywords(5, c=6), ((self.a, 5), {'a': 2, 'c': 6}))

        self.assertEqual(self.a.both(), ((self.a, 3), {'b': 4}))
        self.assertEqual(self.a.both(5), ((self.a, 3, 5), {'b': 4}))
        self.assertEqual(self.a.both(c=6), ((self.a, 3), {'b': 4, 'c': 6}))
        self.assertEqual(self.a.both(5, c=6), ((self.a, 3, 5), {'b': 4, 'c': 6}))

        self.assertEqual(self.A.both(self.a, 5, c=6), ((self.a, 3, 5), {'b': 4, 'c': 6}))

        self.assertEqual(self.a.spec_keywords(), ((self.a,), {'self': 1, 'func': 2}))

    def test_nested(self):
        self.assertEqual(self.a.nested(), ((self.a, 1, 5), {}))
        self.assertEqual(self.a.nested(6), ((self.a, 1, 5, 6), {}))
        self.assertEqual(self.a.nested(d=7), ((self.a, 1, 5), {'d': 7}))
        self.assertEqual(self.a.nested(6, d=7), ((self.a, 1, 5, 6), {'d': 7}))

        self.assertEqual(self.A.nested(self.a, 6, d=7), ((self.a, 1, 5, 6), {'d': 7}))

    def test_over_partial(self):
        self.assertEqual(self.a.over_partial(), ((self.a, 7), {'c': 6}))
        self.assertEqual(self.a.over_partial(5), ((self.a, 7, 5), {'c': 6}))
        self.assertEqual(self.a.over_partial(d=8), ((self.a, 7), {'c': 6, 'd': 8}))
        self.assertEqual(self.a.over_partial(5, d=8), ((self.a, 7, 5), {'c': 6, 'd': 8}))

        self.assertEqual(self.A.over_partial(self.a, 5, d=8), ((self.a, 7, 5), {'c': 6, 'd': 8}))

    def test_bound_method_introspection(self):
        obj = self.a
        self.assertIs(obj.both.__self__, obj)
        self.assertIs(obj.nested.__self__, obj)
        self.assertIs(obj.over_partial.__self__, obj)
        self.assertIs(obj.cls.__self__, self.A)
        self.assertIs(self.A.cls.__self__, self.A)

    def test_unbound_method_retrieval(self):
        obj = self.A
        self.assertFalse(hasattr(obj.both, "__self__"))
        self.assertFalse(hasattr(obj.nested, "__self__"))
        self.assertFalse(hasattr(obj.over_partial, "__self__"))
        self.assertFalse(hasattr(obj.static, "__self__"))
        self.assertFalse(hasattr(self.a.static, "__self__"))

    def test_descriptors(self):
        for obj in [self.A, self.a]:
            with self.subTest(obj=obj):
                self.assertEqual(obj.static(), ((8,), {}))
                self.assertEqual(obj.static(5), ((8, 5), {}))
                self.assertEqual(obj.static(d=8), ((8,), {'d': 8}))
                self.assertEqual(obj.static(5, d=8), ((8, 5), {'d': 8}))

                self.assertEqual(obj.cls(), ((self.A,), {'d': 9}))
                self.assertEqual(obj.cls(5), ((self.A, 5), {'d': 9}))
                self.assertEqual(obj.cls(c=8), ((self.A,), {'c': 8, 'd': 9}))
                self.assertEqual(obj.cls(5, c=8), ((self.A, 5), {'c': 8, 'd': 9}))

    def test_overriding_keywords(self):
        self.assertEqual(self.a.keywords(a=3), ((self.a,), {'a': 3}))
        self.assertEqual(self.A.keywords(self.a, a=3), ((self.a,), {'a': 3}))

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            class B(object):
                method = functools.partialmethod(None, 1)
        with self.assertRaises(TypeError):
            class B:
                method = functools.partialmethod()
        with self.assertRaises(TypeError):
            class B:
                method = functools.partialmethod(func=capture, a=1)

    def test_repr(self):
        self.assertEqual(repr(vars(self.A)['nothing']),
                         'functools.partialmethod({})'.format(capture))
        self.assertEqual(repr(vars(self.A)['positional']),
                         'functools.partialmethod({}, 1)'.format(capture))
        self.assertEqual(repr(vars(self.A)['keywords']),
                         'functools.partialmethod({}, a=2)'.format(capture))
        self.assertEqual(repr(vars(self.A)['spec_keywords']),
                         'functools.partialmethod({}, self=1, func=2)'.format(capture))
        self.assertEqual(repr(vars(self.A)['both']),
                         'functools.partialmethod({}, 3, b=4)'.format(capture))

    def test_abstract(self):
        with torch._dynamo.error_on_graph_break(False):
            class Abstract(abc.ABCMeta):

                @abc.abstractmethod
                def add(self, x, y):
                    pass

                add5 = functools.partialmethod(add, 5)

        self.assertTrue(Abstract.add.__isabstractmethod__)
        self.assertTrue(Abstract.add5.__isabstractmethod__)

        for func in [self.A.static, self.A.cls, self.A.over_partial, self.A.nested, self.A.both]:
            self.assertFalse(getattr(func, '__isabstractmethod__', False))

    def test_positional_only(self):
        def f(a, b, /):
            return a + b

        p = functools.partial(f, 1)
        self.assertEqual(p(2), f(1, 2))


class TestUpdateWrapper(CPythonTestCase):

    def check_wrapper(self, wrapper, wrapped,
                      assigned=functools.WRAPPER_ASSIGNMENTS,
                      updated=functools.WRAPPER_UPDATES):
        # Check attributes were assigned
        for name in assigned:
            self.assertIs(getattr(wrapper, name), getattr(wrapped, name))
        # Check attributes were updated
        for name in updated:
            wrapper_attr = getattr(wrapper, name)
            wrapped_attr = getattr(wrapped, name)
            for key in wrapped_attr:
                if name == "__dict__" and key == "__wrapped__":
                    # __wrapped__ is overwritten by the update code
                    continue
                self.assertIs(wrapped_attr[key], wrapper_attr[key])
        # Check __wrapped__
        self.assertIs(wrapper.__wrapped__, wrapped)


    def _default_update(self):
        def f[T](a:'This is a new annotation'):
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        f.__wrapped__ = "This is a bald faced lie"
        def wrapper(b:'This is the prior annotation'):
            pass
        functools.update_wrapper(wrapper, f)
        return wrapper, f

    def test_default_update(self):
        wrapper, f = self._default_update()
        self.check_wrapper(wrapper, f)
        T, = f.__type_params__
        self.assertIs(wrapper.__wrapped__, f)
        self.assertEqual(wrapper.__name__, 'f')
        self.assertEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.attr, 'This is also a test')
        self.assertEqual(wrapper.__annotations__['a'], 'This is a new annotation')
        self.assertNotIn('b', wrapper.__annotations__)
        self.assertEqual(wrapper.__type_params__, (T,))

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_default_update_doc(self):
        wrapper, f = self._default_update()
        self.assertEqual(wrapper.__doc__, 'This is a test')

    def test_no_update(self):
        def f():
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        def wrapper():
            pass
        functools.update_wrapper(wrapper, f, (), ())
        self.check_wrapper(wrapper, f, (), ())
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.__annotations__, {})
        self.assertFalse(hasattr(wrapper, 'attr'))

    def test_selective_update(self):
        def f():
            pass
        f.attr = 'This is a different test'
        f.dict_attr = dict(a=1, b=2, c=3)
        def wrapper():
            pass
        wrapper.dict_attr = {}
        assign = ('attr',)
        update = ('dict_attr',)
        functools.update_wrapper(wrapper, f, assign, update)
        self.check_wrapper(wrapper, f, assign, update)
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.attr, 'This is a different test')
        self.assertEqual(wrapper.dict_attr, f.dict_attr)

    def test_missing_attributes(self):
        def f():
            pass
        def wrapper():
            pass
        wrapper.dict_attr = {}
        assign = ('attr',)
        update = ('dict_attr',)
        # Missing attributes on wrapped object are ignored
        functools.update_wrapper(wrapper, f, assign, update)
        self.assertNotIn('attr', wrapper.__dict__)
        self.assertEqual(wrapper.dict_attr, {})
        # Wrapper must have expected attributes for updating
        del wrapper.dict_attr
        with self.assertRaises(AttributeError):
            functools.update_wrapper(wrapper, f, assign, update)
        wrapper.dict_attr = 1
        with self.assertRaises(AttributeError):
            functools.update_wrapper(wrapper, f, assign, update)

    @support.requires_docstrings
    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_builtin_update(self):
        # Test for bug #1576241
        def wrapper():
            pass
        functools.update_wrapper(wrapper, max)
        self.assertEqual(wrapper.__name__, 'max')
        self.assertTrue(wrapper.__doc__.startswith('max('))
        self.assertEqual(wrapper.__annotations__, {})

    def test_update_type_wrapper(self):
        def wrapper(*args): pass

        functools.update_wrapper(wrapper, type)
        self.assertEqual(wrapper.__name__, 'type')
        self.assertEqual(wrapper.__annotations__, {})
        self.assertEqual(wrapper.__type_params__, ())


class TestWraps(TestUpdateWrapper):

    def _default_update(self):
        def f():
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        f.__wrapped__ = "This is still a bald faced lie"
        @functools.wraps(f)
        def wrapper():
            pass
        return wrapper, f

    def test_default_update(self):
        wrapper, f = self._default_update()
        self.check_wrapper(wrapper, f)
        self.assertEqual(wrapper.__name__, 'f')
        self.assertEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.attr, 'This is also a test')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_default_update_doc(self):
        wrapper, _ = self._default_update()
        self.assertEqual(wrapper.__doc__, 'This is a test')

    def test_no_update(self):
        def f():
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        @functools.wraps(f, (), ())
        def wrapper():
            pass
        self.check_wrapper(wrapper, f, (), ())
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertFalse(hasattr(wrapper, 'attr'))

    def test_selective_update(self):
        def f():
            pass
        f.attr = 'This is a different test'
        f.dict_attr = dict(a=1, b=2, c=3)
        def add_dict_attr(f):
            f.dict_attr = {}
            return f
        assign = ('attr',)
        update = ('dict_attr',)
        @functools.wraps(f, assign, update)
        @add_dict_attr
        def wrapper():
            pass
        self.check_wrapper(wrapper, f, assign, update)
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.attr, 'This is a different test')
        self.assertEqual(wrapper.dict_attr, f.dict_attr)


class _TestReduce:
    def test_reduce(self):
        with torch._dynamo.error_on_graph_break(False):
            class Squares:
                def __init__(self, max):
                    self.max = max
                    self.sofar = []

                def __len__(self):
                    return len(self.sofar)

                def __getitem__(self, i):
                    if not 0 <= i < self.max: raise IndexError
                    n = len(self.sofar)
                    while n <= i:
                        self.sofar.append(n*n)
                        n += 1
                    return self.sofar[i]
        def add(x, y):
            return x + y
        self.assertEqual(self.reduce(add, ['a', 'b', 'c'], ''), 'abc')
        self.assertEqual(
            self.reduce(add, [['a', 'c'], [], ['d', 'w']], []),
            ['a','c','d','w']
        )
        self.assertEqual(self.reduce(lambda x, y: x*y, range(2,8), 1), 5040)
        self.assertEqual(
            self.reduce(lambda x, y: x*y, range(2,21), 1),
            2432902008176640000
        )
        self.assertEqual(self.reduce(add, Squares(10)), 285)
        self.assertEqual(self.reduce(add, Squares(10), 0), 285)
        self.assertEqual(self.reduce(add, Squares(0), 0), 0)
        with torch._dynamo.error_on_graph_break(False):
            self.assertRaises(TypeError, self.reduce)
            self.assertRaises(TypeError, self.reduce, 42, 42)
            self.assertRaises(TypeError, self.reduce, 42, 42, 42)
        self.assertEqual(self.reduce(42, "1"), "1") # func is never called with one item
        self.assertEqual(self.reduce(42, "", "1"), "1") # func is never called with one item
        with torch._dynamo.error_on_graph_break(False):
            self.assertRaises(TypeError, self.reduce, 42, (42, 42))
            self.assertRaises(TypeError, self.reduce, add, []) # arg 2 must not be empty sequence with no initial value
            self.assertRaises(TypeError, self.reduce, add, "")
            self.assertRaises(TypeError, self.reduce, add, ())
            self.assertRaises(TypeError, self.reduce, add, object())

        with torch._dynamo.error_on_graph_break(False):
            class TestFailingIter:
                def __iter__(self):
                    raise RuntimeError
        self.assertRaises(RuntimeError, self.reduce, add, TestFailingIter())

        self.assertEqual(self.reduce(add, [], None), None)
        self.assertEqual(self.reduce(add, [], 42), 42)

        with torch._dynamo.error_on_graph_break(False):
            class BadSeq:
                def __getitem__(self, index):
                    raise ValueError
        self.assertRaises(ValueError, self.reduce, 42, BadSeq())

    # Test reduce()'s use of iterators.
    def test_iterator_usage(self):
        with torch._dynamo.error_on_graph_break(False):
            class SequenceClass:
                def __init__(self, n):
                    self.n = n
                def __getitem__(self, i):
                    if 0 <= i < self.n:
                        return i
                    else:
                        raise IndexError

        from operator import add
        self.assertEqual(self.reduce(add, SequenceClass(5)), 10)
        self.assertEqual(self.reduce(add, SequenceClass(5), 42), 52)
        self.assertRaises(TypeError, self.reduce, add, SequenceClass(0))
        self.assertEqual(self.reduce(add, SequenceClass(0), 42), 42)
        self.assertEqual(self.reduce(add, SequenceClass(1)), 0)
        self.assertEqual(self.reduce(add, SequenceClass(1), 42), 42)

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(self.reduce(add, d), "".join(d.keys()))


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestReduceC(_TestReduce, CPythonTestCase):
    if c_functools:
        if TEST_WITH_TORCHDYNAMO:
            reduce = functools.reduce
        else:
            reduce = c_functools.reduce


class TestReducePy(_TestReduce, CPythonTestCase):
    reduce = staticmethod(py_functools.reduce)


class _TestCmpToKey:

    def test_cmp_to_key(self):
        def cmp1(x, y):
            return (x > y) - (x < y)
        key = self.cmp_to_key(cmp1)
        self.assertEqual(key(3), key(3))
        self.assertGreater(key(3), key(1))
        self.assertGreaterEqual(key(3), key(3))

        def cmp2(x, y):
            return int(x) - int(y)
        key = self.cmp_to_key(cmp2)
        self.assertEqual(key(4.0), key('4'))
        self.assertLess(key(2), key('35'))
        self.assertLessEqual(key(2), key('35'))
        self.assertNotEqual(key(2), key('35'))

    def test_cmp_to_key_arguments(self):
        def cmp1(x, y):
            return (x > y) - (x < y)
        key = self.cmp_to_key(mycmp=cmp1)
        self.assertEqual(key(obj=3), key(obj=3))
        self.assertGreater(key(obj=3), key(obj=1))
        with self.assertRaises((TypeError, AttributeError)):
            key(3) > 1    # rhs is not a K object
        with self.assertRaises((TypeError, AttributeError)):
            1 < key(3)    # lhs is not a K object
        with self.assertRaises(TypeError):
            key = self.cmp_to_key()             # too few args
        with self.assertRaises(TypeError):
            key = self.cmp_to_key(cmp1, None)   # too many args
        key = self.cmp_to_key(cmp1)
        with self.assertRaises(TypeError):
            key()                                    # too few args
        with self.assertRaises(TypeError):
            key(None, None)                          # too many args

    def test_bad_cmp(self):
        def cmp1(x, y):
            raise ZeroDivisionError
        key = self.cmp_to_key(cmp1)
        with self.assertRaises(ZeroDivisionError):
            key(3) > key(1)

        with torch._dynamo.error_on_graph_break(False):
            class BadCmp:
                def __lt__(self, other):
                    raise ZeroDivisionError
        def cmp1(x, y):
            return BadCmp()
        with self.assertRaises(ZeroDivisionError):
            key(3) > key(1)

    def test_obj_field(self):
        def cmp1(x, y):
            return (x > y) - (x < y)
        key = self.cmp_to_key(mycmp=cmp1)
        self.assertEqual(key(50).obj, 50)

    def test_sort_int(self):
        def mycmp(x, y):
            return y - x
        self.assertEqual(sorted(range(5), key=self.cmp_to_key(mycmp)),
                         [4, 3, 2, 1, 0])

    def test_sort_int_str(self):
        def mycmp(x, y):
            x, y = int(x), int(y)
            return (x > y) - (x < y)
        values = [5, '3', 7, 2, '0', '1', 4, '10', 1]
        values = sorted(values, key=self.cmp_to_key(mycmp))
        self.assertEqual([int(value) for value in values],
                         [0, 1, 1, 2, 3, 4, 5, 7, 10])

    def test_hash(self):
        def mycmp(x, y):
            return y - x
        key = self.cmp_to_key(mycmp)
        k = key(10)
        self.assertRaises(TypeError, hash, k)
        self.assertNotIsInstance(k, collections.abc.Hashable)

    @unittest.skipIf(support.MISSING_C_DOCSTRINGS,
                     "Signature information for builtins requires docstrings")
    def test_cmp_to_signature(self):
        sig = Signature.from_callable(self.cmp_to_key)
        self.assertEqual(str(sig), '(mycmp)')
        def mycmp(x, y):
            return y - x
        sig = Signature.from_callable(self.cmp_to_key(mycmp))
        self.assertEqual(str(sig), '(obj)')



@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestCmpToKeyC(_TestCmpToKey, CPythonTestCase):
    if c_functools:
        if TEST_WITH_TORCHDYNAMO:
            cmp_to_key = functools.cmp_to_key
        else:
            cmp_to_key = c_functools.cmp_to_key

    @support.cpython_only
    def test_disallow_instantiation(self):
        # Ensure that the type disallows instantiation (bpo-43916)
        support.check_disallow_instantiation(
            self, type(c_functools.cmp_to_key(None))
        )


class TestCmpToKeyPy(_TestCmpToKey, CPythonTestCase):
    cmp_to_key = staticmethod(py_functools.cmp_to_key)


class TestTotalOrdering(CPythonTestCase):

    def test_total_ordering_lt(self):
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class A:
                def __init__(self, value):
                    self.value = value
                def __lt__(self, other):
                    return self.value < other.value
                def __eq__(self, other):
                    return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(1) > A(2))

    def test_total_ordering_le(self):
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class A:
                def __init__(self, value):
                    self.value = value
                def __le__(self, other):
                    return self.value <= other.value
                def __eq__(self, other):
                    return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(1) >= A(2))

    def test_total_ordering_gt(self):
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class A:
                def __init__(self, value):
                    self.value = value
                def __gt__(self, other):
                    return self.value > other.value
                def __eq__(self, other):
                    return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(2) < A(1))

    def test_total_ordering_ge(self):
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class A:
                def __init__(self, value):
                    self.value = value
                def __ge__(self, other):
                    return self.value >= other.value
                def __eq__(self, other):
                    return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(2) <= A(1))

    def test_total_ordering_no_overwrite(self):
        # new methods should not overwrite existing
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class A(int):
                pass
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))

    def test_no_operations_defined(self):
        with self.assertRaises(ValueError):
            @functools.total_ordering
            class A:
                pass

    def test_notimplemented(self):
        # Verify NotImplemented results are correctly handled
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class ImplementsLessThan:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsLessThan):
                        return self.value == other.value
                    return False
                def __lt__(self, other):
                    if isinstance(other, ImplementsLessThan):
                        return self.value < other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsLessThanEqualTo:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsLessThanEqualTo):
                        return self.value == other.value
                    return False
                def __le__(self, other):
                    if isinstance(other, ImplementsLessThanEqualTo):
                        return self.value <= other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsGreaterThan:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsGreaterThan):
                        return self.value == other.value
                    return False
                def __gt__(self, other):
                    if isinstance(other, ImplementsGreaterThan):
                        return self.value > other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsGreaterThanEqualTo:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsGreaterThanEqualTo):
                        return self.value == other.value
                    return False
                def __ge__(self, other):
                    if isinstance(other, ImplementsGreaterThanEqualTo):
                        return self.value >= other.value
                    return NotImplemented

        self.assertIs(ImplementsLessThan(1).__le__(1), NotImplemented)
        self.assertIs(ImplementsLessThan(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsLessThan(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__le__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__gt__(1), NotImplemented)

    def test_type_error_when_not_implemented(self):
        # bug 10042; ensure stack overflow does not occur
        # when decorated types return NotImplemented
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class ImplementsLessThan:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsLessThan):
                        return self.value == other.value
                    return False
                def __lt__(self, other):
                    if isinstance(other, ImplementsLessThan):
                        return self.value < other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsGreaterThan:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsGreaterThan):
                        return self.value == other.value
                    return False
                def __gt__(self, other):
                    if isinstance(other, ImplementsGreaterThan):
                        return self.value > other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsLessThanEqualTo:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsLessThanEqualTo):
                        return self.value == other.value
                    return False
                def __le__(self, other):
                    if isinstance(other, ImplementsLessThanEqualTo):
                        return self.value <= other.value
                    return NotImplemented

            @functools.total_ordering
            class ImplementsGreaterThanEqualTo:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ImplementsGreaterThanEqualTo):
                        return self.value == other.value
                    return False
                def __ge__(self, other):
                    if isinstance(other, ImplementsGreaterThanEqualTo):
                        return self.value >= other.value
                    return NotImplemented

            @functools.total_ordering
            class ComparatorNotImplemented:
                def __init__(self, value):
                    self.value = value
                def __eq__(self, other):
                    if isinstance(other, ComparatorNotImplemented):
                        return self.value == other.value
                    return False
                def __lt__(self, other):
                    return NotImplemented

        with self.subTest("LT < 1"), self.assertRaises(TypeError):
            ImplementsLessThan(-1) < 1

        with self.subTest("LT < LE"), self.assertRaises(TypeError):
            ImplementsLessThan(0) < ImplementsLessThanEqualTo(0)

        with self.subTest("LT < GT"), self.assertRaises(TypeError):
            ImplementsLessThan(1) < ImplementsGreaterThan(1)

        with self.subTest("LE <= LT"), self.assertRaises(TypeError):
            ImplementsLessThanEqualTo(2) <= ImplementsLessThan(2)

        with self.subTest("LE <= GE"), self.assertRaises(TypeError):
            ImplementsLessThanEqualTo(3) <= ImplementsGreaterThanEqualTo(3)

        with self.subTest("GT > GE"), self.assertRaises(TypeError):
            ImplementsGreaterThan(4) > ImplementsGreaterThanEqualTo(4)

        with self.subTest("GT > LT"), self.assertRaises(TypeError):
            ImplementsGreaterThan(5) > ImplementsLessThan(5)

        with self.subTest("GE >= GT"), self.assertRaises(TypeError):
            ImplementsGreaterThanEqualTo(6) >= ImplementsGreaterThan(6)

        with self.subTest("GE >= LE"), self.assertRaises(TypeError):
            ImplementsGreaterThanEqualTo(7) >= ImplementsLessThanEqualTo(7)

        with self.subTest("GE when equal"):
            a = ComparatorNotImplemented(8)
            b = ComparatorNotImplemented(8)
            self.assertEqual(a, b)
            with self.assertRaises(TypeError):
                a >= b

        with self.subTest("LE when equal"):
            a = ComparatorNotImplemented(9)
            b = ComparatorNotImplemented(9)
            self.assertEqual(a, b)
            with self.assertRaises(TypeError):
                a <= b

    def test_pickle(self):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for name in '__lt__', '__gt__', '__le__', '__ge__':
                with self.subTest(method=name, proto=proto):
                    method = getattr(Orderable_LT, name)
                    method_copy = pickle.loads(pickle.dumps(method, proto))
                    self.assertIs(method_copy, method)


    def test_total_ordering_for_metaclasses_issue_44605(self):
        with torch._dynamo.error_on_graph_break(False):
            @functools.total_ordering
            class SortableMeta(type):
                def __new__(cls, name, bases, ns):
                    return super().__new__(cls, name, bases, ns)

                def __lt__(self, other):
                    if not isinstance(other, SortableMeta):
                        pass
                    return self.__name__ < other.__name__

                def __eq__(self, other):
                    if not isinstance(other, SortableMeta):
                        pass
                    return self.__name__ == other.__name__

            class B(metaclass=SortableMeta):
                pass

            class A(metaclass=SortableMeta):
                pass

        self.assertTrue(A < B)
        self.assertFalse(A > B)


@functools.total_ordering
class Orderable_LT:
    def __init__(self, value):
        self.value = value
    def __lt__(self, other):
        return self.value < other.value
    def __eq__(self, other):
        return self.value == other.value


class _TestCache:
    # This tests that the pass-through is working as designed.
    # The underlying functionality is tested in TestLRU.

    def test_cache(self):
        @self.module.cache
        def fib(n):
            if n < 2:
                return n
            return fib(n-1) + fib(n-2)
        self.assertEqual([fib(n) for n in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))


class TestCachePy(_TestCache, CPythonTestCase):
    module = py_functools


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestCacheC(_TestCache, CPythonTestCase):
    if c_functools:
        module = c_functools


class _TestLRUU:

    def test_lru(self):
        def orig(x, y):
            return 3 * x + y
        f = self.module.lru_cache(maxsize=20)(orig)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(maxsize, 20)
        self.assertEqual(currsize, 0)
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)

        domain = range(5)
        for i in range(1000):
            x, y = choice(domain), choice(domain)
            actual = f(x, y)
            expected = orig(x, y)
            self.assertEqual(actual, expected)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertTrue(hits > misses)
        self.assertEqual(hits + misses, 1000)
        self.assertEqual(currsize, 20)

        f.cache_clear()   # test clearing
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)
        self.assertEqual(currsize, 0)
        f(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # Test bypassing the cache
        self.assertIs(f.__wrapped__, orig)
        f.__wrapped__(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size zero (which means "never-cache")
        @self.module.lru_cache(0)
        def f():
            nonlocal f_cnt
            f_cnt += 1
            return 20
        self.assertEqual(f.cache_info().maxsize, 0)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 5)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 5)
        self.assertEqual(currsize, 0)

        # test size one
        @self.module.lru_cache(1)
        def f():
            nonlocal f_cnt
            f_cnt += 1
            return 20
        self.assertEqual(f.cache_info().maxsize, 1)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 1)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 4)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size two
        @self.module.lru_cache(2)
        def f(x):
            nonlocal f_cnt
            f_cnt += 1
            return x*10
        self.assertEqual(f.cache_info().maxsize, 2)
        f_cnt = 0
        for x in 7, 9, 7, 9, 7, 9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 7:
            #    *  *              *                          *
            self.assertEqual(f(x), x*10)
        self.assertEqual(f_cnt, 4)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 12)
        self.assertEqual(misses, 4)
        self.assertEqual(currsize, 2)

    def test_lru_no_args(self):
        @self.module.lru_cache
        def square(x):
            return x ** 2

        self.assertEqual(list(map(square, [10, 20, 10])),
                         [100, 400, 100])
        self.assertEqual(square.cache_info().hits, 1)
        self.assertEqual(square.cache_info().misses, 2)
        self.assertEqual(square.cache_info().maxsize, 128)
        self.assertEqual(square.cache_info().currsize, 2)

    def test_lru_bug_35780(self):
        # C version of the lru_cache was not checking to see if
        # the user function call has already modified the cache
        # (this arises in recursive calls and in multi-threading).
        # This cause the cache to have orphan links not referenced
        # by the cache dictionary.

        once = True                 # Modified by f(x) below

        @self.module.lru_cache(maxsize=10)
        def f(x):
            nonlocal once
            rv = f'.{x}.'
            if x == 20 and once:
                once = False
                rv = f(x)
            return rv

        # Fill the cache
        for x in range(15):
            self.assertEqual(f(x), f'.{x}.')
        self.assertEqual(f.cache_info().currsize, 10)

        # Make a recursive call and make sure the cache remains full
        self.assertEqual(f(20), '.20.')
        self.assertEqual(f.cache_info().currsize, 10)

    def test_lru_bug_36650(self):
        # C version of lru_cache was treating a call with an empty **kwargs
        # dictionary as being distinct from a call with no keywords at all.
        # This did not result in an incorrect answer, but it did trigger
        # an unexpected cache miss.

        @self.module.lru_cache()
        def f(x):
            pass

        f(0)
        f(0, **{})
        self.assertEqual(f.cache_info().hits, 1)

    def test_lru_hash_only_once(self):
        # To protect against weird reentrancy bugs and to improve
        # efficiency when faced with slow __hash__ methods, the
        # LRU cache guarantees that it will only call __hash__
        # only once per use as an argument to the cached function.

        @self.module.lru_cache(maxsize=1)
        def f(x, y):
            return x * 3 + y

        # Simulate the integer 5
        mock_int = unittest.mock.Mock()
        mock_int.__mul__ = unittest.mock.Mock(return_value=15)
        mock_int.__hash__ = unittest.mock.Mock(return_value=999)

        # Add to cache:  One use as an argument gives one call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 1)
        self.assertEqual(f.cache_info(), (0, 1, 1, 1))

        # Cache hit: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 1, 1, 1))

        # Cache eviction: No use as an argument gives no additional call
        self.assertEqual(f(6, 2), 20)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 2, 1, 1))

        # Cache miss: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 3)
        self.assertEqual(f.cache_info(), (1, 3, 1, 1))

    def test_lru_reentrancy_with_len(self):
        # Test to make sure the LRU cache code isn't thrown-off by
        # caching the built-in len() function.  Since len() can be
        # cached, we shouldn't use it inside the lru code itself.
        old_len = builtins.len
        try:
            builtins.len = self.module.lru_cache(4)(len)
            for i in [0, 0, 1, 2, 3, 3, 4, 5, 6, 1, 7, 2, 1]:
                self.assertEqual(len('abcdefghijklmn'[:i]), i)
        finally:
            builtins.len = old_len

    def test_lru_star_arg_handling(self):
        # Test regression that arose in ea064ff3c10f
        @self.module.lru_cache()
        def f(*args):
            return args

        self.assertEqual(f(1, 2), (1, 2))
        self.assertEqual(f((1, 2)), ((1, 2),))

    def test_lru_type_error(self):
        # Regression test for issue #28653.
        # lru_cache was leaking when one of the arguments
        # wasn't cacheable.

        @self.module.lru_cache(maxsize=None)
        def infinite_cache(o):
            pass

        @self.module.lru_cache(maxsize=10)
        def limited_cache(o):
            pass

        with self.assertRaises(TypeError):
            infinite_cache([])

        with self.assertRaises(TypeError):
            limited_cache([])

    def test_lru_with_maxsize_none(self):
        @self.module.lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n-1) + fib(n-2)
        self.assertEqual([fib(n) for n in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_lru_with_maxsize_negative(self):
        @self.module.lru_cache(maxsize=-10)
        def eq(n):
            return n
        for i in (0, 1):
            self.assertEqual([eq(n) for n in range(150)], list(range(150)))
        self.assertEqual(eq.cache_info(),
            self.module._CacheInfo(hits=0, misses=300, maxsize=0, currsize=0))

    def test_lru_with_exceptions(self):
        # Verify that user_function exceptions get passed through without
        # creating a hard-to-read chained exception.
        # http://bugs.python.org/issue13177
        for maxsize in (None, 128):
            @self.module.lru_cache(maxsize)
            def func(i):
                return 'abc'[i]
            self.assertEqual(func(0), 'a')
            with self.assertRaises(IndexError) as cm:
                func(15)
            self.assertIsNone(cm.exception.__context__)
            # Verify that the previous exception did not result in a cached entry
            with self.assertRaises(IndexError):
                func(15)

    def test_lru_with_types(self):
        for maxsize in (None, 128):
            @self.module.lru_cache(maxsize=maxsize, typed=True)
            def square(x):
                return x * x
            self.assertEqual(square(3), 9)
            self.assertEqual(type(square(3)), type(9))
            self.assertEqual(square(3.0), 9.0)
            self.assertEqual(type(square(3.0)), type(9.0))
            self.assertEqual(square(x=3), 9)
            self.assertEqual(type(square(x=3)), type(9))
            self.assertEqual(square(x=3.0), 9.0)
            self.assertEqual(type(square(x=3.0)), type(9.0))
            self.assertEqual(square.cache_info().hits, 4)
            self.assertEqual(square.cache_info().misses, 4)

    def test_lru_cache_typed_is_not_recursive(self):
        cached = self.module.lru_cache(typed=True)(repr)

        self.assertEqual(cached(1), '1')
        self.assertEqual(cached(True), 'True')
        self.assertEqual(cached(1.0), '1.0')
        self.assertEqual(cached(0), '0')
        self.assertEqual(cached(False), 'False')
        self.assertEqual(cached(0.0), '0.0')

        self.assertEqual(cached((1,)), '(1,)')
        self.assertEqual(cached((True,)), '(1,)')
        self.assertEqual(cached((1.0,)), '(1,)')
        self.assertEqual(cached((0,)), '(0,)')
        self.assertEqual(cached((False,)), '(0,)')
        self.assertEqual(cached((0.0,)), '(0,)')

        with torch._dynamo.error_on_graph_break(False):
            class T(tuple):
                pass

        self.assertEqual(cached(T((1,))), '(1,)')
        self.assertEqual(cached(T((True,))), '(1,)')
        self.assertEqual(cached(T((1.0,))), '(1,)')
        self.assertEqual(cached(T((0,))), '(0,)')
        self.assertEqual(cached(T((False,))), '(0,)')
        self.assertEqual(cached(T((0.0,))), '(0,)')

    def test_lru_with_keyword_args(self):
        @self.module.lru_cache()
        def fib(n):
            if n < 2:
                return n
            return fib(n=n-1) + fib(n=n-2)
        self.assertEqual(
            [fib(n=number) for number in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        )
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=128, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=128, currsize=0))

    def test_lru_with_keyword_args_maxsize_none(self):
        @self.module.lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n=n-1) + fib(n=n-2)
        self.assertEqual([fib(n=number) for number in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_kwargs_order(self):
        # PEP 468: Preserving Keyword Argument Order
        @self.module.lru_cache(maxsize=10)
        def f(**kwargs):
            return list(kwargs.items())
        self.assertEqual(f(a=1, b=2), [('a', 1), ('b', 2)])
        self.assertEqual(f(b=2, a=1), [('b', 2), ('a', 1)])
        self.assertEqual(f.cache_info(),
            self.module._CacheInfo(hits=0, misses=2, maxsize=10, currsize=2))

    def test_lru_cache_decoration(self):
        def f(zomg: 'zomg_annotation'):
            """f doc string"""
            return 42
        g = self.module.lru_cache()(f)
        for attr in self.module.WRAPPER_ASSIGNMENTS:
            self.assertEqual(getattr(g, attr), getattr(f, attr))

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded(self):
        n, m = 5, 11
        def orig(x, y):
            return 3 * x + y
        f = self.module.lru_cache(maxsize=n*m)(orig)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(currsize, 0)

        start = threading.Event()
        def full(k):
            start.wait(10)
            for _ in range(m):
                self.assertEqual(f(k, 0), orig(k, 0))

        def clear():
            start.wait(10)
            for _ in range(2*m):
                f.cache_clear()

        orig_si = sys.getswitchinterval()
        support.setswitchinterval(1e-6)
        try:
            # create n threads in order to fill cache
            threads = [threading.Thread(target=full, args=[k])
                       for k in range(n)]
            with threading_helper.start_threads(threads):
                start.set()

            hits, misses, maxsize, currsize = f.cache_info()
            if self.module is py_functools:
                # XXX: Why can be not equal?
                self.assertLessEqual(misses, n)
                self.assertLessEqual(hits, m*n - misses)
            else:
                self.assertEqual(misses, n)
                self.assertEqual(hits, m*n - misses)
            self.assertEqual(currsize, n)

            # create n threads in order to fill cache and 1 to clear it
            threads = [threading.Thread(target=clear)]
            threads += [threading.Thread(target=full, args=[k])
                        for k in range(n)]
            start.clear()
            with threading_helper.start_threads(threads):
                start.set()
        finally:
            sys.setswitchinterval(orig_si)

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded2(self):
        # Simultaneous call with the same arguments
        n, m = 5, 7
        start = threading.Barrier(n+1)
        pause = threading.Barrier(n+1)
        stop = threading.Barrier(n+1)
        @self.module.lru_cache(maxsize=m*n)
        def f(x):
            pause.wait(10)
            return 3 * x
        self.assertEqual(f.cache_info(), (0, 0, m*n, 0))
        def test():
            for i in range(m):
                start.wait(10)
                self.assertEqual(f(i), 3 * i)
                stop.wait(10)
        threads = [threading.Thread(target=test) for k in range(n)]
        with threading_helper.start_threads(threads):
            for i in range(m):
                start.wait(10)
                stop.reset()
                pause.wait(10)
                start.reset()
                stop.wait(10)
                pause.reset()
                self.assertEqual(f.cache_info(), (0, (i+1)*n, m*n, i+1))

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded3(self):
        @self.module.lru_cache(maxsize=2)
        def f(x):
            time.sleep(.01)
            return 3 * x
        def test(i, x):
            self.assertEqual(f(x), 3 * x, i)
        threads = [threading.Thread(target=test, args=(i, v))
                   for i, v in enumerate([1, 2, 2, 3, 2])]
        with threading_helper.start_threads(threads):
            pass

    def test_need_for_rlock(self):
        # This will deadlock on an LRU cache that uses a regular lock

        @self.module.lru_cache(maxsize=10)
        def test_func(x):
            'Used to demonstrate a reentrant lru_cache call within a single thread'
            return x

        with torch._dynamo.error_on_graph_break(False):
            class DoubleEq:
                'Demonstrate a reentrant lru_cache call within a single thread'
                def __init__(self, x):
                    self.x = x
                def __hash__(self):
                    return self.x
                def __eq__(self, other):
                    if self.x == 2:
                        test_func(DoubleEq(1))
                    return self.x == other.x

        test_func(DoubleEq(1))                      # Load the cache
        test_func(DoubleEq(2))                      # Load the cache
        self.assertEqual(test_func(DoubleEq(2)),    # Trigger a re-entrant __eq__ call
                         DoubleEq(2))               # Verify the correct return value

    def test_lru_method(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(int):
                f_cnt = 0
                @self.module.lru_cache(2)
                def f(self, x):
                    self.f_cnt += 1
                    return x*10+self
        a = X(5)
        b = X(5)
        c = X(7)
        self.assertEqual(X.f.cache_info(), (0, 0, 2, 0))

        for x in 1, 2, 2, 3, 1, 1, 1, 2, 3, 3:
            self.assertEqual(a.f(x), x*10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 0, 0))
        self.assertEqual(X.f.cache_info(), (4, 6, 2, 2))

        for x in 1, 2, 1, 1, 1, 1, 3, 2, 2, 2:
            self.assertEqual(b.f(x), x*10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 0))
        self.assertEqual(X.f.cache_info(), (10, 10, 2, 2))

        for x in 2, 1, 1, 1, 1, 2, 1, 3, 2, 1:
            self.assertEqual(c.f(x), x*10 + 7)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 5))
        self.assertEqual(X.f.cache_info(), (15, 15, 2, 2))

        self.assertEqual(a.f.cache_info(), X.f.cache_info())
        self.assertEqual(b.f.cache_info(), X.f.cache_info())
        self.assertEqual(c.f.cache_info(), X.f.cache_info())

    def test_pickle(self):
        cls = self.__class__
        for f in cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(proto=proto, func=f):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    self.assertIs(f_copy, f)

    def test_copy(self):
        cls = self.__class__
        def orig(x, y):
            return 3 * x + y
        part = self.module.partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth,
                 self.module.lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.copy(f)
                self.assertIs(f_copy, f)

    def test_deepcopy(self):
        cls = self.__class__
        def orig(x, y):
            return 3 * x + y
        part = self.module.partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth,
                 self.module.lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.deepcopy(f)
                self.assertIs(f_copy, f)

    def test_lru_cache_parameters(self):
        @self.module.lru_cache(maxsize=2)
        def f():
            return 1
        self.assertEqual(f.cache_parameters(), {'maxsize': 2, "typed": False})

        @self.module.lru_cache(maxsize=1000, typed=True)
        def f():
            return 1
        self.assertEqual(f.cache_parameters(), {'maxsize': 1000, "typed": True})

    @support.suppress_immortalization()
    def test_lru_cache_weakrefable(self):
        @self.module.lru_cache
        def test_function(x):
            return x

        with torch._dynamo.error_on_graph_break(False):
            class A:
                @self.module.lru_cache
                def test_method(self, x):
                    return (self, x)

                @staticmethod
                @self.module.lru_cache
                def test_staticmethod(x):
                    return (self, x)

        refs = [weakref.ref(test_function),
                weakref.ref(A.test_method),
                weakref.ref(A.test_staticmethod)]

        for ref in refs:
            self.assertIsNotNone(ref())

        del A
        del test_function
        gc.collect()

        for ref in refs:
            self.assertIsNone(ref())

    def test_common_signatures(self):
        def orig(a, /, b, c=True): ...
        lru = self.module.lru_cache(1)(orig)

        self.assertEqual(str(Signature.from_callable(lru)), '(a, /, b, c=True)')
        self.assertEqual(str(Signature.from_callable(lru.cache_info)), '()')
        self.assertEqual(str(Signature.from_callable(lru.cache_clear)), '()')

    @support.skip_on_s390x
    @unittest.skipIf(support.is_wasi, "WASI has limited C stack")
    def test_lru_recursion(self):

        @self.module.lru_cache
        def fib(n):
            if n <= 1:
                return n
            return fib(n-1) + fib(n-2)

        if not support.Py_DEBUG:
            depth = support.get_c_recursion_limit()*2//7
            with support.infinite_recursion():
                fib(depth)
        if self.module == c_functools:
            fib.cache_clear()
            with support.infinite_recursion():
                with self.assertRaises(RecursionError):
                    fib(10000)


@py_functools.lru_cache()
def py_cached_func(x, y):
    return 3 * x + y

if c_functools:
    @c_functools.lru_cache()
    def c_cached_func(x, y):
        return 3 * x + y


class TestLRUPy(_TestLRUU, CPythonTestCase):
    module = py_functools
    cached_func = py_cached_func,

    @module.lru_cache()
    def cached_meth(self, x, y):
        return 3 * x + y

    @staticmethod
    @module.lru_cache()
    def cached_staticmeth(x, y):
        return 3 * x + y


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestLRUC(_TestLRUU, CPythonTestCase):
    if c_functools:
        module = c_functools
        cached_func = c_cached_func,

        @module.lru_cache()
        def cached_meth(self, x, y):
            return 3 * x + y

        @staticmethod
        @module.lru_cache()
        def cached_staticmeth(x, y):
            return 3 * x + y


class TestSingleDispatch(CPythonTestCase):
    def test_simple_overloads(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        def g_int(i):
            return "integer"
        g.register(int, g_int)
        self.assertEqual(g("str"), "base")
        self.assertEqual(g(1), "integer")
        self.assertEqual(g([1,2,3]), "base")

    def test_mro(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass
            class C(A):
                pass
            class B(A):
                pass
            class D(C, B):
                pass
        def g_A(a):
            return "A"
        def g_B(b):
            return "B"
        g.register(A, g_A)
        g.register(B, g_B)
        self.assertEqual(g(A()), "A")
        self.assertEqual(g(B()), "B")
        self.assertEqual(g(C()), "A")
        self.assertEqual(g(D()), "B")

    def test_register_decorator(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register(int)
        def g_int(i):
            return "int %s" % (i,)
        self.assertEqual(g(""), "base")
        self.assertEqual(g(12), "int 12")
        self.assertIs(g.dispatch(int), g_int)
        self.assertIs(g.dispatch(object), g.dispatch(str))
        # Note: in the assert above this is not g.
        # @singledispatch returns the wrapper.

    def test_wrapping_attributes(self):
        @functools.singledispatch
        def g(obj):
            "Simple test"
            return "Test"
        self.assertEqual(g.__name__, "g")
        if sys.flags.optimize < 2:
            self.assertEqual(g.__doc__, "Simple test")

    @unittest.skipUnless(decimal, 'requires _decimal')
    @support.cpython_only
    def test_c_classes(self):
        @functools.singledispatch
        def g(obj):
            return "base"
        @g.register(decimal.DecimalException)
        def _(obj):
            return obj.args
        subn = decimal.Subnormal("Exponent < Emin")
        rnd = decimal.Rounded("Number got rounded")
        self.assertEqual(g(subn), ("Exponent < Emin",))
        self.assertEqual(g(rnd), ("Number got rounded",))
        @g.register(decimal.Subnormal)
        def _(obj):
            return "Too small to care."
        self.assertEqual(g(subn), "Too small to care.")
        self.assertEqual(g(rnd), ("Number got rounded",))

    def test_compose_mro(self):
        # None of the examples in this test depend on haystack ordering.
        c = collections.abc
        mro = functools._compose_mro
        bases = [c.Sequence, c.MutableMapping, c.Mapping, c.Set]
        for haystack in permutations(bases):
            m = mro(dict, haystack)
            self.assertEqual(m, [dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])
        bases = [c.Container, c.Mapping, c.MutableMapping, collections.OrderedDict]
        for haystack in permutations(bases):
            m = mro(collections.ChainMap, haystack)
            self.assertEqual(m, [collections.ChainMap, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

        # If there's a generic function with implementations registered for
        # both Sized and Container, passing a defaultdict to it results in an
        # ambiguous dispatch which will cause a RuntimeError (see
        # test_mro_conflicts).
        bases = [c.Container, c.Sized, str]
        for haystack in permutations(bases):
            m = mro(collections.defaultdict, [c.Sized, c.Container, str])
            self.assertEqual(m, [collections.defaultdict, dict, c.Sized,
                                 c.Container, object])

        # MutableSequence below is registered directly on D. In other words, it
        # precedes MutableMapping which means single dispatch will always
        # choose MutableSequence here.
        with torch._dynamo.error_on_graph_break(False):
            class D(collections.defaultdict):
                pass
        c.MutableSequence.register(D)
        bases = [c.MutableSequence, c.MutableMapping]
        for haystack in permutations(bases):
            m = mro(D, haystack)
            self.assertEqual(m, [D, c.MutableSequence, c.Sequence, c.Reversible,
                                 collections.defaultdict, dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable, c.Container,
                                 object])

        # Container and Callable are registered on different base classes and
        # a generic function supporting both should always pick the Callable
        # implementation if a C instance is passed.
        with torch._dynamo.error_on_graph_break(False):
            class C(collections.defaultdict):
                def __call__(self):
                    pass
        bases = [c.Sized, c.Callable, c.Container, c.Mapping]
        for haystack in permutations(bases):
            m = mro(C, haystack)
            self.assertEqual(m, [C, c.Callable, collections.defaultdict, dict, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

    def test_register_abc(self):
        c = collections.abc
        d = {"a": "b"}
        l = [1, 2, 3]
        s = {object(), None}
        f = frozenset(s)
        t = (1, 2, 3)
        @functools.singledispatch
        def g(obj):
            return "base"
        self.assertEqual(g(d), "base")
        self.assertEqual(g(l), "base")
        self.assertEqual(g(s), "base")
        self.assertEqual(g(f), "base")
        self.assertEqual(g(t), "base")
        g.register(c.Sized, lambda obj: "sized")
        self.assertEqual(g(d), "sized")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableMapping, lambda obj: "mutablemapping")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(collections.ChainMap, lambda obj: "chainmap")
        self.assertEqual(g(d), "mutablemapping")  # irrelevant ABCs registered
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSequence, lambda obj: "mutablesequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSet, lambda obj: "mutableset")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Mapping, lambda obj: "mapping")
        self.assertEqual(g(d), "mutablemapping")  # not specific enough
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Sequence, lambda obj: "sequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sequence")
        g.register(c.Set, lambda obj: "set")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(dict, lambda obj: "dict")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(list, lambda obj: "list")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(set, lambda obj: "concrete-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(frozenset, lambda obj: "frozen-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "sequence")
        g.register(tuple, lambda obj: "tuple")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "tuple")

    def test_c3_abc(self):
        c = collections.abc
        mro = functools._c3_mro
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                pass
            class B(A):
                def __len__(self):
                    return 0   # implies Sized
            @c.Container.register
            class C(object):
                pass
            class D(object):
                pass   # unrelated
            class X(D, C, B):
                def __call__(self):
                    pass   # implies Callable
        expected = [X, c.Callable, D, C, c.Container, B, c.Sized, A, object]
        for abcs in permutations([c.Sized, c.Callable, c.Container]):
            self.assertEqual(mro(X, abcs=abcs), expected)
        # unrelated ABCs don't appear in the resulting MRO
        many_abcs = [c.Mapping, c.Sized, c.Callable, c.Container, c.Iterable]
        self.assertEqual(mro(X, abcs=many_abcs), expected)

    def test_false_meta(self):
        # see issue23572
        with torch._dynamo.error_on_graph_break(False):
            class MetaA(type):
                def __len__(self):
                    return 0
            class A(metaclass=MetaA):
                pass
            class AA(A):
                pass
            @functools.singledispatch
            def fun(a):
                return 'base A'
            @fun.register(A)
            def _(a):
                return 'fun A'
            aa = AA()
        self.assertEqual(fun(aa), 'fun A')

    def test_mro_conflicts(self):
        c = collections.abc
        @functools.singledispatch
        def g(arg):
            return "base"
        with torch._dynamo.error_on_graph_break(False):
            class O(c.Sized):
                def __len__(self):
                    return 0
            o = O()
        self.assertEqual(g(o), "base")
        g.register(c.Iterable, lambda arg: "iterable")
        g.register(c.Container, lambda arg: "container")
        g.register(c.Sized, lambda arg: "sized")
        g.register(c.Set, lambda arg: "set")
        self.assertEqual(g(o), "sized")
        c.Iterable.register(O)
        self.assertEqual(g(o), "sized")   # because it's explicitly in __mro__
        c.Container.register(O)
        self.assertEqual(g(o), "sized")   # see above: Sized is in __mro__
        c.Set.register(O)
        self.assertEqual(g(o), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Container
        with torch._dynamo.error_on_graph_break(False):
            class P:
                pass
        p = P()
        self.assertEqual(g(p), "base")
        c.Iterable.register(P)
        self.assertEqual(g(p), "iterable")
        c.Container.register(P)
        with self.assertRaises(RuntimeError) as re_one:
            g(p)
        self.assertIn(
            str(re_one.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Iterable'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Iterable'> "
              "or <class 'collections.abc.Container'>")),
        )

        with torch._dynamo.error_on_graph_break(False):
            class Q(c.Sized):
                def __len__(self):
                    return 0
        q = Q()
        self.assertEqual(g(q), "sized")
        c.Iterable.register(Q)
        self.assertEqual(g(q), "sized")   # because it's explicitly in __mro__
        c.Set.register(Q)
        self.assertEqual(g(q), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Iterable
        @functools.singledispatch
        def h(arg):
            return "base"
        @h.register(c.Sized)
        def _(arg):
            return "sized"
        @h.register(c.Container)
        def _(arg):
            return "container"
        # Even though Sized and Container are explicit bases of MutableMapping,
        # this ABC is implicitly registered on defaultdict which makes all of
        # MutableMapping's bases implicit as well from defaultdict's
        # perspective.
        with self.assertRaises(RuntimeError) as re_two:
            h(collections.defaultdict(lambda: 0))
        self.assertIn(
            str(re_two.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        with torch._dynamo.error_on_graph_break(False):
            class R(collections.defaultdict):
                pass
        c.MutableSequence.register(R)
        @functools.singledispatch
        def i(arg):
            return "base"
        @i.register(c.MutableMapping)
        def _(arg):
            return "mapping"
        @i.register(c.MutableSequence)
        def _(arg):
            return "sequence"
        r = R()
        self.assertEqual(i(r), "sequence")
        with torch._dynamo.error_on_graph_break(False):
            class S:
                pass
            class T(S, c.Sized):
                def __len__(self):
                    return 0
        t = T()
        self.assertEqual(h(t), "sized")
        c.Container.register(T)
        self.assertEqual(h(t), "sized")   # because it's explicitly in the MRO
        with torch._dynamo.error_on_graph_break(False):
            class U:
                def __len__(self):
                    return 0
        u = U()
        self.assertEqual(h(u), "sized")   # implicit Sized subclass inferred
                                          # from the existence of __len__()
        c.Container.register(U)
        # There is no preference for registered versus inferred ABCs.
        with self.assertRaises(RuntimeError) as re_three:
            h(u)
        self.assertIn(
            str(re_three.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        with torch._dynamo.error_on_graph_break(False):
            class V(c.Sized, S):
                def __len__(self):
                    return 0
        @functools.singledispatch
        def j(arg):
            return "base"
        @j.register(S)
        def _(arg):
            return "s"
        @j.register(c.Container)
        def _(arg):
            return "container"
        v = V()
        self.assertEqual(j(v), "s")
        c.Container.register(V)
        self.assertEqual(j(v), "container")   # because it ends up right after
                                              # Sized in the MRO

    def test_cache_invalidation(self):
        from collections import UserDict
        import weakref

        class TracingDict(UserDict):
            def __init__(self, *args, **kwargs):
                super(TracingDict, self).__init__(*args, **kwargs)
                self.set_ops = []
                self.get_ops = []
            def __getitem__(self, key):
                result = self.data[key]
                self.get_ops.append(key)
                return result
            def __setitem__(self, key, value):
                self.set_ops.append(key)
                self.data[key] = value
            def clear(self):
                self.data.clear()

        td = TracingDict()
        with support.swap_attr(weakref, "WeakKeyDictionary", lambda: td):
            c = collections.abc
            @functools.singledispatch
            def g(arg):
                return "base"
            d = {}
            l = []
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(g(l), "base")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict, list])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(td.data[list], g.registry[object])
            self.assertEqual(td.data[dict], td.data[list])
            self.assertEqual(g(l), "base")
            self.assertEqual(g(d), "base")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list])
            g.register(list, lambda arg: "list")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict])
            self.assertEqual(td.data[dict],
                             functools._find_impl(dict, g.registry))
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            self.assertEqual(td.data[list],
                             functools._find_impl(list, g.registry))
            class X:
                pass
            c.MutableMapping.register(X)   # Will not invalidate the cache,
                                           # not using ABCs yet.
            self.assertEqual(g(d), "base")
            self.assertEqual(g(l), "list")
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            g.register(c.Sized, lambda arg: "sized")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "sized")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict])
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            self.assertEqual(g(l), "list")
            self.assertEqual(g(d), "sized")
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            g.dispatch(list)
            g.dispatch(dict)
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict,
                                          list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            c.MutableSet.register(X)       # Will invalidate the cache.
            self.assertEqual(len(td), 2)   # Stale cache.
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 1)
            g.register(c.MutableMapping, lambda arg: "mutablemapping")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "mutablemapping")
            self.assertEqual(len(td), 1)
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            g.register(dict, lambda arg: "dict")
            self.assertEqual(g(d), "dict")
            self.assertEqual(g(l), "list")
            g._clear_cache()
            self.assertEqual(len(td), 0)

    def test_annotations(self):
        @functools.singledispatch
        def i(arg):
            return "base"
        @i.register
        def _(arg: collections.abc.Mapping):
            return "mapping"
        @i.register
        def _(arg: "collections.abc.Sequence"):
            return "sequence"
        self.assertEqual(i(None), "base")
        self.assertEqual(i({"a": 1}), "mapping")
        self.assertEqual(i([1, 2, 3]), "sequence")
        self.assertEqual(i((1, 2, 3)), "sequence")
        self.assertEqual(i("str"), "sequence")

        # Registering classes as callables doesn't work with annotations,
        # you need to pass the type explicitly.
        with torch._dynamo.error_on_graph_break(False):
            @i.register(str)
            class _:
                def __init__(self, arg):
                    self.arg = arg

                def __eq__(self, other):
                    return self.arg == other
        self.assertEqual(i("str"), "str")

    def test_method_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def t(self, arg):
                    self.arg = "base"
                @t.register(int)
                def _(self, arg):
                    self.arg = "int"
                @t.register(str)
                def _(self, arg):
                    self.arg = "str"
            a = A()

        a.t(0)
        self.assertEqual(a.arg, "int")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))
        a.t('')
        self.assertEqual(a.arg, "str")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))
        a.t(0.0)
        self.assertEqual(a.arg, "base")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))

    def test_staticmethod_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                @staticmethod
                def t(arg):
                    return arg
                @t.register(int)
                @staticmethod
                def _(arg):
                    return isinstance(arg, int)
                @t.register(str)
                @staticmethod
                def _(arg):
                    return isinstance(arg, str)
            a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)

    def test_slotted_class(self):
        with torch._dynamo.error_on_graph_break(False):
            class Slot:
                __slots__ = ('a', 'b')
                @functools.singledispatchmethod
                def go(self, item, arg):
                    pass

                @go.register
                def _(self, item: int, arg):
                    return item + arg

            s = Slot()
        self.assertEqual(s.go(1, 1), 2)

    def test_classmethod_slotted_class(self):
        with torch._dynamo.error_on_graph_break(False):
            class Slot:
                __slots__ = ('a', 'b')
                @functools.singledispatchmethod
                @classmethod
                def go(cls, item, arg):
                    pass

                @go.register
                @classmethod
                def _(cls, item: int, arg):
                    return item + arg

            s = Slot()
        self.assertEqual(s.go(1, 1), 2)
        self.assertEqual(Slot.go(1, 1), 2)

    def test_staticmethod_slotted_class(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                __slots__ = ['a']
                @functools.singledispatchmethod
                @staticmethod
                def t(arg):
                    return arg
                @t.register(int)
                @staticmethod
                def _(arg):
                    return isinstance(arg, int)
                @t.register(str)
                @staticmethod
                def _(arg):
                    return isinstance(arg, str)
            a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)
        self.assertTrue(a.t(0))
        self.assertTrue(a.t(''))
        self.assertEqual(a.t(0.0), 0.0)

    def test_assignment_behavior(self):
        # see gh-106448
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def t(arg):
                    return arg

        a = A()
        a.t.foo = 'bar'
        a2 = A()
        with self.assertRaises(AttributeError):
            a2.t.foo

    def test_classmethod_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self, arg):
                    self.arg = arg

                @functools.singledispatchmethod
                @classmethod
                def t(cls, arg):
                    return cls("base")
                @t.register(int)
                @classmethod
                def _(cls, arg):
                    return cls("int")
                @t.register(str)
                @classmethod
                def _(cls, arg):
                    return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_callable_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self, arg):
                    self.arg = arg

                @functools.singledispatchmethod
                @classmethod
                def t(cls, arg):
                    return cls("base")

        @A.t.register(int)
        @classmethod
        def _(cls, arg):
            return cls("int")
        @A.t.register(str)
        @classmethod
        def _(cls, arg):
            return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_abstractmethod_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class Abstract(metaclass=abc.ABCMeta):

                @functools.singledispatchmethod
                @abc.abstractmethod
                def add(self, x, y):
                    pass

        self.assertTrue(Abstract.add.__isabstractmethod__)
        self.assertTrue(Abstract.__dict__['add'].__isabstractmethod__)

        with self.assertRaises(TypeError):
            Abstract()

    def test_type_ann_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def t(self, arg):
                    return "base"
                @t.register
                def _(self, arg: int):
                    return "int"
                @t.register
                def _(self, arg: str):
                    return "str"
            a = A()

        self.assertEqual(a.t(0), "int")
        self.assertEqual(a.t(''), "str")
        self.assertEqual(a.t(0.0), "base")

    def test_staticmethod_type_ann_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                @staticmethod
                def t(arg):
                    return arg
                @t.register
                @staticmethod
                def _(arg: int):
                    return isinstance(arg, int)
                @t.register
                @staticmethod
                def _(arg: str):
                    return isinstance(arg, str)
            a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)

    def test_classmethod_type_ann_register(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __init__(self, arg):
                    self.arg = arg

                @functools.singledispatchmethod
                @classmethod
                def t(cls, arg):
                    return cls("base")
                @t.register
                @classmethod
                def _(cls, arg: int):
                    return cls("int")
                @t.register
                @classmethod
                def _(cls, arg: str):
                    return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_method_wrapping_attributes(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def func(self, arg: int) -> str:
                    """My function docstring"""
                    return str(arg)
                @functools.singledispatchmethod
                @classmethod
                def cls_func(cls, arg: int) -> str:
                    """My function docstring"""
                    return str(arg)
                @functools.singledispatchmethod
                @staticmethod
                def static_func(arg: int) -> str:
                    """My function docstring"""
                    return str(arg)

        prefix = A.__qualname__ + '.'
        for meth in (
            A.func,
            A().func,
            A.cls_func,
            A().cls_func,
            A.static_func,
            A().static_func
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__qualname__, prefix + meth.__name__)
                self.assertEqual(meth.__doc__,
                                 ('My function docstring'
                                  if support.HAVE_PY_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__['arg'], int)

        self.assertEqual(A.func.__name__, 'func')
        self.assertEqual(A().func.__name__, 'func')
        self.assertEqual(A.cls_func.__name__, 'cls_func')
        self.assertEqual(A().cls_func.__name__, 'cls_func')
        self.assertEqual(A.static_func.__name__, 'static_func')
        self.assertEqual(A().static_func.__name__, 'static_func')

    def test_double_wrapped_methods(self):
        def classmethod_friendly_decorator(func):
            wrapped = func.__func__
            @classmethod
            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                return wrapped(*args, **kwargs)
            return wrapper

        with torch._dynamo.error_on_graph_break(False):
            class WithoutSingleDispatch:
                @classmethod
                @contextlib.contextmanager
                def cls_context_manager(cls, arg: int) -> str:
                    try:
                        yield str(arg)
                    finally:
                        return 'Done'

                @classmethod_friendly_decorator
                @classmethod
                def decorated_classmethod(cls, arg: int) -> str:
                    return str(arg)

            class WithSingleDispatch:
                @functools.singledispatchmethod
                @classmethod
                @contextlib.contextmanager
                def cls_context_manager(cls, arg: int) -> str:
                    """My function docstring"""
                    try:
                        yield str(arg)
                    finally:
                        return 'Done'

                @functools.singledispatchmethod
                @classmethod_friendly_decorator
                @classmethod
                def decorated_classmethod(cls, arg: int) -> str:
                    """My function docstring"""
                    return str(arg)

        # These are sanity checks
        # to test the test itself is working as expected
        with WithoutSingleDispatch.cls_context_manager(5) as foo:
            without_single_dispatch_foo = foo

        with WithSingleDispatch.cls_context_manager(5) as foo:
            single_dispatch_foo = foo

        self.assertEqual(without_single_dispatch_foo, single_dispatch_foo)
        self.assertEqual(single_dispatch_foo, '5')

        self.assertEqual(
            WithoutSingleDispatch.decorated_classmethod(5),
            WithSingleDispatch.decorated_classmethod(5)
        )

        self.assertEqual(WithSingleDispatch.decorated_classmethod(5), '5')

        # Behavioural checks now follow
        for method_name in ('cls_context_manager', 'decorated_classmethod'):
            with self.subTest(method=method_name):
                self.assertEqual(
                    getattr(WithSingleDispatch, method_name).__name__,
                    getattr(WithoutSingleDispatch, method_name).__name__
                )

                self.assertEqual(
                    getattr(WithSingleDispatch(), method_name).__name__,
                    getattr(WithoutSingleDispatch(), method_name).__name__
                )

        for meth in (
            WithSingleDispatch.cls_context_manager,
            WithSingleDispatch().cls_context_manager,
            WithSingleDispatch.decorated_classmethod,
            WithSingleDispatch().decorated_classmethod
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__doc__,
                                 ('My function docstring'
                                  if support.HAVE_PY_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__['arg'], int)

        self.assertEqual(
            WithSingleDispatch.cls_context_manager.__name__,
            'cls_context_manager'
        )
        self.assertEqual(
            WithSingleDispatch().cls_context_manager.__name__,
            'cls_context_manager'
        )
        self.assertEqual(
            WithSingleDispatch.decorated_classmethod.__name__,
            'decorated_classmethod'
        )
        self.assertEqual(
            WithSingleDispatch().decorated_classmethod.__name__,
            'decorated_classmethod'
        )

    def test_invalid_registrations(self):
        msg_prefix = "Invalid first argument to `register()`: "
        msg_suffix = (
            ". Use either `@register(some_class)` or plain `@register` on an "
            "annotated function."
        )
        @functools.singledispatch
        def i(arg):
            return "base"
        with self.assertRaises(TypeError) as exc:
            @i.register(42)
            def _(arg):
                return "I annotated with a non-type"
        self.assertTrue(str(exc.exception).startswith(msg_prefix + "42"))
        self.assertTrue(str(exc.exception).endswith(msg_suffix))
        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg):
                return "I forgot to annotate"
        self.assertTrue(str(exc.exception).startswith(msg_prefix +
            "<function TestSingleDispatch.test_invalid_registrations.<locals>._"
        ))
        self.assertTrue(str(exc.exception).endswith(msg_suffix))

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Iterable[str]):
                # At runtime, dispatching on generics is impossible.
                # When registering implementations with singledispatch, avoid
                # types from `typing`. Instead, annotate with regular types
                # or ABCs.
                return "I annotated with a generic collection"
        self.assertTrue(str(exc.exception).startswith(
            "Invalid annotation for 'arg'."
        ))
        self.assertTrue(str(exc.exception).endswith(
            'typing.Iterable[str] is not a class.'
        ))

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Union[int, typing.Iterable[str]]):
                return "Invalid Union"
        self.assertTrue(str(exc.exception).startswith(
            "Invalid annotation for 'arg'."
        ))
        self.assertTrue(str(exc.exception).endswith(
            'typing.Union[int, typing.Iterable[str]] not all arguments are classes.'
        ))

    def test_invalid_positional_argument(self):
        @functools.singledispatch
        def f(*args, **kwargs):
            pass
        msg = 'f requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            f()
        msg = 'f requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            f(a=1)

    def test_invalid_positional_argument_singledispatchmethod(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def t(self, *args, **kwargs):
                    pass
        msg = 't requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            A().t()
        msg = 't requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            A().t(a=1)

    def test_union(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            return "typing.Union"

        @f.register
        def _(arg: int | float):
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "typing.Union")
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")
        self.assertEqual(f(1.0), "types.UnionType")

    def test_union_conflict(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            return "typing.Union"

        @f.register
        def _(arg: int | str):
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "types.UnionType")  # last one wins
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")

    def test_union_None(self):
        @functools.singledispatch
        def typing_union(arg):
            return "default"

        @typing_union.register
        def _(arg: typing.Union[str, None]):
            return "typing.Union"

        self.assertEqual(typing_union(1), "default")
        self.assertEqual(typing_union(""), "typing.Union")
        self.assertEqual(typing_union(None), "typing.Union")

        @functools.singledispatch
        def types_union(arg):
            return "default"

        @types_union.register
        def _(arg: int | None):
            return "types.UnionType"

        self.assertEqual(types_union(""), "default")
        self.assertEqual(types_union(1), "types.UnionType")
        self.assertEqual(types_union(None), "types.UnionType")

    def test_register_genericalias(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int], lambda arg: "types.GenericAlias")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int], lambda arg: "typing.GenericAlias")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int] | str, lambda arg: "types.UnionTypes(types.GenericAlias)")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[float] | bytes, lambda arg: "typing.Union[typing.GenericAlias]")

        self.assertEqual(f([1]), "default")
        self.assertEqual(f([1.0]), "default")
        self.assertEqual(f(""), "default")
        self.assertEqual(f(b""), "default")

    def test_register_genericalias_decorator(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int])
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int])
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int] | str)
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int] | str)

    def test_register_genericalias_annotation(self):
        @functools.singledispatch
        def f(arg):
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: list[int]):
                return "types.GenericAlias"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float]):
                return "typing.GenericAlias"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: list[int] | str):
                return "types.UnionType(types.GenericAlias)"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float] | bytes):
                return "typing.Union[typing.GenericAlias]"

        self.assertEqual(f([1]), "default")
        self.assertEqual(f([1.0]), "default")
        self.assertEqual(f(""), "default")
        self.assertEqual(f(b""), "default")

    def test_method_equal_instances(self):
        # gh-127750: Reference to self was cached
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __eq__(self, other):
                    return True
                def __hash__(self):
                    return 1
                @functools.singledispatchmethod
                def t(self, arg):
                    return self

        a = A()
        b = A()
        self.assertIs(a.t(1), a)
        self.assertIs(b.t(2), b)

    def test_method_bad_hash(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __eq__(self, other):
                    raise AssertionError
                def __hash__(self):
                    raise AssertionError
                @functools.singledispatchmethod
                def t(self, arg):
                    pass

        # Should not raise
        A().t(1)
        hash(A().t)
        A().t == A().t

    def test_method_no_reference_loops(self):
        # gh-127750: Created a strong reference to self
        with torch._dynamo.error_on_graph_break(False):
            class A:
                @functools.singledispatchmethod
                def t(self, arg):
                    return weakref.ref(self)

        a = A()
        r = a.t(1)
        self.assertIsNotNone(r())
        del a  # delete a after a.t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

        a = A()
        t = a.t
        del a # delete a before a.t
        support.gc_collect()
        r = t(1)
        self.assertIsNotNone(r())
        del t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

    def test_signatures(self):
        @functools.singledispatch
        def func(item, arg: int) -> str:
            return str(item)
        @func.register
        def _(item: int, arg: bytes) -> str:
            return str(item)

        self.assertEqual(str(Signature.from_callable(func)),
                         '(item, arg: int) -> str')

    def test_method_signatures(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def m(self, item, arg: int) -> str:
                    return str(item)
                @classmethod
                def cm(cls, item, arg: int) -> str:
                    return str(item)
                @functools.singledispatchmethod
                def func(self, item, arg: int) -> str:
                    return str(item)
                @func.register
                def _(self, item, arg: bytes) -> str:
                    return str(item)

                @functools.singledispatchmethod
                @classmethod
                def cls_func(cls, item, arg: int) -> str:
                    return str(arg)
                @func.register
                @classmethod
                def _(cls, item, arg: bytes) -> str:
                    return str(item)

                @functools.singledispatchmethod
                @staticmethod
                def static_func(item, arg: int) -> str:
                    return str(arg)
                @func.register
                @staticmethod
                def _(item, arg: bytes) -> str:
                    return str(item)

        self.assertEqual(str(Signature.from_callable(A.func)),
                         '(self, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A().func)),
                         '(self, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A.cls_func)),
                         '(cls, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A.static_func)),
                         '(item, arg: int) -> str')


class CachedCostItem:
    _cost = 1

    def __init__(self):
        self.lock = py_functools.RLock()

    @py_functools.cached_property
    def cost(self):
        """The cost of the item."""
        with self.lock:
            self._cost += 1
        return self._cost


class OptionallyCachedCostItem:
    _cost = 1

    def get_cost(self):
        """The cost of the item."""
        self._cost += 1
        return self._cost

    cached_cost = py_functools.cached_property(get_cost)


class CachedCostItemWithSlots:
    __slots__ = ('_cost')

    def __init__(self):
        self._cost = 1

    @py_functools.cached_property
    def cost(self):
        raise RuntimeError('never called, slots not supported')


class TestCachedProperty(CPythonTestCase):
    def test_cached(self):
        item = CachedCostItem()
        self.assertEqual(item.cost, 2)
        self.assertEqual(item.cost, 2) # not 3

    def test_cached_attribute_name_differs_from_func_name(self):
        item = OptionallyCachedCostItem()
        self.assertEqual(item.get_cost(), 2)
        self.assertEqual(item.cached_cost, 3)
        self.assertEqual(item.get_cost(), 4)
        self.assertEqual(item.cached_cost, 3)

    def test_object_with_slots(self):
        item = CachedCostItemWithSlots()
        with self.assertRaisesRegex(
                TypeError,
                "No '__dict__' attribute on 'CachedCostItemWithSlots' instance to cache 'cost' property.",
        ):
            item.cost

    def test_immutable_dict(self):
        with torch._dynamo.error_on_graph_break(False):
            class MyMeta(type):
                @py_functools.cached_property
                def prop(self):
                    return True

            class MyClass(metaclass=MyMeta):
                pass

        with self.assertRaisesRegex(
            TypeError,
            "The '__dict__' attribute on 'MyMeta' instance does not support item assignment for caching 'prop' property.",
        ):
            MyClass.prop

    def test_reuse_different_names(self):
        """Disallow this case because decorated function a would not be cached."""
        with self.assertRaises(TypeError) as ctx:
            class ReusedCachedProperty:
                @py_functools.cached_property
                def a(self):
                    pass

                b = a

        self.assertEqual(
            str(ctx.exception),
            str(TypeError("Cannot assign the same cached_property to two different names ('a' and 'b')."))
        )

    def test_reuse_same_name(self):
        """Reusing a cached_property on different classes under the same name is OK."""
        counter = 0

        @py_functools.cached_property
        def _cp(_self):
            nonlocal counter
            counter += 1
            return counter

        with torch._dynamo.error_on_graph_break(False):
            class A:
                cp = _cp

            class B:
                cp = _cp

        a = A()
        b = B()

        self.assertEqual(a.cp, 1)
        self.assertEqual(b.cp, 2)
        self.assertEqual(a.cp, 1)

    def test_set_name_not_called(self):
        cp = py_functools.cached_property(lambda s: None)
        with torch._dynamo.error_on_graph_break(False):
            class Foo:
                pass

        Foo.cp = cp

        with self.assertRaisesRegex(
                TypeError,
                "Cannot use cached_property instance without calling __set_name__ on it.",
        ):
            Foo().cp

    def test_access_from_class(self):
        self.assertIsInstance(CachedCostItem.cost, py_functools.cached_property)

    def test_doc(self):
        self.assertEqual(CachedCostItem.cost.__doc__,
                         ("The cost of the item."
                          if support.HAVE_PY_DOCSTRINGS
                          else None))

    def test_module(self):
        self.assertEqual(CachedCostItem.cost.__module__, CachedCostItem.__module__)

    def test_subclass_with___set__(self):
        """Caching still works for a subclass defining __set__."""
        with torch._dynamo.error_on_graph_break(False):
            class readonly_cached_property(py_functools.cached_property):
                def __set__(self, obj, value):
                    raise AttributeError("read only property")

            class Test:
                def __init__(self, prop):
                    self._prop = prop

                @readonly_cached_property
                def prop(self):
                    return self._prop

        t = Test(1)
        self.assertEqual(t.prop, 1)
        t._prop = 999
        self.assertEqual(t.prop, 1)


if __name__ == '__main__':
    run_tests()
