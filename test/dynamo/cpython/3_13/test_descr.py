# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_descr.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import builtins
import copyreg
import gc
import itertools
import math
import pickle
import random
import string
import sys
import types
import unittest
import warnings
import weakref

from copy import deepcopy
from contextlib import redirect_stdout
from test import support
from test.support.testcase import ExtraAssertions

try:
    import _testcapi
except ImportError:
    _testcapi = None

try:
    import xxsubtype
except ImportError:
    xxsubtype = None


class OperatorsTest(CPythonTestCase):

    def __init__(self, *args, **kwargs):
        CPythonTestCase.__init__(self, *args, **kwargs)
        self.binops = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'matmul': '@',
            'truediv': '/',
            'floordiv': '//',
            'divmod': 'divmod',
            'pow': '**',
            'lshift': '<<',
            'rshift': '>>',
            'and': '&',
            'xor': '^',
            'or': '|',
            'cmp': 'cmp',
            'lt': '<',
            'le': '<=',
            'eq': '==',
            'ne': '!=',
            'gt': '>',
            'ge': '>=',
        }

        for name, expr in list(self.binops.items()):
            if expr.islower():
                expr = expr + "(a, b)"
            else:
                expr = 'a %s b' % expr
            self.binops[name] = expr

        self.unops = {
            'pos': '+',
            'neg': '-',
            'abs': 'abs',
            'invert': '~',
            'int': 'int',
            'float': 'float',
        }

        for name, expr in list(self.unops.items()):
            if expr.islower():
                expr = expr + "(a)"
            else:
                expr = '%s a' % expr
            self.unops[name] = expr

    def unop_test(self, a, res, expr="len(a)", meth="__len__"):
        d = {'a': a}
        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)

        # Find method in parent class
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(), res)

    def binop_test(self, a, b, res, expr="a+b", meth="__add__"):
        d = {'a': a, 'b': b}

        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a, b), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(b), res)

    def sliceop_test(self, a, b, c, res, expr="a[b:c]", meth="__getitem__"):
        d = {'a': a, 'b': b, 'c': c}
        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a, slice(b, c)), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(slice(b, c)), res)

    def setop_test(self, a, b, res, stmt="a+=b", meth="__iadd__"):
        d = {'a': deepcopy(a), 'b': b}
        exec(stmt, d)
        self.assertEqual(d['a'], res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        d['a'] = deepcopy(a)
        m(d['a'], b)
        self.assertEqual(d['a'], res)
        d['a'] = deepcopy(a)
        bm = getattr(d['a'], meth)
        bm(b)
        self.assertEqual(d['a'], res)

    def set2op_test(self, a, b, c, res, stmt="a[b]=c", meth="__setitem__"):
        d = {'a': deepcopy(a), 'b': b, 'c': c}
        exec(stmt, d)
        self.assertEqual(d['a'], res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        d['a'] = deepcopy(a)
        m(d['a'], b, c)
        self.assertEqual(d['a'], res)
        d['a'] = deepcopy(a)
        bm = getattr(d['a'], meth)
        bm(b, c)
        self.assertEqual(d['a'], res)

    def setsliceop_test(self, a, b, c, d, res, stmt="a[b:c]=d", meth="__setitem__"):
        dictionary = {'a': deepcopy(a), 'b': b, 'c': c, 'd': d}
        exec(stmt, dictionary)
        self.assertEqual(dictionary['a'], res)
        t = type(a)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        m = getattr(t, meth)
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        dictionary['a'] = deepcopy(a)
        m(dictionary['a'], slice(b, c), d)
        self.assertEqual(dictionary['a'], res)
        dictionary['a'] = deepcopy(a)
        bm = getattr(dictionary['a'], meth)
        bm(slice(b, c), d)
        self.assertEqual(dictionary['a'], res)

    def test_lists(self):
        # Testing list operations...
        # Asserts are within individual test methods
        self.binop_test([1], [2], [1,2], "a+b", "__add__")
        self.binop_test([1,2,3], 2, 1, "b in a", "__contains__")
        self.binop_test([1,2,3], 4, 0, "b in a", "__contains__")
        self.binop_test([1,2,3], 1, 2, "a[b]", "__getitem__")
        self.sliceop_test([1,2,3], 0, 2, [1,2], "a[b:c]", "__getitem__")
        self.setop_test([1], [2], [1,2], "a+=b", "__iadd__")
        self.setop_test([1,2], 3, [1,2,1,2,1,2], "a*=b", "__imul__")
        self.unop_test([1,2,3], 3, "len(a)", "__len__")
        self.binop_test([1,2], 3, [1,2,1,2,1,2], "a*b", "__mul__")
        self.binop_test([1,2], 3, [1,2,1,2,1,2], "b*a", "__rmul__")
        self.set2op_test([1,2], 1, 3, [1,3], "a[b]=c", "__setitem__")
        self.setsliceop_test([1,2,3,4], 1, 3, [5,6], [1,5,6,4], "a[b:c]=d",
                        "__setitem__")

    def test_dicts(self):
        # Testing dict operations...
        self.binop_test({1:2,3:4}, 1, 1, "b in a", "__contains__")
        self.binop_test({1:2,3:4}, 2, 0, "b in a", "__contains__")
        self.binop_test({1:2,3:4}, 1, 2, "a[b]", "__getitem__")

        d = {1:2, 3:4}
        l1 = []
        for i in list(d.keys()):
            l1.append(i)
        l = []
        for i in iter(d):
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in d.__iter__():
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in dict.__iter__(d):
            l.append(i)
        self.assertEqual(l, l1)
        d = {1:2, 3:4}
        self.unop_test(d, 2, "len(a)", "__len__")
        self.assertEqual(eval(repr(d), {}), d)
        self.assertEqual(eval(d.__repr__(), {}), d)
        self.set2op_test({1:2,3:4}, 2, 3, {1:2,2:3,3:4}, "a[b]=c",
                        "__setitem__")

    # Tests for unary and binary operators
    def number_operators(self, a, b, skip=[]):
        dict = {'a': a, 'b': b}

        for name, expr in self.binops.items():
            if name not in skip:
                name = "__%s__" % name
                if hasattr(a, name):
                    res = eval(expr, dict)
                    self.binop_test(a, b, res, expr, name)

        for name, expr in list(self.unops.items()):
            if name not in skip:
                name = "__%s__" % name
                if hasattr(a, name):
                    res = eval(expr, dict)
                    self.unop_test(a, res, expr, name)

    def test_ints(self):
        # Testing int operations...
        self.number_operators(100, 3)
        # The following crashes in Python 2.2
        self.assertEqual((1).__bool__(), 1)
        self.assertEqual((0).__bool__(), 0)
        # This returns 'NotImplemented' in Python 2.2
        with torch._dynamo.error_on_graph_break(False):
            class C(int):
                def __add__(self, other):
                    return NotImplemented
        self.assertEqual(C(5), 5)
        try:
            C() + ""
        except TypeError:
            pass
        else:
            self.fail("NotImplemented should have caused TypeError")

    def test_floats(self):
        # Testing float operations...
        self.number_operators(100.0, 3.0)

    def test_complexes(self):
        # Testing complex operations...
        self.number_operators(100.0j, 3.0j, skip=['lt', 'le', 'gt', 'ge',
                                                  'int', 'float',
                                                  'floordiv', 'divmod', 'mod'])

        with torch._dynamo.error_on_graph_break(False):
            class Number(complex):
                __slots__ = ['prec']
                def __new__(cls, *args, **kwds):
                    result = complex.__new__(cls, *args)
                    result.prec = kwds.get('prec', 12)
                    return result
                def __repr__(self):
                    prec = self.prec
                    if self.imag == 0.0:
                        return "%.*g" % (prec, self.real)
                    if self.real == 0.0:
                        return "%.*gj" % (prec, self.imag)
                    return "(%.*g+%.*gj)" % (prec, self.real, prec, self.imag)
                __str__ = __repr__

        a = Number(3.14, prec=6)
        self.assertEqual(repr(a), "3.14")
        self.assertEqual(a.prec, 6)

        a = Number(a, prec=2)
        self.assertEqual(repr(a), "3.1")
        self.assertEqual(a.prec, 2)

        a = Number(234.5)
        self.assertEqual(repr(a), "234.5")
        self.assertEqual(a.prec, 12)

    def test_explicit_reverse_methods(self):
        # see issue 9930
        self.assertEqual(complex.__radd__(3j, 4.0), complex(4.0, 3.0))
        self.assertEqual(float.__rsub__(3.0, 1), -2.0)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_spam_lists(self):
        # Testing spamlist operations...
        import copy, xxsubtype as spam

        def spamlist(l, memo=None):
            import xxsubtype as spam
            return spam.spamlist(l)

        # This is an ugly hack:
        copy._deepcopy_dispatch[spam.spamlist] = spamlist

        self.binop_test(spamlist([1]), spamlist([2]), spamlist([1,2]), "a+b",
                       "__add__")
        self.binop_test(spamlist([1,2,3]), 2, 1, "b in a", "__contains__")
        self.binop_test(spamlist([1,2,3]), 4, 0, "b in a", "__contains__")
        self.binop_test(spamlist([1,2,3]), 1, 2, "a[b]", "__getitem__")
        self.sliceop_test(spamlist([1,2,3]), 0, 2, spamlist([1,2]), "a[b:c]",
                          "__getitem__")
        self.setop_test(spamlist([1]), spamlist([2]), spamlist([1,2]), "a+=b",
                        "__iadd__")
        self.setop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "a*=b",
                        "__imul__")
        self.unop_test(spamlist([1,2,3]), 3, "len(a)", "__len__")
        self.binop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "a*b",
                        "__mul__")
        self.binop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "b*a",
                        "__rmul__")
        self.set2op_test(spamlist([1,2]), 1, 3, spamlist([1,3]), "a[b]=c",
                         "__setitem__")
        self.setsliceop_test(spamlist([1,2,3,4]), 1, 3, spamlist([5,6]),
                             spamlist([1,5,6,4]), "a[b:c]=d", "__setitem__")
        with torch._dynamo.error_on_graph_break(False):
            # Test subclassing
            class C(spam.spamlist):
                def foo(self): return 1
        a = C()
        self.assertEqual(a, [])
        self.assertEqual(a.foo(), 1)
        a.append(100)
        self.assertEqual(a, [100])
        self.assertEqual(a.getstate(), 0)
        a.setstate(42)
        self.assertEqual(a.getstate(), 42)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_spam_dicts(self):
        # Testing spamdict operations...
        import copy, xxsubtype as spam
        def spamdict(d, memo=None):
            import xxsubtype as spam
            sd = spam.spamdict()
            for k, v in list(d.items()):
                sd[k] = v
            return sd
        # This is an ugly hack:
        copy._deepcopy_dispatch[spam.spamdict] = spamdict

        self.binop_test(spamdict({1:2,3:4}), 1, 1, "b in a", "__contains__")
        self.binop_test(spamdict({1:2,3:4}), 2, 0, "b in a", "__contains__")
        self.binop_test(spamdict({1:2,3:4}), 1, 2, "a[b]", "__getitem__")
        d = spamdict({1:2,3:4})
        l1 = []
        for i in list(d.keys()):
            l1.append(i)
        l = []
        for i in iter(d):
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in d.__iter__():
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in type(spamdict({})).__iter__(d):
            l.append(i)
        self.assertEqual(l, l1)
        straightd = {1:2, 3:4}
        spamd = spamdict(straightd)
        self.unop_test(spamd, 2, "len(a)", "__len__")
        self.unop_test(spamd, repr(straightd), "repr(a)", "__repr__")
        self.set2op_test(spamdict({1:2,3:4}), 2, 3, spamdict({1:2,2:3,3:4}),
                   "a[b]=c", "__setitem__")
        with torch._dynamo.error_on_graph_break(False):
            # Test subclassing
            class C(spam.spamdict):
                def foo(self): return 1
        a = C()
        self.assertEqual(list(a.items()), [])
        self.assertEqual(a.foo(), 1)
        a['foo'] = 'bar'
        self.assertEqual(list(a.items()), [('foo', 'bar')])
        self.assertEqual(a.getstate(), 0)
        a.setstate(100)
        self.assertEqual(a.getstate(), 100)

    def test_wrap_lenfunc_bad_cast(self):
        self.assertEqual(range(sys.maxsize).__len__(), sys.maxsize)


class ClassPropertiesAndMethods(CPythonTestCase, ExtraAssertions):

    def test_python_dicts(self):
        # Testing Python subclass of dict...
        self.assertTrue(issubclass(dict, dict))
        self.assertIsInstance({}, dict)
        d = dict()
        self.assertEqual(d, {})
        self.assertIs(d.__class__, dict)
        self.assertIsInstance(d, dict)
        with torch._dynamo.error_on_graph_break(False):
            class C(dict):
                state = -1
                def __init__(self_local, *a, **kw):
                    if a:
                        self.assertEqual(len(a), 1)
                        self_local.state = a[0]
                    if kw:
                        for k, v in list(kw.items()):
                            self_local[v] = k
                def __getitem__(self, key):
                    return self.get(key, 0)
                def __setitem__(self_local, key, value):
                    self.assertIsInstance(key, int)
                    dict.__setitem__(self_local, key, value)
                def setstate(self, state):
                    self.state = state
                def getstate(self):
                    return self.state
        self.assertTrue(issubclass(C, dict))
        a1 = C(12)
        self.assertEqual(a1.state, 12)
        a2 = C(foo=1, bar=2)
        self.assertEqual(a2[1] == 'foo' and a2[2], 'bar')
        a = C()
        self.assertEqual(a.state, -1)
        self.assertEqual(a.getstate(), -1)
        a.setstate(0)
        self.assertEqual(a.state, 0)
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.state, 10)
        self.assertEqual(a.getstate(), 10)
        self.assertEqual(a[42], 0)
        a[42] = 24
        self.assertEqual(a[42], 24)
        N = 50
        for i in range(N):
            a[i] = C()
            for j in range(N):
                a[i][j] = i*j
        for i in range(N):
            for j in range(N):
                self.assertEqual(a[i][j], i*j)

    def test_python_lists(self):
        with torch._dynamo.error_on_graph_break(False):
            # Testing Python subclass of list...
            class C(list):
                def __getitem__(self, i):
                    if isinstance(i, slice):
                        return i.start, i.stop
                    return list.__getitem__(self, i) + 100
        a = C()
        a.extend([0,1,2])
        self.assertEqual(a[0], 100)
        self.assertEqual(a[1], 101)
        self.assertEqual(a[2], 102)
        self.assertEqual(a[100:200], (100,200))

    def test_metaclass(self):
        with torch._dynamo.error_on_graph_break(False):
            # Testing metaclasses...
            class C(metaclass=type):
                def __init__(self):
                    self.__state = 0
                def getstate(self):
                    return self.__state
                def setstate(self, state):
                    self.__state = state
        a = C()
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.getstate(), 10)
        with torch._dynamo.error_on_graph_break(False):
            class _metaclass(type):
                def myself(cls): return cls
            class D(metaclass=_metaclass):
                pass
        self.assertEqual(D.myself(), D)
        d = D()
        self.assertEqual(d.__class__, D)
        with torch._dynamo.error_on_graph_break(False):
            class M1(type):
                def __new__(cls, name, bases, dict):
                    dict['__spam__'] = 1
                    return type.__new__(cls, name, bases, dict)
            class C(metaclass=M1):
                pass
        self.assertEqual(C.__spam__, 1)
        c = C()
        self.assertEqual(c.__spam__, 1)

        with torch._dynamo.error_on_graph_break(False):
            class _instance(object):
                pass
            class M2(object):
                @staticmethod
                def __new__(cls, name, bases, dict):
                    self = object.__new__(cls)
                    self.name = name
                    self.bases = bases
                    self.dict = dict
                    return self
                def __call__(self):
                    it = _instance()
                    # Early binding of methods
                    for key in self.dict:
                        if key.startswith("__"):
                            continue
                        setattr(it, key, self.dict[key].__get__(it, self))
                    return it
            class C(metaclass=M2):
                def spam(self):
                    return 42
        self.assertEqual(C.name, 'C')
        self.assertEqual(C.bases, ())
        self.assertIn('spam', C.dict)
        c = C()
        self.assertEqual(c.spam(), 42)

        # More metaclass examples

        with torch._dynamo.error_on_graph_break(False):
            class autosuper(type):
                # Automatically add __super to the class
                # This trick only works for dynamic classes
                def __new__(metaclass, name, bases, dict):
                    cls = super(autosuper, metaclass).__new__(metaclass,
                                                            name, bases, dict)
                    # Name mangling for __super removes leading underscores
                    while name[:1] == "_":
                        name = name[1:]
                    if name:
                        name = "_%s__super" % name
                    else:
                        name = "__super"
                    setattr(cls, name, super(cls))
                    return cls
            class A(metaclass=autosuper):
                def meth(self):
                    return "A"
            class B(A):
                def meth(self):
                    return "B" + self.__super.meth()
            class C(A):
                def meth(self):
                    return "C" + self.__super.meth()
            class D(C, B):
                def meth(self):
                    return "D" + self.__super.meth()
        self.assertEqual(D().meth(), "DCBA")
        with torch._dynamo.error_on_graph_break(False):
            class E(B, C):
                def meth(self):
                    return "E" + self.__super.meth()
        self.assertEqual(E().meth(), "EBCA")

        with torch._dynamo.error_on_graph_break(False):
            class autoproperty(type):
                # Automatically create property attributes when methods
                # named _get_x and/or _set_x are found
                def __new__(metaclass, name, bases, dict):
                    hits = {}
                    for key, val in dict.items():
                        if key.startswith("_get_"):
                            key = key[5:]
                            get, set = hits.get(key, (None, None))
                            get = val
                            hits[key] = get, set
                        elif key.startswith("_set_"):
                            key = key[5:]
                            get, set = hits.get(key, (None, None))
                            set = val
                            hits[key] = get, set
                    for key, (get, set) in hits.items():
                        dict[key] = property(get, set)
                    return super(autoproperty, metaclass).__new__(metaclass,
                                                                name, bases, dict)
        with torch._dynamo.error_on_graph_break(False):
            class A(metaclass=autoproperty):
                def _get_x(self):
                    return -self.__x
                def _set_x(self, x):
                    self.__x = -x
        a = A()
        self.assertNotHasAttr(a, "x")
        a.x = 12
        self.assertEqual(a.x, 12)
        self.assertEqual(a._A__x, -12)

        with torch._dynamo.error_on_graph_break(False):
            class multimetaclass(autoproperty, autosuper):
                # Merge of multiple cooperating metaclasses
                pass
            class A(metaclass=multimetaclass):
                def _get_x(self):
                    return "A"
            class B(A):
                def _get_x(self):
                    return "B" + self.__super._get_x()
            class C(A):
                def _get_x(self):
                    return "C" + self.__super._get_x()
            class D(C, B):
                def _get_x(self):
                    return "D" + self.__super._get_x()
        self.assertEqual(D().x, "DCBA")

        with torch._dynamo.error_on_graph_break(False):
            # Make sure type(x) doesn't call x.__class__.__init__
            class T(type):
                counter = 0
                def __init__(self, *args):
                    T.counter += 1
            class C(metaclass=T):
                pass
        self.assertEqual(T.counter, 1)
        a = C()
        self.assertEqual(type(a), C)
        self.assertEqual(T.counter, 1)

        with torch._dynamo.error_on_graph_break(False):
            class C(object): pass
        c = C()
        try: c()
        except TypeError: pass
        else: self.fail("calling object w/o call method should raise "
                        "TypeError")

        with torch._dynamo.error_on_graph_break(False):
            # Testing code to find most derived baseclass
            class A(type):
                def __new__(*args, **kwargs):
                    return type.__new__(*args, **kwargs)

            class B(object):
                pass

            class C(object, metaclass=A):
                pass

            # The most derived metaclass of D is A rather than type.
            class D(B, C):
                pass
        self.assertIs(A, type(D))

        # issue1294232: correct metaclass calculation
        new_calls = []  # to check the order of __new__ calls
        with torch._dynamo.error_on_graph_break(False):
            class AMeta(type):
                @staticmethod
                def __new__(mcls, name, bases, ns):
                    new_calls.append('AMeta')
                    return super().__new__(mcls, name, bases, ns)
                @classmethod
                def __prepare__(mcls, name, bases):
                    return {}

            class BMeta(AMeta):
                @staticmethod
                def __new__(mcls, name, bases, ns):
                    new_calls.append('BMeta')
                    return super().__new__(mcls, name, bases, ns)
                @classmethod
                def __prepare__(mcls, name, bases):
                    ns = super().__prepare__(name, bases)
                    ns['BMeta_was_here'] = True
                    return ns

            class A(metaclass=AMeta):
                pass
        self.assertEqual(['AMeta'], new_calls)
        new_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class B(metaclass=BMeta):
                pass
        # BMeta.__new__ calls AMeta.__new__ with super:
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class C(A, B):
                pass
        # The most derived metaclass is BMeta:
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        # BMeta.__prepare__ should've been called:
        self.assertIn('BMeta_was_here', C.__dict__)

        with torch._dynamo.error_on_graph_break(False):
            # The order of the bases shouldn't matter:
            class C2(B, A):
                pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', C2.__dict__)

        with torch._dynamo.error_on_graph_break(False):
            # Check correct metaclass calculation when a metaclass is declared:
            class D(C, metaclass=type):
                pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', D.__dict__)

        with torch._dynamo.error_on_graph_break(False):
            class E(C, metaclass=AMeta):
                pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', E.__dict__)

        # Special case: the given metaclass isn't a class,
        # so there is no metaclass calculation.
        marker = object()
        def func(*args, **kwargs):
            return marker
        with torch._dynamo.error_on_graph_break(False):
            class X(metaclass=func):
                pass
            class Y(object, metaclass=func):
                pass
            class Z(D, metaclass=func):
                pass
        self.assertIs(marker, X)
        self.assertIs(marker, Y)
        self.assertIs(marker, Z)

        # The given metaclass is a class,
        # but not a descendant of type.
        prepare_calls = []  # to track __prepare__ calls
        with torch._dynamo.error_on_graph_break(False):
            class ANotMeta:
                def __new__(mcls, *args, **kwargs):
                    new_calls.append('ANotMeta')
                    return super().__new__(mcls)
                @classmethod
                def __prepare__(mcls, name, bases):
                    prepare_calls.append('ANotMeta')
                    return {}
            class BNotMeta(ANotMeta):
                def __new__(mcls, *args, **kwargs):
                    new_calls.append('BNotMeta')
                    return super().__new__(mcls)
                @classmethod
                def __prepare__(mcls, name, bases):
                    prepare_calls.append('BNotMeta')
                    return super().__prepare__(name, bases)

            class A(metaclass=ANotMeta):
                pass
        self.assertIs(ANotMeta, type(A))
        self.assertEqual(['ANotMeta'], prepare_calls)
        prepare_calls.clear()
        self.assertEqual(['ANotMeta'], new_calls)
        new_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class B(metaclass=BNotMeta):
                pass
        self.assertIs(BNotMeta, type(B))
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class C(A, B):
                pass
        self.assertIs(BNotMeta, type(C))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class C2(B, A):
                pass
        self.assertIs(BNotMeta, type(C2))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        # This is a TypeError, because of a metaclass conflict:
        # BNotMeta is neither a subclass, nor a superclass of type
        with torch._dynamo.error_on_graph_break(False):
            with self.assertRaises(TypeError):
                class D(C, metaclass=type):
                    pass

            class E(C, metaclass=ANotMeta):
                pass
        self.assertIs(BNotMeta, type(E))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class F(object(), C):
                pass
        self.assertIs(BNotMeta, type(F))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        with torch._dynamo.error_on_graph_break(False):
            class F2(C, object()):
                pass
        self.assertIs(BNotMeta, type(F2))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        # TypeError: BNotMeta is neither a
        # subclass, nor a superclass of int
        with torch._dynamo.error_on_graph_break(False):
            with self.assertRaises(TypeError):
                class X(C, int()):
                    pass
            with self.assertRaises(TypeError):
                class X(int(), C):
                    pass

    def test_module_subclasses(self):
        # Testing Python subclass of module...
        log = []
        MT = type(sys)
        with torch._dynamo.error_on_graph_break(False):
            class MM(MT):
                def __init__(self, name):
                    MT.__init__(self, name)
                def __getattribute__(self, name):
                    log.append(("getattr", name))
                    return MT.__getattribute__(self, name)
                def __setattr__(self, name, value):
                    log.append(("setattr", name, value))
                    MT.__setattr__(self, name, value)
                def __delattr__(self, name):
                    log.append(("delattr", name))
                    MT.__delattr__(self, name)
        a = MM("a")
        a.foo = 12
        x = a.foo
        del a.foo
        self.assertEqual(log, [("setattr", "foo", 12),
                               ("getattr", "foo"),
                               ("delattr", "foo")])

        # https://bugs.python.org/issue1174712
        try:
            with torch._dynamo.error_on_graph_break(False):
                class Module(types.ModuleType, str):
                    pass
        except TypeError:
            pass
        else:
            self.fail("inheriting from ModuleType and str at the same time "
                    "should fail")

        # Issue 34805: Verify that definition order is retained
        def random_name():
            return ''.join(random.choices(string.ascii_letters, k=10))
        with torch._dynamo.error_on_graph_break(False):
            class A:
                pass
        subclasses = [type(random_name(), (A,), {}) for i in range(100)]
        self.assertEqual(A.__subclasses__(), subclasses)

    def test_multiple_inheritance(self):
        # Testing multiple inheritance...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __init__(self):
                    self.__state = 0
                def getstate(self):
                    return self.__state
                def setstate(self, state):
                    self.__state = state
        a = C()
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.getstate(), 10)
        with torch._dynamo.error_on_graph_break(False):
            class D(dict, C):
                def __init__(self):
                    dict.__init__(self)
                    C.__init__(self)
        d = D()
        self.assertEqual(list(d.keys()), [])
        d["hello"] = "world"
        self.assertEqual(list(d.items()), [("hello", "world")])
        self.assertEqual(d["hello"], "world")
        self.assertEqual(d.getstate(), 0)
        d.setstate(10)
        self.assertEqual(d.getstate(), 10)
        self.assertEqual(D.__mro__, (D, dict, C, object))

        # SF bug #442833
        with torch._dynamo.error_on_graph_break(False):
            class Node(object):
                def __int__(self):
                    return int(self.foo())
                def foo(self):
                    return "23"
            class Frag(Node, list):
                def foo(self):
                    return "42"
        self.assertEqual(Node().__int__(), 23)
        self.assertEqual(int(Node()), 23)
        self.assertEqual(Frag().__int__(), 42)
        self.assertEqual(int(Frag()), 42)

    def test_diamond_inheritance(self):
        # Testing multiple inheritance special cases...
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def spam(self): return "A"
        self.assertEqual(A().spam(), "A")
        with torch._dynamo.error_on_graph_break(False):
            class B(A):
                def boo(self): return "B"
                def spam(self): return "B"
        self.assertEqual(B().spam(), "B")
        self.assertEqual(B().boo(), "B")
        with torch._dynamo.error_on_graph_break(False):
            class C(A):
                def boo(self): return "C"
        self.assertEqual(C().spam(), "A")
        self.assertEqual(C().boo(), "C")
        with torch._dynamo.error_on_graph_break(False):
            class D(B, C): pass
        self.assertEqual(D().spam(), "B")
        self.assertEqual(D().boo(), "B")
        self.assertEqual(D.__mro__, (D, B, C, A, object))
        with torch._dynamo.error_on_graph_break(False):
            class E(C, B): pass
        self.assertEqual(E().spam(), "B")
        self.assertEqual(E().boo(), "C")
        self.assertEqual(E.__mro__, (E, C, B, A, object))
        # MRO order disagreement
        try:
            with torch._dynamo.error_on_graph_break(False):
                class F(D, E): pass
        except TypeError:
            pass
        else:
            self.fail("expected MRO order disagreement (F)")
        try:
            with torch._dynamo.error_on_graph_break(False):
                class G(E, D): pass
        except TypeError:
            pass
        else:
            self.fail("expected MRO order disagreement (G)")

    # see thread python-dev/2002-October/029035.html
    def test_ex5_from_c3_switch(self):
        # Testing ex5 from C3 switch discussion...
        with torch._dynamo.error_on_graph_break(False):
            class A(object): pass
            class B(object): pass
            class C(object): pass
            class X(A): pass
            class Y(A): pass
            class Z(X,B,Y,C): pass
        self.assertEqual(Z.__mro__, (Z, X, B, Y, A, C, object))

    # see "A Monotonic Superclass Linearization for Dylan",
    # by Kim Barrett et al. (OOPSLA 1996)
    def test_monotonicity(self):
        # Testing MRO monotonicity...
        with torch._dynamo.error_on_graph_break(False):
            class Boat(object): pass
            class DayBoat(Boat): pass
            class WheelBoat(Boat): pass
            class EngineLess(DayBoat): pass
            class SmallMultihull(DayBoat): pass
            class PedalWheelBoat(EngineLess,WheelBoat): pass
            class SmallCatamaran(SmallMultihull): pass
            class Pedalo(PedalWheelBoat,SmallCatamaran): pass

        self.assertEqual(PedalWheelBoat.__mro__,
              (PedalWheelBoat, EngineLess, DayBoat, WheelBoat, Boat, object))
        self.assertEqual(SmallCatamaran.__mro__,
              (SmallCatamaran, SmallMultihull, DayBoat, Boat, object))
        self.assertEqual(Pedalo.__mro__,
              (Pedalo, PedalWheelBoat, EngineLess, SmallCatamaran,
               SmallMultihull, DayBoat, WheelBoat, Boat, object))

    # see "A Monotonic Superclass Linearization for Dylan",
    # by Kim Barrett et al. (OOPSLA 1996)
    def test_consistency_with_epg(self):
        # Testing consistency with EPG...
        with torch._dynamo.error_on_graph_break(False):
            class Pane(object): pass
            class ScrollingMixin(object): pass
            class EditingMixin(object): pass
            class ScrollablePane(Pane,ScrollingMixin): pass
            class EditablePane(Pane,EditingMixin): pass
            class EditableScrollablePane(ScrollablePane,EditablePane): pass

        self.assertEqual(EditableScrollablePane.__mro__,
              (EditableScrollablePane, ScrollablePane, EditablePane, Pane,
                ScrollingMixin, EditingMixin, object))

    def test_mro_disagreement(self):
        # Testing error messages for MRO disagreement...
        mro_err_msg = ("Cannot create a consistent method resolution "
                       "order (MRO) for bases ")

        def raises(exc, expected, callable, *args):
            try:
                callable(*args)
            except exc as msg:
                # the exact msg is generally considered an impl detail
                if support.check_impl_detail():
                    if not str(msg).startswith(expected):
                        self.fail("Message %r, expected %r" %
                                  (str(msg), expected))
            else:
                self.fail("Expected %s" % exc)

        with torch._dynamo.error_on_graph_break(False):
            class A(object): pass
            class B(A): pass
            class C(object): pass

        # Test some very simple errors
        raises(TypeError, "duplicate base class A",
               type, "X", (A, A), {})
        raises(TypeError, mro_err_msg,
               type, "X", (A, B), {})
        raises(TypeError, mro_err_msg,
               type, "X", (A, C, B), {})
        # Test a slightly more complex error
        with torch._dynamo.error_on_graph_break(False):
            class GridLayout(object): pass
            class HorizontalGrid(GridLayout): pass
            class VerticalGrid(GridLayout): pass
            class HVGrid(HorizontalGrid, VerticalGrid): pass
            class VHGrid(VerticalGrid, HorizontalGrid): pass
        raises(TypeError, mro_err_msg,
               type, "ConfusedGrid", (HVGrid, VHGrid), {})

    def test_object_class(self):
        # Testing object class...
        a = object()
        self.assertEqual(a.__class__, object)
        self.assertEqual(type(a), object)
        b = object()
        self.assertNotEqual(a, b)
        self.assertNotHasAttr(a, "foo")
        try:
            a.foo = 12
        except (AttributeError, TypeError):
            pass
        else:
            self.fail("object() should not allow setting a foo attribute")
        self.assertNotHasAttr(object(), "__dict__")

        with torch._dynamo.error_on_graph_break(False):
            class Cdict(object):
                pass
        x = Cdict()
        self.assertEqual(x.__dict__, {})
        x.foo = 1
        self.assertEqual(x.foo, 1)
        self.assertEqual(x.__dict__, {'foo': 1})

    def test_object_class_assignment_between_heaptypes_and_nonheaptypes(self):
        with torch._dynamo.error_on_graph_break(False):
            class SubType(types.ModuleType):
                a = 1

        m = types.ModuleType("m")
        self.assertTrue(m.__class__ is types.ModuleType)
        self.assertFalse(hasattr(m, "a"))

        m.__class__ = SubType
        self.assertTrue(m.__class__ is SubType)
        self.assertTrue(hasattr(m, "a"))

        m.__class__ = types.ModuleType
        self.assertTrue(m.__class__ is types.ModuleType)
        self.assertFalse(hasattr(m, "a"))

        # Make sure that builtin immutable objects don't support __class__
        # assignment, because the object instances may be interned.
        # We set __slots__ = () to ensure that the subclasses are
        # memory-layout compatible, and thus otherwise reasonable candidates
        # for __class__ assignment.

        # The following types have immutable instances, but are not
        # subclassable and thus don't need to be checked:
        #   NoneType, bool

        with torch._dynamo.error_on_graph_break(False):
            class MyInt(int):
                __slots__ = ()
        with self.assertRaises(TypeError):
            (1).__class__ = MyInt

        with torch._dynamo.error_on_graph_break(False):
            class MyFloat(float):
                __slots__ = ()
        with self.assertRaises(TypeError):
            (1.0).__class__ = MyFloat

        with torch._dynamo.error_on_graph_break(False):
            class MyComplex(complex):
                __slots__ = ()
        with self.assertRaises(TypeError):
            (1 + 2j).__class__ = MyComplex

        with torch._dynamo.error_on_graph_break(False):
            class MyStr(str):
                __slots__ = ()
        with self.assertRaises(TypeError):
            "a".__class__ = MyStr

        with torch._dynamo.error_on_graph_break(False):
            class MyBytes(bytes):
                __slots__ = ()
        with self.assertRaises(TypeError):
            b"a".__class__ = MyBytes

        with torch._dynamo.error_on_graph_break(False):
            class MyTuple(tuple):
                __slots__ = ()
        with self.assertRaises(TypeError):
            ().__class__ = MyTuple

        with torch._dynamo.error_on_graph_break(False):
            class MyFrozenSet(frozenset):
                __slots__ = ()
        with self.assertRaises(TypeError):
            frozenset().__class__ = MyFrozenSet

    @unittest.expectedFailure
    def test_slots(self):
        # Testing __slots__...
        with torch._dynamo.error_on_graph_break(False):
            class C0(object):
                __slots__ = []
        x = C0()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, "foo")

        with torch._dynamo.error_on_graph_break(False):
            class C1(object):
                __slots__ = ['a']
        x = C1()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, "a")
        x.a = 1
        self.assertEqual(x.a, 1)
        x.a = None
        self.assertEqual(x.a, None)
        del x.a
        self.assertNotHasAttr(x, "a")

        with torch._dynamo.error_on_graph_break(False):
            class C3(object):
                __slots__ = ['a', 'b', 'c']
            x = C3()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, 'a')
        self.assertNotHasAttr(x, 'b')
        self.assertNotHasAttr(x, 'c')
        x.a = 1
        x.b = 2
        x.c = 3
        self.assertEqual(x.a, 1)
        self.assertEqual(x.b, 2)
        self.assertEqual(x.c, 3)

        with torch._dynamo.error_on_graph_break(False):
            class C4(object):
                """Validate name mangling"""
                __slots__ = ['__a']
                def __init__(self, value):
                    self.__a = value
                def get(self):
                    return self.__a
            x = C4(5)
        self.assertNotHasAttr(x, '__dict__')
        self.assertNotHasAttr(x, '__a')
        self.assertEqual(x.get(), 5)
        try:
            x.__a = 6
        except AttributeError:
            pass
        else:
            self.fail("Double underscored names not mangled")

        # Make sure slot names are proper identifiers
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = [None]
        except TypeError:
            pass
        else:
            self.fail("[None] slots not caught")
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = ["foo bar"]
        except TypeError:
            pass
        else:
            self.fail("['foo bar'] slots not caught")
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = ["foo\0bar"]
        except TypeError:
            pass
        else:
            self.fail("['foo\\0bar'] slots not caught")
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = ["1"]
        except TypeError:
            pass
        else:
            self.fail("['1'] slots not caught")
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = [""]
        except TypeError:
            pass
        else:
            self.fail("[''] slots not caught")

        with torch._dynamo.error_on_graph_break(False):
            class WithValidIdentifiers(object):
                __slots__ = ["a", "a_b", "_a", "A0123456789Z"]

        # Test a single string is not expanded as a sequence.
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                __slots__ = "abc"
        c = C()
        c.abc = 5
        self.assertEqual(c.abc, 5)

        # Test unicode slot names
        # Test a single unicode string is not expanded as a sequence.
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                __slots__ = "abc"
        c = C()
        c.abc = 5
        self.assertEqual(c.abc, 5)

        # _unicode_to_string used to modify slots in certain circumstances
        slots = ("foo", "bar")
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                __slots__ = slots
        x = C()
        x.foo = 5
        self.assertEqual(x.foo, 5)
        self.assertIs(type(slots[0]), str)
        # this used to leak references
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = [chr(128)]
        except (TypeError, UnicodeEncodeError):
            pass
        else:
            self.fail("[chr(128)] slots not caught")

        # Test leaks
        with torch._dynamo.error_on_graph_break(False):
            class Counted(object):
                counter = 0    # counts the number of instances alive
                def __init__(self):
                    Counted.counter += 1
                def __del__(self):
                    Counted.counter -= 1
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                __slots__ = ['a', 'b', 'c']
        x = C()
        x.a = Counted()
        x.b = Counted()
        x.c = Counted()
        self.assertEqual(Counted.counter, 3)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        x = D()
        x.a = Counted()
        x.z = Counted()
        self.assertEqual(Counted.counter, 2)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)
        with torch._dynamo.error_on_graph_break(False):
            class E(D):
                __slots__ = ['e']
        x = E()
        x.a = Counted()
        x.z = Counted()
        x.e = Counted()
        self.assertEqual(Counted.counter, 3)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)

        # Test cyclical leaks [SF bug 519621]
        with torch._dynamo.error_on_graph_break(False):
            class F(object):
                __slots__ = ['a', 'b']
        s = F()
        s.a = [Counted(), s]
        self.assertEqual(Counted.counter, 1)
        s = None
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)

        # Test lookup leaks [SF bug 572567]
        if hasattr(gc, 'get_objects'):
            with torch._dynamo.error_on_graph_break(False):
                class G(object):
                    def __eq__(self, other):
                        return False
            g = G()
            orig_objects = len(gc.get_objects())
            for i in range(10):
                g==g
            new_objects = len(gc.get_objects())
            self.assertEqual(orig_objects, new_objects)

        with torch._dynamo.error_on_graph_break(False):
            class H(object):
                __slots__ = ['a', 'b']
                def __init__(self):
                    self.a = 1
                    self.b = 2
                def __del__(self_):
                    self.assertEqual(self_.a, 1)
                    self.assertEqual(self_.b, 2)
        with support.captured_output('stderr') as s:
            h = H()
            del h
        self.assertEqual(s.getvalue(), '')

        with torch._dynamo.error_on_graph_break(False):
            class X(object):
                __slots__ = "a"
        with self.assertRaises(AttributeError):
            del X().a

        # Inherit from object on purpose to check some backwards compatibility paths
        with torch._dynamo.error_on_graph_break(False):
            class X(object):
                __slots__ = "a"
        with self.assertRaisesRegex(AttributeError, "'test.test_descr.ClassPropertiesAndMethods.test_slots.<locals>.X' object has no attribute 'a'"):
            X().a

        # Test string subclass in `__slots__`, see gh-98783
        with torch._dynamo.error_on_graph_break(False):
            class SubStr(str):
                pass
        with torch._dynamo.error_on_graph_break(False):
            class X(object):
                __slots__ = (SubStr('x'),)
        X().x = 1
        with self.assertRaisesRegex(AttributeError, "'X' object has no attribute 'a'"):
            X().a

    def test_slots_special(self):
        # Testing __dict__ and __weakref__ in __slots__...
        with torch._dynamo.error_on_graph_break(False):
            class D(object):
                __slots__ = ["__dict__"]
        a = D()
        self.assertHasAttr(a, "__dict__")
        self.assertNotHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

        with torch._dynamo.error_on_graph_break(False):
            class W(object):
                __slots__ = ["__weakref__"]
        a = W()
        self.assertHasAttr(a, "__weakref__")
        self.assertNotHasAttr(a, "__dict__")
        try:
            a.foo = 42
        except AttributeError:
            pass
        else:
            self.fail("shouldn't be allowed to set a.foo")

        with torch._dynamo.error_on_graph_break(False):
            class C1(W, D):
                __slots__ = []
        a = C1()
        self.assertHasAttr(a, "__dict__")
        self.assertHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

        with torch._dynamo.error_on_graph_break(False):
            class C2(D, W):
                __slots__ = []
        a = C2()
        self.assertHasAttr(a, "__dict__")
        self.assertHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

    def test_slots_special2(self):
        # Testing __qualname__ and __classcell__ in __slots__
        with torch._dynamo.error_on_graph_break(False):
            class Meta(type):
                def __new__(cls, name, bases, namespace, attr):
                    self.assertIn(attr, namespace)
                    return super().__new__(cls, name, bases, namespace)

        with torch._dynamo.error_on_graph_break(False):
            class C1:
                def __init__(self):
                    self.b = 42
        with torch._dynamo.error_on_graph_break(False):
            class C2(C1, metaclass=Meta, attr="__classcell__"):
                __slots__ = ["__classcell__"]
                def __init__(self):
                    super().__init__()
        self.assertIsInstance(C2.__dict__["__classcell__"],
                              types.MemberDescriptorType)
        c = C2()
        self.assertEqual(c.b, 42)
        self.assertNotHasAttr(c, "__classcell__")
        c.__classcell__ = 42
        self.assertEqual(c.__classcell__, 42)
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class C3:
                    __classcell__ = 42
                    __slots__ = ["__classcell__"]

        with torch._dynamo.error_on_graph_break(False):
            class Q1(metaclass=Meta, attr="__qualname__"):
                __slots__ = ["__qualname__"]
        self.assertEqual(Q1.__qualname__, C1.__qualname__[:-2] + "Q1")
        self.assertIsInstance(Q1.__dict__["__qualname__"],
                              types.MemberDescriptorType)
        q = Q1()
        self.assertNotHasAttr(q, "__qualname__")
        q.__qualname__ = "q"
        self.assertEqual(q.__qualname__, "q")
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class Q2:
                    __qualname__ = object()
                    __slots__ = ["__qualname__"]

    def test_slots_descriptor(self):
        # Issue2115: slot descriptors did not correctly check
        # the type of the given object
        import abc
        with torch._dynamo.error_on_graph_break(False):
            class MyABC(metaclass=abc.ABCMeta):
                __slots__ = "a"

        with torch._dynamo.error_on_graph_break(False):
            class Unrelated(object):
                pass
        MyABC.register(Unrelated)

        u = Unrelated()
        self.assertIsInstance(u, MyABC)

        # This used to crash
        self.assertRaises(TypeError, MyABC.a.__set__, u, 3)

    def test_dynamics(self):
        # Testing class attribute propagation...
        with torch._dynamo.error_on_graph_break(False):
            class D(object):
                pass
            class E(D):
                pass
            class F(D):
                pass
        D.foo = 1
        self.assertEqual(D.foo, 1)
        # Test that dynamic attributes are inherited
        self.assertEqual(E.foo, 1)
        self.assertEqual(F.foo, 1)
        # Test dynamic instances
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                pass
        a = C()
        self.assertNotHasAttr(a, "foobar")
        C.foobar = 2
        self.assertEqual(a.foobar, 2)
        C.method = lambda self: 42
        self.assertEqual(a.method(), 42)
        C.__repr__ = lambda self: "C()"
        self.assertEqual(repr(a), "C()")
        C.__int__ = lambda self: 100
        self.assertEqual(int(a), 100)
        self.assertEqual(a.foobar, 2)
        self.assertNotHasAttr(a, "spam")
        def mygetattr(self, name):
            if name == "spam":
                return "spam"
            raise AttributeError
        C.__getattr__ = mygetattr
        self.assertEqual(a.spam, "spam")
        a.new = 12
        self.assertEqual(a.new, 12)
        def mysetattr(self, name, value):
            if name == "spam":
                raise AttributeError
            return object.__setattr__(self, name, value)
        C.__setattr__ = mysetattr
        with self.assertRaises(AttributeError):
            a.spam = "not spam"

        self.assertEqual(a.spam, "spam")
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        d = D()
        d.foo = 1
        self.assertEqual(d.foo, 1)

        # Test handling of int*seq and seq*int
        with torch._dynamo.error_on_graph_break(False):
            class I(int):
                pass
        self.assertEqual("a"*I(2), "aa")
        self.assertEqual(I(2)*"a", "aa")
        self.assertEqual(2*I(3), 6)
        self.assertEqual(I(3)*2, 6)
        self.assertEqual(I(3)*I(2), 6)

        # Test comparison of classes with dynamic metaclasses
        with torch._dynamo.error_on_graph_break(False):
            class dynamicmetaclass(type):
                pass
            class someclass(metaclass=dynamicmetaclass):
                pass
        self.assertNotEqual(someclass, object)

    def test_errors(self):
        # Testing errors...
        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(list, dict):
                    pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from both list and dict should be illegal")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object, None):
                    pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from non-type should be illegal")
        with torch._dynamo.error_on_graph_break(False):
            class Classic:
                pass

        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(type(len)):
                    pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from CFunction should be illegal")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = 1
        except TypeError:
            pass
        else:
            self.fail("__slots__ = 1 should be illegal")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class C(object):
                    __slots__ = [1]
        except TypeError:
            pass
        else:
            self.fail("__slots__ = [1] should be illegal")

        with torch._dynamo.error_on_graph_break(False):
            class M1(type):
                pass
            class M2(type):
                pass
            class A1(object, metaclass=M1):
                pass
            class A2(object, metaclass=M2):
                pass
        try:
            with torch._dynamo.error_on_graph_break(False):
                class B(A1, A2):
                    pass
        except TypeError:
            pass
        else:
            self.fail("finding the most derived metaclass should have failed")

    def test_classmethods(self):
        # Testing class methods...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def foo(*a): return a
                goo = classmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (C, 1))
        self.assertEqual(c.goo(1), (C, 1))
        self.assertEqual(c.foo(1), (c, 1))
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        d = D()
        self.assertEqual(D.goo(1), (D, 1))
        self.assertEqual(d.goo(1), (D, 1))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        # Test for a specific crash (SF bug 528132)
        def f(cls, arg):
            "f docstring"
            return (cls, arg)
        ff = classmethod(f)
        self.assertEqual(ff.__get__(0, int)(42), (int, 42))
        self.assertEqual(ff.__get__(0)(42), (int, 42))

        # Test super() with classmethods (SF bug 535444)
        self.assertEqual(C.goo.__self__, C)
        self.assertEqual(D.goo.__self__, D)
        self.assertEqual(super(D,D).goo.__self__, D)
        self.assertEqual(super(D,d).goo.__self__, D)
        self.assertEqual(super(D,D).goo(), (D,))
        self.assertEqual(super(D,d).goo(), (D,))

        # Verify that a non-callable will raise
        meth = classmethod(1).__get__(1)
        self.assertRaises(TypeError, meth)

        # Verify that classmethod() doesn't allow keyword args
        try:
            classmethod(f, kw=1)
        except TypeError:
            pass
        else:
            self.fail("classmethod shouldn't accept keyword args")

        cm = classmethod(f)
        cm_dict = {'__annotations__': {},
                   '__doc__': (
                       "f docstring"
                       if support.HAVE_PY_DOCSTRINGS
                       else None
                    ),
                   '__module__': __name__,
                   '__name__': 'f',
                   '__qualname__': f.__qualname__}
        self.assertEqual(cm.__dict__, cm_dict)

        cm.x = 42
        self.assertEqual(cm.x, 42)
        self.assertEqual(cm.__dict__, {"x" : 42, **cm_dict})
        del cm.x
        self.assertNotHasAttr(cm, "x")

    @support.refcount_test
    def test_refleaks_in_classmethod___init__(self):
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        cm = classmethod(None)
        refs_before = gettotalrefcount()
        for i in range(100):
            cm.__init__(None)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_classmethods_in_c(self):
        # Testing C-based class methods...
        import xxsubtype as spam
        a = (1, 2, 3)
        d = {'abc': 123}
        x, a1, d1 = spam.spamlist.classmeth(*a, **d)
        self.assertEqual(x, spam.spamlist)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        x, a1, d1 = spam.spamlist().classmeth(*a, **d)
        self.assertEqual(x, spam.spamlist)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        spam_cm = spam.spamlist.__dict__['classmeth']
        x2, a2, d2 = spam_cm(spam.spamlist, *a, **d)
        self.assertEqual(x2, spam.spamlist)
        self.assertEqual(a2, a1)
        self.assertEqual(d2, d1)
        with torch._dynamo.error_on_graph_break(False):
            class SubSpam(spam.spamlist): pass
        x2, a2, d2 = spam_cm(SubSpam, *a, **d)
        self.assertEqual(x2, SubSpam)
        self.assertEqual(a2, a1)
        self.assertEqual(d2, d1)

        with self.assertRaises(TypeError) as cm:
            spam_cm()
        self.assertEqual(
            str(cm.exception),
            "descriptor 'classmeth' of 'xxsubtype.spamlist' "
            "object needs an argument")

        with self.assertRaises(TypeError) as cm:
            spam_cm(spam.spamlist())
        self.assertEqual(
            str(cm.exception),
            "descriptor 'classmeth' for type 'xxsubtype.spamlist' "
            "needs a type, not a 'xxsubtype.spamlist' as arg 2")

        with self.assertRaises(TypeError) as cm:
            spam_cm(list)
        expected_errmsg = (
            "descriptor 'classmeth' requires a subtype of 'xxsubtype.spamlist' "
            "but received 'list'")
        self.assertEqual(str(cm.exception), expected_errmsg)

        with self.assertRaises(TypeError) as cm:
            spam_cm.__get__(None, list)
        self.assertEqual(str(cm.exception), expected_errmsg)

    def test_staticmethods(self):
        # Testing static methods...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def foo(*a): return a
                goo = staticmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (1,))
        self.assertEqual(c.goo(1), (1,))
        self.assertEqual(c.foo(1), (c, 1,))
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        d = D()
        self.assertEqual(D.goo(1), (1,))
        self.assertEqual(d.goo(1), (1,))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        sm = staticmethod(None)
        self.assertEqual(sm.__dict__, {'__doc__': None.__doc__})
        sm.x = 42
        self.assertEqual(sm.x, 42)
        self.assertEqual(sm.__dict__, {"x" : 42, '__doc__': None.__doc__})
        del sm.x
        self.assertNotHasAttr(sm, "x")

    @support.refcount_test
    def test_refleaks_in_staticmethod___init__(self):
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        sm = staticmethod(None)
        refs_before = gettotalrefcount()
        for i in range(100):
            sm.__init__(None)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_staticmethods_in_c(self):
        # Testing C-based static methods...
        import xxsubtype as spam
        a = (1, 2, 3)
        d = {"abc": 123}
        x, a1, d1 = spam.spamlist.staticmeth(*a, **d)
        self.assertEqual(x, None)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        x, a1, d2 = spam.spamlist().staticmeth(*a, **d)
        self.assertEqual(x, None)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)

    def test_classic(self):
        # Testing classic classes...
        with torch._dynamo.error_on_graph_break(False):
            class C:
                def foo(*a): return a
                goo = classmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (C, 1))
        self.assertEqual(c.goo(1), (C, 1))
        self.assertEqual(c.foo(1), (c, 1))
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        d = D()
        self.assertEqual(D.goo(1), (D, 1))
        self.assertEqual(d.goo(1), (D, 1))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        with torch._dynamo.error_on_graph_break(False):
            class E: # *not* subclassing from C
                foo = C.foo
        self.assertEqual(E().foo.__func__, C.foo) # i.e., unbound
        self.assertTrue(repr(C.foo.__get__(C())).startswith("<bound method "))

    def test_compattr(self):
        # Testing computed attributes...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                class computed_attribute(object):
                    def __init__(self, get, set=None, delete=None):
                        self.__get = get
                        self.__set = set
                        self.__delete = delete
                    def __get__(self, obj, type=None):
                        return self.__get(obj)
                    def __set__(self, obj, value):
                        return self.__set(obj, value)
                    def __delete__(self, obj):
                        return self.__delete(obj)
                def __init__(self):
                    self.__x = 0
                def __get_x(self):
                    x = self.__x
                    self.__x = x+1
                    return x
                def __set_x(self, x):
                    self.__x = x
                def __delete_x(self):
                    del self.__x
                x = computed_attribute(__get_x, __set_x, __delete_x)
        a = C()
        self.assertEqual(a.x, 0)
        self.assertEqual(a.x, 1)
        a.x = 10
        self.assertEqual(a.x, 10)
        self.assertEqual(a.x, 11)
        del a.x
        self.assertNotHasAttr(a, 'x')

    def test_newslots(self):
        # Testing __new__ slot override...
        with torch._dynamo.error_on_graph_break(False):
            class C(list):
                def __new__(cls):
                    self = list.__new__(cls)
                    self.foo = 1
                    return self
                def __init__(self):
                    self.foo = self.foo + 2
        a = C()
        self.assertEqual(a.foo, 3)
        self.assertEqual(a.__class__, C)
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        b = D()
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, D)

    @unittest.expectedFailure
    def test_bad_new(self):
        self.assertRaises(TypeError, object.__new__)
        self.assertRaises(TypeError, object.__new__, '')
        self.assertRaises(TypeError, list.__new__, object)
        self.assertRaises(TypeError, object.__new__, list)
        class C(object):
            __new__ = list.__new__
        self.assertRaises(TypeError, C)
        class C(list):
            __new__ = object.__new__
        self.assertRaises(TypeError, C)

    def test_object_new(self):
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                pass
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A())
        self.assertRaises(TypeError, object.__init__, A(), 5)

        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def __init__(self, foo):
                    self.foo = foo
        object.__new__(A)
        object.__new__(A, 5)
        object.__init__(A(3))
        self.assertRaises(TypeError, object.__init__, A(3), 5)

        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def __new__(cls, foo):
                    return object.__new__(cls)
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A(3))
        object.__init__(A(3), 5)

        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def __new__(cls, foo):
                    return object.__new__(cls)
                def __init__(self, foo):
                    self.foo = foo
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A(3))
        self.assertRaises(TypeError, object.__init__, A(3), 5)

    @unittest.expectedFailure
    def test_restored_object_new(self):
        class A(object):
            def __new__(cls, *args, **kwargs):
                raise AssertionError
        self.assertRaises(AssertionError, A)
        class B(A):
            __new__ = object.__new__
            def __init__(self, foo):
                self.foo = foo
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            b = B(3)
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, B)
        del B.__new__
        self.assertRaises(AssertionError, B)
        del A.__new__
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            b = B(3)
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, B)

    def test_altmro(self):
        # Testing mro() and overriding it...
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def f(self): return "A"
            class B(A):
                pass
            class C(A):
                def f(self): return "C"
            class D(B, C):
                pass
        self.assertEqual(A.mro(), [A, object])
        self.assertEqual(A.__mro__, (A, object))
        self.assertEqual(B.mro(), [B, A, object])
        self.assertEqual(B.__mro__, (B, A, object))
        self.assertEqual(C.mro(), [C, A, object])
        self.assertEqual(C.__mro__, (C, A, object))
        self.assertEqual(D.mro(), [D, B, C, A, object])
        self.assertEqual(D.__mro__, (D, B, C, A, object))
        self.assertEqual(D().f(), "C")

        with torch._dynamo.error_on_graph_break(False):
            class PerverseMetaType(type):
                def mro(cls):
                    L = type.mro(cls)
                    L.reverse()
                    return L
            class X(D,B,C,A, metaclass=PerverseMetaType):
                pass
        self.assertEqual(X.__mro__, (object, A, C, B, D, X))
        self.assertEqual(X().f(), "A")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class _metaclass(type):
                    def mro(self):
                        return [self, dict, object]
                class X(object, metaclass=_metaclass):
                    pass
            # In CPython, the class creation above already raises
            # TypeError, as a protection against the fact that
            # instances of X would segfault it.  In other Python
            # implementations it would be ok to let the class X
            # be created, but instead get a clean TypeError on the
            # __setitem__ below.
            x = object.__new__(X)
            x[5] = 6
        except TypeError:
            pass
        else:
            self.fail("devious mro() return not caught")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class _metaclass(type):
                    def mro(self):
                        return [1]
                class X(object, metaclass=_metaclass):
                    pass
        except TypeError:
            pass
        else:
            self.fail("non-class mro() return not caught")

        try:
            with torch._dynamo.error_on_graph_break(False):
                class _metaclass(type):
                    def mro(self):
                        return 1
                class X(object, metaclass=_metaclass):
                    pass
        except TypeError:
            pass
        else:
            self.fail("non-sequence mro() return not caught")

    def test_overloading(self):
        # Testing operator overloading...

        with torch._dynamo.error_on_graph_break(False):
            class B(object):
                "Intermediate class because object doesn't have a __setattr__"

            class C(B):
                def __getattr__(self, name):
                    if name == "foo":
                        return ("getattr", name)
                    else:
                        raise AttributeError
                def __setattr__(self, name, value):
                    if name == "foo":
                        self.setattr = (name, value)
                    else:
                        return B.__setattr__(self, name, value)
                def __delattr__(self, name):
                    if name == "foo":
                        self.delattr = name
                    else:
                        return B.__delattr__(self, name)

                def __getitem__(self, key):
                    return ("getitem", key)
                def __setitem__(self, key, value):
                    self.setitem = (key, value)
                def __delitem__(self, key):
                    self.delitem = key

        a = C()
        self.assertEqual(a.foo, ("getattr", "foo"))
        a.foo = 12
        self.assertEqual(a.setattr, ("foo", 12))
        del a.foo
        self.assertEqual(a.delattr, "foo")

        self.assertEqual(a[12], ("getitem", 12))
        a[12] = 21
        self.assertEqual(a.setitem, (12, 21))
        del a[12]
        self.assertEqual(a.delitem, 12)

        self.assertEqual(a[0:10], ("getitem", slice(0, 10)))
        a[0:10] = "foo"
        self.assertEqual(a.setitem, (slice(0, 10), "foo"))
        del a[0:10]
        self.assertEqual(a.delitem, (slice(0, 10)))

    def test_load_attr_extended_arg(self):
        # https://github.com/python/cpython/issues/91625
        with torch._dynamo.error_on_graph_break(False):
            class Numbers:
                def __getattr__(self, attr):
                    return int(attr.lstrip("_"))
        attrs = ", ".join(f"Z._{n:03d}" for n in range(280))
        code = f"def number_attrs(Z):\n    return [ {attrs} ]"
        ns = {}
        exec(code, ns)
        number_attrs = ns["number_attrs"]
        # Warm up the function for quickening (PEP 659)
        for _ in range(30):
            self.assertEqual(number_attrs(Numbers()), list(range(280)))

    def test_methods(self):
        # Testing methods...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __init__(self, x):
                    self.x = x
                def foo(self):
                    return self.x
        c1 = C(1)
        self.assertEqual(c1.foo(), 1)
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                boo = C.foo
                goo = c1.foo
        d2 = D(2)
        self.assertEqual(d2.foo(), 2)
        self.assertEqual(d2.boo(), 2)
        self.assertEqual(d2.goo(), 1)
        with torch._dynamo.error_on_graph_break(False):
            class E(object):
                foo = C.foo
        self.assertEqual(E().foo.__func__, C.foo) # i.e., unbound
        self.assertTrue(repr(C.foo.__get__(C(1))).startswith("<bound method "))

    @support.impl_detail("testing error message from implementation")
    def test_methods_in_c(self):
        # This test checks error messages in builtin method descriptor.
        # It is allowed that other Python implementations use
        # different error messages.
        set_add = set.add

        expected_errmsg = "unbound method set.add() needs an argument"

        with self.assertRaises(TypeError) as cm:
            set_add()
        self.assertEqual(cm.exception.args[0], expected_errmsg)

        expected_errmsg = "descriptor 'add' for 'set' objects doesn't apply to a 'int' object"

        with self.assertRaises(TypeError) as cm:
            set_add(0)
        self.assertEqual(cm.exception.args[0], expected_errmsg)

        with self.assertRaises(TypeError) as cm:
            set_add.__get__(0)
        self.assertEqual(cm.exception.args[0], expected_errmsg)

    def test_special_method_lookup(self):
        # The lookup of special methods bypasses __getattr__ and
        # __getattribute__, but they still can be descriptors.

        def run_context(manager):
            with manager:
                pass
        def iden(self):
            return self
        def hello(self):
            return b"hello"
        def empty_seq(self):
            return []
        def zero(self):
            return 0
        def complex_num(self):
            return 1j
        def stop(self):
            raise StopIteration
        def return_true(self, thing=None):
            return True
        def do_isinstance(obj):
            return isinstance(int, obj)
        def do_issubclass(obj):
            return issubclass(int, obj)
        def do_dict_missing(checker):
            with torch._dynamo.error_on_graph_break(False):
                class DictSub(checker.__class__, dict):
                    pass
            self.assertEqual(DictSub()["hi"], 4)
        def some_number(self_, key):
            self.assertEqual(key, "hi")
            return 4
        def swallow(*args): pass
        def format_impl(self, spec):
            return "hello"

        # It would be nice to have every special method tested here, but I'm
        # only listing the ones I can remember outside of typeobject.c, since it
        # does it right.
        specials = [
            ("__bytes__", bytes, hello, set(), {}),
            ("__reversed__", reversed, empty_seq, set(), {}),
            ("__length_hint__", list, zero, set(),
             {"__iter__" : iden, "__next__" : stop}),
            ("__sizeof__", sys.getsizeof, zero, set(), {}),
            ("__instancecheck__", do_isinstance, return_true, set(), {}),
            ("__missing__", do_dict_missing, some_number,
             set(("__class__",)), {}),
            ("__subclasscheck__", do_issubclass, return_true,
             set(("__bases__",)), {}),
            ("__enter__", run_context, iden, set(), {"__exit__" : swallow}),
            ("__exit__", run_context, swallow, set(), {"__enter__" : iden}),
            ("__complex__", complex, complex_num, set(), {}),
            ("__format__", format, format_impl, set(), {}),
            ("__floor__", math.floor, zero, set(), {}),
            ("__trunc__", math.trunc, zero, set(), {}),
            ("__ceil__", math.ceil, zero, set(), {}),
            ("__dir__", dir, empty_seq, set(), {}),
            ("__round__", round, zero, set(), {}),
            ]

        with torch._dynamo.error_on_graph_break(False):
            class Checker(object):
                def __getattr__(self, attr, test=self):
                    test.fail("__getattr__ called with {0}".format(attr))
                def __getattribute__(self, attr, test=self):
                    if attr not in ok:
                        test.fail("__getattribute__ called with {0}".format(attr))
                    return object.__getattribute__(self, attr)
            class SpecialDescr(object):
                def __init__(self, impl):
                    self.impl = impl
                def __get__(self, obj, owner):
                    record.append(1)
                    return self.impl.__get__(obj, owner)
            class MyException(Exception):
                pass
            class ErrDescr(object):
                def __get__(self, obj, owner):
                    raise MyException

        for name, runner, meth_impl, ok, env in specials:
            with torch._dynamo.error_on_graph_break(False):
                class X(Checker):
                    pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, meth_impl)
            runner(X())

            record = []
            with torch._dynamo.error_on_graph_break(False):
                class X(Checker):
                    pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, SpecialDescr(meth_impl))
            runner(X())
            self.assertEqual(record, [1], name)

            with torch._dynamo.error_on_graph_break(False):
                class X(Checker):
                    pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, ErrDescr())
            self.assertRaises(MyException, runner, X())

    def test_specials(self):
        # Testing special operators...
        # Test operators like __hash__ for which a built-in default exists

        # Test the default behavior for static classes
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __getitem__(self, i):
                    if 0 <= i < 10: return i
                    raise IndexError
        c1 = C()
        c2 = C()
        self.assertFalse(not c1)
        self.assertNotEqual(id(c1), id(c2))
        hash(c1)
        hash(c2)
        self.assertEqual(c1, c1)
        self.assertTrue(c1 != c2)
        self.assertFalse(c1 != c1)
        self.assertFalse(c1 == c2)
        # Note that the module name appears in str/repr, and that varies
        # depending on whether this test is run standalone or from a framework.
        self.assertGreaterEqual(str(c1).find('C object at '), 0)
        self.assertEqual(str(c1), repr(c1))
        self.assertNotIn(-1, c1)
        for i in range(10):
            self.assertIn(i, c1)
        self.assertNotIn(10, c1)
        # Test the default behavior for dynamic classes
        with torch._dynamo.error_on_graph_break(False):
            class D(object):
                def __getitem__(self, i):
                    if 0 <= i < 10: return i
                    raise IndexError
        d1 = D()
        d2 = D()
        self.assertFalse(not d1)
        self.assertNotEqual(id(d1), id(d2))
        hash(d1)
        hash(d2)
        self.assertEqual(d1, d1)
        self.assertNotEqual(d1, d2)
        self.assertFalse(d1 != d1)
        self.assertFalse(d1 == d2)
        # Note that the module name appears in str/repr, and that varies
        # depending on whether this test is run standalone or from a framework.
        self.assertGreaterEqual(str(d1).find('D object at '), 0)
        self.assertEqual(str(d1), repr(d1))
        self.assertNotIn(-1, d1)
        for i in range(10):
            self.assertIn(i, d1)
        self.assertNotIn(10, d1)
        # Test overridden behavior
        with torch._dynamo.error_on_graph_break(False):
            class Proxy(object):
                def __init__(self, x):
                    self.x = x
                def __bool__(self):
                    return not not self.x
                def __hash__(self):
                    return hash(self.x)
                def __eq__(self, other):
                    return self.x == other
                def __ne__(self, other):
                    return self.x != other
                def __ge__(self, other):
                    return self.x >= other
                def __gt__(self, other):
                    return self.x > other
                def __le__(self, other):
                    return self.x <= other
                def __lt__(self, other):
                    return self.x < other
                def __str__(self):
                    return "Proxy:%s" % self.x
                def __repr__(self):
                    return "Proxy(%r)" % self.x
                def __contains__(self, value):
                    return value in self.x
        p0 = Proxy(0)
        p1 = Proxy(1)
        p_1 = Proxy(-1)
        self.assertFalse(p0)
        self.assertFalse(not p1)
        self.assertEqual(hash(p0), hash(0))
        self.assertEqual(p0, p0)
        self.assertNotEqual(p0, p1)
        self.assertFalse(p0 != p0)
        self.assertEqual(not p0, p1)
        self.assertTrue(p0 < p1)
        self.assertTrue(p0 <= p1)
        self.assertTrue(p1 > p0)
        self.assertTrue(p1 >= p0)
        self.assertEqual(str(p0), "Proxy:0")
        self.assertEqual(repr(p0), "Proxy(0)")
        p10 = Proxy(range(10))
        self.assertNotIn(-1, p10)
        for i in range(10):
            self.assertIn(i, p10)
        self.assertNotIn(10, p10)

    def test_weakrefs(self):
        # Testing weak references...
        import weakref
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                pass
        c = C()
        r = weakref.ref(c)
        self.assertEqual(r(), c)
        del c
        support.gc_collect()
        self.assertEqual(r(), None)
        del r
        with torch._dynamo.error_on_graph_break(False):
            class NoWeak(object):
                __slots__ = ['foo']
        no = NoWeak()
        try:
            weakref.ref(no)
        except TypeError as msg:
            self.assertIn("weak reference", str(msg))
        else:
            self.fail("weakref.ref(no) should be illegal")
        with torch._dynamo.error_on_graph_break(False):
            class Weak(object):
                __slots__ = ['foo', '__weakref__']
        yes = Weak()
        r = weakref.ref(yes)
        self.assertEqual(r(), yes)
        del yes
        support.gc_collect()
        self.assertEqual(r(), None)
        del r

    def test_properties(self):
        # Testing property...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def getx(self):
                    return self.__x
                def setx(self, value):
                    self.__x = value
                def delx(self):
                    del self.__x
                x = property(getx, setx, delx, doc="I'm the x property.")
        a = C()
        self.assertNotHasAttr(a, "x")
        a.x = 42
        self.assertEqual(a._C__x, 42)
        self.assertEqual(a.x, 42)
        del a.x
        self.assertNotHasAttr(a, "x")
        self.assertNotHasAttr(a, "_C__x")
        C.x.__set__(a, 100)
        self.assertEqual(C.x.__get__(a), 100)
        C.x.__delete__(a)
        self.assertNotHasAttr(a, "x")

        raw = C.__dict__['x']
        self.assertIsInstance(raw, property)

        attrs = dir(raw)
        self.assertIn("__doc__", attrs)
        self.assertIn("fget", attrs)
        self.assertIn("fset", attrs)
        self.assertIn("fdel", attrs)

        self.assertEqual(raw.__doc__, "I'm the x property.")
        self.assertIs(raw.fget, C.__dict__['getx'])
        self.assertIs(raw.fset, C.__dict__['setx'])
        self.assertIs(raw.fdel, C.__dict__['delx'])

        for attr in "fget", "fset", "fdel":
            try:
                setattr(raw, attr, 42)
            except AttributeError as msg:
                if str(msg).find('readonly') < 0:
                    self.fail("when setting readonly attr %r on a property, "
                              "got unexpected AttributeError msg %r" % (attr, str(msg)))
            else:
                self.fail("expected AttributeError from trying to set readonly %r "
                          "attr on a property" % attr)

        raw.__doc__ = 42
        self.assertEqual(raw.__doc__, 42)

        with torch._dynamo.error_on_graph_break(False):
            class D(object):
                __getitem__ = property(lambda s: 1/0)

        d = D()
        try:
            for i in d:
                str(i)
        except ZeroDivisionError:
            pass
        else:
            self.fail("expected ZeroDivisionError from bad property")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_properties_doc_attrib(self):
        with torch._dynamo.error_on_graph_break(False):
            class E(object):
                def getter(self):
                    "getter method"
                    return 0
                def setter(self_, value):
                    "setter method"
                    pass
                prop = property(getter)
                self.assertEqual(prop.__doc__, "getter method")
                prop2 = property(fset=setter)
                self.assertEqual(prop2.__doc__, None)

    @support.cpython_only
    def test_testcapi_no_segfault(self):
        # this segfaulted in 2.5b2
        try:
            import _testcapi
        except ImportError:
            pass
        else:
            with torch._dynamo.error_on_graph_break(False):
                class X(object):
                    p = property(_testcapi.test_with_docstring)

    def test_properties_plus(self):
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                foo = property(doc="hello")
                @foo.getter
                def foo(self):
                    return self._foo
                @foo.setter
                def foo(self, value):
                    self._foo = abs(value)
                @foo.deleter
                def foo(self):
                    del self._foo
        c = C()
        self.assertEqual(C.foo.__doc__, "hello")
        self.assertNotHasAttr(c, "foo")
        c.foo = -42
        self.assertHasAttr(c, '_foo')
        self.assertEqual(c._foo, 42)
        self.assertEqual(c.foo, 42)
        del c.foo
        self.assertNotHasAttr(c, '_foo')
        self.assertNotHasAttr(c, "foo")

        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                @C.foo.deleter
                def foo(self):
                    try:
                        del self._foo
                    except AttributeError:
                        pass
        d = D()
        d.foo = 24
        self.assertEqual(d.foo, 24)
        del d.foo
        del d.foo

        with torch._dynamo.error_on_graph_break(False):
            class E(object):
                @property
                def foo(self):
                    return self._foo
                @foo.setter
                def foo(self, value):
                    raise RuntimeError
                @foo.setter
                def foo(self, value):
                    self._foo = abs(value)
                @foo.deleter
                def foo(self, value=None):
                    del self._foo

        e = E()
        e.foo = -42
        self.assertEqual(e.foo, 42)
        del e.foo

        with torch._dynamo.error_on_graph_break(False):
            class F(E):
                @E.foo.deleter
                def foo(self):
                    del self._foo
                @foo.setter
                def foo(self, value):
                    self._foo = max(0, value)
        f = F()
        f.foo = -10
        self.assertEqual(f.foo, 0)
        del f.foo

    def test_dict_constructors(self):
        # Testing dict constructor ...
        d = dict()
        self.assertEqual(d, {})
        d = dict({})
        self.assertEqual(d, {})
        d = dict({1: 2, 'a': 'b'})
        self.assertEqual(d, {1: 2, 'a': 'b'})
        self.assertEqual(d, dict(list(d.items())))
        self.assertEqual(d, dict(iter(d.items())))
        d = dict({'one':1, 'two':2})
        self.assertEqual(d, dict(one=1, two=2))
        self.assertEqual(d, dict(**d))
        self.assertEqual(d, dict({"one": 1}, two=2))
        self.assertEqual(d, dict([("two", 2)], one=1))
        self.assertEqual(d, dict([("one", 100), ("two", 200)], **d))
        self.assertEqual(d, dict(**d))

        for badarg in 0, 0, 0j, "0", [0], (0,):
            try:
                dict(badarg)
            except TypeError:
                pass
            except ValueError:
                if badarg == "0":
                    # It's a sequence, and its elements are also sequences (gotta
                    # love strings <wink>), but they aren't of length 2, so this
                    # one seemed better as a ValueError than a TypeError.
                    pass
                else:
                    self.fail("no TypeError from dict(%r)" % badarg)
            else:
                self.fail("no TypeError from dict(%r)" % badarg)

        with self.assertRaises(TypeError):
            dict({}, {})

        with torch._dynamo.error_on_graph_break(False):
            class Mapping:
                # Lacks a .keys() method; will be added later.
                dict = {1:2, 3:4, 'a':1j}

        try:
            dict(Mapping())
        except TypeError:
            pass
        else:
            self.fail("no TypeError from dict(incomplete mapping)")

        Mapping.keys = lambda self: list(self.dict.keys())
        Mapping.__getitem__ = lambda self, i: self.dict[i]
        d = dict(Mapping())
        self.assertEqual(d, Mapping.dict)

        # Init from sequence of iterable objects, each producing a 2-sequence.
        with torch._dynamo.error_on_graph_break(False):
            class AddressBookEntry:
                def __init__(self, first, last):
                    self.first = first
                    self.last = last
                def __iter__(self):
                    return iter([self.first, self.last])

        d = dict([AddressBookEntry('Tim', 'Warsaw'),
                  AddressBookEntry('Barry', 'Peters'),
                  AddressBookEntry('Tim', 'Peters'),
                  AddressBookEntry('Barry', 'Warsaw')])
        self.assertEqual(d, {'Barry': 'Warsaw', 'Tim': 'Peters'})

        d = dict(zip(range(4), range(1, 5)))
        self.assertEqual(d, dict([(i, i+1) for i in range(4)]))

        # Bad sequence lengths.
        for bad in [('tooshort',)], [('too', 'long', 'by 1')]:
            try:
                dict(bad)
            except ValueError:
                pass
            else:
                self.fail("no ValueError from dict(%r)" % bad)

    def test_dir(self):
        # Testing dir() ...
        junk = 12
        self.assertEqual(dir(), ['junk', 'self'])
        del junk

        # Just make sure these don't blow up!
        for arg in 2, 2, 2j, 2e0, [2], "2", b"2", (2,), {2:2}, type, self.test_dir:
            dir(arg)

        # Test dir on new-style classes.  Since these have object as a
        # base class, a lot more gets sucked in.
        def interesting(strings):
            return [s for s in strings if not s.startswith('_')]

        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                Cdata = 1
                def Cmethod(self): pass

        cstuff = ['Cdata', 'Cmethod']
        self.assertEqual(interesting(dir(C)), cstuff)

        c = C()
        self.assertEqual(interesting(dir(c)), cstuff)
        ## self.assertIn('__self__', dir(C.Cmethod))

        c.cdata = 2
        c.cmethod = lambda self: 0
        self.assertEqual(interesting(dir(c)), cstuff + ['cdata', 'cmethod'])
        ## self.assertIn('__self__', dir(c.Cmethod))

        with torch._dynamo.error_on_graph_break(False):
            class A(C):
                Adata = 1
                def Amethod(self): pass

        astuff = ['Adata', 'Amethod'] + cstuff
        self.assertEqual(interesting(dir(A)), astuff)
        ## self.assertIn('__self__', dir(A.Amethod))
        a = A()
        self.assertEqual(interesting(dir(a)), astuff)
        a.adata = 42
        a.amethod = lambda self: 3
        self.assertEqual(interesting(dir(a)), astuff + ['adata', 'amethod'])
        ## self.assertIn('__self__', dir(a.Amethod))

        # Try a module subclass.
        with torch._dynamo.error_on_graph_break(False):
            class M(type(sys)):
                pass
        minstance = M("m")
        minstance.b = 2
        minstance.a = 1
        default_attributes = ['__name__', '__doc__', '__package__',
                              '__loader__', '__spec__']
        names = [x for x in dir(minstance) if x not in default_attributes]
        self.assertEqual(names, ['a', 'b'])

        with torch._dynamo.error_on_graph_break(False):
            class M2(M):
                def getdict(self):
                    return "Not a dict!"
                __dict__ = property(getdict)

        m2instance = M2("m2")
        m2instance.b = 2
        m2instance.a = 1
        self.assertEqual(m2instance.__dict__, "Not a dict!")
        with self.assertRaises(TypeError):
            dir(m2instance)

        # Two essentially featureless objects, (Ellipsis just inherits stuff
        # from object.
        self.assertEqual(dir(object()), dir(Ellipsis))

        # Nasty test case for proxied objects
        with torch._dynamo.error_on_graph_break(False):
            class Wrapper(object):
                def __init__(self, obj):
                    self.__obj = obj
                def __repr__(self):
                    return "Wrapper(%s)" % repr(self.__obj)
                def __getitem__(self, key):
                    return Wrapper(self.__obj[key])
                def __len__(self):
                    return len(self.__obj)
                def __getattr__(self, name):
                    return Wrapper(getattr(self.__obj, name))

            class C(object):
                def __getclass(self):
                    return Wrapper(type(self))
                __class__ = property(__getclass)

        dir(C()) # This used to segfault

    def test_supers(self):
        # Testing super...

        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                def meth(self, a):
                    return "A(%r)" % a

        self.assertEqual(A().meth(1), "A(1)")

        with torch._dynamo.error_on_graph_break(False):
            class B(A):
                def __init__(self):
                    self.__super = super(B, self)
                def meth(self, a):
                    return "B(%r)" % a + self.__super.meth(a)

        self.assertEqual(B().meth(2), "B(2)A(2)")

        with torch._dynamo.error_on_graph_break(False):
            class C(A):
                def meth(self, a):
                    return "C(%r)" % a + self.__super.meth(a)
        C._C__super = super(C)

        self.assertEqual(C().meth(3), "C(3)A(3)")

        with torch._dynamo.error_on_graph_break(False):
            class D(C, B):
                def meth(self, a):
                    return "D(%r)" % a + super(D, self).meth(a)

        self.assertEqual(D().meth(4), "D(4)C(4)B(4)A(4)")

        # Test for subclassing super

        with torch._dynamo.error_on_graph_break(False):
            class mysuper(super):
                def __init__(self, *args):
                    return super(mysuper, self).__init__(*args)

            class E(D):
                def meth(self, a):
                    return "E(%r)" % a + mysuper(E, self).meth(a)

        self.assertEqual(E().meth(5), "E(5)D(5)C(5)B(5)A(5)")

        with torch._dynamo.error_on_graph_break(False):
            class F(E):
                def meth(self, a):
                    s = self.__super # == mysuper(F, self)
                    return "F(%r)[%s]" % (a, s.__class__.__name__) + s.meth(a)
        F._F__super = mysuper(F)

        self.assertEqual(F().meth(6), "F(6)[mysuper]E(6)D(6)C(6)B(6)A(6)")

        # Make sure certain errors are raised

        try:
            super(D, 42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D, 42)")

        try:
            super(D, C())
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D, C())")

        try:
            super(D).__get__(12)
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D).__get__(12)")

        try:
            super(D).__get__(C())
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D).__get__(C())")

        # Make sure data descriptors can be overridden and accessed via super
        # (new feature in Python 2.3)

        with torch._dynamo.error_on_graph_break(False):
            class DDbase(object):
                def getx(self): return 42
                x = property(getx)

            class DDsub(DDbase):
                def getx(self): return "hello"
                x = property(getx)

        dd = DDsub()
        self.assertEqual(dd.x, "hello")
        self.assertEqual(super(DDsub, dd).x, 42)

        # Ensure that super() lookup of descriptor from classmethod
        # works (SF ID# 743627)

        with torch._dynamo.error_on_graph_break(False):
            class Base(object):
                aProp = property(lambda self: "foo")

            class Sub(Base):
                @classmethod
                def test(klass):
                    return super(Sub,klass).aProp

        self.assertEqual(Sub.test(), Base.aProp)

        # Verify that super() doesn't allow keyword args
        with self.assertRaises(TypeError):
            super(Base, kw=1)

    def test_basic_inheritance(self):
        # Testing inheritance from basic types...

        with torch._dynamo.error_on_graph_break(False):
            class hexint(int):
                def __repr__(self):
                    return hex(self)
                def __add__(self, other):
                    return hexint(int.__add__(self, other))
                # (Note that overriding __radd__ doesn't work,
                # because the int type gets first dibs.)
        self.assertEqual(repr(hexint(7) + 9), "0x10")
        self.assertEqual(repr(hexint(1000) + 7), "0x3ef")
        a = hexint(12345)
        self.assertEqual(a, 12345)
        self.assertEqual(int(a), 12345)
        self.assertIs(int(a).__class__, int)
        self.assertEqual(hash(a), hash(12345))
        self.assertIs((+a).__class__, int)
        self.assertIs((a >> 0).__class__, int)
        self.assertIs((a << 0).__class__, int)
        self.assertIs((hexint(0) << 12).__class__, int)
        self.assertIs((hexint(0) >> 12).__class__, int)

        with torch._dynamo.error_on_graph_break(False):
            class octlong(int):
                __slots__ = []
                def __str__(self):
                    return oct(self)
                def __add__(self, other):
                    return self.__class__(super(octlong, self).__add__(other))
                __radd__ = __add__
        self.assertEqual(str(octlong(3) + 5), "0o10")
        # (Note that overriding __radd__ here only seems to work
        # because the example uses a short int left argument.)
        self.assertEqual(str(5 + octlong(3000)), "0o5675")
        a = octlong(12345)
        self.assertEqual(a, 12345)
        self.assertEqual(int(a), 12345)
        self.assertEqual(hash(a), hash(12345))
        self.assertIs(int(a).__class__, int)
        self.assertIs((+a).__class__, int)
        self.assertIs((-a).__class__, int)
        self.assertIs((-octlong(0)).__class__, int)
        self.assertIs((a >> 0).__class__, int)
        self.assertIs((a << 0).__class__, int)
        self.assertIs((a - 0).__class__, int)
        self.assertIs((a * 1).__class__, int)
        self.assertIs((a ** 1).__class__, int)
        self.assertIs((a // 1).__class__, int)
        self.assertIs((1 * a).__class__, int)
        self.assertIs((a | 0).__class__, int)
        self.assertIs((a ^ 0).__class__, int)
        self.assertIs((a & -1).__class__, int)
        self.assertIs((octlong(0) << 12).__class__, int)
        self.assertIs((octlong(0) >> 12).__class__, int)
        self.assertIs(abs(octlong(0)).__class__, int)

        # Because octlong overrides __add__, we can't check the absence of +0
        # optimizations using octlong.
        with torch._dynamo.error_on_graph_break(False):
            class longclone(int):
                pass
        a = longclone(1)
        self.assertIs((a + 0).__class__, int)
        self.assertIs((0 + a).__class__, int)

        # Check that negative clones don't segfault
        a = longclone(-1)
        self.assertEqual(a.__dict__, {})
        self.assertEqual(int(a), -1)  # self.assertTrue PyNumber_Long() copies the sign bit

        with torch._dynamo.error_on_graph_break(False):
            class precfloat(float):
                __slots__ = ['prec']
                def __init__(self, value=0.0, prec=12):
                    self.prec = int(prec)
                def __repr__(self):
                    return "%.*g" % (self.prec, self)
        self.assertEqual(repr(precfloat(1.1)), "1.1")
        a = precfloat(12345)
        self.assertEqual(a, 12345.0)
        self.assertEqual(float(a), 12345.0)
        self.assertIs(float(a).__class__, float)
        self.assertEqual(hash(a), hash(12345.0))
        self.assertIs((+a).__class__, float)

        with torch._dynamo.error_on_graph_break(False):
            class madcomplex(complex):
                def __repr__(self):
                    return "%.17gj%+.17g" % (self.imag, self.real)
        a = madcomplex(-3, 4)
        self.assertEqual(repr(a), "4j-3")
        base = complex(-3, 4)
        self.assertEqual(base.__class__, complex)
        self.assertEqual(a, base)
        self.assertEqual(complex(a), base)
        self.assertEqual(complex(a).__class__, complex)
        a = madcomplex(a)  # just trying another form of the constructor
        self.assertEqual(repr(a), "4j-3")
        self.assertEqual(a, base)
        self.assertEqual(complex(a), base)
        self.assertEqual(complex(a).__class__, complex)
        self.assertEqual(hash(a), hash(base))
        self.assertEqual((+a).__class__, complex)
        self.assertEqual((a + 0).__class__, complex)
        self.assertEqual(a + 0, base)
        self.assertEqual((a - 0).__class__, complex)
        self.assertEqual(a - 0, base)
        self.assertEqual((a * 1).__class__, complex)
        self.assertEqual(a * 1, base)
        self.assertEqual((a / 1).__class__, complex)
        self.assertEqual(a / 1, base)

        with torch._dynamo.error_on_graph_break(False):
            class madtuple(tuple):
                _rev = None
                def rev(self):
                    if self._rev is not None:
                        return self._rev
                    L = list(self)
                    L.reverse()
                    self._rev = self.__class__(L)
                    return self._rev
        a = madtuple((1,2,3,4,5,6,7,8,9,0))
        self.assertEqual(a, (1,2,3,4,5,6,7,8,9,0))
        self.assertEqual(a.rev(), madtuple((0,9,8,7,6,5,4,3,2,1)))
        self.assertEqual(a.rev().rev(), madtuple((1,2,3,4,5,6,7,8,9,0)))
        for i in range(512):
            t = madtuple(range(i))
            u = t.rev()
            v = u.rev()
            self.assertEqual(v, t)
        a = madtuple((1,2,3,4,5))
        self.assertEqual(tuple(a), (1,2,3,4,5))
        self.assertIs(tuple(a).__class__, tuple)
        self.assertEqual(hash(a), hash((1,2,3,4,5)))
        self.assertIs(a[:].__class__, tuple)
        self.assertIs((a * 1).__class__, tuple)
        self.assertIs((a * 0).__class__, tuple)
        self.assertIs((a + ()).__class__, tuple)
        a = madtuple(())
        self.assertEqual(tuple(a), ())
        self.assertIs(tuple(a).__class__, tuple)
        self.assertIs((a + a).__class__, tuple)
        self.assertIs((a * 0).__class__, tuple)
        self.assertIs((a * 1).__class__, tuple)
        self.assertIs((a * 2).__class__, tuple)
        self.assertIs(a[:].__class__, tuple)

        with torch._dynamo.error_on_graph_break(False):
            class madstring(str):
                _rev = None
                def rev(self):
                    if self._rev is not None:
                        return self._rev
                    L = list(self)
                    L.reverse()
                    self._rev = self.__class__("".join(L))
                    return self._rev
        s = madstring("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(s, "abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(s.rev(), madstring("zyxwvutsrqponmlkjihgfedcba"))
        self.assertEqual(s.rev().rev(), madstring("abcdefghijklmnopqrstuvwxyz"))
        for i in range(256):
            s = madstring("".join(map(chr, range(i))))
            t = s.rev()
            u = t.rev()
            self.assertEqual(u, s)
        s = madstring("12345")
        self.assertEqual(str(s), "12345")
        self.assertIs(str(s).__class__, str)

        base = "\x00" * 5
        s = madstring(base)
        self.assertEqual(s, base)
        self.assertEqual(str(s), base)
        self.assertIs(str(s).__class__, str)
        self.assertEqual(hash(s), hash(base))
        self.assertEqual({s: 1}[base], 1)
        self.assertEqual({base: 1}[s], 1)
        self.assertIs((s + "").__class__, str)
        self.assertEqual(s + "", base)
        self.assertIs(("" + s).__class__, str)
        self.assertEqual("" + s, base)
        self.assertIs((s * 0).__class__, str)
        self.assertEqual(s * 0, "")
        self.assertIs((s * 1).__class__, str)
        self.assertEqual(s * 1, base)
        self.assertIs((s * 2).__class__, str)
        self.assertEqual(s * 2, base + base)
        self.assertIs(s[:].__class__, str)
        self.assertEqual(s[:], base)
        self.assertIs(s[0:0].__class__, str)
        self.assertEqual(s[0:0], "")
        self.assertIs(s.strip().__class__, str)
        self.assertEqual(s.strip(), base)
        self.assertIs(s.lstrip().__class__, str)
        self.assertEqual(s.lstrip(), base)
        self.assertIs(s.rstrip().__class__, str)
        self.assertEqual(s.rstrip(), base)
        identitytab = {}
        self.assertIs(s.translate(identitytab).__class__, str)
        self.assertEqual(s.translate(identitytab), base)
        self.assertIs(s.replace("x", "x").__class__, str)
        self.assertEqual(s.replace("x", "x"), base)
        self.assertIs(s.ljust(len(s)).__class__, str)
        self.assertEqual(s.ljust(len(s)), base)
        self.assertIs(s.rjust(len(s)).__class__, str)
        self.assertEqual(s.rjust(len(s)), base)
        self.assertIs(s.center(len(s)).__class__, str)
        self.assertEqual(s.center(len(s)), base)
        self.assertIs(s.lower().__class__, str)
        self.assertEqual(s.lower(), base)

        with torch._dynamo.error_on_graph_break(False):
            class madunicode(str):
                _rev = None
                def rev(self):
                    if self._rev is not None:
                        return self._rev
                    L = list(self)
                    L.reverse()
                    self._rev = self.__class__("".join(L))
                    return self._rev
        u = madunicode("ABCDEF")
        self.assertEqual(u, "ABCDEF")
        self.assertEqual(u.rev(), madunicode("FEDCBA"))
        self.assertEqual(u.rev().rev(), madunicode("ABCDEF"))
        base = "12345"
        u = madunicode(base)
        self.assertEqual(str(u), base)
        self.assertIs(str(u).__class__, str)
        self.assertEqual(hash(u), hash(base))
        self.assertEqual({u: 1}[base], 1)
        self.assertEqual({base: 1}[u], 1)
        self.assertIs(u.strip().__class__, str)
        self.assertEqual(u.strip(), base)
        self.assertIs(u.lstrip().__class__, str)
        self.assertEqual(u.lstrip(), base)
        self.assertIs(u.rstrip().__class__, str)
        self.assertEqual(u.rstrip(), base)
        self.assertIs(u.replace("x", "x").__class__, str)
        self.assertEqual(u.replace("x", "x"), base)
        self.assertIs(u.replace("xy", "xy").__class__, str)
        self.assertEqual(u.replace("xy", "xy"), base)
        self.assertIs(u.center(len(u)).__class__, str)
        self.assertEqual(u.center(len(u)), base)
        self.assertIs(u.ljust(len(u)).__class__, str)
        self.assertEqual(u.ljust(len(u)), base)
        self.assertIs(u.rjust(len(u)).__class__, str)
        self.assertEqual(u.rjust(len(u)), base)
        self.assertIs(u.lower().__class__, str)
        self.assertEqual(u.lower(), base)
        self.assertIs(u.upper().__class__, str)
        self.assertEqual(u.upper(), base)
        self.assertIs(u.capitalize().__class__, str)
        self.assertEqual(u.capitalize(), base)
        self.assertIs(u.title().__class__, str)
        self.assertEqual(u.title(), base)
        self.assertIs((u + "").__class__, str)
        self.assertEqual(u + "", base)
        self.assertIs(("" + u).__class__, str)
        self.assertEqual("" + u, base)
        self.assertIs((u * 0).__class__, str)
        self.assertEqual(u * 0, "")
        self.assertIs((u * 1).__class__, str)
        self.assertEqual(u * 1, base)
        self.assertIs((u * 2).__class__, str)
        self.assertEqual(u * 2, base + base)
        self.assertIs(u[:].__class__, str)
        self.assertEqual(u[:], base)
        self.assertIs(u[0:0].__class__, str)
        self.assertEqual(u[0:0], "")

        with torch._dynamo.error_on_graph_break(False):
            class sublist(list):
                pass
        a = sublist(range(5))
        self.assertEqual(a, list(range(5)))
        a.append("hello")
        self.assertEqual(a, list(range(5)) + ["hello"])
        a[5] = 5
        self.assertEqual(a, list(range(6)))
        a.extend(range(6, 20))
        self.assertEqual(a, list(range(20)))
        a[-5:] = []
        self.assertEqual(a, list(range(15)))
        del a[10:15]
        self.assertEqual(len(a), 10)
        self.assertEqual(a, list(range(10)))
        self.assertEqual(list(a), list(range(10)))
        self.assertEqual(a[0], 0)
        self.assertEqual(a[9], 9)
        self.assertEqual(a[-10], 0)
        self.assertEqual(a[-1], 9)
        self.assertEqual(a[:5], list(range(5)))

        ## class CountedInput(file):
        ##    """Counts lines read by self.readline().
        ##
        ##     self.lineno is the 0-based ordinal of the last line read, up to
        ##     a maximum of one greater than the number of lines in the file.
        ##
        ##     self.ateof is true if and only if the final "" line has been read,
        ##     at which point self.lineno stops incrementing, and further calls
        ##     to readline() continue to return "".
        ##     """
        ##
        ##     lineno = 0
        ##     ateof = 0
        ##     def readline(self):
        ##         if self.ateof:
        ##             return ""
        ##         s = file.readline(self)
        ##         # Next line works too.
        ##         # s = super(CountedInput, self).readline()
        ##         self.lineno += 1
        ##         if s == "":
        ##             self.ateof = 1
        ##        return s
        ##
        ## f = file(name=os_helper.TESTFN, mode='w')
        ## lines = ['a\n', 'b\n', 'c\n']
        ## try:
        ##     f.writelines(lines)
        ##     f.close()
        ##     f = CountedInput(os_helper.TESTFN)
        ##     for (i, expected) in zip(range(1, 5) + [4], lines + 2 * [""]):
        ##         got = f.readline()
        ##         self.assertEqual(expected, got)
        ##         self.assertEqual(f.lineno, i)
        ##         self.assertEqual(f.ateof, (i > len(lines)))
        ##     f.close()
        ## finally:
        ##     try:
        ##         f.close()
        ##     except:
        ##         pass
        ##     os_helper.unlink(os_helper.TESTFN)

    def test_keywords(self):
        # Testing keyword args to basic type constructors ...
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            int(x=1)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            float(x=2)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            bool(x=2)
        self.assertEqual(complex(imag=42, real=666), complex(666, 42))
        self.assertEqual(str(object=500), '500')
        self.assertEqual(str(object=b'abc', errors='strict'), 'abc')
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            tuple(sequence=range(3))
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            list(sequence=(0, 1, 2))
        # note: as of Python 2.3, dict() no longer has an "items" keyword arg

        for constructor in (int, float, int, complex, str, str,
                            tuple, list):
            try:
                constructor(bogus_keyword_arg=1)
            except TypeError:
                pass
            else:
                self.fail("expected TypeError from bogus keyword argument to %r"
                            % constructor)

    def test_str_subclass_as_dict_key(self):
        # Testing a str subclass used as dict key ..

        with torch._dynamo.error_on_graph_break(False):
            class cistr(str):
                """Subclass of str that computes __eq__ case-insensitively.

                Also computes a hash code of the string in canonical form.
                """

                def __init__(self, value):
                    self.canonical = value.lower()
                    self.hashcode = hash(self.canonical)

                def __eq__(self, other):
                    if not isinstance(other, cistr):
                        other = cistr(other)
                    return self.canonical == other.canonical

                def __hash__(self):
                    return self.hashcode

        self.assertEqual(cistr('ABC'), 'abc')
        self.assertEqual('aBc', cistr('ABC'))
        self.assertEqual(str(cistr('ABC')), 'ABC')

        d = {cistr('one'): 1, cistr('two'): 2, cistr('tHree'): 3}
        self.assertEqual(d[cistr('one')], 1)
        self.assertEqual(d[cistr('tWo')], 2)
        self.assertEqual(d[cistr('THrEE')], 3)
        self.assertIn(cistr('ONe'), d)
        self.assertEqual(d.get(cistr('thrEE')), 3)

    def test_classic_comparisons(self):
        # Testing classic comparisons...
        with torch._dynamo.error_on_graph_break(False):
            class classic:
                pass

        for base in (classic, int, object):
            with torch._dynamo.error_on_graph_break(False):
                class C(base):
                    def __init__(self, value):
                        self.value = int(value)
                    def __eq__(self, other):
                        if isinstance(other, C):
                            return self.value == other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value == other
                        return NotImplemented
                    def __ne__(self, other):
                        if isinstance(other, C):
                            return self.value != other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value != other
                        return NotImplemented
                    def __lt__(self, other):
                        if isinstance(other, C):
                            return self.value < other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value < other
                        return NotImplemented
                    def __le__(self, other):
                        if isinstance(other, C):
                            return self.value <= other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value <= other
                        return NotImplemented
                    def __gt__(self, other):
                        if isinstance(other, C):
                            return self.value > other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value > other
                        return NotImplemented
                    def __ge__(self, other):
                        if isinstance(other, C):
                            return self.value >= other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value >= other
                        return NotImplemented

            c1 = C(1)
            c2 = C(2)
            c3 = C(3)
            self.assertEqual(c1, 1)
            c = {1: c1, 2: c2, 3: c3}
            for x in 1, 2, 3:
                for y in 1, 2, 3:
                    for op in "<", "<=", "==", "!=", ">", ">=":
                        self.assertEqual(eval("c[x] %s c[y]" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("c[x] %s y" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("x %s c[y]" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))

    def test_rich_comparisons(self):
        # Testing rich comparisons...
        with torch._dynamo.error_on_graph_break(False):
            class Z(complex):
                pass
        z = Z(1)
        self.assertEqual(z, 1+0j)
        self.assertEqual(1+0j, z)
        with torch._dynamo.error_on_graph_break(False):
            class ZZ(complex):
                def __eq__(self, other):
                    try:
                        return abs(self - other) <= 1e-6
                    except:
                        return NotImplemented
        zz = ZZ(1.0000003)
        self.assertEqual(zz, 1+0j)
        self.assertEqual(1+0j, zz)

        with torch._dynamo.error_on_graph_break(False):
            class classic:
                pass
        for base in (classic, int, object, list):
            with torch._dynamo.error_on_graph_break(False):
                class C(base):
                    def __init__(self, value):
                        self.value = int(value)
                    def __cmp__(self_, other):
                        self.fail("shouldn't call __cmp__")
                    def __eq__(self, other):
                        if isinstance(other, C):
                            return self.value == other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value == other
                        return NotImplemented
                    def __ne__(self, other):
                        if isinstance(other, C):
                            return self.value != other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value != other
                        return NotImplemented
                    def __lt__(self, other):
                        if isinstance(other, C):
                            return self.value < other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value < other
                        return NotImplemented
                    def __le__(self, other):
                        if isinstance(other, C):
                            return self.value <= other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value <= other
                        return NotImplemented
                    def __gt__(self, other):
                        if isinstance(other, C):
                            return self.value > other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value > other
                        return NotImplemented
                    def __ge__(self, other):
                        if isinstance(other, C):
                            return self.value >= other.value
                        if isinstance(other, int) or isinstance(other, int):
                            return self.value >= other
                        return NotImplemented
            c1 = C(1)
            c2 = C(2)
            c3 = C(3)
            self.assertEqual(c1, 1)
            c = {1: c1, 2: c2, 3: c3}
            for x in 1, 2, 3:
                for y in 1, 2, 3:
                    for op in "<", "<=", "==", "!=", ">", ">=":
                        self.assertEqual(eval("c[x] %s c[y]" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("c[x] %s y" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("x %s c[y]" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))

    def test_descrdoc(self):
        # Testing descriptor doc strings...
        from _io import FileIO
        def check(descr, what):
            self.assertEqual(descr.__doc__, what)
        check(FileIO.closed, "True if the file is closed") # getset descriptor
        check(complex.real, "the real part of a complex number") # member descriptor

    def test_doc_descriptor(self):
        # Testing __doc__ descriptor...
        # SF bug 542984
        with torch._dynamo.error_on_graph_break(False):
            class DocDescr(object):
                def __get__(self, object, otype):
                    if object:
                        object = object.__class__.__name__ + ' instance'
                    if otype:
                        otype = otype.__name__
                    return 'object=%s; type=%s' % (object, otype)
            class NewClass:
                __doc__ = DocDescr()
        self.assertEqual(NewClass.__doc__, 'object=None; type=NewClass')
        self.assertEqual(NewClass().__doc__, 'object=NewClass instance; type=NewClass')

    def test_set_class(self):
        # Testing __class__ assignment...
        with torch._dynamo.error_on_graph_break(False):
            class C(object): pass
            class D(object): pass
            class E(object): pass
            class F(D, E): pass
        for cls in C, D, E, F:
            for cls2 in C, D, E, F:
                x = cls()
                x.__class__ = cls2
                self.assertIs(x.__class__, cls2)
                x.__class__ = cls
                self.assertIs(x.__class__, cls)
        def cant(x, C):
            try:
                x.__class__ = C
            except TypeError:
                pass
            else:
                self.fail("shouldn't allow %r.__class__ = %r" % (x, C))
            try:
                delattr(x, "__class__")
            except (TypeError, AttributeError):
                pass
            else:
                self.fail("shouldn't allow del %r.__class__" % x)
        cant(C(), list)
        cant(list(), C)
        cant(C(), 1)
        cant(C(), object)
        cant(object(), list)
        cant(list(), object)
        with torch._dynamo.error_on_graph_break(False):
            class Int(int): __slots__ = []
        cant(True, int)
        cant(2, bool)
        o = object()
        cant(o, int)
        cant(o, type(None))
        del o
        with torch._dynamo.error_on_graph_break(False):
            class G(object):
                __slots__ = ["a", "b"]
            class H(object):
                __slots__ = ["b", "a"]
            class I(object):
                __slots__ = ["a", "b"]
            class J(object):
                __slots__ = ["c", "b"]
            class K(object):
                __slots__ = ["a", "b", "d"]
            class L(H):
                __slots__ = ["e"]
            class M(I):
                __slots__ = ["e"]
            class N(J):
                __slots__ = ["__weakref__"]
            class P(J):
                __slots__ = ["__dict__"]
            class Q(J):
                pass
            class R(J):
                __slots__ = ["__dict__", "__weakref__"]

        for cls, cls2 in ((G, H), (G, I), (I, H), (Q, R), (R, Q)):
            x = cls()
            x.a = 1
            x.__class__ = cls2
            self.assertIs(x.__class__, cls2,
                   "assigning %r as __class__ for %r silently failed" % (cls2, x))
            self.assertEqual(x.a, 1)
            x.__class__ = cls
            self.assertIs(x.__class__, cls,
                   "assigning %r as __class__ for %r silently failed" % (cls, x))
            self.assertEqual(x.a, 1)
        for cls in G, J, K, L, M, N, P, R, list, Int:
            for cls2 in G, J, K, L, M, N, P, R, list, Int:
                if cls is cls2:
                    continue
                cant(cls(), cls2)

        # Issue5283: when __class__ changes in __del__, the wrong
        # type gets DECREF'd.
        with torch._dynamo.error_on_graph_break(False):
            class O(object):
                pass
            class A(object):
                def __del__(self):
                    self.__class__ = O
        l = [A() for x in range(100)]
        del l

    def test_set_dict(self):
        # Testing __dict__ assignment...
        with torch._dynamo.error_on_graph_break(False):
            class C(object): pass
        a = C()
        a.__dict__ = {'b': 1}
        self.assertEqual(a.b, 1)
        def cant(x, dict):
            try:
                x.__dict__ = dict
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("shouldn't allow %r.__dict__ = %r" % (x, dict))
        cant(a, None)
        cant(a, [])
        cant(a, 1)
        del a.__dict__ # Deleting __dict__ is allowed

        with torch._dynamo.error_on_graph_break(False):
            class Base(object):
                pass
        def verify_dict_readonly(x):
            """
            x has to be an instance of a class inheriting from Base.
            """
            cant(x, {})
            try:
                del x.__dict__
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("shouldn't allow del %r.__dict__" % x)
            dict_descr = Base.__dict__["__dict__"]
            try:
                dict_descr.__set__(x, {})
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("dict_descr allowed access to %r's dict" % x)

        # Classes don't allow __dict__ assignment and have readonly dicts
        with torch._dynamo.error_on_graph_break(False):
            class Meta1(type, Base):
                pass
            class Meta2(Base, type):
                pass
            class D(object, metaclass=Meta1):
                pass
            class E(object, metaclass=Meta2):
                pass
        for cls in C, D, E:
            verify_dict_readonly(cls)
            class_dict = cls.__dict__
            try:
                class_dict["spam"] = "eggs"
            except TypeError:
                pass
            else:
                self.fail("%r's __dict__ can be modified" % cls)

        # Modules also disallow __dict__ assignment
        with torch._dynamo.error_on_graph_break(False):
            class Module1(types.ModuleType, Base):
                pass
            class Module2(Base, types.ModuleType):
                pass
        for ModuleType in Module1, Module2:
            mod = ModuleType("spam")
            verify_dict_readonly(mod)
            mod.__dict__["spam"] = "eggs"

        # Exception's __dict__ can be replaced, but not deleted
        # (at least not any more than regular exception's __dict__ can
        # be deleted; on CPython it is not the case, whereas on PyPy they
        # can, just like any other new-style instance's __dict__.)
        def can_delete_dict(e):
            try:
                del e.__dict__
            except (TypeError, AttributeError):
                return False
            else:
                return True
        with torch._dynamo.error_on_graph_break(False):
            class Exception1(Exception, Base):
                pass
            class Exception2(Base, Exception):
                pass
        for ExceptionType in Exception, Exception1, Exception2:
            e = ExceptionType()
            e.__dict__ = {"a": 1}
            self.assertEqual(e.a, 1)
            self.assertEqual(can_delete_dict(e), can_delete_dict(ValueError()))

    def test_binary_operator_override(self):
        # Testing overrides of binary operations...
        with torch._dynamo.error_on_graph_break(False):
            class I(int):
                def __repr__(self):
                    return "I(%r)" % int(self)
                def __add__(self, other):
                    return I(int(self) + int(other))
                __radd__ = __add__
                def __pow__(self, other, mod=None):
                    if mod is None:
                        return I(pow(int(self), int(other)))
                    else:
                        return I(pow(int(self), int(other), int(mod)))
                def __rpow__(self, other, mod=None):
                    if mod is None:
                        return I(pow(int(other), int(self), mod))
                    else:
                        return I(pow(int(other), int(self), int(mod)))

        self.assertEqual(repr(I(1) + I(2)), "I(3)")
        self.assertEqual(repr(I(1) + 2), "I(3)")
        self.assertEqual(repr(1 + I(2)), "I(3)")
        self.assertEqual(repr(I(2) ** I(3)), "I(8)")
        self.assertEqual(repr(2 ** I(3)), "I(8)")
        self.assertEqual(repr(I(2) ** 3), "I(8)")
        self.assertEqual(repr(pow(I(2), I(3), I(5))), "I(3)")
        with torch._dynamo.error_on_graph_break(False):
            class S(str):
                def __eq__(self, other):
                    return self.lower() == other.lower()

    def test_subclass_propagation(self):
        # Testing propagation of slot functions to subclasses...
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                pass
            class B(A):
                pass
            class C(A):
                pass
            class D(B, C):
                pass
        d = D()
        orig_hash = hash(d) # related to id(d) in platform-dependent ways
        A.__hash__ = lambda self: 42
        self.assertEqual(hash(d), 42)
        C.__hash__ = lambda self: 314
        self.assertEqual(hash(d), 314)
        B.__hash__ = lambda self: 144
        self.assertEqual(hash(d), 144)
        D.__hash__ = lambda self: 100
        self.assertEqual(hash(d), 100)
        D.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del D.__hash__
        self.assertEqual(hash(d), 144)
        B.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del B.__hash__
        self.assertEqual(hash(d), 314)
        C.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del C.__hash__
        self.assertEqual(hash(d), 42)
        A.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del A.__hash__
        self.assertEqual(hash(d), orig_hash)
        d.foo = 42
        d.bar = 42
        self.assertEqual(d.foo, 42)
        self.assertEqual(d.bar, 42)
        def __getattribute__(self, name):
            if name == "foo":
                return 24
            return object.__getattribute__(self, name)
        A.__getattribute__ = __getattribute__
        self.assertEqual(d.foo, 24)
        self.assertEqual(d.bar, 42)
        def __getattr__(self, name):
            if name in ("spam", "foo", "bar"):
                return "hello"
            raise AttributeError(name)
        B.__getattr__ = __getattr__
        self.assertEqual(d.spam, "hello")
        self.assertEqual(d.foo, 24)
        self.assertEqual(d.bar, 42)
        del A.__getattribute__
        self.assertEqual(d.foo, 42)
        del d.foo
        self.assertEqual(d.foo, "hello")
        self.assertEqual(d.bar, 42)
        del B.__getattr__
        try:
            d.foo
        except AttributeError:
            pass
        else:
            self.fail("d.foo should be undefined now")

        # Test a nasty bug in recurse_down_subclasses()
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                pass
            class B(A):
                pass
        del B
        support.gc_collect()
        A.__setitem__ = lambda *a: None # crash

    def test_buffer_inheritance(self):
        # Testing that buffer interface is inherited ...

        import binascii
        # SF bug [#470040] ParseTuple t# vs subclasses.

        with torch._dynamo.error_on_graph_break(False):
            class MyBytes(bytes):
                pass
        base = b'abc'
        m = MyBytes(base)
        # b2a_hex uses the buffer interface to get its argument's value, via
        # PyArg_ParseTuple 't#' code.
        self.assertEqual(binascii.b2a_hex(m), binascii.b2a_hex(base))

        with torch._dynamo.error_on_graph_break(False):
            class MyInt(int):
                pass
        m = MyInt(42)
        try:
            binascii.b2a_hex(m)
            self.fail('subclass of int should not have a buffer interface')
        except TypeError:
            pass

    def test_str_of_str_subclass(self):
        # Testing __str__ defined in subclass of str ...
        import binascii

        with torch._dynamo.error_on_graph_break(False):
            class octetstring(str):
                def __str__(self):
                    return binascii.b2a_hex(self.encode('ascii')).decode("ascii")
                def __repr__(self):
                    return self + " repr"

        o = octetstring('A')
        self.assertEqual(type(o), octetstring)
        self.assertEqual(type(str(o)), str)
        self.assertEqual(type(repr(o)), str)
        self.assertEqual(ord(o), 0x41)
        self.assertEqual(str(o), '41')
        self.assertEqual(repr(o), 'A repr')
        self.assertEqual(o.__str__(), '41')
        self.assertEqual(o.__repr__(), 'A repr')

    def test_repr_with_module_str_subclass(self):
        # gh-98783
        with torch._dynamo.error_on_graph_break(False):
            class StrSub(str):
                pass
            class Some:
                pass
        Some.__module__ = StrSub('example')
        self.assertIsInstance(repr(Some), str)  # should not crash
        self.assertIsInstance(repr(Some()), str)  # should not crash

    def test_keyword_arguments(self):
        # Testing keyword arguments to __init__, __call__...
        def f(a): return a
        self.assertEqual(f.__call__(a=42), 42)
        ba = bytearray()
        bytearray.__init__(ba, 'abc\xbd\u20ac',
                           encoding='latin1', errors='replace')
        self.assertEqual(ba, b'abc\xbd?')

    def test_recursive_call(self):
        # Testing recursive __call__() by setting to instance of class...
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                pass

        A.__call__ = A()
        with self.assertRaises(RecursionError):
            A()()

    def test_delete_hook(self):
        # Testing __del__ hook...
        log = []
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __del__(self):
                    log.append(1)
        c = C()
        self.assertEqual(log, [])
        del c
        support.gc_collect()
        self.assertEqual(log, [1])

        with torch._dynamo.error_on_graph_break(False):
            class D(object): pass
        d = D()
        try: del d[0]
        except TypeError: pass
        else: self.fail("invalid del() didn't raise TypeError")

    def test_hash_inheritance(self):
        # Testing hash of mutable subclasses...

        with torch._dynamo.error_on_graph_break(False):
            class mydict(dict):
                pass
        d = mydict()
        try:
            hash(d)
        except TypeError:
            pass
        else:
            self.fail("hash() of dict subclass should fail")

        with torch._dynamo.error_on_graph_break(False):
            class mylist(list):
                pass
        d = mylist()
        try:
            hash(d)
        except TypeError:
            pass
        else:
            self.fail("hash() of list subclass should fail")

    def test_str_operations(self):
        try: 'a' + 5
        except TypeError: pass
        else: self.fail("'' + 5 doesn't raise TypeError")

        try: ''.split('')
        except ValueError: pass
        else: self.fail("''.split('') doesn't raise ValueError")

        try: ''.join([0])
        except TypeError: pass
        else: self.fail("''.join([0]) doesn't raise TypeError")

        try: ''.rindex('5')
        except ValueError: pass
        else: self.fail("''.rindex('5') doesn't raise ValueError")

        try: '%(n)s' % None
        except TypeError: pass
        else: self.fail("'%(n)s' % None doesn't raise TypeError")

        try: '%(n' % {}
        except ValueError: pass
        else: self.fail("'%(n' % {} '' doesn't raise ValueError")

        try: '%*s' % ('abc')
        except TypeError: pass
        else: self.fail("'%*s' % ('abc') doesn't raise TypeError")

        try: '%*.*s' % ('abc', 5)
        except TypeError: pass
        else: self.fail("'%*.*s' % ('abc', 5) doesn't raise TypeError")

        try: '%s' % (1, 2)
        except TypeError: pass
        else: self.fail("'%s' % (1, 2) doesn't raise TypeError")

        try: '%' % None
        except ValueError: pass
        else: self.fail("'%' % None doesn't raise ValueError")

        self.assertEqual('534253'.isdigit(), 1)
        self.assertEqual('534253x'.isdigit(), 0)
        self.assertEqual('%c' % 5, '\x05')
        self.assertEqual('%c' % '5', '5')

    def test_deepcopy_recursive(self):
        # Testing deepcopy of recursive objects...
        with torch._dynamo.error_on_graph_break(False):
            class Node:
                pass
        a = Node()
        b = Node()
        a.b = b
        b.a = a
        z = deepcopy(a) # This blew up before

    def test_uninitialized_modules(self):
        # Testing uninitialized module objects...
        from types import ModuleType as M
        m = M.__new__(M)
        str(m)
        self.assertNotHasAttr(m, "__name__")
        self.assertNotHasAttr(m, "__file__")
        self.assertNotHasAttr(m, "foo")
        self.assertFalse(m.__dict__)   # None or {} are both reasonable answers
        m.foo = 1
        self.assertEqual(m.__dict__, {"foo": 1})

    def test_funny_new(self):
        # Testing __new__ returning something unexpected...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __new__(cls, arg):
                    if isinstance(arg, str): return [1, 2, 3]
                    elif isinstance(arg, int): return object.__new__(D)
                    else: return object.__new__(cls)
            class D(C):
                def __init__(self, arg):
                    self.foo = arg
        self.assertEqual(C("1"), [1, 2, 3])
        self.assertEqual(D("1"), [1, 2, 3])
        d = D(None)
        self.assertEqual(d.foo, None)
        d = C(1)
        self.assertIsInstance(d, D)
        self.assertEqual(d.foo, 1)
        d = D(1)
        self.assertIsInstance(d, D)
        self.assertEqual(d.foo, 1)

        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                @staticmethod
                def __new__(*args):
                    return args
        self.assertEqual(C(1, 2), (C, 1, 2))
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        self.assertEqual(D(1, 2), (D, 1, 2))

        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                @classmethod
                def __new__(*args):
                    return args
        self.assertEqual(C(1, 2), (C, C, 1, 2))
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
        self.assertEqual(D(1, 2), (D, D, 1, 2))

    def test_imul_bug(self):
        # Testing for __imul__ problems...
        # SF bug 544647
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __imul__(self, other):
                    return (self, other)
        x = C()
        y = x
        y *= 1.0
        self.assertEqual(y, (x, 1.0))
        y = x
        y *= 2
        self.assertEqual(y, (x, 2))
        y = x
        y *= 3
        self.assertEqual(y, (x, 3))
        y = x
        y *= 1<<100
        self.assertEqual(y, (x, 1<<100))
        y = x
        y *= None
        self.assertEqual(y, (x, None))
        y = x
        y *= "foo"
        self.assertEqual(y, (x, "foo"))

    def test_copy_setstate(self):
        # Testing that copy.*copy() correctly uses __setstate__...
        import copy
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __init__(self, foo=None):
                    self.foo = foo
                    self.__foo = foo
                def setfoo(self, foo=None):
                    self.foo = foo
                def getfoo(self):
                    return self.__foo
                def __getstate__(self):
                    return [self.foo]
                def __setstate__(self_, lst):
                    self.assertEqual(len(lst), 1)
                    self_.__foo = self_.foo = lst[0]
        a = C(42)
        a.setfoo(24)
        self.assertEqual(a.foo, 24)
        self.assertEqual(a.getfoo(), 42)
        b = copy.copy(a)
        self.assertEqual(b.foo, 24)
        self.assertEqual(b.getfoo(), 24)
        b = copy.deepcopy(a)
        self.assertEqual(b.foo, 24)
        self.assertEqual(b.getfoo(), 24)

    def test_slices(self):
        # Testing cases with slices and overridden __getitem__ ...

        # Strings
        self.assertEqual("hello"[:4], "hell")
        self.assertEqual("hello"[slice(4)], "hell")
        self.assertEqual(str.__getitem__("hello", slice(4)), "hell")
        with torch._dynamo.error_on_graph_break(False):
            class S(str):
                def __getitem__(self, x):
                    return str.__getitem__(self, x)
        self.assertEqual(S("hello")[:4], "hell")
        self.assertEqual(S("hello")[slice(4)], "hell")
        self.assertEqual(S("hello").__getitem__(slice(4)), "hell")
        # Tuples
        self.assertEqual((1,2,3)[:2], (1,2))
        self.assertEqual((1,2,3)[slice(2)], (1,2))
        self.assertEqual(tuple.__getitem__((1,2,3), slice(2)), (1,2))
        with torch._dynamo.error_on_graph_break(False):
            class T(tuple):
                def __getitem__(self, x):
                    return tuple.__getitem__(self, x)
        self.assertEqual(T((1,2,3))[:2], (1,2))
        self.assertEqual(T((1,2,3))[slice(2)], (1,2))
        self.assertEqual(T((1,2,3)).__getitem__(slice(2)), (1,2))
        # Lists
        self.assertEqual([1,2,3][:2], [1,2])
        self.assertEqual([1,2,3][slice(2)], [1,2])
        self.assertEqual(list.__getitem__([1,2,3], slice(2)), [1,2])
        with torch._dynamo.error_on_graph_break(False):
            class L(list):
                def __getitem__(self, x):
                    return list.__getitem__(self, x)
        self.assertEqual(L([1,2,3])[:2], [1,2])
        self.assertEqual(L([1,2,3])[slice(2)], [1,2])
        self.assertEqual(L([1,2,3]).__getitem__(slice(2)), [1,2])
        # Now do lists and __setitem__
        a = L([1,2,3])
        a[slice(1, 3)] = [3,2]
        self.assertEqual(a, [1,3,2])
        a[slice(0, 2, 1)] = [3,1]
        self.assertEqual(a, [3,1,2])
        a.__setitem__(slice(1, 3), [2,1])
        self.assertEqual(a, [3,2,1])
        a.__setitem__(slice(0, 2, 1), [2,3])
        self.assertEqual(a, [2,3,1])

    def test_subtype_resurrection(self):
        # Testing resurrection of new-style instance...

        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                container = []

                def __del__(self):
                    # resurrect the instance
                    C.container.append(self)

            c = C()
        c.attr = 42

        # The most interesting thing here is whether this blows up, due to
        # flawed GC tracking logic in typeobject.c's call_finalizer() (a 2.2.1
        # bug).
        del c

        support.gc_collect()
        self.assertEqual(len(C.container), 1)

        # Make c mortal again, so that the test framework with -l doesn't report
        # it as a leak.
        del C.__del__

    def test_slots_trash(self):
        # Testing slot trash...
        # Deallocating deeply nested slotted trash caused stack overflows
        class trash(object):
            __slots__ = ['x']
            def __init__(self, x):
                self.x = x
        o = None
        for i in range(50000):
            o = trash(o)
        del o

    def test_slots_multiple_inheritance(self):
        # SF bug 575229, multiple inheritance w/ slots dumps core
        with torch._dynamo.error_on_graph_break(False):
            class A(object):
                __slots__=()
            class B(object):
                pass
            class C(A,B) :
                __slots__=()
        if support.check_impl_detail():
            self.assertEqual(C.__basicsize__, B.__basicsize__)
        self.assertHasAttr(C, '__dict__')
        self.assertHasAttr(C, '__weakref__')
        C().x = 2

    def test_rmul(self):
        # Testing correct invocation of __rmul__...
        # SF patch 592646
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __mul__(self, other):
                    return "mul"
                def __rmul__(self, other):
                    return "rmul"
        a = C()
        self.assertEqual(a*2, "mul")
        self.assertEqual(a*2.2, "mul")
        self.assertEqual(2*a, "rmul")
        self.assertEqual(2.2*a, "rmul")

    def test_ipow(self):
        # Testing correct invocation of __ipow__...
        # [SF bug 620179]
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __ipow__(self, other):
                    pass
        a = C()
        a **= 2

    def test_ipow_returns_not_implemented(self):
        with torch._dynamo.error_on_graph_break(False):
            class A:
                def __ipow__(self, other):
                    return NotImplemented

            class B(A):
                def __rpow__(self, other):
                    return 1

            class C(A):
                def __pow__(self, other):
                    return 2
        a = A()
        b = B()
        c = C()

        a **= b
        self.assertEqual(a, 1)

        c **= b
        self.assertEqual(c, 2)

    def test_no_ipow(self):
        with torch._dynamo.error_on_graph_break(False):
            class B:
                def __rpow__(self, other):
                    return 1

        a = object()
        b = B()
        a **= b
        self.assertEqual(a, 1)

    def test_ipow_exception_text(self):
        x = None
        with self.assertRaises(TypeError) as cm:
            x **= 2
        self.assertIn('unsupported operand type(s) for **=', str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            y = x ** 2
        self.assertIn('unsupported operand type(s) for **', str(cm.exception))

    def test_mutable_bases(self):
        # Testing mutable bases...

        # stuff that should work:
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                pass
            class C2(object):
                def __getattribute__(self, attr):
                    if attr == 'a':
                        return 2
                    else:
                        return super(C2, self).__getattribute__(attr)
                def meth(self):
                    return 1
            class D(C):
                pass
            class E(D):
                pass
        d = D()
        e = E()
        D.__bases__ = (C,)
        D.__bases__ = (C2,)
        self.assertEqual(d.meth(), 1)
        self.assertEqual(e.meth(), 1)
        self.assertEqual(d.a, 2)
        self.assertEqual(e.a, 2)
        self.assertEqual(C2.__subclasses__(), [D])

        try:
            del D.__bases__
        except (TypeError, AttributeError):
            pass
        else:
            self.fail("shouldn't be able to delete .__bases__")

        try:
            D.__bases__ = ()
        except TypeError as msg:
            if str(msg) == "a new-style class can't have only classic bases":
                self.fail("wrong error message for .__bases__ = ()")
        else:
            self.fail("shouldn't be able to set .__bases__ to ()")

        try:
            D.__bases__ = (D,)
        except TypeError:
            pass
        else:
            # actually, we'll have crashed by here...
            self.fail("shouldn't be able to create inheritance cycles")

        try:
            D.__bases__ = (C, C)
        except TypeError:
            pass
        else:
            self.fail("didn't detect repeated base classes")

        try:
            D.__bases__ = (E,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't be able to create inheritance cycles")

    def test_builtin_bases(self):
        # Make sure all the builtin types can have their base queried without
        # segfaulting. See issue #5787.
        builtin_types = [tp for tp in builtins.__dict__.values()
                         if isinstance(tp, type)]
        for tp in builtin_types:
            object.__getattribute__(tp, "__bases__")
            if tp is not object:
                if tp is ExceptionGroup:
                    num_bases = 2
                else:
                    num_bases = 1
                self.assertEqual(len(tp.__bases__), num_bases, tp)

        with torch._dynamo.error_on_graph_break(False):
            class L(list):
                pass

            class C(object):
                pass

            class D(C):
                pass

        try:
            L.__bases__ = (dict,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't turn list subclass into dict subclass")

        try:
            list.__bases__ = (dict,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't be able to assign to list.__bases__")

        try:
            D.__bases__ = (C, list)
        except TypeError:
            pass
        else:
            self.fail("best_base calculation found wanting")

    def test_unsubclassable_types(self):
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class X(type(None)):
                    pass
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class X(object, type(None)):
                    pass
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class X(type(None), object):
                    pass
        with torch._dynamo.error_on_graph_break(False):
            class O(object):
                pass
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class X(O, type(None)):
                    pass
        with self.assertRaises(TypeError):
            with torch._dynamo.error_on_graph_break(False):
                class X(type(None), O):
                    pass

        with torch._dynamo.error_on_graph_break(False):
            class X(object):
                pass
        with self.assertRaises(TypeError):
            X.__bases__ = type(None),
        with self.assertRaises(TypeError):
            X.__bases__ = object, type(None)
        with self.assertRaises(TypeError):
            X.__bases__ = type(None), object
        with self.assertRaises(TypeError):
            X.__bases__ = O, type(None)
        with self.assertRaises(TypeError):
            X.__bases__ = type(None), O

    def test_mutable_bases_with_failing_mro(self):
        with torch._dynamo.error_on_graph_break(False):
            # Testing mutable bases with failing mro...
            class WorkOnce(type):
                def __new__(self, name, bases, ns):
                    self.flag = 0
                    return super(WorkOnce, self).__new__(WorkOnce, name, bases, ns)
                def mro(self):
                    if self.flag > 0:
                        raise RuntimeError("bozo")
                    else:
                        self.flag += 1
                        return type.mro(self)

            class WorkAlways(type):
                def mro(self):
                    # this is here to make sure that .mro()s aren't called
                    # with an exception set (which was possible at one point).
                    # An error message will be printed in a debug build.
                    # What's a good way to test for this?
                    return type.mro(self)

            class C(object):
                pass

            class C2(object):
                pass

            class D(C):
                pass

            class E(D):
                pass

            class F(D, metaclass=WorkOnce):
                pass

            class G(D, metaclass=WorkAlways):
                pass

        # Immediate subclasses have their mro's adjusted in alphabetical
        # order, so E's will get adjusted before adjusting F's fails.  We
        # check here that E's gets restored.

        E_mro_before = E.__mro__
        D_mro_before = D.__mro__

        try:
            D.__bases__ = (C2,)
        except RuntimeError:
            self.assertEqual(E.__mro__, E_mro_before)
            self.assertEqual(D.__mro__, D_mro_before)
        else:
            self.fail("exception not propagated")

    def test_mutable_bases_catch_mro_conflict(self):
        with torch._dynamo.error_on_graph_break(False):
            # Testing mutable bases catch mro conflict...
            class A(object):
                pass

            class B(object):
                pass

            class C(A, B):
                pass

            class D(A, B):
                pass

            class E(C, D):
                pass

        try:
            C.__bases__ = (B, A)
        except TypeError:
            pass
        else:
            self.fail("didn't catch MRO conflict")

    def test_mutable_names(self):
        # Testing mutable names...
        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                pass

        # C.__module__ could be 'test_descr' or '__main__'
        mod = C.__module__

        C.__name__ = 'D'
        self.assertEqual((C.__module__, C.__name__), (mod, 'D'))

        C.__name__ = 'D.E'
        self.assertEqual((C.__module__, C.__name__), (mod, 'D.E'))

    def test_evil_type_name(self):
        # A badly placed Py_DECREF in type_set_name led to arbitrary code
        # execution while the type structure was not in a sane state, and a
        # possible segmentation fault as a result.  See bug #16447.
        class Nasty(str):
            def __del__(self):
                C.__name__ = "other"

        class C:
            pass

        C.__name__ = Nasty("abc")
        C.__name__ = "normal"

    def test_subclass_right_op(self):
        # Testing correct dispatch of subclass overloading __r<op>__...

        # This code tests various cases where right-dispatch of a subclass
        # should be preferred over left-dispatch of a base class.

        # Case 1: subclass of int; this tests code in abstract.c::binary_op1()

        class B(int):
            def __floordiv__(self, other):
                return "B.__floordiv__"
            def __rfloordiv__(self, other):
                return "B.__rfloordiv__"

        self.assertEqual(B(1) // 1, "B.__floordiv__")
        self.assertEqual(1 // B(1), "B.__rfloordiv__")

        # Case 2: subclass of object; this is just the baseline for case 3

        class C(object):
            def __floordiv__(self, other):
                return "C.__floordiv__"
            def __rfloordiv__(self, other):
                return "C.__rfloordiv__"

        self.assertEqual(C() // 1, "C.__floordiv__")
        self.assertEqual(1 // C(), "C.__rfloordiv__")

        # Case 3: subclass of new-style class; here it gets interesting

        class D(C):
            def __floordiv__(self, other):
                return "D.__floordiv__"
            def __rfloordiv__(self, other):
                return "D.__rfloordiv__"

        self.assertEqual(D() // C(), "D.__floordiv__")
        self.assertEqual(C() // D(), "D.__rfloordiv__")

        # Case 4: this didn't work right in 2.2.2 and 2.3a1

        class E(C):
            pass

        self.assertEqual(E.__rfloordiv__, C.__rfloordiv__)

        self.assertEqual(E() // 1, "C.__floordiv__")
        self.assertEqual(1 // E(), "C.__rfloordiv__")
        self.assertEqual(E() // C(), "C.__floordiv__")
        self.assertEqual(C() // E(), "C.__floordiv__") # This one would fail

    @support.impl_detail("testing an internal kind of method object")
    def test_meth_class_get(self):
        # Testing __get__ method of METH_CLASS C methods...
        # Full coverage of descrobject.c::classmethod_get()

        # Baseline
        arg = [1, 2, 3]
        res = {1: None, 2: None, 3: None}
        self.assertEqual(dict.fromkeys(arg), res)
        self.assertEqual({}.fromkeys(arg), res)

        # Now get the descriptor
        descr = dict.__dict__["fromkeys"]

        # More baseline using the descriptor directly
        self.assertEqual(descr.__get__(None, dict)(arg), res)
        self.assertEqual(descr.__get__({})(arg), res)

        # Now check various error cases
        try:
            descr.__get__(None, None)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, None)")
        try:
            descr.__get__(42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(42)")
        try:
            descr.__get__(None, 42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, 42)")
        try:
            descr.__get__(None, int)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, int)")

    def test_isinst_isclass(self):
        with torch._dynamo.error_on_graph_break(False):
            # Testing proxy isinstance() and isclass()...
            class Proxy(object):
                def __init__(self, obj):
                    self.__obj = obj
                def __getattribute__(self, name):
                    if name.startswith("_Proxy__"):
                        return object.__getattribute__(self, name)
                    else:
                        return getattr(self.__obj, name)
            # Test with a classic class
            class C:
                pass
            a = C()
            pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        with torch._dynamo.error_on_graph_break(False):
            # Test with a classic subclass
            class D(C):
                pass
            a = D()
            pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        with torch._dynamo.error_on_graph_break(False):
            # Test with a new-style class
            class C(object):
                pass
            a = C()
            pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        # Test with a new-style subclass
        with torch._dynamo.error_on_graph_break(False):
            class D(C):
                pass
            a = D()
            pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test

    def test_proxy_super(self):
        # Testing super() for a proxy object...
        with torch._dynamo.error_on_graph_break(False):
            class Proxy(object):
                def __init__(self, obj):
                    self.__obj = obj
                def __getattribute__(self, name):
                    if name.startswith("_Proxy__"):
                        return object.__getattribute__(self, name)
                    else:
                        return getattr(self.__obj, name)

            class B(object):
                def f(self):
                    return "B.f"

            class C(B):
                def f(self):
                    return super(C, self).f() + "->C.f"

            obj = C()
            p = Proxy(obj)
        self.assertEqual(C.__dict__["f"](p), "B.f->C.f")

    def test_carloverre(self):
        # Testing prohibition of Carlo Verre's hack...
        try:
            object.__setattr__(str, "foo", 42)
        except TypeError:
            pass
        else:
            self.fail("Carlo Verre __setattr__ succeeded!")
        try:
            object.__delattr__(str, "lower")
        except TypeError:
            pass
        else:
            self.fail("Carlo Verre __delattr__ succeeded!")

    def test_carloverre_multi_inherit_valid(self):
        with torch._dynamo.error_on_graph_break(False):
            class A(type):
                def __setattr__(cls, key, value):
                    type.__setattr__(cls, key, value)

            class B:
                pass

            class C(B, A):
                pass

            obj = C('D', (object,), {})
        try:
            obj.test = True
        except TypeError:
            self.fail("setattr through direct base types should be legal")

    def test_carloverre_multi_inherit_invalid(self):
        with torch._dynamo.error_on_graph_break(False):
            class A(type):
                def __setattr__(cls, key, value):
                    object.__setattr__(cls, key, value)  # this should fail!

            class B:
                pass

            class C(B, A):
                pass

            obj = C('D', (object,), {})
        try:
            obj.test = True
        except TypeError:
            pass
        else:
            self.fail("setattr through indirect base types should be rejected")

    def test_weakref_segfault(self):
        # Testing weakref segfault...
        # SF 742911
        import weakref

        class Provoker:
            def __init__(self, referrent):
                self.ref = weakref.ref(referrent)

            def __del__(self):
                x = self.ref()

        class Oops(object):
            pass

        o = Oops()
        o.whatever = Provoker(o)
        del o

    @support.requires_resource('cpu')
    def test_wrapper_segfault(self):
        # SF 927248: deeply nested wrappers could cause stack overflow
        f = lambda:None
        for i in range(1000000):
            f = f.__call__
        f = None

    def test_file_fault(self):
        # Testing sys.stdout is changed in getattr...
        class StdoutGuard:
            def __getattr__(self, attr):
                sys.stdout = sys.__stdout__
                raise RuntimeError(f"Premature access to sys.stdout.{attr}")

        with redirect_stdout(StdoutGuard()):
            with self.assertRaises(RuntimeError):
                print("Oops!")

    def test_vicious_descriptor_nonsense(self):
        # Testing vicious_descriptor_nonsense...

        # A potential segfault spotted by Thomas Wouters in mail to
        # python-dev 2003-04-17, turned into an example & fixed by Michael
        # Hudson just less than four months later...

        class Evil(object):
            def __hash__(self):
                return hash('attr')
            def __eq__(self, other):
                try:
                    del C.attr
                except AttributeError:
                    # possible race condition
                    pass
                return 0

        class Descr(object):
            def __get__(self, ob, type=None):
                return 1

        class C(object):
            attr = Descr()

        c = C()
        c.__dict__[Evil()] = 0

        self.assertEqual(c.attr, 1)
        # this makes a crash more likely:
        support.gc_collect()
        self.assertNotHasAttr(c, 'attr')

    def test_init(self):
        # SF 1155938
        class Foo(object):
            def __init__(self):
                return 10
        try:
            Foo()
        except TypeError:
            pass
        else:
            self.fail("did not test __init__() for None return")

    def assertNotOrderable(self, a, b):
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a >= b

    def test_method_wrapper(self):
        # Testing method-wrapper objects...
        # <type 'method-wrapper'> did not support any reflection before 2.5
        l = []
        self.assertTrue(l.__add__ == l.__add__)
        self.assertFalse(l.__add__ != l.__add__)
        self.assertFalse(l.__add__ == [].__add__)
        self.assertTrue(l.__add__ != [].__add__)
        self.assertFalse(l.__add__ == l.__mul__)
        self.assertTrue(l.__add__ != l.__mul__)
        self.assertNotOrderable(l.__add__, l.__add__)
        self.assertEqual(l.__add__.__name__, '__add__')
        self.assertIs(l.__add__.__self__, l)
        self.assertIs(l.__add__.__objclass__, list)
        self.assertEqual(l.__add__.__doc__, list.__add__.__doc__)
        # hash([].__add__) should not be based on hash([])
        hash(l.__add__)

    def test_builtin_function_or_method(self):
        # Not really belonging to test_descr, but introspection and
        # comparison on <type 'builtin_function_or_method'> seems not
        # to be tested elsewhere
        l = []
        self.assertTrue(l.append == l.append)
        self.assertFalse(l.append != l.append)
        self.assertFalse(l.append == [].append)
        self.assertTrue(l.append != [].append)
        self.assertFalse(l.append == l.pop)
        self.assertTrue(l.append != l.pop)
        self.assertNotOrderable(l.append, l.append)
        self.assertEqual(l.append.__name__, 'append')
        self.assertIs(l.append.__self__, l)
        # self.assertIs(l.append.__objclass__, list) --- could be added?
        self.assertEqual(l.append.__doc__, list.append.__doc__)
        # hash([].append) should not be based on hash([])
        hash(l.append)

    def test_special_unbound_method_types(self):
        # Testing objects of <type 'wrapper_descriptor'>...
        self.assertTrue(list.__add__ == list.__add__)
        self.assertFalse(list.__add__ != list.__add__)
        self.assertFalse(list.__add__ == list.__mul__)
        self.assertTrue(list.__add__ != list.__mul__)
        self.assertNotOrderable(list.__add__, list.__add__)
        self.assertEqual(list.__add__.__name__, '__add__')
        self.assertIs(list.__add__.__objclass__, list)

        # Testing objects of <type 'method_descriptor'>...
        self.assertTrue(list.append == list.append)
        self.assertFalse(list.append != list.append)
        self.assertFalse(list.append == list.pop)
        self.assertTrue(list.append != list.pop)
        self.assertNotOrderable(list.append, list.append)
        self.assertEqual(list.append.__name__, 'append')
        self.assertIs(list.append.__objclass__, list)

    def test_not_implemented(self):
        # Testing NotImplemented...
        # all binary methods should be able to return a NotImplemented

        def specialmethod(self, other):
            return NotImplemented

        def check(expr, x, y):
            with (
                self.subTest(expr=expr, x=x, y=y),
                self.assertRaises(TypeError),
            ):
                exec(expr, {'x': x, 'y': y})

        N1 = sys.maxsize + 1    # might trigger OverflowErrors instead of
                                # TypeErrors
        N2 = sys.maxsize         # if sizeof(int) < sizeof(long), might trigger
                                #   ValueErrors instead of TypeErrors
        for name, expr, iexpr in [
                ('__add__',      'x + y',                   'x += y'),
                ('__sub__',      'x - y',                   'x -= y'),
                ('__mul__',      'x * y',                   'x *= y'),
                ('__matmul__',   'x @ y',                   'x @= y'),
                ('__truediv__',  'x / y',                   'x /= y'),
                ('__floordiv__', 'x // y',                  'x //= y'),
                ('__mod__',      'x % y',                   'x %= y'),
                ('__divmod__',   'divmod(x, y)',            None),
                ('__pow__',      'x ** y',                  'x **= y'),
                ('__lshift__',   'x << y',                  'x <<= y'),
                ('__rshift__',   'x >> y',                  'x >>= y'),
                ('__and__',      'x & y',                   'x &= y'),
                ('__or__',       'x | y',                   'x |= y'),
                ('__xor__',      'x ^ y',                   'x ^= y')]:
            # Defines 'left' magic method:
            A = type('A', (), {name: specialmethod})
            a = A()
            check(expr, a, a)
            check(expr, a, N1)
            check(expr, a, N2)
            # Defines 'right' magic method:
            rname = '__r' + name[2:]
            B = type('B', (), {rname: specialmethod})
            b = B()
            check(expr, b, b)
            check(expr, a, b)
            check(expr, b, a)
            check(expr, b, N1)
            check(expr, b, N2)
            check(expr, N1, b)
            check(expr, N2, b)
            if iexpr:
                check(iexpr, a, a)
                check(iexpr, a, N1)
                check(iexpr, a, N2)
                iname = '__i' + name[2:]
                C = type('C', (), {iname: specialmethod})
                c = C()
                check(iexpr, c, a)
                check(iexpr, c, N1)
                check(iexpr, c, N2)

    def test_assign_slice(self):
        # ceval.c's assign_slice used to check for
        # tp->tp_as_sequence->sq_slice instead of
        # tp->tp_as_sequence->sq_ass_slice

        with torch._dynamo.error_on_graph_break(False):
            class C(object):
                def __setitem__(self, idx, value):
                    self.value = value

            c = C()
        c[1:2] = 3
        self.assertEqual(c.value, 3)

    def test_set_and_no_get(self):
        # See
        # http://mail.python.org/pipermail/python-dev/2010-January/095637.html
        with torch._dynamo.error_on_graph_break(False):
            class Descr(object):

                def __init__(self, name):
                    self.name = name

                def __set__(self, obj, value):
                    obj.__dict__[self.name] = value
            descr = Descr("a")

            class X(object):
                a = descr

        x = X()
        self.assertIs(x.a, descr)
        x.a = 42
        self.assertEqual(x.a, 42)
        with torch._dynamo.error_on_graph_break(False):

            # Also check type_getattro for correctness.
            class Meta(type):
                pass
            class X(metaclass=Meta):
                pass
        X.a = 42
        Meta.a = Descr("a")
        self.assertEqual(X.a, 42)

    def test_getattr_hooks(self):
        # issue 4230

        with torch._dynamo.error_on_graph_break(False):
            class Descriptor(object):
                counter = 0
                def __get__(self, obj, objtype=None):
                    def getter(name):
                        self.counter += 1
                        raise AttributeError(name)
                    return getter

            descr = Descriptor()
            class A(object):
                __getattribute__ = descr
            class B(object):
                __getattr__ = descr
            class C(object):
                __getattribute__ = descr
                __getattr__ = descr

        self.assertRaises(AttributeError, getattr, A(), "attr")
        self.assertEqual(descr.counter, 1)
        self.assertRaises(AttributeError, getattr, B(), "attr")
        self.assertEqual(descr.counter, 2)
        self.assertRaises(AttributeError, getattr, C(), "attr")
        self.assertEqual(descr.counter, 4)

        with torch._dynamo.error_on_graph_break(False):
            class EvilGetattribute(object):
                # This used to segfault
                def __getattr__(self, name):
                    raise AttributeError(name)
                def __getattribute__(self, name):
                    del EvilGetattribute.__getattr__
                    for i in range(5):
                        gc.collect()
                    raise AttributeError(name)

        self.assertRaises(AttributeError, getattr, EvilGetattribute(), "attr")

    def test_type___getattribute__(self):
        self.assertRaises(TypeError, type.__getattribute__, list, type)

    def test_abstractmethods(self):
        # type pretends not to have __abstractmethods__.
        self.assertRaises(AttributeError, getattr, type, "__abstractmethods__")
        with torch._dynamo.error_on_graph_break(False):
            class meta(type):
                pass
        self.assertRaises(AttributeError, getattr, meta, "__abstractmethods__")
        with torch._dynamo.error_on_graph_break(False):
            class X(object):
                pass
        with self.assertRaises(AttributeError):
            del X.__abstractmethods__

    def test_gh55664(self):
        # gh-55664: issue a warning when the
        # __dict__ of a class contains non-string keys
        with self.assertWarnsRegex(RuntimeWarning, 'MyClass'):
            MyClass = type('MyClass', (), {1: 2})

        class meta(type):
            def __new__(mcls, name, bases, ns):
                ns[1] = 2
                return super().__new__(mcls, name, bases, ns)

        with self.assertWarnsRegex(RuntimeWarning, 'MyClass'):
            MyClass = meta('MyClass', (), {})

    def test_proxy_call(self):
        with torch._dynamo.error_on_graph_break(False):
            class FakeStr:
                __class__ = str

        fake_str = FakeStr()
        # isinstance() reads __class__
        self.assertIsInstance(fake_str, str)

        # call a method descriptor
        with self.assertRaises(TypeError):
            str.split(fake_str)

        # call a slot wrapper descriptor
        with self.assertRaises(TypeError):
            str.__add__(fake_str, "abc")

    def test_specialized_method_calls_check_types(self):
        # https://github.com/python/cpython/issues/92063
        class Thing:
            pass
        thing = Thing()
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
                list.sort(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
                str.split(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_NOARGS
                str.upper(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST
                str.strip(thing)
        from collections import deque
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_O
                deque.append(thing, thing)

    def test_repr_as_str(self):
        # Issue #11603: crash or infinite loop when rebinding __str__ as
        # __repr__.
        class Foo:
            pass
        Foo.__repr__ = Foo.__str__
        foo = Foo()
        self.assertRaises(RecursionError, str, foo)
        self.assertRaises(RecursionError, repr, foo)

    def test_mixing_slot_wrappers(self):
        with torch._dynamo.error_on_graph_break(False):
            class X(dict):
                __setattr__ = dict.__setitem__
                __neg__ = dict.copy
            x = X()
        x.y = 42
        self.assertEqual(x["y"], 42)
        self.assertEqual(x, -x)

    def test_wrong_class_slot_wrapper(self):
        # Check bpo-37619: a wrapper descriptor taken from the wrong class
        # should raise an exception instead of silently being ignored
        with torch._dynamo.error_on_graph_break(False):
            class A(int):
                __eq__ = str.__eq__
                __add__ = str.__add__
            a = A()
        with self.assertRaises(TypeError):
            a == a
        with self.assertRaises(TypeError):
            a + a

    def test_slot_shadows_class_variable(self):
        with self.assertRaises(ValueError) as cm:
            class X:
                __slots__ = ["foo"]
                foo = None
        m = str(cm.exception)
        self.assertEqual("'foo' in __slots__ conflicts with class variable", m)

    def test_set_doc(self):
        with torch._dynamo.error_on_graph_break(False):
            class X:
                "elephant"
        X.__doc__ = "banana"
        self.assertEqual(X.__doc__, "banana")

        with self.assertRaises(TypeError) as cm:
            type(list).__dict__["__doc__"].__set__(list, "blah")
        self.assertIn("cannot set '__doc__' attribute of immutable type 'list'", str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            type(X).__dict__["__doc__"].__delete__(X)
        self.assertIn("cannot delete '__doc__' attribute of immutable type 'X'", str(cm.exception))
        self.assertEqual(X.__doc__, "banana")

    def test_qualname(self):
        descriptors = [str.lower, complex.real, float.real, int.__add__]
        types = ['method', 'member', 'getset', 'wrapper']

        # make sure we have an example of each type of descriptor
        for d, n in zip(descriptors, types):
            self.assertEqual(type(d).__name__, n + '_descriptor')

        for d in descriptors:
            qualname = d.__objclass__.__qualname__ + '.' + d.__name__
            self.assertEqual(d.__qualname__, qualname)

        self.assertEqual(str.lower.__qualname__, 'str.lower')
        self.assertEqual(complex.real.__qualname__, 'complex.real')
        self.assertEqual(float.real.__qualname__, 'float.real')
        self.assertEqual(int.__add__.__qualname__, 'int.__add__')

        with torch._dynamo.error_on_graph_break(False):
            class X:
                pass
        with self.assertRaises(TypeError):
            del X.__qualname__

        self.assertRaises(TypeError, type.__dict__['__qualname__'].__set__,
                          str, 'Oink')

        global Y

        with torch._dynamo.error_on_graph_break(False):
            class Y:
                class Inside:
                    pass
        self.assertEqual(Y.__qualname__, 'Y')
        self.assertEqual(Y.Inside.__qualname__, 'Y.Inside')

    def test_qualname_dict(self):
        ns = {'__qualname__': 'some.name'}
        tp = type('Foo', (), ns)
        self.assertEqual(tp.__qualname__, 'some.name')
        self.assertNotIn('__qualname__', tp.__dict__)
        self.assertEqual(ns, {'__qualname__': 'some.name'})

        ns = {'__qualname__': 1}
        self.assertRaises(TypeError, type, 'Foo', (), ns)

    def test_cycle_through_dict(self):
        # See bug #1469629
        class X(dict):
            def __init__(self):
                dict.__init__(self)
                self.__dict__ = self
        x = X()
        x.attr = 42
        wr = weakref.ref(x)
        del x
        support.gc_collect()
        self.assertIsNone(wr())
        for o in gc.get_objects():
            self.assertIsNot(type(o), X)

    def test_object_new_and_init_with_parameters(self):
        # See issue #1683368
        with torch._dynamo.error_on_graph_break(False):
            class OverrideNeither:
                pass

        self.assertRaises(TypeError, OverrideNeither, 1)
        self.assertRaises(TypeError, OverrideNeither, kw=1)

        with torch._dynamo.error_on_graph_break(False):
            class OverrideNew:
                def __new__(cls, foo, kw=0, *args, **kwds):
                    return object.__new__(cls, *args, **kwds)
            class OverrideInit:
                def __init__(self, foo, kw=0, *args, **kwargs):
                    return object.__init__(self, *args, **kwargs)
            class OverrideBoth(OverrideNew, OverrideInit):
                pass

        for case in OverrideNew, OverrideInit, OverrideBoth:
            case(1)
            case(1, kw=2)
            self.assertRaises(TypeError, case, 1, 2, 3)
            self.assertRaises(TypeError, case, 1, 2, foo=3)

    def test_subclassing_does_not_duplicate_dict_descriptors(self):
        with torch._dynamo.error_on_graph_break(False):
            class Base:
                pass
            class Sub(Base):
                pass
        self.assertIn("__dict__", Base.__dict__)
        self.assertNotIn("__dict__", Sub.__dict__)

    def test_bound_method_repr(self):
        class Foo:
            def method(self):
                pass
        self.assertRegex(repr(Foo().method),
            r"<bound method .*Foo\.method of <.*Foo object at .*>>")


        class Base:
            def method(self):
                pass
        class Derived1(Base):
            pass
        class Derived2(Base):
            def method(self):
                pass
        base = Base()
        derived1 = Derived1()
        derived2 = Derived2()
        super_d2 = super(Derived2, derived2)
        self.assertRegex(repr(base.method),
            r"<bound method .*Base\.method of <.*Base object at .*>>")
        self.assertRegex(repr(derived1.method),
            r"<bound method .*Base\.method of <.*Derived1 object at .*>>")
        self.assertRegex(repr(derived2.method),
            r"<bound method .*Derived2\.method of <.*Derived2 object at .*>>")
        self.assertRegex(repr(super_d2.method),
            r"<bound method .*Base\.method of <.*Derived2 object at .*>>")

        class Foo:
            @classmethod
            def method(cls):
                pass
        foo = Foo()
        self.assertRegex(repr(foo.method), # access via instance
            r"<bound method .*Foo\.method of <class '.*Foo'>>")
        self.assertRegex(repr(Foo.method), # access via the class
            r"<bound method .*Foo\.method of <class '.*Foo'>>")


        class MyCallable:
            def __call__(self, arg):
                pass
        func = MyCallable() # func has no __name__ or __qualname__ attributes
        instance = object()
        method = types.MethodType(func, instance)
        self.assertRegex(repr(method),
            r"<bound method \? of <object object at .*>>")
        func.__name__ = "name"
        self.assertRegex(repr(method),
            r"<bound method name of <object object at .*>>")
        func.__qualname__ = "qualname"
        self.assertRegex(repr(method),
            r"<bound method qualname of <object object at .*>>")

    @unittest.skipIf(_testcapi is None, 'need the _testcapi module')
    def test_bpo25750(self):
        # bpo-25750: calling a descriptor (implemented as built-in
        # function with METH_FASTCALL) should not crash CPython if the
        # descriptor deletes itself from the class.
        class Descr:
            __get__ = _testcapi.bad_get

        class X:
            descr = Descr()
            def __new__(cls):
                cls.descr = None
                # Create this large list to corrupt some unused memory
                cls.lst = [2**i for i in range(10000)]
        X.descr

    @support.suppress_immortalization()
    def test_remove_subclass(self):
        # bpo-46417: when the last subclass of a type is deleted,
        # remove_subclass() clears the internal dictionary of subclasses:
        # set PyTypeObject.tp_subclasses to NULL. remove_subclass() is called
        # when a type is deallocated.
        class Parent:
            pass
        self.assertEqual(Parent.__subclasses__(), [])

        class Child(Parent):
            pass
        self.assertEqual(Parent.__subclasses__(), [Child])

        del Child
        gc.collect()
        self.assertEqual(Parent.__subclasses__(), [])

    def test_instance_method_get_behavior(self):
        # test case for gh-113157

        class A:
            def meth(self):
                return self

        class B:
            pass

        a = A()
        b = B()
        b.meth = a.meth.__get__(b, B)
        self.assertEqual(b.meth(), a)

    def test_attr_raise_through_property(self):
        # test case for gh-103272
        class A:
            def __getattr__(self, name):
                raise ValueError("FOO")

            @property
            def foo(self):
                return self.__getattr__("asdf")

        with self.assertRaisesRegex(ValueError, "FOO"):
            A().foo

        # test case for gh-103551
        class B:
            @property
            def __getattr__(self, name):
                raise ValueError("FOO")

            @property
            def foo(self):
                raise NotImplementedError("BAR")

        with self.assertRaisesRegex(NotImplementedError, "BAR"):
            B().foo


class DictProxyTests(CPythonTestCase):
    def setUp(self):
        super().setUp()
        class C(object):
            def meth(self):
                pass
        self.C = C

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_keys(self):
        # Testing dict-proxy keys...
        it = self.C.__dict__.keys()
        self.assertNotIsInstance(it, list)
        keys = list(it)
        keys.sort()
        self.assertEqual(keys, ['__dict__', '__doc__', '__firstlineno__',
                                '__module__',
                                '__static_attributes__', '__weakref__',
                                'meth'])

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_values(self):
        # Testing dict-proxy values...
        it = self.C.__dict__.values()
        self.assertNotIsInstance(it, list)
        values = list(it)
        self.assertEqual(len(values), 7)

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_items(self):
        # Testing dict-proxy iteritems...
        it = self.C.__dict__.items()
        self.assertNotIsInstance(it, list)
        keys = [item[0] for item in it]
        keys.sort()
        self.assertEqual(keys, ['__dict__', '__doc__', '__firstlineno__',
                                '__module__',
                                '__static_attributes__', '__weakref__',
                                'meth'])

    def test_dict_type_with_metaclass(self):
        # Testing type of __dict__ when metaclass set...
        with torch._dynamo.error_on_graph_break(False):
            class B(object):
                pass
            class M(type):
                pass
            class C(metaclass=M):
                # In 2.3a1, C.__dict__ was a real dict rather than a dict proxy
                pass
        self.assertEqual(type(C.__dict__), type(B.__dict__))

    def test_repr(self):
        # Testing mappingproxy.__repr__.
        # We can't blindly compare with the repr of another dict as ordering
        # of keys and values is arbitrary and may differ.
        r = repr(self.C.__dict__)
        self.assertTrue(r.startswith('mappingproxy('), r)
        self.assertTrue(r.endswith(')'), r)
        for k, v in self.C.__dict__.items():
            self.assertIn('{!r}: {!r}'.format(k, v), r)


class AAAPTypesLongInitTest(CPythonTestCase):
    # This is in its own TestCase so that it can be run before any other tests.
    # (Hence the 'AAA' in the test class name: to make it the first
    # item in a list sorted by name, like
    # unittest.TestLoader.getTestCaseNames() does.)
    def test_pytype_long_ready(self):
        # Testing SF bug 551412 ...

        # This dumps core when SF bug 551412 isn't fixed --
        # but only when test_descr.py is run separately.
        # (That can't be helped -- as soon as PyType_Ready()
        # is called for PyLong_Type, the bug is gone.)
        class UserLong(object):
            def __pow__(self, *args):
                pass
        try:
            pow(0, UserLong(), 0)
        except:
            pass

        # Another segfault only when run early
        # (before PyType_Ready(tuple) is called)
        type.mro(tuple)


class MiscTests(CPythonTestCase):
    def test_type_lookup_mro_reference(self):
        # Issue #14199: _PyType_Lookup() has to keep a strong reference to
        # the type MRO because it may be modified during the lookup, if
        # __bases__ is set during the lookup for example.
        class MyKey(object):
            def __hash__(self):
                return hash('mykey')

            def __eq__(self, other):
                X.__bases__ = (Base2,)

        class Base(object):
            mykey = 'from Base'
            mykey2 = 'from Base'

        class Base2(object):
            mykey = 'from Base2'
            mykey2 = 'from Base2'

        with self.assertWarnsRegex(RuntimeWarning, 'X'):
            X = type('X', (Base,), {MyKey(): 5})

        # Note that the access below uses getattr() rather than normally
        # accessing the attribute.  That is done to avoid the bytecode
        # specializer activating on repeated runs of the test.

        # mykey is read from Base
        self.assertEqual(getattr(X, 'mykey'), 'from Base')
        # mykey2 is read from Base2 because MyKey.__eq__ has set __bases__
        self.assertEqual(getattr(X, 'mykey2'), 'from Base2')


class PicklingTests(CPythonTestCase):

    def _check_reduce(self, proto, obj, args=(), kwargs={}, state=None,
                      listitems=None, dictitems=None):
        if proto >= 2:
            reduce_value = obj.__reduce_ex__(proto)
            if kwargs:
                self.assertEqual(reduce_value[0], copyreg.__newobj_ex__)
                self.assertEqual(reduce_value[1], (type(obj), args, kwargs))
            else:
                self.assertEqual(reduce_value[0], copyreg.__newobj__)
                self.assertEqual(reduce_value[1], (type(obj),) + args)
            self.assertEqual(reduce_value[2], state)
            if listitems is not None:
                self.assertListEqual(list(reduce_value[3]), listitems)
            else:
                self.assertIsNone(reduce_value[3])
            if dictitems is not None:
                self.assertDictEqual(dict(reduce_value[4]), dictitems)
            else:
                self.assertIsNone(reduce_value[4])
        else:
            base_type = type(obj).__base__
            reduce_value = (copyreg._reconstructor,
                            (type(obj),
                             base_type,
                             None if base_type is object else base_type(obj)))
            if state is not None:
                reduce_value += (state,)
            self.assertEqual(obj.__reduce_ex__(proto), reduce_value)
            self.assertEqual(obj.__reduce__(), reduce_value)

    def test_reduce(self):
        protocols = range(pickle.HIGHEST_PROTOCOL + 1)
        args = (-101, "spam")
        kwargs = {'bacon': -201, 'fish': -301}
        state = {'cheese': -401}

        class C1:
            def __getnewargs__(self):
                return args
        obj = C1()
        for proto in protocols:
            self._check_reduce(proto, obj, args)

        for name, value in state.items():
            setattr(obj, name, value)
        for proto in protocols:
            self._check_reduce(proto, obj, args, state=state)

        class C2:
            def __getnewargs__(self):
                return "bad args"
        obj = C2()
        for proto in protocols:
            if proto >= 2:
                with self.assertRaises(TypeError):
                    obj.__reduce_ex__(proto)

        class C3:
            def __getnewargs_ex__(self):
                return (args, kwargs)
        obj = C3()
        for proto in protocols:
            if proto >= 2:
                self._check_reduce(proto, obj, args, kwargs)

        class C4:
            def __getnewargs_ex__(self):
                return (args, "bad dict")
        class C5:
            def __getnewargs_ex__(self):
                return ("bad tuple", kwargs)
        class C6:
            def __getnewargs_ex__(self):
                return ()
        class C7:
            def __getnewargs_ex__(self):
                return "bad args"
        for proto in protocols:
            for cls in C4, C5, C6, C7:
                obj = cls()
                if proto >= 2:
                    with self.assertRaises((TypeError, ValueError)):
                        obj.__reduce_ex__(proto)

        class C9:
            def __getnewargs_ex__(self):
                return (args, {})
        obj = C9()
        for proto in protocols:
            self._check_reduce(proto, obj, args)

        class C10:
            def __getnewargs_ex__(self):
                raise IndexError
        obj = C10()
        for proto in protocols:
            if proto >= 2:
                with self.assertRaises(IndexError):
                    obj.__reduce_ex__(proto)

        class C11:
            def __getstate__(self):
                return state
        obj = C11()
        for proto in protocols:
            self._check_reduce(proto, obj, state=state)

        class C12:
            def __getstate__(self):
                return "not dict"
        obj = C12()
        for proto in protocols:
            self._check_reduce(proto, obj, state="not dict")

        class C13:
            def __getstate__(self):
                raise IndexError
        obj = C13()
        for proto in protocols:
            with self.assertRaises(IndexError):
                obj.__reduce_ex__(proto)
            if proto < 2:
                with self.assertRaises(IndexError):
                    obj.__reduce__()

        class C14:
            __slots__ = tuple(state)
            def __init__(self):
                for name, value in state.items():
                    setattr(self, name, value)

        obj = C14()
        for proto in protocols:
            if proto >= 2:
                self._check_reduce(proto, obj, state=(None, state))
            else:
                with self.assertRaises(TypeError):
                    obj.__reduce_ex__(proto)
                with self.assertRaises(TypeError):
                    obj.__reduce__()

        class C15(dict):
            pass
        obj = C15({"quebec": -601})
        for proto in protocols:
            self._check_reduce(proto, obj, dictitems=dict(obj))

        class C16(list):
            pass
        obj = C16(["yukon"])
        for proto in protocols:
            self._check_reduce(proto, obj, listitems=list(obj))

    def test_special_method_lookup(self):
        protocols = range(pickle.HIGHEST_PROTOCOL + 1)
        class Picky:
            def __getstate__(self):
                return {}

            def __getattr__(self, attr):
                if attr in ("__getnewargs__", "__getnewargs_ex__"):
                    raise AssertionError(attr)
                return None
        for protocol in protocols:
            state = {} if protocol >= 2 else None
            self._check_reduce(protocol, Picky(), state=state)

    def _assert_is_copy(self, obj, objcopy, msg=None):
        """Utility method to verify if two objects are copies of each others.
        """
        if msg is None:
            msg = "{!r} is not a copy of {!r}".format(obj, objcopy)
        if type(obj).__repr__ is object.__repr__:
            # We have this limitation for now because we use the object's repr
            # to help us verify that the two objects are copies. This allows
            # us to delegate the non-generic verification logic to the objects
            # themselves.
            raise ValueError("object passed to _assert_is_copy must " +
                             "override the __repr__ method.")
        self.assertIsNot(obj, objcopy, msg=msg)
        self.assertIs(type(obj), type(objcopy), msg=msg)
        if hasattr(obj, '__dict__'):
            self.assertDictEqual(obj.__dict__, objcopy.__dict__, msg=msg)
            self.assertIsNot(obj.__dict__, objcopy.__dict__, msg=msg)
        if hasattr(obj, '__slots__'):
            self.assertListEqual(obj.__slots__, objcopy.__slots__, msg=msg)
            for slot in obj.__slots__:
                self.assertEqual(
                    hasattr(obj, slot), hasattr(objcopy, slot), msg=msg)
                self.assertEqual(getattr(obj, slot, None),
                                 getattr(objcopy, slot, None), msg=msg)
        self.assertEqual(repr(obj), repr(objcopy), msg=msg)

    @staticmethod
    def _generate_pickle_copiers():
        """Utility method to generate the many possible pickle configurations.
        """
        class PickleCopier:
            "This class copies object using pickle."
            def __init__(self, proto, dumps, loads):
                self.proto = proto
                self.dumps = dumps
                self.loads = loads
            def copy(self, obj):
                return self.loads(self.dumps(obj, self.proto))
            def __repr__(self):
                # We try to be as descriptive as possible here since this is
                # the string which we will allow us to tell the pickle
                # configuration we are using during debugging.
                return ("PickleCopier(proto={}, dumps={}.{}, loads={}.{})"
                        .format(self.proto,
                                self.dumps.__module__, self.dumps.__qualname__,
                                self.loads.__module__, self.loads.__qualname__))
        return (PickleCopier(*args) for args in
                   itertools.product(range(pickle.HIGHEST_PROTOCOL + 1),
                                     {pickle.dumps, pickle._dumps},
                                     {pickle.loads, pickle._loads}))

    def test_pickle_slots(self):
        # Tests pickling of classes with __slots__.

        # Pickling of classes with __slots__ but without __getstate__ should
        # fail (if using protocol 0 or 1)
        global C
        class C:
            __slots__ = ['a']
        with self.assertRaises(TypeError):
            pickle.dumps(C(), 0)

        global D
        class D(C):
            pass
        with self.assertRaises(TypeError):
            pickle.dumps(D(), 0)

        class C:
            "A class with __getstate__ and __setstate__ implemented."
            __slots__ = ['a']
            def __getstate__(self):
                state = getattr(self, '__dict__', {}).copy()
                for cls in type(self).__mro__:
                    for slot in cls.__dict__.get('__slots__', ()):
                        try:
                            state[slot] = getattr(self, slot)
                        except AttributeError:
                            pass
                return state
            def __setstate__(self, state):
                for k, v in state.items():
                    setattr(self, k, v)
            def __repr__(self):
                return "%s()<%r>" % (type(self).__name__, self.__getstate__())

        class D(C):
            "A subclass of a class with slots."
            pass

        global E
        class E(C):
            "A subclass with an extra slot."
            __slots__ = ['b']

        # Now it should work
        for pickle_copier in self._generate_pickle_copiers():
            with self.subTest(pickle_copier=pickle_copier):
                x = C()
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x.a = 42
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x = D()
                x.a = 42
                x.b = 100
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x = E()
                x.a = 42
                x.b = "foo"
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

    def test_reduce_copying(self):
        # Tests pickling and copying new-style classes and objects.
        global C1
        class C1:
            "The state of this class is copyable via its instance dict."
            ARGS = (1, 2)
            NEED_DICT_COPYING = True
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b
            def __repr__(self):
                return "C1(%r, %r)" % (self.a, self.b)

        global C2
        class C2(list):
            "A list subclass copyable via __getnewargs__."
            ARGS = (1, 2)
            NEED_DICT_COPYING = False
            def __new__(cls, a, b):
                self = super().__new__(cls)
                self.a = a
                self.b = b
                return self
            def __init__(self, *args):
                super().__init__()
                # This helps testing that __init__ is not called during the
                # unpickling process, which would cause extra appends.
                self.append("cheese")
            @classmethod
            def __getnewargs__(cls):
                return cls.ARGS
            def __repr__(self):
                return "C2(%r, %r)<%r>" % (self.a, self.b, list(self))

        global C3
        class C3(list):
            "A list subclass copyable via __getstate__."
            ARGS = (1, 2)
            NEED_DICT_COPYING = False
            def __init__(self, a, b):
                self.a = a
                self.b = b
                # This helps testing that __init__ is not called during the
                # unpickling process, which would cause extra appends.
                self.append("cheese")
            @classmethod
            def __getstate__(cls):
                return cls.ARGS
            def __setstate__(self, state):
                a, b = state
                self.a = a
                self.b = b
            def __repr__(self):
                return "C3(%r, %r)<%r>" % (self.a, self.b, list(self))

        global C4
        class C4(int):
            "An int subclass copyable via __getnewargs__."
            ARGS = ("hello", "world", 1)
            NEED_DICT_COPYING = False
            def __new__(cls, a, b, value):
                self = super().__new__(cls, value)
                self.a = a
                self.b = b
                return self
            @classmethod
            def __getnewargs__(cls):
                return cls.ARGS
            def __repr__(self):
                return "C4(%r, %r)<%r>" % (self.a, self.b, int(self))

        global C5
        class C5(int):
            "An int subclass copyable via __getnewargs_ex__."
            ARGS = (1, 2)
            KWARGS = {'value': 3}
            NEED_DICT_COPYING = False
            def __new__(cls, a, b, *, value=0):
                self = super().__new__(cls, value)
                self.a = a
                self.b = b
                return self
            @classmethod
            def __getnewargs_ex__(cls):
                return (cls.ARGS, cls.KWARGS)
            def __repr__(self):
                return "C5(%r, %r)<%r>" % (self.a, self.b, int(self))

        test_classes = (C1, C2, C3, C4, C5)
        # Testing copying through pickle
        pickle_copiers = self._generate_pickle_copiers()
        for cls, pickle_copier in itertools.product(test_classes, pickle_copiers):
            with self.subTest(cls=cls, pickle_copier=pickle_copier):
                kwargs = getattr(cls, 'KWARGS', {})
                obj = cls(*cls.ARGS, **kwargs)
                proto = pickle_copier.proto
                objcopy = pickle_copier.copy(obj)
                self._assert_is_copy(obj, objcopy)
                # For test classes that supports this, make sure we didn't go
                # around the reduce protocol by simply copying the attribute
                # dictionary. We clear attributes using the previous copy to
                # not mutate the original argument.
                if proto >= 2 and not cls.NEED_DICT_COPYING:
                    objcopy.__dict__.clear()
                    objcopy2 = pickle_copier.copy(objcopy)
                    self._assert_is_copy(obj, objcopy2)

        # Testing copying through copy.deepcopy()
        for cls in test_classes:
            with self.subTest(cls=cls):
                kwargs = getattr(cls, 'KWARGS', {})
                obj = cls(*cls.ARGS, **kwargs)
                objcopy = deepcopy(obj)
                self._assert_is_copy(obj, objcopy)
                # For test classes that supports this, make sure we didn't go
                # around the reduce protocol by simply copying the attribute
                # dictionary. We clear attributes using the previous copy to
                # not mutate the original argument.
                if not cls.NEED_DICT_COPYING:
                    objcopy.__dict__.clear()
                    objcopy2 = deepcopy(objcopy)
                    self._assert_is_copy(obj, objcopy2)

    def test_issue24097(self):
        # Slot name is freed inside __getattr__ and is later used.
        class S(str):  # Not interned
            pass
        class A:
            __slotnames__ = [S('spam')]
            def __getattr__(self, attr):
                if attr == 'spam':
                    A.__slotnames__[:] = [S('spam')]
                    return 42
                else:
                    raise AttributeError

        import copyreg
        expected = (copyreg.__newobj__, (A,), (None, {'spam': 42}), None, None)
        self.assertEqual(A().__reduce_ex__(2), expected)  # Shouldn't crash

    def test_object_reduce(self):
        # Issue #29914
        # __reduce__() takes no arguments
        object().__reduce__()
        with self.assertRaises(TypeError):
            object().__reduce__(0)
        # __reduce_ex__() takes one integer argument
        object().__reduce_ex__(0)
        with self.assertRaises(TypeError):
            object().__reduce_ex__()
        with self.assertRaises(TypeError):
            object().__reduce_ex__(None)


class SharedKeyTests(CPythonTestCase):

    @support.cpython_only
    def test_subclasses(self):
        # Verify that subclasses can share keys (per PEP 412)
        class A:
            pass
        class B(A):
            pass

        #Shrink keys by repeatedly creating instances
        [(A(), B()) for _ in range(30)]

        a, b = A(), B()
        self.assertEqual(sys.getsizeof(vars(a)), sys.getsizeof(vars(b)))
        self.assertLess(sys.getsizeof(vars(a)), sys.getsizeof({"a":1}))
        # Initial hash table can contain only one or two elements.
        # Set 6 attributes to cause internal resizing.
        a.x, a.y, a.z, a.w, a.v, a.u = range(6)
        self.assertNotEqual(sys.getsizeof(vars(a)), sys.getsizeof(vars(b)))
        a2 = A()
        self.assertGreater(sys.getsizeof(vars(a)), sys.getsizeof(vars(a2)))
        self.assertLess(sys.getsizeof(vars(a2)), sys.getsizeof({"a":1}))
        self.assertLess(sys.getsizeof(vars(b)), sys.getsizeof({"a":1}))


class DebugHelperMeta(type):
    """
    Sets default __doc__ and simplifies repr() output.
    """
    def __new__(mcls, name, bases, attrs):
        if attrs.get('__doc__') is None:
            attrs['__doc__'] = name  # helps when debugging with gdb
        return type.__new__(mcls, name, bases, attrs)
    def __repr__(cls):
        return repr(cls.__name__)


class MroTest(CPythonTestCase):
    """
    Regressions for some bugs revealed through
    mcsl.mro() customization (typeobject.c: mro_internal()) and
    cls.__bases__ assignment (typeobject.c: type_set_bases()).
    """

    def setUp(self):
        super().setUp()
        self.step = 0
        self.ready = False

    def step_until(self, limit):
        ret = (self.step < limit)
        if ret:
            self.step += 1
        return ret

    def test_incomplete_set_bases_on_self(self):
        """
        type_set_bases must be aware that type->tp_mro can be NULL.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if self.step_until(1):
                    assert cls.__mro__ is None
                    cls.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            pass

    def test_reent_set_bases_on_base(self):
        """
        Deep reentrancy must not over-decref old_mro.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if cls.__mro__ is not None and cls.__name__ == 'B':
                    # 4-5 steps are usually enough to make it crash somewhere
                    if self.step_until(10):
                        A.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            pass
        class B(A):
            pass
        B.__bases__ += ()

    def test_reent_set_bases_on_direct_base(self):
        """
        Similar to test_reent_set_bases_on_base, but may crash differently.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                base = cls.__bases__[0]
                if base is not object:
                    if self.step_until(5):
                        base.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            pass
        class B(A):
            pass
        class C(B):
            pass

    def test_reent_set_bases_tp_base_cycle(self):
        """
        type_set_bases must check for an inheritance cycle not only through
        MRO of the type, which may be not yet updated in case of reentrance,
        but also through tp_base chain, which is assigned before diving into
        inner calls to mro().

        Otherwise, the following snippet can loop forever:
            do {
                // ...
                type = type->tp_base;
            } while (type != NULL);

        Functions that rely on tp_base (like solid_base and PyType_IsSubtype)
        would not be happy in that case, causing a stack overflow.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if self.ready:
                    if cls.__name__ == 'B1':
                        B2.__bases__ = (B1,)
                    if cls.__name__ == 'B2':
                        B1.__bases__ = (B2,)
                return type.mro(cls)

        class A(metaclass=M):
            pass
        class B1(A):
            pass
        class B2(A):
            pass

        self.ready = True
        with self.assertRaises(TypeError):
            B1.__bases__ += ()

    def test_tp_subclasses_cycle_in_update_slots(self):
        """
        type_set_bases must check for reentrancy upon finishing its job
        by updating tp_subclasses of old/new bases of the type.
        Otherwise, an implicit inheritance cycle through tp_subclasses
        can break functions that recurse on elements of that field
        (like recurse_down_subclasses and mro_hierarchy) eventually
        leading to a stack overflow.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if self.ready and cls.__name__ == 'C':
                    self.ready = False
                    C.__bases__ = (B2,)
                return type.mro(cls)

        class A(metaclass=M):
            pass
        class B1(A):
            pass
        class B2(A):
            pass
        class C(A):
            pass

        self.ready = True
        C.__bases__ = (B1,)
        B1.__bases__ = (C,)

        self.assertEqual(C.__bases__, (B2,))
        self.assertEqual(B2.__subclasses__(), [C])
        self.assertEqual(B1.__subclasses__(), [])

        self.assertEqual(B1.__bases__, (C,))
        self.assertEqual(C.__subclasses__(), [B1])

    def test_tp_subclasses_cycle_error_return_path(self):
        """
        The same as test_tp_subclasses_cycle_in_update_slots, but tests
        a code path executed on error (goto bail).
        """
        class E(Exception):
            pass
        class M(DebugHelperMeta):
            def mro(cls):
                if self.ready and cls.__name__ == 'C':
                    if C.__bases__ == (B2,):
                        self.ready = False
                    else:
                        C.__bases__ = (B2,)
                        raise E
                return type.mro(cls)

        class A(metaclass=M):
            pass
        class B1(A):
            pass
        class B2(A):
            pass
        class C(A):
            pass

        self.ready = True
        with self.assertRaises(E):
            C.__bases__ = (B1,)
        B1.__bases__ = (C,)

        self.assertEqual(C.__bases__, (B2,))
        self.assertEqual(C.__mro__, tuple(type.mro(C)))

    def test_incomplete_extend(self):
        """
        Extending an uninitialized type with type->tp_mro == NULL must
        throw a reasonable TypeError exception, instead of failing
        with PyErr_BadInternalCall.
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if cls.__mro__ is None and cls.__name__ != 'X':
                    with self.assertRaises(TypeError):
                        class X(cls):
                            pass

                return type.mro(cls)

        class A(metaclass=M):
            pass

    def test_incomplete_super(self):
        """
        Attribute lookup on a super object must be aware that
        its target type can be uninitialized (type->tp_mro == NULL).
        """
        class M(DebugHelperMeta):
            def mro(cls):
                if cls.__mro__ is None:
                    with self.assertRaises(AttributeError):
                        super(cls, cls).xxx

                return type.mro(cls)

        class A(metaclass=M):
            pass

    def test_disappearing_custom_mro(self):
        """
        gh-92112: A custom mro() returning a result conflicting with
        __bases__ and deleting itself caused a double free.
        """
        class B:
            pass

        class M(DebugHelperMeta):
            def mro(cls):
                del M.mro
                return (B,)

        with self.assertRaises(TypeError):
            class A(metaclass=M):
                pass


if __name__ == "__main__":
    run_tests()
