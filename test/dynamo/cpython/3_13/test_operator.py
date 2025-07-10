# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_operator.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

import unittest
import inspect
import pickle
import sys
from decimal import Decimal
from fractions import Fraction

from test import support
from test.support import import_helper


py_operator = import_helper.import_fresh_module('operator',
                                                blocked=['_operator'])
c_operator = import_helper.import_fresh_module('operator',
                                               fresh=['_operator'])

class Seq1:
    def __init__(self, lst):
        self.lst = lst
    def __len__(self):
        return len(self.lst)
    def __getitem__(self, i):
        return self.lst[i]
    def __add__(self, other):
        return self.lst + other.lst
    def __mul__(self, other):
        return self.lst * other
    def __rmul__(self, other):
        return other * self.lst

class Seq2(object):
    def __init__(self, lst):
        self.lst = lst
    def __len__(self):
        return len(self.lst)
    def __getitem__(self, i):
        return self.lst[i]
    def __add__(self, other):
        return self.lst + other.lst
    def __mul__(self, other):
        return self.lst * other
    def __rmul__(self, other):
        return other * self.lst

class BadIterable:
    def __iter__(self):
        raise ZeroDivisionError


class OperatorTestCase:
    def test___all__(self):
        operator = self.module
        actual_all = set(operator.__all__)
        computed_all = set()
        for name in vars(operator):
            if name.startswith('__'):
                continue
            value = getattr(operator, name)
            if value.__module__ in ('operator', '_operator'):
                computed_all.add(name)
        self.assertSetEqual(computed_all, actual_all)

    def test_lt(self):
        operator = self.module
        self.assertRaises(TypeError, operator.lt)
        self.assertRaises(TypeError, operator.lt, 1j, 2j)
        self.assertFalse(operator.lt(1, 0))
        self.assertFalse(operator.lt(1, 0.0))
        self.assertFalse(operator.lt(1, 1))
        self.assertFalse(operator.lt(1, 1.0))
        self.assertTrue(operator.lt(1, 2))
        self.assertTrue(operator.lt(1, 2.0))

    def test_le(self):
        operator = self.module
        self.assertRaises(TypeError, operator.le)
        self.assertRaises(TypeError, operator.le, 1j, 2j)
        self.assertFalse(operator.le(1, 0))
        self.assertFalse(operator.le(1, 0.0))
        self.assertTrue(operator.le(1, 1))
        self.assertTrue(operator.le(1, 1.0))
        self.assertTrue(operator.le(1, 2))
        self.assertTrue(operator.le(1, 2.0))

    def test_eq(self):
        operator = self.module
        class C(object):
            def __eq__(self, other):
                raise SyntaxError
        self.assertRaises(TypeError, operator.eq)
        self.assertRaises(SyntaxError, operator.eq, C(), C())
        self.assertFalse(operator.eq(1, 0))
        self.assertFalse(operator.eq(1, 0.0))
        self.assertTrue(operator.eq(1, 1))
        self.assertTrue(operator.eq(1, 1.0))
        self.assertFalse(operator.eq(1, 2))
        self.assertFalse(operator.eq(1, 2.0))

    def test_ne(self):
        operator = self.module
        class C(object):
            def __ne__(self, other):
                raise SyntaxError
        self.assertRaises(TypeError, operator.ne)
        self.assertRaises(SyntaxError, operator.ne, C(), C())
        self.assertTrue(operator.ne(1, 0))
        self.assertTrue(operator.ne(1, 0.0))
        self.assertFalse(operator.ne(1, 1))
        self.assertFalse(operator.ne(1, 1.0))
        self.assertTrue(operator.ne(1, 2))
        self.assertTrue(operator.ne(1, 2.0))

    def test_ge(self):
        operator = self.module
        self.assertRaises(TypeError, operator.ge)
        self.assertRaises(TypeError, operator.ge, 1j, 2j)
        self.assertTrue(operator.ge(1, 0))
        self.assertTrue(operator.ge(1, 0.0))
        self.assertTrue(operator.ge(1, 1))
        self.assertTrue(operator.ge(1, 1.0))
        self.assertFalse(operator.ge(1, 2))
        self.assertFalse(operator.ge(1, 2.0))

    def test_gt(self):
        operator = self.module
        self.assertRaises(TypeError, operator.gt)
        self.assertRaises(TypeError, operator.gt, 1j, 2j)
        self.assertTrue(operator.gt(1, 0))
        self.assertTrue(operator.gt(1, 0.0))
        self.assertFalse(operator.gt(1, 1))
        self.assertFalse(operator.gt(1, 1.0))
        self.assertFalse(operator.gt(1, 2))
        self.assertFalse(operator.gt(1, 2.0))

    def test_abs(self):
        operator = self.module
        self.assertRaises(TypeError, operator.abs)
        self.assertRaises(TypeError, operator.abs, None)
        self.assertEqual(operator.abs(-1), 1)
        self.assertEqual(operator.abs(1), 1)

    def test_add(self):
        operator = self.module
        self.assertRaises(TypeError, operator.add)
        self.assertRaises(TypeError, operator.add, None, None)
        self.assertEqual(operator.add(3, 4), 7)

    def test_bitwise_and(self):
        operator = self.module
        self.assertRaises(TypeError, operator.and_)
        self.assertRaises(TypeError, operator.and_, None, None)
        self.assertEqual(operator.and_(0xf, 0xa), 0xa)

    def test_concat(self):
        operator = self.module
        self.assertRaises(TypeError, operator.concat)
        self.assertRaises(TypeError, operator.concat, None, None)
        self.assertEqual(operator.concat('py', 'thon'), 'python')
        self.assertEqual(operator.concat([1, 2], [3, 4]), [1, 2, 3, 4])
        self.assertEqual(operator.concat(Seq1([5, 6]), Seq1([7])), [5, 6, 7])
        self.assertEqual(operator.concat(Seq2([5, 6]), Seq2([7])), [5, 6, 7])
        self.assertRaises(TypeError, operator.concat, 13, 29)

    def test_countOf(self):
        operator = self.module
        self.assertRaises(TypeError, operator.countOf)
        self.assertRaises(TypeError, operator.countOf, None, None)
        self.assertRaises(ZeroDivisionError, operator.countOf, BadIterable(), 1)
        self.assertEqual(operator.countOf([1, 2, 1, 3, 1, 4], 3), 1)
        self.assertEqual(operator.countOf([1, 2, 1, 3, 1, 4], 5), 0)
        # is but not ==
        nan = float("nan")
        self.assertEqual(operator.countOf([nan, nan, 21], nan), 2)
        # == but not is
        self.assertEqual(operator.countOf([{}, 1, {}, 2], {}), 2)

    def test_delitem(self):
        operator = self.module
        a = [4, 3, 2, 1]
        self.assertRaises(TypeError, operator.delitem, a)
        self.assertRaises(TypeError, operator.delitem, a, None)
        self.assertIsNone(operator.delitem(a, 1))
        self.assertEqual(a, [4, 2, 1])

    def test_floordiv(self):
        operator = self.module
        self.assertRaises(TypeError, operator.floordiv, 5)
        self.assertRaises(TypeError, operator.floordiv, None, None)
        self.assertEqual(operator.floordiv(5, 2), 2)

    def test_truediv(self):
        operator = self.module
        self.assertRaises(TypeError, operator.truediv, 5)
        self.assertRaises(TypeError, operator.truediv, None, None)
        self.assertEqual(operator.truediv(5, 2), 2.5)

    def test_getitem(self):
        operator = self.module
        a = range(10)
        self.assertRaises(TypeError, operator.getitem)
        self.assertRaises(TypeError, operator.getitem, a, None)
        self.assertEqual(operator.getitem(a, 2), 2)

    def test_indexOf(self):
        operator = self.module
        self.assertRaises(TypeError, operator.indexOf)
        self.assertRaises(TypeError, operator.indexOf, None, None)
        self.assertRaises(ZeroDivisionError, operator.indexOf, BadIterable(), 1)
        self.assertEqual(operator.indexOf([4, 3, 2, 1], 3), 1)
        self.assertRaises(ValueError, operator.indexOf, [4, 3, 2, 1], 0)
        nan = float("nan")
        self.assertEqual(operator.indexOf([nan, nan, 21], nan), 0)
        self.assertEqual(operator.indexOf([{}, 1, {}, 2], {}), 0)
        it = iter('leave the iterator at exactly the position after the match')
        self.assertEqual(operator.indexOf(it, 'a'), 2)
        self.assertEqual(next(it), 'v')

    def test_invert(self):
        operator = self.module
        self.assertRaises(TypeError, operator.invert)
        self.assertRaises(TypeError, operator.invert, None)
        self.assertEqual(operator.inv(4), -5)

    def test_lshift(self):
        operator = self.module
        self.assertRaises(TypeError, operator.lshift)
        self.assertRaises(TypeError, operator.lshift, None, 42)
        self.assertEqual(operator.lshift(5, 1), 10)
        self.assertEqual(operator.lshift(5, 0), 5)
        self.assertRaises(ValueError, operator.lshift, 2, -1)

    def test_mod(self):
        operator = self.module
        self.assertRaises(TypeError, operator.mod)
        self.assertRaises(TypeError, operator.mod, None, 42)
        self.assertEqual(operator.mod(5, 2), 1)

    def test_mul(self):
        operator = self.module
        self.assertRaises(TypeError, operator.mul)
        self.assertRaises(TypeError, operator.mul, None, None)
        self.assertEqual(operator.mul(5, 2), 10)

    def test_matmul(self):
        operator = self.module
        self.assertRaises(TypeError, operator.matmul)
        self.assertRaises(TypeError, operator.matmul, 42, 42)
        class M:
            def __matmul__(self, other):
                return other - 1
        self.assertEqual(M() @ 42, 41)

    def test_neg(self):
        operator = self.module
        self.assertRaises(TypeError, operator.neg)
        self.assertRaises(TypeError, operator.neg, None)
        self.assertEqual(operator.neg(5), -5)
        self.assertEqual(operator.neg(-5), 5)
        self.assertEqual(operator.neg(0), 0)
        self.assertEqual(operator.neg(-0), 0)

    def test_bitwise_or(self):
        operator = self.module
        self.assertRaises(TypeError, operator.or_)
        self.assertRaises(TypeError, operator.or_, None, None)
        self.assertEqual(operator.or_(0xa, 0x5), 0xf)

    def test_pos(self):
        operator = self.module
        self.assertRaises(TypeError, operator.pos)
        self.assertRaises(TypeError, operator.pos, None)
        self.assertEqual(operator.pos(5), 5)
        self.assertEqual(operator.pos(-5), -5)
        self.assertEqual(operator.pos(0), 0)
        self.assertEqual(operator.pos(-0), 0)

    def test_pow(self):
        operator = self.module
        self.assertRaises(TypeError, operator.pow)
        self.assertRaises(TypeError, operator.pow, None, None)
        self.assertEqual(operator.pow(3,5), 3**5)
        self.assertRaises(TypeError, operator.pow, 1)
        self.assertRaises(TypeError, operator.pow, 1, 2, 3)

    def test_rshift(self):
        operator = self.module
        self.assertRaises(TypeError, operator.rshift)
        self.assertRaises(TypeError, operator.rshift, None, 42)
        self.assertEqual(operator.rshift(5, 1), 2)
        self.assertEqual(operator.rshift(5, 0), 5)
        self.assertRaises(ValueError, operator.rshift, 2, -1)

    def test_contains(self):
        operator = self.module
        self.assertRaises(TypeError, operator.contains)
        self.assertRaises(TypeError, operator.contains, None, None)
        self.assertRaises(ZeroDivisionError, operator.contains, BadIterable(), 1)
        self.assertTrue(operator.contains(range(4), 2))
        self.assertFalse(operator.contains(range(4), 5))

    def test_setitem(self):
        operator = self.module
        a = list(range(3))
        self.assertRaises(TypeError, operator.setitem, a)
        self.assertRaises(TypeError, operator.setitem, a, None, None)
        self.assertIsNone(operator.setitem(a, 0, 2))
        self.assertEqual(a, [2, 1, 2])
        self.assertRaises(IndexError, operator.setitem, a, 4, 2)

    def test_sub(self):
        operator = self.module
        self.assertRaises(TypeError, operator.sub)
        self.assertRaises(TypeError, operator.sub, None, None)
        self.assertEqual(operator.sub(5, 2), 3)

    def test_truth(self):
        operator = self.module
        class C(object):
            def __bool__(self):
                raise SyntaxError
        self.assertRaises(TypeError, operator.truth)
        self.assertRaises(SyntaxError, operator.truth, C())
        self.assertTrue(operator.truth(5))
        self.assertTrue(operator.truth([0]))
        self.assertFalse(operator.truth(0))
        self.assertFalse(operator.truth([]))

    def test_bitwise_xor(self):
        operator = self.module
        self.assertRaises(TypeError, operator.xor)
        self.assertRaises(TypeError, operator.xor, None, None)
        self.assertEqual(operator.xor(0xb, 0xc), 0x7)

    def test_is(self):
        operator = self.module
        a = b = 'xyzpdq'
        c = a[:3] + b[3:]
        self.assertRaises(TypeError, operator.is_)
        self.assertTrue(operator.is_(a, b))
        self.assertFalse(operator.is_(a,c))

    def test_is_not(self):
        operator = self.module
        a = b = 'xyzpdq'
        c = a[:3] + b[3:]
        self.assertRaises(TypeError, operator.is_not)
        self.assertFalse(operator.is_not(a, b))
        self.assertTrue(operator.is_not(a,c))

    def test_attrgetter(self):
        operator = self.module
        class A:
            pass
        a = A()
        a.name = 'arthur'
        f = operator.attrgetter('name')
        self.assertEqual(f(a), 'arthur')
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 'dent')
        self.assertRaises(TypeError, f, a, surname='dent')
        f = operator.attrgetter('rank')
        self.assertRaises(AttributeError, f, a)
        self.assertRaises(TypeError, operator.attrgetter, 2)
        self.assertRaises(TypeError, operator.attrgetter)

        # multiple gets
        record = A()
        record.x = 'X'
        record.y = 'Y'
        record.z = 'Z'
        self.assertEqual(operator.attrgetter('x','z','y')(record), ('X', 'Z', 'Y'))
        self.assertRaises(TypeError, operator.attrgetter, ('x', (), 'y'))

        class C(object):
            def __getattr__(self, name):
                raise SyntaxError
        self.assertRaises(SyntaxError, operator.attrgetter('foo'), C())

        # recursive gets
        a = A()
        a.name = 'arthur'
        a.child = A()
        a.child.name = 'thomas'
        f = operator.attrgetter('child.name')
        self.assertEqual(f(a), 'thomas')
        self.assertRaises(AttributeError, f, a.child)
        f = operator.attrgetter('name', 'child.name')
        self.assertEqual(f(a), ('arthur', 'thomas'))
        f = operator.attrgetter('name', 'child.name', 'child.child.name')
        self.assertRaises(AttributeError, f, a)
        f = operator.attrgetter('child.')
        self.assertRaises(AttributeError, f, a)
        f = operator.attrgetter('.child')
        self.assertRaises(AttributeError, f, a)

        a.child.child = A()
        a.child.child.name = 'johnson'
        f = operator.attrgetter('child.child.name')
        self.assertEqual(f(a), 'johnson')
        f = operator.attrgetter('name', 'child.name', 'child.child.name')
        self.assertEqual(f(a), ('arthur', 'thomas', 'johnson'))

    def test_itemgetter(self):
        operator = self.module
        a = 'ABCDE'
        f = operator.itemgetter(2)
        self.assertEqual(f(a), 'C')
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 3)
        self.assertRaises(TypeError, f, a, size=3)
        f = operator.itemgetter(10)
        self.assertRaises(IndexError, f, a)

        class C(object):
            def __getitem__(self, name):
                raise SyntaxError
        self.assertRaises(SyntaxError, operator.itemgetter(42), C())

        f = operator.itemgetter('name')
        self.assertRaises(TypeError, f, a)
        self.assertRaises(TypeError, operator.itemgetter)

        d = dict(key='val')
        f = operator.itemgetter('key')
        self.assertEqual(f(d), 'val')
        f = operator.itemgetter('nonkey')
        self.assertRaises(KeyError, f, d)

        # example used in the docs
        inventory = [('apple', 3), ('banana', 2), ('pear', 5), ('orange', 1)]
        getcount = operator.itemgetter(1)
        self.assertEqual(list(map(getcount, inventory)), [3, 2, 5, 1])
        self.assertEqual(sorted(inventory, key=getcount),
            [('orange', 1), ('banana', 2), ('apple', 3), ('pear', 5)])

        # multiple gets
        data = list(map(str, range(20)))
        self.assertEqual(operator.itemgetter(2,10,5)(data), ('2', '10', '5'))
        self.assertRaises(TypeError, operator.itemgetter(2, 'x', 5), data)

        # interesting indices
        t = tuple('abcde')
        self.assertEqual(operator.itemgetter(-1)(t), 'e')
        self.assertEqual(operator.itemgetter(slice(2, 4))(t), ('c', 'd'))

        # interesting sequences
        class T(tuple):
            'Tuple subclass'
            pass
        self.assertEqual(operator.itemgetter(0)(T('abc')), 'a')
        self.assertEqual(operator.itemgetter(0)(['a', 'b', 'c']), 'a')
        self.assertEqual(operator.itemgetter(0)(range(100, 200)), 100)

    def test_methodcaller(self):
        operator = self.module
        self.assertRaises(TypeError, operator.methodcaller)
        self.assertRaises(TypeError, operator.methodcaller, 12)
        class A:
            def foo(self, *args, **kwds):
                return args[0] + args[1]
            def bar(self, f=42):
                return f
            def baz(*args, **kwds):
                return kwds['name'], kwds['self']
        a = A()
        f = operator.methodcaller('foo')
        self.assertRaises(IndexError, f, a)
        f = operator.methodcaller('foo', 1, 2)
        self.assertEqual(f(a), 3)
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, a, 3)
        self.assertRaises(TypeError, f, a, spam=3)
        f = operator.methodcaller('bar')
        self.assertEqual(f(a), 42)
        self.assertRaises(TypeError, f, a, a)
        f = operator.methodcaller('bar', f=5)
        self.assertEqual(f(a), 5)
        f = operator.methodcaller('baz', name='spam', self='eggs')
        self.assertEqual(f(a), ('spam', 'eggs'))

    def test_inplace(self):
        operator = self.module
        class C(object):
            def __iadd__     (self, other): return "iadd"
            def __iand__     (self, other): return "iand"
            def __ifloordiv__(self, other): return "ifloordiv"
            def __ilshift__  (self, other): return "ilshift"
            def __imod__     (self, other): return "imod"
            def __imul__     (self, other): return "imul"
            def __imatmul__  (self, other): return "imatmul"
            def __ior__      (self, other): return "ior"
            def __ipow__     (self, other): return "ipow"
            def __irshift__  (self, other): return "irshift"
            def __isub__     (self, other): return "isub"
            def __itruediv__ (self, other): return "itruediv"
            def __ixor__     (self, other): return "ixor"
            def __getitem__(self, other): return 5  # so that C is a sequence
        c = C()
        self.assertEqual(operator.iadd     (c, 5), "iadd")
        self.assertEqual(operator.iand     (c, 5), "iand")
        self.assertEqual(operator.ifloordiv(c, 5), "ifloordiv")
        self.assertEqual(operator.ilshift  (c, 5), "ilshift")
        self.assertEqual(operator.imod     (c, 5), "imod")
        self.assertEqual(operator.imul     (c, 5), "imul")
        self.assertEqual(operator.imatmul  (c, 5), "imatmul")
        self.assertEqual(operator.ior      (c, 5), "ior")
        self.assertEqual(operator.ipow     (c, 5), "ipow")
        self.assertEqual(operator.irshift  (c, 5), "irshift")
        self.assertEqual(operator.isub     (c, 5), "isub")
        self.assertEqual(operator.itruediv (c, 5), "itruediv")
        self.assertEqual(operator.ixor     (c, 5), "ixor")
        self.assertEqual(operator.iconcat  (c, c), "iadd")

    def test_iconcat_without_getitem(self):
        operator = self.module

        msg = "'int' object can't be concatenated"
        with self.assertRaisesRegex(TypeError, msg):
            operator.iconcat(1, 0.5)

    def test_index(self):
        operator = self.module
        class X:
            def __index__(self):
                return 1

        self.assertEqual(operator.index(X()), 1)
        self.assertEqual(operator.index(0), 0)
        self.assertEqual(operator.index(1), 1)
        self.assertEqual(operator.index(2), 2)
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(1.5)
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(Fraction(3, 7))
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(Decimal(1))
        with self.assertRaises((AttributeError, TypeError)):
            operator.index(None)

    def test_not_(self):
        operator = self.module
        class C:
            def __bool__(self):
                raise SyntaxError
        self.assertRaises(TypeError, operator.not_)
        self.assertRaises(SyntaxError, operator.not_, C())
        self.assertFalse(operator.not_(5))
        self.assertFalse(operator.not_([0]))
        self.assertTrue(operator.not_(0))
        self.assertTrue(operator.not_([]))

    def test_length_hint(self):
        operator = self.module
        class X(object):
            def __init__(self, value):
                self.value = value

            def __length_hint__(self):
                if type(self.value) is type:
                    raise self.value
                else:
                    return self.value

        self.assertEqual(operator.length_hint([], 2), 0)
        self.assertEqual(operator.length_hint(iter([1, 2, 3])), 3)

        self.assertEqual(operator.length_hint(X(2)), 2)
        self.assertEqual(operator.length_hint(X(NotImplemented), 4), 4)
        self.assertEqual(operator.length_hint(X(TypeError), 12), 12)
        with self.assertRaises(TypeError):
            operator.length_hint(X("abc"))
        with self.assertRaises(ValueError):
            operator.length_hint(X(-2))
        with self.assertRaises(LookupError):
            operator.length_hint(X(LookupError))

        class Y: pass

        msg = "'str' object cannot be interpreted as an integer"
        with self.assertRaisesRegex(TypeError, msg):
            operator.length_hint(X(2), "abc")
        self.assertEqual(operator.length_hint(Y(), 10), 10)

    def test_call(self):
        operator = self.module

        def func(*args, **kwargs): return args, kwargs

        self.assertEqual(operator.call(func), ((), {}))
        self.assertEqual(operator.call(func, 0, 1), ((0, 1), {}))
        self.assertEqual(operator.call(func, a=2, obj=3),
                         ((), {"a": 2, "obj": 3}))
        self.assertEqual(operator.call(func, 0, 1, a=2, obj=3),
                         ((0, 1), {"a": 2, "obj": 3}))

    def test_dunder_is_original(self):
        operator = self.module

        names = [name for name in dir(operator) if not name.startswith('_')]
        for name in names:
            orig = getattr(operator, name)
            dunder = getattr(operator, '__' + name.strip('_') + '__', None)
            if dunder:
                self.assertIs(dunder, orig)

    @support.requires_docstrings
    def test_attrgetter_signature(self):
        operator = self.module
        sig = inspect.signature(operator.attrgetter)
        self.assertEqual(str(sig), '(attr, /, *attrs)')
        sig = inspect.signature(operator.attrgetter('x', 'z', 'y'))
        self.assertEqual(str(sig), '(obj, /)')

    @support.requires_docstrings
    def test_itemgetter_signature(self):
        operator = self.module
        sig = inspect.signature(operator.itemgetter)
        self.assertEqual(str(sig), '(item, /, *items)')
        sig = inspect.signature(operator.itemgetter(2, 3, 5))
        self.assertEqual(str(sig), '(obj, /)')

    @support.requires_docstrings
    def test_methodcaller_signature(self):
        operator = self.module
        sig = inspect.signature(operator.methodcaller)
        self.assertEqual(str(sig), '(name, /, *args, **kwargs)')
        sig = inspect.signature(operator.methodcaller('foo', 2, y=3))
        self.assertEqual(str(sig), '(obj, /)')


class PyOperatorTestCase(OperatorTestCase, __TestCase):
    module = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class COperatorTestCase(OperatorTestCase, __TestCase):
    module = c_operator


class OperatorPickleTestCase:
    def copy(self, obj, proto):
        with support.swap_item(sys.modules, 'operator', self.module):
            pickled = pickle.dumps(obj, proto)
        with support.swap_item(sys.modules, 'operator', self.module2):
            return pickle.loads(pickled)

    def test_attrgetter(self):
        attrgetter = self.module.attrgetter
        class A:
            pass
        a = A()
        a.x = 'X'
        a.y = 'Y'
        a.z = 'Z'
        a.t = A()
        a.t.u = A()
        a.t.u.v = 'V'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = attrgetter('x')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # multiple gets
                f = attrgetter('x', 'y', 'z')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # recursive gets
                f = attrgetter('t.u.v')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))

    def test_itemgetter(self):
        itemgetter = self.module.itemgetter
        a = 'ABCDE'
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = itemgetter(2)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # multiple gets
                f = itemgetter(2, 0, 4)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))

    def test_methodcaller(self):
        methodcaller = self.module.methodcaller
        class A:
            def foo(self, *args, **kwds):
                return args[0] + args[1]
            def bar(self, f=42):
                return f
            def baz(*args, **kwds):
                return kwds['name'], kwds['self']
        a = A()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                f = methodcaller('bar')
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # positional args
                f = methodcaller('foo', 1, 2)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                # keyword args
                f = methodcaller('bar', f=5)
                f2 = self.copy(f, proto)
                self.assertEqual(repr(f2), repr(f))
                self.assertEqual(f2(a), f(a))
                f = methodcaller('baz', self='eggs', name='spam')
                f2 = self.copy(f, proto)
                # Can't test repr consistently with multiple keyword args
                self.assertEqual(f2(a), f(a))

class PyPyOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    module = py_operator
    module2 = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class PyCOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    module = py_operator
    module2 = c_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class CPyOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    module = c_operator
    module2 = py_operator

@unittest.skipUnless(c_operator, 'requires _operator')
class CCOperatorPickleTestCase(OperatorPickleTestCase, __TestCase):
    module = c_operator
    module2 = c_operator


if __name__ == "__main__":
    run_tests()
