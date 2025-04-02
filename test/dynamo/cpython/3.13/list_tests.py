# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHDYNAMO,
    run_tests,
)

if TEST_WITH_TORCHDYNAMO:
    __TestCase = CPythonTestCase
else:
    __TestCase = unittest.TestCase

# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

"""
Tests common to list and UserList.UserList
"""

import sys
from functools import cmp_to_key

import seq_tests
from test.support import ALWAYS_EQ, NEVER_EQ, get_c_recursion_limit


class CommonTest(seq_tests.CommonTest):

    def test_init(self):
        # Iterable arg is optional
        self.assertEqual(self.type2test([]), self.type2test())

        # Init clears previous values
        a = self.type2test([1, 2, 3])
        a.__init__()
        self.assertEqual(a, self.type2test([]))

        # Init overwrites previous values
        a = self.type2test([1, 2, 3])
        a.__init__([4, 5, 6])
        self.assertEqual(a, self.type2test([4, 5, 6]))

        # Mutables always return a new object
        b = self.type2test(a)
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(a, b)

    def test_getitem_error(self):
        a = []
        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a['a']

    def test_setitem_error(self):
        a = []
        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a['a'] = "python"

    def test_repr(self):
        l0 = []
        l2 = [0, 1, 2]
        a0 = self.type2test(l0)
        a2 = self.type2test(l2)

        self.assertEqual(str(a0), str(l0))
        self.assertEqual(repr(a0), repr(l0))
        self.assertEqual(repr(a2), repr(l2))
        self.assertEqual(str(a2), "[0, 1, 2]")
        self.assertEqual(repr(a2), "[0, 1, 2]")

        a2.append(a2)
        a2.append(3)
        self.assertEqual(str(a2), "[0, 1, 2, [...], 3]")
        self.assertEqual(repr(a2), "[0, 1, 2, [...], 3]")

    def test_repr_deep(self):
        a = self.type2test([])
        for i in range(get_c_recursion_limit() + 1):
            a = self.type2test([a])
        self.assertRaises(RecursionError, repr, a)

    def test_set_subscript(self):
        a = self.type2test(range(20))
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 0), [1,2,3])
        self.assertRaises(TypeError, a.__setitem__, slice(0, 10), 1)
        self.assertRaises(ValueError, a.__setitem__, slice(0, 10, 2), [1,2])
        self.assertRaises(TypeError, a.__getitem__, 'x', 1)
        a[slice(2,10,3)] = [1,2,3]
        self.assertEqual(a, self.type2test([0, 1, 1, 3, 4, 2, 6, 7, 3,
                                            9, 10, 11, 12, 13, 14, 15,
                                            16, 17, 18, 19]))

    def test_reversed(self):
        a = self.type2test(range(20))
        r = reversed(a)
        self.assertEqual(list(r), self.type2test(range(19, -1, -1)))
        self.assertRaises(StopIteration, next, r)
        self.assertEqual(list(reversed(self.type2test())),
                         self.type2test())
        # Bug 3689: make sure list-reversed-iterator doesn't have __len__
        self.assertRaises(TypeError, len, reversed([1,2,3]))

    def test_setitem(self):
        a = self.type2test([0, 1])
        a[0] = 0
        a[1] = 100
        self.assertEqual(a, self.type2test([0, 100]))
        a[-1] = 200
        self.assertEqual(a, self.type2test([0, 200]))
        a[-2] = 100
        self.assertEqual(a, self.type2test([100, 200]))
        self.assertRaises(IndexError, a.__setitem__, -3, 200)
        self.assertRaises(IndexError, a.__setitem__, 2, 200)

        a = self.type2test([])
        self.assertRaises(IndexError, a.__setitem__, 0, 200)
        self.assertRaises(IndexError, a.__setitem__, -1, 200)
        self.assertRaises(TypeError, a.__setitem__)

        a = self.type2test([0,1,2,3,4])
        a[0] = 1
        a[1] = 2
        a[2] = 3
        self.assertEqual(a, self.type2test([1,2,3,3,4]))
        a[0] = 5
        a[1] = 6
        a[2] = 7
        self.assertEqual(a, self.type2test([5,6,7,3,4]))
        a[-2] = 88
        a[-1] = 99
        self.assertEqual(a, self.type2test([5,6,7,88,99]))
        a[-2] = 8
        a[-1] = 9
        self.assertEqual(a, self.type2test([5,6,7,8,9]))

        msg = "list indices must be integers or slices"
        with self.assertRaisesRegex(TypeError, msg):
            a['a'] = "python"

    def test_delitem(self):
        a = self.type2test([0, 1])
        del a[1]
        self.assertEqual(a, [0])
        del a[0]
        self.assertEqual(a, [])

        a = self.type2test([0, 1])
        del a[-2]
        self.assertEqual(a, [1])
        del a[-1]
        self.assertEqual(a, [])

        a = self.type2test([0, 1])
        self.assertRaises(IndexError, a.__delitem__, -3)
        self.assertRaises(IndexError, a.__delitem__, 2)

        a = self.type2test([])
        self.assertRaises(IndexError, a.__delitem__, 0)

        self.assertRaises(TypeError, a.__delitem__)

    def test_setslice(self):
        l = [0, 1]
        a = self.type2test(l)

        for i in range(-3, 4):
            a[:i] = l[:i]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[:i] = a[:i]
            self.assertEqual(a2, a)
            a[i:] = l[i:]
            self.assertEqual(a, l)
            a2 = a[:]
            a2[i:] = a[i:]
            self.assertEqual(a2, a)
            for j in range(-3, 4):
                a[i:j] = l[i:j]
                self.assertEqual(a, l)
                a2 = a[:]
                a2[i:j] = a[i:j]
                self.assertEqual(a2, a)

        aa2 = a2[:]
        aa2[:0] = [-2, -1]
        self.assertEqual(aa2, [-2, -1, 0, 1])
        aa2[0:] = []
        self.assertEqual(aa2, [])

        a = self.type2test([1, 2, 3, 4, 5])
        a[:-1] = a
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 5]))
        a = self.type2test([1, 2, 3, 4, 5])
        a[1:] = a
        self.assertEqual(a, self.type2test([1, 1, 2, 3, 4, 5]))
        a = self.type2test([1, 2, 3, 4, 5])
        a[1:-1] = a
        self.assertEqual(a, self.type2test([1, 1, 2, 3, 4, 5, 5]))

        a = self.type2test([])
        a[:] = tuple(range(10))
        self.assertEqual(a, self.type2test(range(10)))

        self.assertRaises(TypeError, a.__setitem__, slice(0, 1, 5))

        self.assertRaises(TypeError, a.__setitem__)

    def test_delslice(self):
        a = self.type2test([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, self.type2test([]))

        a = self.type2test([0, 1])
        del a[1:2]
        del a[0:1]
        self.assertEqual(a, self.type2test([]))

        a = self.type2test([0, 1])
        del a[-2:-1]
        self.assertEqual(a, self.type2test([1]))

        a = self.type2test([0, 1])
        del a[-2:-1]
        self.assertEqual(a, self.type2test([1]))

        a = self.type2test([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, self.type2test([]))

        a = self.type2test([0, 1])
        del a[1:]
        del a[:1]
        self.assertEqual(a, self.type2test([]))

        a = self.type2test([0, 1])
        del a[-1:]
        self.assertEqual(a, self.type2test([0]))

        a = self.type2test([0, 1])
        del a[-1:]
        self.assertEqual(a, self.type2test([0]))

        a = self.type2test([0, 1])
        del a[:]
        self.assertEqual(a, self.type2test([]))

    def test_append(self):
        a = self.type2test([])
        a.append(0)
        a.append(1)
        a.append(2)
        self.assertEqual(a, self.type2test([0, 1, 2]))

        self.assertRaises(TypeError, a.append)

    def test_extend(self):
        a1 = self.type2test([0])
        a2 = self.type2test((0, 1))
        a = a1[:]
        a.extend(a2)
        self.assertEqual(a, a1 + a2)

        a.extend(self.type2test([]))
        self.assertEqual(a, a1 + a2)

        a.extend(a)
        self.assertEqual(a, self.type2test([0, 0, 1, 0, 0, 1]))

        a = self.type2test("spam")
        a.extend("eggs")
        self.assertEqual(a, list("spameggs"))

        self.assertRaises(TypeError, a.extend, None)
        self.assertRaises(TypeError, a.extend)

        # overflow test. issue1621
        class CustomIter:
            def __iter__(self):
                return self
            def __next__(self):
                raise StopIteration
            def __length_hint__(self):
                return sys.maxsize
        a = self.type2test([1,2,3,4])
        a.extend(CustomIter())
        self.assertEqual(a, [1,2,3,4])


    def test_insert(self):
        a = self.type2test([0, 1, 2])
        a.insert(0, -2)
        a.insert(1, -1)
        a.insert(2, 0)
        self.assertEqual(a, [-2, -1, 0, 0, 1, 2])

        b = a[:]
        b.insert(-2, "foo")
        b.insert(-200, "left")
        b.insert(200, "right")
        self.assertEqual(b, self.type2test(["left",-2,-1,0,0,"foo",1,2,"right"]))

        self.assertRaises(TypeError, a.insert)

    def test_pop(self):
        a = self.type2test([-1, 0, 1])
        a.pop()
        self.assertEqual(a, [-1, 0])
        a.pop(0)
        self.assertEqual(a, [0])
        self.assertRaises(IndexError, a.pop, 5)
        a.pop(0)
        self.assertEqual(a, [])
        self.assertRaises(IndexError, a.pop)
        self.assertRaises(TypeError, a.pop, 42, 42)
        a = self.type2test([0, 10, 20, 30, 40])

    def test_remove(self):
        a = self.type2test([0, 0, 1])
        a.remove(1)
        self.assertEqual(a, [0, 0])
        a.remove(0)
        self.assertEqual(a, [0])
        a.remove(0)
        self.assertEqual(a, [])

        self.assertRaises(ValueError, a.remove, 0)

        self.assertRaises(TypeError, a.remove)

        a = self.type2test([1, 2])
        self.assertRaises(ValueError, a.remove, NEVER_EQ)
        self.assertEqual(a, [1, 2])
        a.remove(ALWAYS_EQ)
        self.assertEqual(a, [2])
        a = self.type2test([ALWAYS_EQ])
        a.remove(1)
        self.assertEqual(a, [])
        a = self.type2test([ALWAYS_EQ])
        a.remove(NEVER_EQ)
        self.assertEqual(a, [])
        a = self.type2test([NEVER_EQ])
        self.assertRaises(ValueError, a.remove, ALWAYS_EQ)

        class BadExc(Exception):
            pass

        class BadCmp:
            def __eq__(self, other):
                if other == 2:
                    raise BadExc()
                return False

        a = self.type2test([0, 1, 2, 3])
        self.assertRaises(BadExc, a.remove, BadCmp())

        class BadCmp2:
            def __eq__(self, other):
                raise BadExc()

        d = self.type2test('abcdefghcij')
        d.remove('c')
        self.assertEqual(d, self.type2test('abdefghcij'))
        d.remove('c')
        self.assertEqual(d, self.type2test('abdefghij'))
        self.assertRaises(ValueError, d.remove, 'c')
        self.assertEqual(d, self.type2test('abdefghij'))

        # Handle comparison errors
        d = self.type2test(['a', 'b', BadCmp2(), 'c'])
        e = self.type2test(d)
        self.assertRaises(BadExc, d.remove, 'c')
        for x, y in zip(d, e):
            # verify that original order and values are retained.
            self.assertIs(x, y)

    def test_index(self):
        super().test_index()
        a = self.type2test([-2, -1, 0, 0, 1, 2])
        a.remove(0)
        self.assertRaises(ValueError, a.index, 2, 0, 4)
        self.assertEqual(a, self.type2test([-2, -1, 0, 1, 2]))

        # Test modifying the list during index's iteration
        class EvilCmp:
            def __init__(self, victim):
                self.victim = victim
            def __eq__(self, other):
                del self.victim[:]
                return False
        a = self.type2test()
        a[:] = [EvilCmp(a) for _ in range(100)]
        # This used to seg fault before patch #1005778
        self.assertRaises(ValueError, a.index, None)

    def test_reverse(self):
        u = self.type2test([-2, -1, 0, 1, 2])
        u2 = u[:]
        u.reverse()
        self.assertEqual(u, [2, 1, 0, -1, -2])
        u.reverse()
        self.assertEqual(u, u2)

        self.assertRaises(TypeError, u.reverse, 42)

    def test_clear(self):
        u = self.type2test([2, 3, 4])
        u.clear()
        self.assertEqual(u, [])

        u = self.type2test([])
        u.clear()
        self.assertEqual(u, [])

        u = self.type2test([])
        u.append(1)
        u.clear()
        u.append(2)
        self.assertEqual(u, [2])

        self.assertRaises(TypeError, u.clear, None)

    def test_copy(self):
        u = self.type2test([1, 2, 3])
        v = u.copy()
        self.assertEqual(v, [1, 2, 3])

        u = self.type2test([])
        v = u.copy()
        self.assertEqual(v, [])

        # test that it's indeed a copy and not a reference
        u = self.type2test(['a', 'b'])
        v = u.copy()
        v.append('i')
        self.assertEqual(u, ['a', 'b'])
        self.assertEqual(v, u + ['i'])

        # test that it's a shallow, not a deep copy
        u = self.type2test([1, 2, [3, 4], 5])
        v = u.copy()
        self.assertEqual(u, v)
        self.assertIs(v[3], u[3])

        self.assertRaises(TypeError, u.copy, None)

    def test_sort(self):
        u = self.type2test([1, 0])
        u.sort()
        self.assertEqual(u, [0, 1])

        u = self.type2test([2,1,0,-1,-2])
        u.sort()
        self.assertEqual(u, self.type2test([-2,-1,0,1,2]))

        self.assertRaises(TypeError, u.sort, 42, 42)

        def revcmp(a, b):
            if a == b:
                return 0
            elif a < b:
                return 1
            else: # a > b
                return -1
        u.sort(key=cmp_to_key(revcmp))
        self.assertEqual(u, self.type2test([2,1,0,-1,-2]))

        # The following dumps core in unpatched Python 1.5:
        def myComparison(x,y):
            xmod, ymod = x%3, y%7
            if xmod == ymod:
                return 0
            elif xmod < ymod:
                return -1
            else: # xmod > ymod
                return 1
        z = self.type2test(range(12))
        z.sort(key=cmp_to_key(myComparison))

        self.assertRaises(TypeError, z.sort, 2)

        def selfmodifyingComparison(x,y):
            z.append(1)
            if x == y:
                return 0
            elif x < y:
                return -1
            else: # x > y
                return 1
        self.assertRaises(ValueError, z.sort,
                          key=cmp_to_key(selfmodifyingComparison))

        self.assertRaises(TypeError, z.sort, 42, 42, 42, 42)

    def test_slice(self):
        u = self.type2test("spam")
        u[:2] = "h"
        self.assertEqual(u, list("ham"))

    def test_iadd(self):
        super().test_iadd()
        u = self.type2test([0, 1])
        u2 = u
        u += [2, 3]
        self.assertIs(u, u2)

        u = self.type2test("spam")
        u += "eggs"
        self.assertEqual(u, self.type2test("spameggs"))

        self.assertRaises(TypeError, u.__iadd__, None)

    def test_imul(self):
        super().test_imul()
        s = self.type2test([])
        oldid = id(s)
        s *= 10
        self.assertEqual(id(s), oldid)

    def test_extendedslicing(self):
        #  subscript
        a = self.type2test([0,1,2,3,4])

        #  deletion
        del a[::2]
        self.assertEqual(a, self.type2test([1,3]))
        a = self.type2test(range(5))
        del a[1::2]
        self.assertEqual(a, self.type2test([0,2,4]))
        a = self.type2test(range(5))
        del a[1::-2]
        self.assertEqual(a, self.type2test([0,2,3,4]))
        a = self.type2test(range(10))
        del a[::1000]
        self.assertEqual(a, self.type2test([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #  assignment
        a = self.type2test(range(10))
        a[::2] = [-1]*5
        self.assertEqual(a, self.type2test([-1, 1, -1, 3, -1, 5, -1, 7, -1, 9]))
        a = self.type2test(range(10))
        a[::-4] = [10]*3
        self.assertEqual(a, self.type2test([0, 10, 2, 3, 4, 10, 6, 7, 8 ,10]))
        a = self.type2test(range(4))
        a[::-1] = a
        self.assertEqual(a, self.type2test([3, 2, 1, 0]))
        a = self.type2test(range(10))
        b = a[:]
        c = a[:]
        a[2:3] = self.type2test(["two", "elements"])
        b[slice(2,3)] = self.type2test(["two", "elements"])
        c[2:3:] = self.type2test(["two", "elements"])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        a = self.type2test(range(10))
        a[::2] = tuple(range(5))
        self.assertEqual(a, self.type2test([0, 1, 1, 3, 2, 5, 3, 7, 4, 9]))
        # test issue7788
        a = self.type2test(range(10))
        del a[9::1<<333]

    def test_constructor_exception_handling(self):
        # Bug #1242657
        class F(object):
            def __iter__(self):
                raise KeyboardInterrupt
        self.assertRaises(KeyboardInterrupt, list, F())

    def test_exhausted_iterator(self):
        a = self.type2test([1, 2, 3])
        exhit = iter(a)
        empit = iter(a)
        for x in exhit:  # exhaust the iterator
            next(empit)  # not exhausted
        a.append(9)
        self.assertEqual(list(exhit), [])
        self.assertEqual(list(empit), [9])
        self.assertEqual(a, self.type2test([1, 2, 3, 9]))

        # gh-115733: Crash when iterating over exhausted iterator
        exhit = iter(self.type2test([1, 2, 3]))
        for _ in exhit:
            next(exhit, 1)
