# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.13/Lib/test/test_isinstance.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase
# ======= END DYNAMO PATCH =======

# Tests some corner cases with isinstance() and issubclass().  While these
# tests use new style classes and properties, they actually do whitebox
# testing of error conditions uncovered when using extension types.

import unittest
import typing
from test import support



class TestIsInstanceExceptions(__TestCase):
    # Test to make sure that an AttributeError when accessing the instance's
    # class's bases is masked.  This was actually a bug in Python 2.2 and
    # 2.2.1 where the exception wasn't caught but it also wasn't being cleared
    # (leading to an "undetected error" in the debug build).  Set up is,
    # isinstance(inst, cls) where:
    #
    # - cls isn't a type, or a tuple
    # - cls has a __bases__ attribute
    # - inst has a __class__ attribute
    # - inst.__class__ as no __bases__ attribute
    #
    # Sounds complicated, I know, but this mimics a situation where an
    # extension type raises an AttributeError when its __bases__ attribute is
    # gotten.  In that case, isinstance() should return False.
    def test_class_has_no_bases(self):
        class I(object):
            def getclass(self):
                # This must return an object that has no __bases__ attribute
                return None
            __class__ = property(getclass)

        class C(object):
            def getbases(self):
                return ()
            __bases__ = property(getbases)

        self.assertEqual(False, isinstance(I(), C()))

    # Like above except that inst.__class__.__bases__ raises an exception
    # other than AttributeError
    def test_bases_raises_other_than_attribute_error(self):
        class E(object):
            def getbases(self):
                raise RuntimeError
            __bases__ = property(getbases)

        class I(object):
            def getclass(self):
                return E()
            __class__ = property(getclass)

        class C(object):
            def getbases(self):
                return ()
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, isinstance, I(), C())

    # Here's a situation where getattr(cls, '__bases__') raises an exception.
    # If that exception is not AttributeError, it should not get masked
    def test_dont_mask_non_attribute_error(self):
        class I: pass

        class C(object):
            def getbases(self):
                raise RuntimeError
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, isinstance, I(), C())

    # Like above, except that getattr(cls, '__bases__') raises an
    # AttributeError, which /should/ get masked as a TypeError
    def test_mask_attribute_error(self):
        class I: pass

        class C(object):
            def getbases(self):
                raise AttributeError
            __bases__ = property(getbases)

        self.assertRaises(TypeError, isinstance, I(), C())

    # check that we don't mask non AttributeErrors
    # see: http://bugs.python.org/issue1574217
    def test_isinstance_dont_mask_non_attribute_error(self):
        class C(object):
            def getclass(self):
                raise RuntimeError
            __class__ = property(getclass)

        c = C()
        self.assertRaises(RuntimeError, isinstance, c, bool)

        # test another code path
        class D: pass
        self.assertRaises(RuntimeError, isinstance, c, D)


# These tests are similar to above, but tickle certain code paths in
# issubclass() instead of isinstance() -- really PyObject_IsSubclass()
# vs. PyObject_IsInstance().
class TestIsSubclassExceptions(__TestCase):
    def test_dont_mask_non_attribute_error(self):
        class C(object):
            def getbases(self):
                raise RuntimeError
            __bases__ = property(getbases)

        class S(C): pass

        self.assertRaises(RuntimeError, issubclass, C(), S())

    def test_mask_attribute_error(self):
        class C(object):
            def getbases(self):
                raise AttributeError
            __bases__ = property(getbases)

        class S(C): pass

        self.assertRaises(TypeError, issubclass, C(), S())

    # Like above, but test the second branch, where the __bases__ of the
    # second arg (the cls arg) is tested.  This means the first arg must
    # return a valid __bases__, and it's okay for it to be a normal --
    # unrelated by inheritance -- class.
    def test_dont_mask_non_attribute_error_in_cls_arg(self):
        class B: pass

        class C(object):
            def getbases(self):
                raise RuntimeError
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, issubclass, B, C())

    def test_mask_attribute_error_in_cls_arg(self):
        class B: pass

        class C(object):
            def getbases(self):
                raise AttributeError
            __bases__ = property(getbases)

        self.assertRaises(TypeError, issubclass, B, C())



# meta classes for creating abstract classes and instances
class AbstractClass(object):
    def __init__(self, bases):
        self.bases = bases

    def getbases(self):
        return self.bases
    __bases__ = property(getbases)

    def __call__(self):
        return AbstractInstance(self)

class AbstractInstance(object):
    def __init__(self, klass):
        self.klass = klass

    def getclass(self):
        return self.klass
    __class__ = property(getclass)

# abstract classes
AbstractSuper = AbstractClass(bases=())

AbstractChild = AbstractClass(bases=(AbstractSuper,))

# normal classes
class Super:
    pass

class Child(Super):
    pass

class TestIsInstanceIsSubclass(__TestCase):
    # Tests to ensure that isinstance and issubclass work on abstract
    # classes and instances.  Before the 2.2 release, TypeErrors were
    # raised when boolean values should have been returned.  The bug was
    # triggered by mixing 'normal' classes and instances were with
    # 'abstract' classes and instances.  This case tries to test all
    # combinations.

    def test_isinstance_normal(self):
        # normal instances
        self.assertEqual(True, isinstance(Super(), Super))
        self.assertEqual(False, isinstance(Super(), Child))
        self.assertEqual(False, isinstance(Super(), AbstractSuper))
        self.assertEqual(False, isinstance(Super(), AbstractChild))

        self.assertEqual(True, isinstance(Child(), Super))
        self.assertEqual(False, isinstance(Child(), AbstractSuper))

    def test_isinstance_abstract(self):
        # abstract instances
        self.assertEqual(True, isinstance(AbstractSuper(), AbstractSuper))
        self.assertEqual(False, isinstance(AbstractSuper(), AbstractChild))
        self.assertEqual(False, isinstance(AbstractSuper(), Super))
        self.assertEqual(False, isinstance(AbstractSuper(), Child))

        self.assertEqual(True, isinstance(AbstractChild(), AbstractChild))
        self.assertEqual(True, isinstance(AbstractChild(), AbstractSuper))
        self.assertEqual(False, isinstance(AbstractChild(), Super))
        self.assertEqual(False, isinstance(AbstractChild(), Child))

    def test_isinstance_with_or_union(self):
        self.assertTrue(isinstance(Super(), Super | int))
        self.assertFalse(isinstance(None, str | int))
        self.assertTrue(isinstance(3, str | int))
        self.assertTrue(isinstance("", str | int))
        self.assertTrue(isinstance([], typing.List | typing.Tuple))
        self.assertTrue(isinstance(2, typing.List | int))
        self.assertFalse(isinstance(2, typing.List | typing.Tuple))
        self.assertTrue(isinstance(None, int | None))
        self.assertFalse(isinstance(3.14, int | str))
        with self.assertRaises(TypeError):
            isinstance(2, list[int])
        with self.assertRaises(TypeError):
            isinstance(2, list[int] | int)
        with self.assertRaises(TypeError):
            isinstance(2, float | str | list[int] | int)



    def test_subclass_normal(self):
        # normal classes
        self.assertEqual(True, issubclass(Super, Super))
        self.assertEqual(False, issubclass(Super, AbstractSuper))
        self.assertEqual(False, issubclass(Super, Child))

        self.assertEqual(True, issubclass(Child, Child))
        self.assertEqual(True, issubclass(Child, Super))
        self.assertEqual(False, issubclass(Child, AbstractSuper))
        self.assertTrue(issubclass(typing.List, typing.List|typing.Tuple))
        self.assertFalse(issubclass(int, typing.List|typing.Tuple))

    def test_subclass_abstract(self):
        # abstract classes
        self.assertEqual(True, issubclass(AbstractSuper, AbstractSuper))
        self.assertEqual(False, issubclass(AbstractSuper, AbstractChild))
        self.assertEqual(False, issubclass(AbstractSuper, Child))

        self.assertEqual(True, issubclass(AbstractChild, AbstractChild))
        self.assertEqual(True, issubclass(AbstractChild, AbstractSuper))
        self.assertEqual(False, issubclass(AbstractChild, Super))
        self.assertEqual(False, issubclass(AbstractChild, Child))

    def test_subclass_tuple(self):
        # test with a tuple as the second argument classes
        self.assertEqual(True, issubclass(Child, (Child,)))
        self.assertEqual(True, issubclass(Child, (Super,)))
        self.assertEqual(False, issubclass(Super, (Child,)))
        self.assertEqual(True, issubclass(Super, (Child, Super)))
        self.assertEqual(False, issubclass(Child, ()))
        self.assertEqual(True, issubclass(Super, (Child, (Super,))))

        self.assertEqual(True, issubclass(int, (int, (float, int))))
        self.assertEqual(True, issubclass(str, (str, (Child, str))))

    def test_subclass_recursion_limit(self):
        # make sure that issubclass raises RecursionError before the C stack is
        # blown
        with support.infinite_recursion():
            self.assertRaises(RecursionError, blowstack, issubclass, str, str)

    def test_isinstance_recursion_limit(self):
        # make sure that issubclass raises RecursionError before the C stack is
        # blown
        with support.infinite_recursion():
            self.assertRaises(RecursionError, blowstack, isinstance, '', str)

    def test_subclass_with_union(self):
        self.assertTrue(issubclass(int, int | float | int))
        self.assertTrue(issubclass(str, str | Child | str))
        self.assertFalse(issubclass(dict, float|str))
        self.assertFalse(issubclass(object, float|str))
        with self.assertRaises(TypeError):
            issubclass(2, Child | Super)
        with self.assertRaises(TypeError):
            issubclass(int, list[int] | Child)

    def test_issubclass_refcount_handling(self):
        # bpo-39382: abstract_issubclass() didn't hold item reference while
        # peeking in the bases tuple, in the single inheritance case.
        class A:
            @property
            def __bases__(self):
                return (int, )

        class B:
            def __init__(self):
                # setting this here increases the chances of exhibiting the bug,
                # probably due to memory layout changes.
                self.x = 1

            @property
            def __bases__(self):
                return (A(), )

        self.assertEqual(True, issubclass(B(), int))

    def test_infinite_recursion_in_bases(self):
        class X:
            @property
            def __bases__(self):
                return self.__bases__
        with support.infinite_recursion(25):
            self.assertRaises(RecursionError, issubclass, X(), int)
            self.assertRaises(RecursionError, issubclass, int, X())
            self.assertRaises(RecursionError, isinstance, 1, X())

    def test_infinite_recursion_via_bases_tuple(self):
        """Regression test for bpo-30570."""
        class Failure(object):
            def __getattr__(self, attr):
                return (self, None)
        with support.infinite_recursion():
            with self.assertRaises(RecursionError):
                issubclass(Failure(), int)

    def test_infinite_cycle_in_bases(self):
        """Regression test for bpo-30570."""
        class X:
            @property
            def __bases__(self):
                return (self, self, self)
        with support.infinite_recursion():
            self.assertRaises(RecursionError, issubclass, X(), int)

    def test_infinitely_many_bases(self):
        """Regression test for bpo-30570."""
        class X:
            def __getattr__(self, attr):
                self.assertEqual(attr, "__bases__")
                class A:
                    pass
                class B:
                    pass
                A.__getattr__ = B.__getattr__ = X.__getattr__
                return (A(), B())
        with support.infinite_recursion(25):
            self.assertRaises(RecursionError, issubclass, X(), int)


def blowstack(fxn, arg, compare_to):
    # Make sure that calling isinstance with a deeply nested tuple for its
    # argument will raise RecursionError eventually.
    tuple_arg = (compare_to,)
    for cnt in range(support.exceeds_recursion_limit()):
        tuple_arg = (tuple_arg,)
        fxn(arg, tuple_arg)


if __name__ == '__main__':
    run_tests()
