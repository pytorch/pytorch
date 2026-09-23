# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_subclassinit.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import types
import unittest


class Test(CPythonTestCase):
    def test_init_subclass(self):
        class A:
            initialized = False

            def __init_subclass__(cls):
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            pass

        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_dict(self):
        class A(dict):
            initialized = False

            def __init_subclass__(cls):
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            pass

        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_kwargs(self):
        class A:
            def __init_subclass__(cls, **kwargs):
                cls.kwargs = kwargs

        class B(A, x=3):
            pass

        self.assertEqual(B.kwargs, dict(x=3))

    def test_init_subclass_error(self):
        class A:
            def __init_subclass__(cls):
                raise RuntimeError

        with self.assertRaises(RuntimeError):
            class B(A):
                pass

    def test_init_subclass_wrong(self):
        class A:
            def __init_subclass__(cls, whatever):
                pass

        with self.assertRaises(TypeError):
            class B(A):
                pass

    def test_init_subclass_skipped(self):
        class BaseWithInit:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.initialized = cls

        class BaseWithoutInit(BaseWithInit):
            pass

        class A(BaseWithoutInit):
            pass

        self.assertIs(A.initialized, A)
        self.assertIs(BaseWithoutInit.initialized, BaseWithoutInit)

    def test_init_subclass_diamond(self):
        class Base:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.calls = []

        class Left(Base):
            pass

        class Middle:
            def __init_subclass__(cls, middle, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.calls += [middle]

        class Right(Base):
            def __init_subclass__(cls, right="right", **kwargs):
                super().__init_subclass__(**kwargs)
                cls.calls += [right]

        class A(Left, Middle, Right, middle="middle"):
            pass

        self.assertEqual(A.calls, ["right", "middle"])
        self.assertEqual(Left.calls, [])
        self.assertEqual(Right.calls, [])

    def test_set_name(self):
        class Descriptor:
            def __set_name__(self, owner, name):
                self.owner = owner
                self.name = name

        class A:
            d = Descriptor()

        self.assertEqual(A.d.name, "d")
        self.assertIs(A.d.owner, A)

    def test_set_name_metaclass(self):
        class Meta(type):
            def __new__(cls, name, bases, ns):
                ret = super().__new__(cls, name, bases, ns)
                self.assertEqual(ret.d.name, "d")
                self.assertIs(ret.d.owner, ret)
                return 0

        class Descriptor:
            def __set_name__(self, owner, name):
                self.owner = owner
                self.name = name

        class A(metaclass=Meta):
            d = Descriptor()
        self.assertEqual(A, 0)

    def test_set_name_error(self):
        class Descriptor:
            def __set_name__(self, owner, name):
                1/0

        with self.assertRaises(ZeroDivisionError) as cm:
            class NotGoingToWork:
                attr = Descriptor()

        notes = cm.exception.__notes__
        self.assertRegex(str(notes), r'\bNotGoingToWork\b')
        self.assertRegex(str(notes), r'\battr\b')
        self.assertRegex(str(notes), r'\bDescriptor\b')

    def test_set_name_wrong(self):
        class Descriptor:
            def __set_name__(self):
                pass

        with self.assertRaises(TypeError) as cm:
            class NotGoingToWork:
                attr = Descriptor()

        notes = cm.exception.__notes__
        self.assertRegex(str(notes), r'\bNotGoingToWork\b')
        self.assertRegex(str(notes), r'\battr\b')
        self.assertRegex(str(notes), r'\bDescriptor\b')

    def test_set_name_lookup(self):
        resolved = []
        class NonDescriptor:
            def __getattr__(self, name):
                resolved.append(name)

        class A:
            d = NonDescriptor()

        self.assertNotIn('__set_name__', resolved,
                         '__set_name__ is looked up in instance dict')

    def test_set_name_init_subclass(self):
        class Descriptor:
            def __set_name__(self, owner, name):
                self.owner = owner
                self.name = name

        class Meta(type):
            def __new__(cls, name, bases, ns):
                self = super().__new__(cls, name, bases, ns)
                self.meta_owner = self.owner
                self.meta_name = self.name
                return self

        class A:
            def __init_subclass__(cls):
                cls.owner = cls.d.owner
                cls.name = cls.d.name

        class B(A, metaclass=Meta):
            d = Descriptor()

        self.assertIs(B.owner, B)
        self.assertEqual(B.name, 'd')
        self.assertIs(B.meta_owner, B)
        self.assertEqual(B.name, 'd')

    def test_set_name_modifying_dict(self):
        notified = []
        class Descriptor:
            def __set_name__(self, owner, name):
                setattr(owner, name + 'x', None)
                notified.append(name)

        class A:
            a = Descriptor()
            b = Descriptor()
            c = Descriptor()
            d = Descriptor()
            e = Descriptor()

        self.assertCountEqual(notified, ['a', 'b', 'c', 'd', 'e'])

    def test_errors(self):
        class MyMeta(type):
            pass

        with self.assertRaises(TypeError):
            class MyClass(metaclass=MyMeta, otherarg=1):
                pass

        with self.assertRaises(TypeError):
            types.new_class("MyClass", (object,),
                            dict(metaclass=MyMeta, otherarg=1))
        types.prepare_class("MyClass", (object,),
                            dict(metaclass=MyMeta, otherarg=1))

        class MyMeta(type):
            def __init__(self, name, bases, namespace, otherarg):
                super().__init__(name, bases, namespace)

        with self.assertRaises(TypeError):
            class MyClass2(metaclass=MyMeta, otherarg=1):
                pass

        class MyMeta(type):
            def __new__(cls, name, bases, namespace, otherarg):
                return super().__new__(cls, name, bases, namespace)

            def __init__(self, name, bases, namespace, otherarg):
                super().__init__(name, bases, namespace)
                self.otherarg = otherarg

        class MyClass3(metaclass=MyMeta, otherarg=1):
            pass

        self.assertEqual(MyClass3.otherarg, 1)

    def test_errors_changed_pep487(self):
        # These tests failed before Python 3.6, PEP 487
        class MyMeta(type):
            def __new__(cls, name, bases, namespace):
                return super().__new__(cls, name=name, bases=bases,
                                       dict=namespace)

        with self.assertRaises(TypeError):
            class MyClass(metaclass=MyMeta):
                pass

        class MyMeta(type):
            def __new__(cls, name, bases, namespace, otherarg):
                self = super().__new__(cls, name, bases, namespace)
                self.otherarg = otherarg
                return self

        class MyClass2(metaclass=MyMeta, otherarg=1):
            pass

        self.assertEqual(MyClass2.otherarg, 1)

    def test_type(self):
        t = type('NewClass', (object,), {})
        self.assertIsInstance(t, type)
        self.assertEqual(t.__name__, 'NewClass')

        with self.assertRaises(TypeError):
            type(name='NewClass', bases=(object,), dict={})


if __name__ == "__main__":
    run_tests()
