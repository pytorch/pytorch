# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_abc.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

# Note: each test is run with Python and C versions of ABCMeta. Except for
# test_ABC_helper(), which assures that abc.ABC is an instance of abc.ABCMeta.

"""Unit tests for abc.py."""

import unittest

import abc
import _py_abc
from inspect import isabstract

def test_factory(abc_ABCMeta, abc_get_cache_token):
    class TestLegacyAPI(CPythonTestCase):

        def test_abstractproperty_basics(self):
            @abc.abstractproperty
            def foo(self): pass
            self.assertTrue(foo.__isabstractmethod__)
            def bar(self): pass
            self.assertFalse(hasattr(bar, "__isabstractmethod__"))

            class C(metaclass=abc_ABCMeta):
                @abc.abstractproperty
                def foo(self): return 3
            self.assertRaises(TypeError, C)
            class D(C):
                @property
                def foo(self): return super().foo
            self.assertEqual(D().foo, 3)
            self.assertFalse(getattr(D.foo, "__isabstractmethod__", False))

        def test_abstractclassmethod_basics(self):
            @abc.abstractclassmethod
            def foo(cls): pass
            self.assertTrue(foo.__isabstractmethod__)
            @classmethod
            def bar(cls): pass
            self.assertFalse(getattr(bar, "__isabstractmethod__", False))

            class C(metaclass=abc_ABCMeta):
                @abc.abstractclassmethod
                def foo(cls): return cls.__name__
            self.assertRaises(TypeError, C)
            class D(C):
                @classmethod
                def foo(cls): return super().foo()
            self.assertEqual(D.foo(), 'D')
            self.assertEqual(D().foo(), 'D')

        def test_abstractstaticmethod_basics(self):
            @abc.abstractstaticmethod
            def foo(): pass
            self.assertTrue(foo.__isabstractmethod__)
            @staticmethod
            def bar(): pass
            self.assertFalse(getattr(bar, "__isabstractmethod__", False))

            class C(metaclass=abc_ABCMeta):
                @abc.abstractstaticmethod
                def foo(): return 3
            self.assertRaises(TypeError, C)
            class D(C):
                @staticmethod
                def foo(): return 4
            self.assertEqual(D.foo(), 4)
            self.assertEqual(D().foo(), 4)


    class TestABC(CPythonTestCase):

        def test_ABC_helper(self):
            # create an ABC using the helper class and perform basic checks
            class C(abc.ABC):
                @classmethod
                @abc.abstractmethod
                def foo(cls): return cls.__name__
            self.assertEqual(type(C), abc.ABCMeta)
            self.assertRaises(TypeError, C)
            class D(C):
                @classmethod
                def foo(cls): return super().foo()
            self.assertEqual(D.foo(), 'D')

        def test_abstractmethod_basics(self):
            @abc.abstractmethod
            def foo(self): pass
            self.assertTrue(foo.__isabstractmethod__)
            def bar(self): pass
            self.assertFalse(hasattr(bar, "__isabstractmethod__"))

        def test_abstractproperty_basics(self):
            @property
            @abc.abstractmethod
            def foo(self): pass
            self.assertTrue(foo.__isabstractmethod__)
            def bar(self): pass
            self.assertFalse(getattr(bar, "__isabstractmethod__", False))

            class C(metaclass=abc_ABCMeta):
                @property
                @abc.abstractmethod
                def foo(self): return 3
            self.assertRaises(TypeError, C)
            class D(C):
                @C.foo.getter
                def foo(self): return super().foo
            self.assertEqual(D().foo, 3)

        def test_abstractclassmethod_basics(self):
            @classmethod
            @abc.abstractmethod
            def foo(cls): pass
            self.assertTrue(foo.__isabstractmethod__)
            @classmethod
            def bar(cls): pass
            self.assertFalse(getattr(bar, "__isabstractmethod__", False))

            class C(metaclass=abc_ABCMeta):
                @classmethod
                @abc.abstractmethod
                def foo(cls): return cls.__name__
            self.assertRaises(TypeError, C)
            class D(C):
                @classmethod
                def foo(cls): return super().foo()
            self.assertEqual(D.foo(), 'D')
            self.assertEqual(D().foo(), 'D')

        def test_abstractstaticmethod_basics(self):
            @staticmethod
            @abc.abstractmethod
            def foo(): pass
            self.assertTrue(foo.__isabstractmethod__)
            @staticmethod
            def bar(): pass
            self.assertFalse(getattr(bar, "__isabstractmethod__", False))

            class C(metaclass=abc_ABCMeta):
                @staticmethod
                @abc.abstractmethod
                def foo(): return 3
            self.assertRaises(TypeError, C)
            class D(C):
                @staticmethod
                def foo(): return 4
            self.assertEqual(D.foo(), 4)
            self.assertEqual(D().foo(), 4)

        def test_object_new_with_one_abstractmethod(self):
            class C(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def method_one(self):
                    pass
            msg = r"class C without an implementation for abstract method 'method_one'"
            self.assertRaisesRegex(TypeError, msg, C)

        def test_object_new_with_many_abstractmethods(self):
            class C(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def method_one(self):
                    pass
                @abc.abstractmethod
                def method_two(self):
                    pass
            msg = r"class C without an implementation for abstract methods 'method_one', 'method_two'"
            self.assertRaisesRegex(TypeError, msg, C)

        def test_abstractmethod_integration(self):
            for abstractthing in [abc.abstractmethod, abc.abstractproperty,
                                  abc.abstractclassmethod,
                                  abc.abstractstaticmethod]:
                class C(metaclass=abc_ABCMeta):
                    @abstractthing
                    def foo(self): pass  # abstract
                    def bar(self): pass  # concrete
                self.assertEqual(C.__abstractmethods__, {"foo"})
                self.assertRaises(TypeError, C)  # because foo is abstract
                self.assertTrue(isabstract(C))
                class D(C):
                    def bar(self): pass  # concrete override of concrete
                self.assertEqual(D.__abstractmethods__, {"foo"})
                self.assertRaises(TypeError, D)  # because foo is still abstract
                self.assertTrue(isabstract(D))
                class E(D):
                    def foo(self): pass
                self.assertEqual(E.__abstractmethods__, set())
                E()  # now foo is concrete, too
                self.assertFalse(isabstract(E))
                class F(E):
                    @abstractthing
                    def bar(self): pass  # abstract override of concrete
                self.assertEqual(F.__abstractmethods__, {"bar"})
                self.assertRaises(TypeError, F)  # because bar is abstract now
                self.assertTrue(isabstract(F))

        def test_descriptors_with_abstractmethod(self):
            class C(metaclass=abc_ABCMeta):
                @property
                @abc.abstractmethod
                def foo(self): return 3
                @foo.setter
                @abc.abstractmethod
                def foo(self, val): pass
            self.assertRaises(TypeError, C)
            class D(C):
                @C.foo.getter
                def foo(self): return super().foo
            self.assertRaises(TypeError, D)
            class E(D):
                @D.foo.setter
                def foo(self, val): pass
            self.assertEqual(E().foo, 3)
            # check that the property's __isabstractmethod__ descriptor does the
            # right thing when presented with a value that fails truth testing:
            class NotBool(object):
                def __bool__(self):
                    raise ValueError()
                __len__ = __bool__
            with self.assertRaises(ValueError):
                class F(C):
                    def bar(self):
                        pass
                    bar.__isabstractmethod__ = NotBool()
                    foo = property(bar)


        def test_customdescriptors_with_abstractmethod(self):
            class Descriptor:
                def __init__(self, fget, fset=None):
                    self._fget = fget
                    self._fset = fset
                def getter(self, callable):
                    return Descriptor(callable, self._fget)
                def setter(self, callable):
                    return Descriptor(self._fget, callable)
                @property
                def __isabstractmethod__(self):
                    return (getattr(self._fget, '__isabstractmethod__', False)
                            or getattr(self._fset, '__isabstractmethod__', False))
            class C(metaclass=abc_ABCMeta):
                @Descriptor
                @abc.abstractmethod
                def foo(self): return 3
                @foo.setter
                @abc.abstractmethod
                def foo(self, val): pass
            self.assertRaises(TypeError, C)
            class D(C):
                @C.foo.getter
                def foo(self): return super().foo
            self.assertRaises(TypeError, D)
            class E(D):
                @D.foo.setter
                def foo(self, val): pass
            self.assertFalse(E.foo.__isabstractmethod__)

        def test_metaclass_abc(self):
            # Metaclasses can be ABCs, too.
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def x(self):
                    pass
            self.assertEqual(A.__abstractmethods__, {"x"})
            class meta(type, A):
                def x(self):
                    return 1
            class C(metaclass=meta):
                pass

        def test_registration_basics(self):
            class A(metaclass=abc_ABCMeta):
                pass
            class B(object):
                pass
            b = B()
            self.assertFalse(issubclass(B, A))
            self.assertFalse(issubclass(B, (A,)))
            self.assertNotIsInstance(b, A)
            self.assertNotIsInstance(b, (A,))
            B1 = A.register(B)
            self.assertTrue(issubclass(B, A))
            self.assertTrue(issubclass(B, (A,)))
            self.assertIsInstance(b, A)
            self.assertIsInstance(b, (A,))
            self.assertIs(B1, B)
            class C(B):
                pass
            c = C()
            self.assertTrue(issubclass(C, A))
            self.assertTrue(issubclass(C, (A,)))
            self.assertIsInstance(c, A)
            self.assertIsInstance(c, (A,))

        def test_register_as_class_deco(self):
            class A(metaclass=abc_ABCMeta):
                pass
            @A.register
            class B(object):
                pass
            b = B()
            self.assertTrue(issubclass(B, A))
            self.assertTrue(issubclass(B, (A,)))
            self.assertIsInstance(b, A)
            self.assertIsInstance(b, (A,))
            @A.register
            class C(B):
                pass
            c = C()
            self.assertTrue(issubclass(C, A))
            self.assertTrue(issubclass(C, (A,)))
            self.assertIsInstance(c, A)
            self.assertIsInstance(c, (A,))
            self.assertIs(C, A.register(C))

        def test_isinstance_invalidation(self):
            class A(metaclass=abc_ABCMeta):
                pass
            class B:
                pass
            b = B()
            self.assertFalse(isinstance(b, A))
            self.assertFalse(isinstance(b, (A,)))
            token_old = abc_get_cache_token()
            A.register(B)
            token_new = abc_get_cache_token()
            self.assertGreater(token_new, token_old)
            self.assertTrue(isinstance(b, A))
            self.assertTrue(isinstance(b, (A,)))

        def test_registration_builtins(self):
            class A(metaclass=abc_ABCMeta):
                pass
            A.register(int)
            self.assertIsInstance(42, A)
            self.assertIsInstance(42, (A,))
            self.assertTrue(issubclass(int, A))
            self.assertTrue(issubclass(int, (A,)))
            class B(A):
                pass
            B.register(str)
            class C(str): pass
            self.assertIsInstance("", A)
            self.assertIsInstance("", (A,))
            self.assertTrue(issubclass(str, A))
            self.assertTrue(issubclass(str, (A,)))
            self.assertTrue(issubclass(C, A))
            self.assertTrue(issubclass(C, (A,)))

        def test_registration_edge_cases(self):
            class A(metaclass=abc_ABCMeta):
                pass
            A.register(A)  # should pass silently
            class A1(A):
                pass
            self.assertRaises(RuntimeError, A1.register, A)  # cycles not allowed
            class B(object):
                pass
            A1.register(B)  # ok
            A1.register(B)  # should pass silently
            class C(A):
                pass
            A.register(C)  # should pass silently
            self.assertRaises(RuntimeError, C.register, A)  # cycles not allowed
            C.register(B)  # ok

        def test_register_non_class(self):
            class A(metaclass=abc_ABCMeta):
                pass
            self.assertRaisesRegex(TypeError, "Can only register classes",
                                   A.register, 4)

        def test_registration_transitiveness(self):
            class A(metaclass=abc_ABCMeta):
                pass
            self.assertTrue(issubclass(A, A))
            self.assertTrue(issubclass(A, (A,)))
            class B(metaclass=abc_ABCMeta):
                pass
            self.assertFalse(issubclass(A, B))
            self.assertFalse(issubclass(A, (B,)))
            self.assertFalse(issubclass(B, A))
            self.assertFalse(issubclass(B, (A,)))
            class C(metaclass=abc_ABCMeta):
                pass
            A.register(B)
            class B1(B):
                pass
            self.assertTrue(issubclass(B1, A))
            self.assertTrue(issubclass(B1, (A,)))
            class C1(C):
                pass
            B1.register(C1)
            self.assertFalse(issubclass(C, B))
            self.assertFalse(issubclass(C, (B,)))
            self.assertFalse(issubclass(C, B1))
            self.assertFalse(issubclass(C, (B1,)))
            self.assertTrue(issubclass(C1, A))
            self.assertTrue(issubclass(C1, (A,)))
            self.assertTrue(issubclass(C1, B))
            self.assertTrue(issubclass(C1, (B,)))
            self.assertTrue(issubclass(C1, B1))
            self.assertTrue(issubclass(C1, (B1,)))
            C1.register(int)
            class MyInt(int):
                pass
            self.assertTrue(issubclass(MyInt, A))
            self.assertTrue(issubclass(MyInt, (A,)))
            self.assertIsInstance(42, A)
            self.assertIsInstance(42, (A,))

        def test_issubclass_bad_arguments(self):
            class A(metaclass=abc_ABCMeta):
                pass

            with self.assertRaises(TypeError):
                issubclass({}, A)  # unhashable

            with self.assertRaises(TypeError):
                issubclass(42, A)  # No __mro__

            # Python version supports any iterable as __mro__.
            # But it's implementation detail and don't emulate it in C version.
            class C:
                __mro__ = 42  # __mro__ is not tuple

            with self.assertRaises(TypeError):
                issubclass(C(), A)

            # bpo-34441: Check that issubclass() doesn't crash on bogus
            # classes.
            bogus_subclasses = [
                None,
                lambda x: [],
                lambda: 42,
                lambda: [42],
            ]

            for i, func in enumerate(bogus_subclasses):
                class S(metaclass=abc_ABCMeta):
                    __subclasses__ = func

                with self.subTest(i=i):
                    with self.assertRaises(TypeError):
                        issubclass(int, S)

            # Also check that issubclass() propagates exceptions raised by
            # __subclasses__.
            class CustomError(Exception): ...
            exc_msg = "exception from __subclasses__"

            def raise_exc():
                raise CustomError(exc_msg)

            class S(metaclass=abc_ABCMeta):
                __subclasses__ = raise_exc

            with self.assertRaisesRegex(CustomError, exc_msg):
                issubclass(int, S)

        def test_subclasshook(self):
            class A(metaclass=abc.ABCMeta):
                @classmethod
                def __subclasshook__(cls, C):
                    if cls is A:
                        return 'foo' in C.__dict__
                    return NotImplemented
            self.assertFalse(issubclass(A, A))
            self.assertFalse(issubclass(A, (A,)))
            class B:
                foo = 42
            self.assertTrue(issubclass(B, A))
            self.assertTrue(issubclass(B, (A,)))
            class C:
                spam = 42
            self.assertFalse(issubclass(C, A))
            self.assertFalse(issubclass(C, (A,)))

        def test_all_new_methods_are_called(self):
            class A(metaclass=abc_ABCMeta):
                pass
            class B(object):
                counter = 0
                def __new__(cls):
                    B.counter += 1
                    return super().__new__(cls)
            class C(A, B):
                pass
            self.assertEqual(B.counter, 0)
            C()
            self.assertEqual(B.counter, 1)

        def test_ABC_has___slots__(self):
            self.assertTrue(hasattr(abc.ABC, '__slots__'))

        def test_tricky_new_works(self):
            def with_metaclass(meta, *bases):
                class metaclass(type):
                    def __new__(cls, name, this_bases, d):
                        return meta(name, bases, d)
                return type.__new__(metaclass, 'temporary_class', (), {})
            class A: ...
            class B: ...
            class C(with_metaclass(abc_ABCMeta, A, B)):
                pass
            self.assertEqual(C.__class__, abc_ABCMeta)

        def test_update_del(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            del A.foo
            self.assertEqual(A.__abstractmethods__, {'foo'})
            self.assertFalse(hasattr(A, 'foo'))

            abc.update_abstractmethods(A)

            self.assertEqual(A.__abstractmethods__, set())
            A()


        def test_update_new_abstractmethods(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def bar(self):
                    pass

            @abc.abstractmethod
            def updated_foo(self):
                pass

            A.foo = updated_foo
            abc.update_abstractmethods(A)
            self.assertEqual(A.__abstractmethods__, {'foo', 'bar'})
            msg = "class A without an implementation for abstract methods 'bar', 'foo'"
            self.assertRaisesRegex(TypeError, msg, A)

        def test_update_implementation(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            class B(A):
                pass

            msg = "class B without an implementation for abstract method 'foo'"
            self.assertRaisesRegex(TypeError, msg, B)
            self.assertEqual(B.__abstractmethods__, {'foo'})

            B.foo = lambda self: None

            abc.update_abstractmethods(B)

            B()
            self.assertEqual(B.__abstractmethods__, set())

        def test_update_as_decorator(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            def class_decorator(cls):
                cls.foo = lambda self: None
                return cls

            @abc.update_abstractmethods
            @class_decorator
            class B(A):
                pass

            B()
            self.assertEqual(B.__abstractmethods__, set())

        def test_update_non_abc(self):
            class A:
                pass

            @abc.abstractmethod
            def updated_foo(self):
                pass

            A.foo = updated_foo
            abc.update_abstractmethods(A)
            A()
            self.assertFalse(hasattr(A, '__abstractmethods__'))

        def test_update_del_implementation(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            class B(A):
                def foo(self):
                    pass

            B()

            del B.foo

            abc.update_abstractmethods(B)

            msg = "class B without an implementation for abstract method 'foo'"
            self.assertRaisesRegex(TypeError, msg, B)

        def test_update_layered_implementation(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            class B(A):
                pass

            class C(B):
                def foo(self):
                    pass

            C()

            del C.foo

            abc.update_abstractmethods(C)

            msg = "class C without an implementation for abstract method 'foo'"
            self.assertRaisesRegex(TypeError, msg, C)

        def test_update_multi_inheritance(self):
            class A(metaclass=abc_ABCMeta):
                @abc.abstractmethod
                def foo(self):
                    pass

            class B(metaclass=abc_ABCMeta):
                def foo(self):
                    pass

            class C(B, A):
                @abc.abstractmethod
                def foo(self):
                    pass

            self.assertEqual(C.__abstractmethods__, {'foo'})

            del C.foo

            abc.update_abstractmethods(C)

            self.assertEqual(C.__abstractmethods__, set())

            C()


    class TestABCWithInitSubclass(CPythonTestCase):
        def test_works_with_init_subclass(self):
            class abc_ABC(metaclass=abc_ABCMeta):
                __slots__ = ()
            saved_kwargs = {}
            class ReceivesClassKwargs:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__()
                    saved_kwargs.update(kwargs)
            class Receiver(ReceivesClassKwargs, abc_ABC, x=1, y=2, z=3):
                pass
            self.assertEqual(saved_kwargs, dict(x=1, y=2, z=3))

        def test_positional_only_and_kwonlyargs_with_init_subclass(self):
            saved_kwargs = {}

            class A:
                def __init_subclass__(cls, **kwargs):
                    super().__init_subclass__()
                    saved_kwargs.update(kwargs)

            class B(A, metaclass=abc_ABCMeta, name="test"):
                pass
            self.assertEqual(saved_kwargs, dict(name="test"))

    return TestLegacyAPI, TestABC, TestABCWithInitSubclass

test_factory.__test__ = False

TestLegacyAPI_Py, TestABC_Py, TestABCWithInitSubclass_Py = test_factory(abc.ABCMeta,
                                                                        abc.get_cache_token)
TestLegacyAPI_C, TestABC_C, TestABCWithInitSubclass_C = test_factory(_py_abc.ABCMeta,
                                                                     _py_abc.get_cache_token)

if __name__ == "__main__":
    run_tests()
