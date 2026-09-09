# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_super.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Unit tests for zero-argument super() & related machinery."""

import textwrap
import threading
import unittest
from unittest.mock import patch
from test.support import import_helper, threading_helper


ADAPTIVE_WARMUP_DELAY = 2


class A:
    def f(self):
        return 'A'
    @classmethod
    def cm(cls):
        return (cls, 'A')

class B(A):
    def f(self):
        return super().f() + 'B'
    @classmethod
    def cm(cls):
        return (cls, super().cm(), 'B')

class C(A):
    def f(self):
        return super().f() + 'C'
    @classmethod
    def cm(cls):
        return (cls, super().cm(), 'C')

class D(C, B):
    def f(self):
        return super().f() + 'D'
    def cm(cls):
        return (cls, super().cm(), 'D')

class E(D):
    pass

class F(E):
    f = E.f

class G(A):
    pass


class TestSuper(CPythonTestCase):

    def tearDown(self):
        # This fixes the damage that test_various___class___pathologies does.
        nonlocal __class__
        __class__ = TestSuper
        super().tearDown()

    def test_basics_working(self):
        self.assertEqual(D().f(), 'ABCD')

    def test_class_getattr_working(self):
        self.assertEqual(D.f(D()), 'ABCD')

    def test_subclass_no_override_working(self):
        self.assertEqual(E().f(), 'ABCD')
        self.assertEqual(E.f(E()), 'ABCD')

    def test_unbound_method_transfer_working(self):
        self.assertEqual(F().f(), 'ABCD')
        self.assertEqual(F.f(F()), 'ABCD')

    def test_class_methods_still_working(self):
        self.assertEqual(A.cm(), (A, 'A'))
        self.assertEqual(A().cm(), (A, 'A'))
        self.assertEqual(G.cm(), (G, 'A'))
        self.assertEqual(G().cm(), (G, 'A'))

    def test_super_in_class_methods_working(self):
        d = D()
        self.assertEqual(d.cm(), (d, (D, (D, (D, 'A'), 'B'), 'C'), 'D'))
        e = E()
        self.assertEqual(e.cm(), (e, (E, (E, (E, 'A'), 'B'), 'C'), 'D'))

    def test_super_with_closure(self):
        # Issue4360: super() did not work in a function that
        # contains a closure
        class E(A):
            def f(self):
                def nested():
                    self
                return super().f() + 'E'

        self.assertEqual(E().f(), 'AE')

    def test_various___class___pathologies(self):
        # See issue #12370
        class X(A):
            def f(self):
                return super().f()
            __class__ = 413
        x = X()
        self.assertEqual(x.f(), 'A')
        self.assertEqual(x.__class__, 413)
        class X:
            x = __class__
            def f():
                __class__
        self.assertIs(X.x, type(self))
        with self.assertRaises(NameError) as e:
            exec("""class X:
                __class__
                def f():
                    __class__""", globals(), {})
        self.assertIs(type(e.exception), NameError) # Not UnboundLocalError
        class X:
            global __class__
            __class__ = 42
            def f():
                __class__
        self.assertEqual(globals()["__class__"], 42)
        del globals()["__class__"]
        self.assertNotIn("__class__", X.__dict__)
        class X:
            nonlocal __class__
            __class__ = 42
            def f():
                __class__
        self.assertEqual(__class__, 42)

    def test___class___instancemethod(self):
        # See issue #14857
        class X:
            def f(self):
                return __class__
        self.assertIs(X().f(), X)

    def test___class___classmethod(self):
        # See issue #14857
        class X:
            @classmethod
            def f(cls):
                return __class__
        self.assertIs(X.f(), X)

    def test___class___staticmethod(self):
        # See issue #14857
        class X:
            @staticmethod
            def f():
                return __class__
        self.assertIs(X.f(), X)

    def test___class___new(self):
        # See issue #23722
        # Ensure zero-arg super() works as soon as type.__new__() is completed
        test_class = None

        class Meta(type):
            def __new__(cls, name, bases, namespace):
                nonlocal test_class
                self = super().__new__(cls, name, bases, namespace)
                test_class = self.f()
                return self

        class A(metaclass=Meta):
            @staticmethod
            def f():
                return __class__

        self.assertIs(test_class, A)

    def test___class___delayed(self):
        # See issue #23722
        test_namespace = None

        class Meta(type):
            def __new__(cls, name, bases, namespace):
                nonlocal test_namespace
                test_namespace = namespace
                return None

        class A(metaclass=Meta):
            @staticmethod
            def f():
                return __class__

        self.assertIs(A, None)

        B = type("B", (), test_namespace)
        self.assertIs(B.f(), B)

    def test___class___mro(self):
        # See issue #23722
        test_class = None

        class Meta(type):
            def mro(self):
                # self.f() doesn't work yet...
                self.__dict__["f"]()
                return super().mro()

        class A(metaclass=Meta):
            def f():
                nonlocal test_class
                test_class = __class__

        self.assertIs(test_class, A)

    def test___classcell___expected_behaviour(self):
        # See issue #23722
        class Meta(type):
            def __new__(cls, name, bases, namespace):
                nonlocal namespace_snapshot
                namespace_snapshot = namespace.copy()
                return super().__new__(cls, name, bases, namespace)

        # __classcell__ is injected into the class namespace by the compiler
        # when at least one method needs it, and should be omitted otherwise
        namespace_snapshot = None
        class WithoutClassRef(metaclass=Meta):
            pass
        self.assertNotIn("__classcell__", namespace_snapshot)

        # With zero-arg super() or an explicit __class__ reference,
        # __classcell__ is the exact cell reference to be populated by
        # type.__new__
        namespace_snapshot = None
        class WithClassRef(metaclass=Meta):
            def f(self):
                return __class__

        class_cell = namespace_snapshot["__classcell__"]
        method_closure = WithClassRef.f.__closure__
        self.assertEqual(len(method_closure), 1)
        self.assertIs(class_cell, method_closure[0])
        # Ensure the cell reference *doesn't* get turned into an attribute
        with self.assertRaises(AttributeError):
            WithClassRef.__classcell__

    def test___classcell___missing(self):
        # See issue #23722
        # Some metaclasses may not pass the original namespace to type.__new__
        # We test that case here by forcibly deleting __classcell__
        class Meta(type):
            def __new__(cls, name, bases, namespace):
                namespace.pop('__classcell__', None)
                return super().__new__(cls, name, bases, namespace)

        # The default case should continue to work without any errors
        class WithoutClassRef(metaclass=Meta):
            pass

        # With zero-arg super() or an explicit __class__ reference, we expect
        # __build_class__ to raise a RuntimeError complaining that
        # __class__ was not set, and asking if __classcell__ was propagated
        # to type.__new__.
        expected_error = '__class__ not set.*__classcell__ propagated'
        with self.assertRaisesRegex(RuntimeError, expected_error):
            class WithClassRef(metaclass=Meta):
                def f(self):
                    return __class__

    def test___classcell___overwrite(self):
        # See issue #23722
        # Overwriting __classcell__ with nonsense is explicitly prohibited
        class Meta(type):
            def __new__(cls, name, bases, namespace, cell):
                namespace['__classcell__'] = cell
                return super().__new__(cls, name, bases, namespace)

        for bad_cell in (None, 0, "", object()):
            with self.subTest(bad_cell=bad_cell):
                with self.assertRaises(TypeError):
                    class A(metaclass=Meta, cell=bad_cell):
                        pass

    def test___classcell___wrong_cell(self):
        # See issue #23722
        # Pointing the cell reference at the wrong class is also prohibited
        class Meta(type):
            def __new__(cls, name, bases, namespace):
                cls = super().__new__(cls, name, bases, namespace)
                B = type("B", (), namespace)
                return cls

        with self.assertRaises(TypeError):
            class A(metaclass=Meta):
                def f(self):
                    return __class__

    def test_obscure_super_errors(self):
        def f():
            super()
        with self.assertRaisesRegex(RuntimeError, r"no arguments"):
            f()

        class C:
            def f():
                super()
        with self.assertRaisesRegex(RuntimeError, r"no arguments"):
            C.f()

        def f(x):
            del x
            super()
        with self.assertRaisesRegex(RuntimeError, r"arg\[0\] deleted"):
            f(None)

        class X:
            def f(x):
                nonlocal __class__
                del __class__
                super()
        with self.assertRaisesRegex(RuntimeError, r"empty __class__ cell"):
            X().f()

    def test_cell_as_self(self):
        class X:
            def meth(self):
                super()

        def f():
            k = X()
            def g():
                return k
            return g
        c = f().__closure__[0]
        self.assertRaises(TypeError, X.meth, c)

    def test_super_init_leaks(self):
        # Issue #26718: super.__init__ leaked memory if called multiple times.
        # This will be caught by regrtest.py -R if this leak.
        # NOTE: Despite the use in the test a direct call of super.__init__
        # is not endorsed.
        sp = super(float, 1.0)
        for i in range(1000):
            super.__init__(sp, int, i)

    def test_super_argcount(self):
        with self.assertRaisesRegex(TypeError, "expected at most"):
            super(int, int, int)

    def test_super_argtype(self):
        with self.assertRaisesRegex(TypeError, "argument 1 must be a type"):
            super(1, int)

    def test_shadowed_global(self):
        source = textwrap.dedent(
            """
            class super:
                msg = "truly super"

            class C:
                def method(self):
                    return super().msg
            """,
        )
        with import_helper.ready_to_import(name="shadowed_super", source=source):
            import shadowed_super
        self.assertEqual(shadowed_super.C().method(), "truly super")
        import_helper.unload("shadowed_super")

    def test_shadowed_local(self):
        class super:
            msg = "quite super"

        class C:
            def method(self):
                return super().msg

        self.assertEqual(C().method(), "quite super")

    def test_shadowed_dynamic(self):
        class MySuper:
            msg = "super super"

        class C:
            def method(self):
                return super().msg

        with patch(f"{__name__}.super", MySuper) as m:
            self.assertEqual(C().method(), "super super")

    def test_shadowed_dynamic_two_arg(self):
        call_args = []
        class MySuper:
            def __init__(self, *args):
                call_args.append(args)
            msg = "super super"

        class C:
            def method(self):
                return super(1, 2).msg

        with patch(f"{__name__}.super", MySuper) as m:
            self.assertEqual(C().method(), "super super")
            self.assertEqual(call_args, [(1, 2)])

    def test_attribute_error(self):
        class C:
            def method(self):
                return super().msg

        with self.assertRaisesRegex(AttributeError, "'super' object has no attribute 'msg'"):
            C().method()

    def test_bad_first_arg(self):
        class C:
            def method(self):
                return super(1, self).method()

        with self.assertRaisesRegex(TypeError, "argument 1 must be a type"):
            C().method()

    def test_supercheck_fail(self):
        class C:
            def method(self, type_, obj):
                return super(type_, obj).method()

        c = C()
        err_msg = (
            r"super\(type, obj\): obj \({} {}\) is not "
            r"an instance or subtype of type \({}\)."
        )

        cases = (
            (int, c, int.__name__, C.__name__, "instance of"),
            # obj is instance of type
            (C, list(), C.__name__, list.__name__, "instance of"),
            # obj is type itself
            (C, list, C.__name__, list.__name__, "type"),
        )

        for case in cases:
            with self.subTest(case=case):
                type_, obj, type_str, obj_str, instance_or_type = case
                regex = err_msg.format(instance_or_type, obj_str, type_str)

                with self.assertRaisesRegex(TypeError, regex):
                    c.method(type_, obj)

    def test_super___class__(self):
        class C:
            def method(self):
                return super().__class__

        self.assertEqual(C().method(), super)

    def test_super_subclass___class__(self):
        class mysuper(super):
            pass

        class C:
            def method(self):
                return mysuper(C, self).__class__

        self.assertEqual(C().method(), mysuper)

    def test_unusual_getattro(self):
        class MyType(type):
            pass

        def test(name):
            mytype = MyType(name, (MyType,), {})
            super(MyType, type(mytype)).__setattr__(mytype, "bar", 1)
            self.assertEqual(mytype.bar, 1)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            test("foo1")

    def test_reassigned_new(self):
        class A:
            def __new__(cls):
                pass

            def __init_subclass__(cls):
                if "__new__" not in cls.__dict__:
                    cls.__new__ = cls.__new__

        class B(A):
            pass

        class C(B):
            def __new__(cls):
                return super().__new__(cls)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            C()

    def test_mixed_staticmethod_hierarchy(self):
        # This test is just a desugared version of `test_reassigned_new`
        class A:
            @staticmethod
            def some(cls, *args, **kwargs):
                self.assertFalse(args)
                self.assertFalse(kwargs)

        class B(A):
            def some(cls, *args, **kwargs):
                return super().some(cls, *args, **kwargs)

        class C(B):
            @staticmethod
            def some(cls):
                return super().some(cls)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            C.some(C)

    @threading_helper.requires_working_threading()
    def test___class___modification_multithreaded(self):
        """ Note: this test isn't actually testing anything on its own.
        It requires a sys audithook to be set to crash on older Python.
        This should be the case anyways as our test suite sets
        an audit hook.
        """

        class Foo:
            pass

        class Bar:
            pass

        thing = Foo()
        def work():
            foo = thing
            for _ in range(200):
                foo.__class__ = Bar
                type(foo)
                foo.__class__ = Foo
                type(foo)


        threads = []
        for _ in range(6):
            thread = threading.Thread(target=work)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


if __name__ == "__main__":
    run_tests()
