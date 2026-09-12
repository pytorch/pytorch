# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_genericclass.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
from test import support
from test.support.import_helper import import_module


class TestMROEntry(CPythonTestCase):
    def test_mro_entry_signature(self):
        tested = []
        class B: ...
        class C:
            def __mro_entries__(self, *args, **kwargs):
                tested.extend([args, kwargs])
                return (C,)
        c = C()
        self.assertEqual(tested, [])
        class D(B, c): ...
        self.assertEqual(tested[0], ((B, c),))
        self.assertEqual(tested[1], {})

    def test_mro_entry(self):
        tested = []
        class A: ...
        class B: ...
        class C:
            def __mro_entries__(self, bases):
                tested.append(bases)
                return (self.__class__,)
        c = C()
        self.assertEqual(tested, [])
        class D(A, c, B): ...
        self.assertEqual(tested[-1], (A, c, B))
        self.assertEqual(D.__bases__, (A, C, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, C, B, object))
        d = D()
        class E(d): ...
        self.assertEqual(tested[-1], (d,))
        self.assertEqual(E.__bases__, (D,))

    def test_mro_entry_none(self):
        tested = []
        class A: ...
        class B: ...
        class C:
            def __mro_entries__(self, bases):
                tested.append(bases)
                return ()
        c = C()
        self.assertEqual(tested, [])
        class D(A, c, B): ...
        self.assertEqual(tested[-1], (A, c, B))
        self.assertEqual(D.__bases__, (A, B))
        self.assertEqual(D.__orig_bases__, (A, c, B))
        self.assertEqual(D.__mro__, (D, A, B, object))
        class E(c): ...
        self.assertEqual(tested[-1], (c,))
        self.assertEqual(E.__bases__, (object,))
        self.assertEqual(E.__orig_bases__, (c,))
        self.assertEqual(E.__mro__, (E, object))

    def test_mro_entry_with_builtins(self):
        tested = []
        class A: ...
        class C:
            def __mro_entries__(self, bases):
                tested.append(bases)
                return (dict,)
        c = C()
        self.assertEqual(tested, [])
        class D(A, c): ...
        self.assertEqual(tested[-1], (A, c))
        self.assertEqual(D.__bases__, (A, dict))
        self.assertEqual(D.__orig_bases__, (A, c))
        self.assertEqual(D.__mro__, (D, A, dict, object))

    def test_mro_entry_with_builtins_2(self):
        tested = []
        class C:
            def __mro_entries__(self, bases):
                tested.append(bases)
                return (C,)
        c = C()
        self.assertEqual(tested, [])
        class D(c, dict): ...
        self.assertEqual(tested[-1], (c, dict))
        self.assertEqual(D.__bases__, (C, dict))
        self.assertEqual(D.__orig_bases__, (c, dict))
        self.assertEqual(D.__mro__, (D, C, dict, object))

    def test_mro_entry_errors(self):
        class C_too_many:
            def __mro_entries__(self, bases, something, other):
                return ()
        c = C_too_many()
        with self.assertRaises(TypeError):
            class D(c): ...
        class C_too_few:
            def __mro_entries__(self):
                return ()
        d = C_too_few()
        with self.assertRaises(TypeError):
            class E(d): ...

    def test_mro_entry_errors_2(self):
        class C_not_callable:
            __mro_entries__ = "Surprise!"
        c = C_not_callable()
        with self.assertRaises(TypeError):
            class D(c): ...
        class C_not_tuple:
            def __mro_entries__(self):
                return object
        c = C_not_tuple()
        with self.assertRaises(TypeError):
            class E(c): ...

    def test_mro_entry_metaclass(self):
        meta_args = []
        class Meta(type):
            def __new__(mcls, name, bases, ns):
                meta_args.extend([mcls, name, bases, ns])
                return super().__new__(mcls, name, bases, ns)
        class A: ...
        class C:
            def __mro_entries__(self, bases):
                return (A,)
        c = C()
        class D(c, metaclass=Meta):
            x = 1
        self.assertEqual(meta_args[0], Meta)
        self.assertEqual(meta_args[1], 'D')
        self.assertEqual(meta_args[2], (A,))
        self.assertEqual(meta_args[3]['x'], 1)
        self.assertEqual(D.__bases__, (A,))
        self.assertEqual(D.__orig_bases__, (c,))
        self.assertEqual(D.__mro__, (D, A, object))
        self.assertEqual(D.__class__, Meta)

    def test_mro_entry_type_call(self):
        # Substitution should _not_ happen in direct type call
        class C:
            def __mro_entries__(self, bases):
                return ()
        c = C()
        with self.assertRaisesRegex(TypeError,
                                    "MRO entry resolution; "
                                    "use types.new_class()"):
            type('Bad', (c,), {})


class TestClassGetitem(CPythonTestCase):
    def test_class_getitem(self):
        getitem_args = []
        class C:
            def __class_getitem__(*args, **kwargs):
                getitem_args.extend([args, kwargs])
                return None
        C[int, str]
        self.assertEqual(getitem_args[0], (C, (int, str)))
        self.assertEqual(getitem_args[1], {})

    def test_class_getitem_format(self):
        class C:
            def __class_getitem__(cls, item):
                return f'C[{item.__name__}]'
        self.assertEqual(C[int], 'C[int]')
        self.assertEqual(C[C], 'C[C]')

    def test_class_getitem_inheritance(self):
        class C:
            def __class_getitem__(cls, item):
                return f'{cls.__name__}[{item.__name__}]'
        class D(C): ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_inheritance_2(self):
        class C:
            def __class_getitem__(cls, item):
                return 'Should not see this'
        class D(C):
            def __class_getitem__(cls, item):
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_classmethod(self):
        class C:
            @classmethod
            def __class_getitem__(cls, item):
                return f'{cls.__name__}[{item.__name__}]'
        class D(C): ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_patched(self):
        class C:
            def __init_subclass__(cls):
                def __class_getitem__(cls, item):
                    return f'{cls.__name__}[{item.__name__}]'
                cls.__class_getitem__ = classmethod(__class_getitem__)
        class D(C): ...
        self.assertEqual(D[int], 'D[int]')
        self.assertEqual(D[D], 'D[D]')

    def test_class_getitem_with_builtins(self):
        class A(dict):
            called_with = None

            def __class_getitem__(cls, item):
                cls.called_with = item
        class B(A):
            pass
        self.assertIs(B.called_with, None)
        B[int]
        self.assertIs(B.called_with, int)

    def test_class_getitem_errors(self):
        class C_too_few:
            def __class_getitem__(cls):
                return None
        with self.assertRaises(TypeError):
            C_too_few[int]

        class C_too_many:
            def __class_getitem__(cls, one, two):
                return None
        with self.assertRaises(TypeError):
            C_too_many[int]

    def test_class_getitem_errors_2(self):
        class C:
            def __class_getitem__(cls, item):
                return None
        with self.assertRaises(TypeError):
            C()[int]

        class E: ...
        e = E()
        e.__class_getitem__ = lambda cls, item: 'This will not work'
        with self.assertRaises(TypeError):
            e[int]

        class C_not_callable:
            __class_getitem__ = "Surprise!"
        with self.assertRaises(TypeError):
            C_not_callable[int]

        class C_is_none(tuple):
            __class_getitem__ = None
        with self.assertRaisesRegex(TypeError, "C_is_none"):
            C_is_none[int]

    def test_class_getitem_metaclass(self):
        class Meta(type):
            def __class_getitem__(cls, item):
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(Meta[int], 'Meta[int]')

    def test_class_getitem_with_metaclass(self):
        class Meta(type): pass
        class C(metaclass=Meta):
            def __class_getitem__(cls, item):
                return f'{cls.__name__}[{item.__name__}]'
        self.assertEqual(C[int], 'C[int]')

    def test_class_getitem_metaclass_first(self):
        class Meta(type):
            def __getitem__(cls, item):
                return 'from metaclass'
        class C(metaclass=Meta):
            def __class_getitem__(cls, item):
                return 'from __class_getitem__'
        self.assertEqual(C[int], 'from metaclass')


@support.cpython_only
@unittest.skip("imports _testcapi")
class CAPITest(CPythonTestCase):

    def test_c_class(self):
        _testcapi = import_module("_testcapi")
        Generic = _testcapi.Generic
        GenericAlias = _testcapi.GenericAlias
        self.assertIsInstance(Generic.__class_getitem__(int), GenericAlias)

        IntGeneric = Generic[int]
        self.assertIs(type(IntGeneric), GenericAlias)
        self.assertEqual(IntGeneric.__mro_entries__(()), (int,))
        class C(IntGeneric):
            pass
        self.assertEqual(C.__bases__, (int,))
        self.assertEqual(C.__orig_bases__, (IntGeneric,))
        self.assertEqual(C.__mro__, (C, int, object))


if __name__ == "__main__":
    run_tests()
