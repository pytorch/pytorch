# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_dynamicclassattribute.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase


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

# Test case for DynamicClassAttribute
# more tests are in test_descr

import abc
import sys
import unittest
from types import DynamicClassAttribute

class PropertyBase(Exception):
    pass

class PropertyGet(PropertyBase):
    pass

class PropertySet(PropertyBase):
    pass

class PropertyDel(PropertyBase):
    pass

class BaseClass(object):
    def __init__(self):
        self._spam = 5

    @DynamicClassAttribute
    def spam(self):
        """BaseClass.getter"""
        return self._spam

    @spam.setter
    def spam(self, value):
        self._spam = value

    @spam.deleter
    def spam(self):
        del self._spam

class SubClass(BaseClass):

    spam = BaseClass.__dict__['spam']

    @spam.getter
    def spam(self):
        """SubClass.getter"""
        raise PropertyGet(self._spam)

    @spam.setter
    def spam(self, value):
        raise PropertySet(self._spam)

    @spam.deleter
    def spam(self):
        raise PropertyDel(self._spam)

class PropertyDocBase(object):
    _spam = 1
    def _get_spam(self):
        return self._spam
    spam = DynamicClassAttribute(_get_spam, doc="spam spam spam")

class PropertyDocSub(PropertyDocBase):
    spam = PropertyDocBase.__dict__['spam']
    @spam.getter
    def spam(self):
        """The decorator does not use this doc string"""
        return self._spam

class PropertySubNewGetter(BaseClass):
    spam = BaseClass.__dict__['spam']
    @spam.getter
    def spam(self):
        """new docstring"""
        return 5

class PropertyNewGetter(object):
    @DynamicClassAttribute
    def spam(self):
        """original docstring"""
        return 1
    @spam.getter
    def spam(self):
        """new docstring"""
        return 8

class ClassWithAbstractVirtualProperty(metaclass=abc.ABCMeta):
    @DynamicClassAttribute
    @abc.abstractmethod
    def color():
        pass

class ClassWithPropertyAbstractVirtual(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @DynamicClassAttribute
    def color():
        pass

class PropertyTests(CPythonTestCase):
    def test_property_decorator_baseclass(self):
        # see #1620
        base = BaseClass()
        self.assertEqual(base.spam, 5)
        self.assertEqual(base._spam, 5)
        base.spam = 10
        self.assertEqual(base.spam, 10)
        self.assertEqual(base._spam, 10)
        delattr(base, "spam")
        self.assertTrue(not hasattr(base, "spam"))
        self.assertTrue(not hasattr(base, "_spam"))
        base.spam = 20
        self.assertEqual(base.spam, 20)
        self.assertEqual(base._spam, 20)

    def test_property_decorator_subclass(self):
        # see #1620
        sub = SubClass()
        self.assertRaises(PropertyGet, getattr, sub, "spam")
        self.assertRaises(PropertySet, setattr, sub, "spam", None)
        self.assertRaises(PropertyDel, delattr, sub, "spam")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_subclass_doc(self):
        sub = SubClass()
        self.assertEqual(sub.__class__.__dict__['spam'].__doc__, "SubClass.getter")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_baseclass_doc(self):
        base = BaseClass()
        self.assertEqual(base.__class__.__dict__['spam'].__doc__, "BaseClass.getter")

    def test_property_decorator_doc(self):
        base = PropertyDocBase()
        sub = PropertyDocSub()
        self.assertEqual(base.__class__.__dict__['spam'].__doc__, "spam spam spam")
        self.assertEqual(sub.__class__.__dict__['spam'].__doc__, "spam spam spam")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_getter_doc_override(self):
        newgettersub = PropertySubNewGetter()
        self.assertEqual(newgettersub.spam, 5)
        self.assertEqual(newgettersub.__class__.__dict__['spam'].__doc__, "new docstring")
        newgetter = PropertyNewGetter()
        self.assertEqual(newgetter.spam, 8)
        self.assertEqual(newgetter.__class__.__dict__['spam'].__doc__, "new docstring")

    def test_property___isabstractmethod__descriptor(self):
        for val in (True, False, [], [1], '', '1'):
            class C(object):
                def foo(self):
                    pass
                foo.__isabstractmethod__ = val
                foo = DynamicClassAttribute(foo)
            self.assertIs(C.__dict__['foo'].__isabstractmethod__, bool(val))

        # check that the DynamicClassAttribute's __isabstractmethod__ descriptor does the
        # right thing when presented with a value that fails truth testing:
        class NotBool(object):
            def __bool__(self):
                raise ValueError()
            __len__ = __bool__
        with self.assertRaises(ValueError):
            class C(object):
                def foo(self):
                    pass
                foo.__isabstractmethod__ = NotBool()
                foo = DynamicClassAttribute(foo)

    def test_abstract_virtual(self):
        self.assertRaises(TypeError, ClassWithAbstractVirtualProperty)
        self.assertRaises(TypeError, ClassWithPropertyAbstractVirtual)
        class APV(ClassWithPropertyAbstractVirtual):
            pass
        self.assertRaises(TypeError, APV)
        class AVP(ClassWithAbstractVirtualProperty):
            pass
        self.assertRaises(TypeError, AVP)
        class Okay1(ClassWithAbstractVirtualProperty):
            @DynamicClassAttribute
            def color(self):
                return self._color
            def __init__(self):
                self._color = 'cyan'
        with self.assertRaises(AttributeError):
            Okay1.color
        self.assertEqual(Okay1().color, 'cyan')
        class Okay2(ClassWithAbstractVirtualProperty):
            @DynamicClassAttribute
            def color(self):
                return self._color
            def __init__(self):
                self._color = 'magenta'
        with self.assertRaises(AttributeError):
            Okay2.color
        self.assertEqual(Okay2().color, 'magenta')


# Issue 5890: subclasses of DynamicClassAttribute do not preserve method __doc__ strings
class PropertySub(DynamicClassAttribute):
    """This is a subclass of DynamicClassAttribute"""

class PropertySubSlots(DynamicClassAttribute):
    """This is a subclass of DynamicClassAttribute that defines __slots__"""
    __slots__ = ()

class PropertySubclassTests(CPythonTestCase):

    @unittest.skipIf(hasattr(PropertySubSlots, '__doc__'),
            "__doc__ is already present, __slots__ will have no effect")
    def test_slots_docstring_copy_exception(self):
        try:
            class Foo(object):
                @PropertySubSlots
                def spam(self):
                    """Trying to copy this docstring will raise an exception"""
                    return 1
                print('\n',spam.__doc__)
        except AttributeError:
            pass
        else:
            raise Exception("AttributeError not raised")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_docstring_copy(self):
        class Foo(object):
            @PropertySub
            def spam(self):
                """spam wrapped in DynamicClassAttribute subclass"""
                return 1
        self.assertEqual(
            Foo.__dict__['spam'].__doc__,
            "spam wrapped in DynamicClassAttribute subclass")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_setter_copies_getter_docstring(self):
        class Foo(object):
            def __init__(self): self._spam = 1
            @PropertySub
            def spam(self):
                """spam wrapped in DynamicClassAttribute subclass"""
                return self._spam
            @spam.setter
            def spam(self, value):
                """this docstring is ignored"""
                self._spam = value
        foo = Foo()
        self.assertEqual(foo.spam, 1)
        foo.spam = 2
        self.assertEqual(foo.spam, 2)
        self.assertEqual(
            Foo.__dict__['spam'].__doc__,
            "spam wrapped in DynamicClassAttribute subclass")
        class FooSub(Foo):
            spam = Foo.__dict__['spam']
            @spam.setter
            def spam(self, value):
                """another ignored docstring"""
                self._spam = 'eggs'
        foosub = FooSub()
        self.assertEqual(foosub.spam, 1)
        foosub.spam = 7
        self.assertEqual(foosub.spam, 'eggs')
        self.assertEqual(
            FooSub.__dict__['spam'].__doc__,
            "spam wrapped in DynamicClassAttribute subclass")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_new_getter_new_docstring(self):

        class Foo(object):
            @PropertySub
            def spam(self):
                """a docstring"""
                return 1
            @spam.getter
            def spam(self):
                """a new docstring"""
                return 2
        self.assertEqual(Foo.__dict__['spam'].__doc__, "a new docstring")
        class FooBase(object):
            @PropertySub
            def spam(self):
                """a docstring"""
                return 1
        class Foo2(FooBase):
            spam = FooBase.__dict__['spam']
            @spam.getter
            def spam(self):
                """a new docstring"""
                return 2
        self.assertEqual(Foo.__dict__['spam'].__doc__, "a new docstring")



if __name__ == '__main__':
    run_tests()
