# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_property.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Test case for property
# more tests are in test_descr

import sys
import unittest
from test import support

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

    @property
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

    @BaseClass.spam.getter
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
    spam = property(_get_spam, doc="spam spam spam")

class PropertyDocSub(PropertyDocBase):
    @PropertyDocBase.spam.getter
    def spam(self):
        """The decorator does not use this doc string"""
        return self._spam

class PropertySubNewGetter(BaseClass):
    @BaseClass.spam.getter
    def spam(self):
        """new docstring"""
        return 5

class PropertyNewGetter(object):
    @property
    def spam(self):
        """original docstring"""
        return 1
    @spam.getter
    def spam(self):
        """new docstring"""
        return 8

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
        self.assertEqual(sub.__class__.spam.__doc__, "SubClass.getter")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_baseclass_doc(self):
        base = BaseClass()
        self.assertEqual(base.__class__.spam.__doc__, "BaseClass.getter")

    def test_property_decorator_doc(self):
        base = PropertyDocBase()
        sub = PropertyDocSub()
        self.assertEqual(base.__class__.spam.__doc__, "spam spam spam")
        self.assertEqual(sub.__class__.spam.__doc__, "spam spam spam")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_getter_doc_override(self):
        newgettersub = PropertySubNewGetter()
        self.assertEqual(newgettersub.spam, 5)
        self.assertEqual(newgettersub.__class__.spam.__doc__, "new docstring")
        newgetter = PropertyNewGetter()
        self.assertEqual(newgetter.spam, 8)
        self.assertEqual(newgetter.__class__.spam.__doc__, "new docstring")

    def test_property___isabstractmethod__descriptor(self):
        for val in (True, False, [], [1], '', '1'):
            class C(object):
                def foo(self):
                    pass
                foo.__isabstractmethod__ = val
                foo = property(foo)
            self.assertIs(C.foo.__isabstractmethod__, bool(val))

        # check that the property's __isabstractmethod__ descriptor does the
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
                foo = property(foo)
            C.foo.__isabstractmethod__

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_builtin_doc_writable(self):
        p = property(doc='basic')
        self.assertEqual(p.__doc__, 'basic')
        p.__doc__ = 'extended'
        self.assertEqual(p.__doc__, 'extended')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_doc_writable(self):
        class PropertyWritableDoc(object):

            @property
            def spam(self):
                """Eggs"""
                return "eggs"

        sub = PropertyWritableDoc()
        self.assertEqual(sub.__class__.spam.__doc__, 'Eggs')
        sub.__class__.spam.__doc__ = 'Spam'
        self.assertEqual(sub.__class__.spam.__doc__, 'Spam')

    @support.refcount_test
    def test_refleaks_in___init__(self):
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        fake_prop = property('fget', 'fset', 'fdel', 'doc')
        refs_before = gettotalrefcount()
        for i in range(100):
            fake_prop.__init__('fget', 'fset', 'fdel', 'doc')
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.refcount_test
    def test_gh_115618(self):
        # Py_XDECREF() was improperly called for None argument
        # in property methods.
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        prop = property()
        refs_before = gettotalrefcount()
        for i in range(100):
            prop = prop.getter(None)
        self.assertIsNone(prop.fget)
        for i in range(100):
            prop = prop.setter(None)
        self.assertIsNone(prop.fset)
        for i in range(100):
            prop = prop.deleter(None)
        self.assertIsNone(prop.fdel)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    def test_property_name(self):
        def getter(self):
            return 42

        def setter(self, value):
            pass

        class A:
            @property
            def foo(self):
                return 1

            @foo.setter
            def oof(self, value):
                pass

            bar = property(getter)
            baz = property(None, setter)

        self.assertEqual(A.foo.__name__, 'foo')
        self.assertEqual(A.oof.__name__, 'oof')
        self.assertEqual(A.bar.__name__, 'bar')
        self.assertEqual(A.baz.__name__, 'baz')

        A.quux = property(getter)
        self.assertEqual(A.quux.__name__, 'getter')
        A.quux.__name__ = 'myquux'
        self.assertEqual(A.quux.__name__, 'myquux')
        self.assertEqual(A.bar.__name__, 'bar')  # not affected
        A.quux.__name__ = None
        self.assertIsNone(A.quux.__name__)

        with self.assertRaisesRegex(
            AttributeError, "'property' object has no attribute '__name__'"
        ):
            property(None, setter).__name__

        with self.assertRaisesRegex(
            AttributeError, "'property' object has no attribute '__name__'"
        ):
            property(1).__name__

        class Err:
            def __getattr__(self, attr):
                raise RuntimeError('fail')

        p = property(Err())
        with self.assertRaisesRegex(RuntimeError, 'fail'):
            p.__name__

        p.__name__ = 'not_fail'
        self.assertEqual(p.__name__, 'not_fail')

    def test_property_set_name_incorrect_args(self):
        p = property()

        for i in (0, 1, 3):
            with self.assertRaisesRegex(
                TypeError,
                fr'^__set_name__\(\) takes 2 positional arguments but {i} were given$'
            ):
                p.__set_name__(*([0] * i))

    def test_property_setname_on_property_subclass(self):
        # https://github.com/python/cpython/issues/100942
        # Copy was setting the name field without first
        # verifying that the copy was an actual property
        # instance.  As a result, the code below was
        # causing a segfault.

        class pro(property):
            def __new__(typ, *args, **kwargs):
                return "abcdef"

        class A:
            pass

        p = property.__new__(pro)
        p.__set_name__(A, 1)
        np = p.getter(lambda self: 1)

# Issue 5890: subclasses of property do not preserve method __doc__ strings
class PropertySub(property):
    """This is a subclass of property"""

class PropertySubWoDoc(property):
    pass

class PropertySubSlots(property):
    """This is a subclass of property that defines __slots__"""
    __slots__ = ()

class PropertySubclassTests(CPythonTestCase):

    @support.requires_docstrings
    def test_slots_docstring_copy_exception(self):
        # A special case error that we preserve despite the GH-98963 behavior
        # that would otherwise silently ignore this error.
        # This came from commit b18500d39d791c879e9904ebac293402b4a7cd34
        # as part of https://bugs.python.org/issue5890 which allowed docs to
        # be set via property subclasses in the first place.
        with self.assertRaises(AttributeError):
            class Foo(object):
                @PropertySubSlots
                def spam(self):
                    """Trying to copy this docstring will raise an exception"""
                    return 1

    def test_property_with_slots_no_docstring(self):
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            __slots__ = ("foo",)

        p = slotted_prop()  # no AttributeError
        self.assertIsNone(getattr(p, "__doc__", None))

        def undocumented_getter():
            return 4

        p = slotted_prop(undocumented_getter)  # New in 3.12: no AttributeError
        self.assertIsNone(getattr(p, "__doc__", None))

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_docstring_silently_dropped(self):
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            __slots__ = ("foo",)

        p = slotted_prop(doc="what's up")  # no AttributeError
        self.assertIsNone(p.__doc__)

        def documented_getter():
            """getter doc."""
            return 4

        # Historical behavior: A docstring from a getter always raises.
        # (matches test_slots_docstring_copy_exception above).
        with self.assertRaises(AttributeError):
            p = slotted_prop(documented_getter)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_and_doc_slot_docstring_present(self):
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            __slots__ = ("foo", "__doc__")

        p = slotted_prop(doc="what's up")
        self.assertEqual("what's up", p.__doc__)  # new in 3.12: This gets set.

        def documented_getter():
            """what's up getter doc?"""
            return 4

        p = slotted_prop(documented_getter)
        self.assertEqual("what's up getter doc?", p.__doc__)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_issue41287(self):

        self.assertEqual(PropertySub.__doc__, "This is a subclass of property",
                         "Docstring of `property` subclass is ignored")

        doc = PropertySub(None, None, None, "issue 41287 is fixed").__doc__
        self.assertEqual(doc, "issue 41287 is fixed",
                         "Subclasses of `property` ignores `doc` constructor argument")

        def getter(x):
            """Getter docstring"""

        def getter_wo_doc(x):
            pass

        for ps in property, PropertySub, PropertySubWoDoc:
            doc = ps(getter, None, None, "issue 41287 is fixed").__doc__
            self.assertEqual(doc, "issue 41287 is fixed",
                             "Getter overrides explicit property docstring (%s)" % ps.__name__)

            doc = ps(getter, None, None, None).__doc__
            self.assertEqual(doc, "Getter docstring", "Getter docstring is not picked-up (%s)" % ps.__name__)

            doc = ps(getter_wo_doc, None, None, "issue 41287 is fixed").__doc__
            self.assertEqual(doc, "issue 41287 is fixed",
                             "Getter overrides explicit property docstring (%s)" % ps.__name__)

            doc = ps(getter_wo_doc, None, None, None).__doc__
            self.assertIsNone(doc, "Property class doc appears in instance __doc__ (%s)" % ps.__name__)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_docstring_copy(self):
        class Foo(object):
            @PropertySub
            def spam(self):
                """spam wrapped in property subclass"""
                return 1
        self.assertEqual(
            Foo.spam.__doc__,
            "spam wrapped in property subclass")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_docstring_copy2(self):
        """
        Property tries to provide the best docstring it finds for its instances.
        If a user-provided docstring is available, it is preserved on copies.
        If no docstring is available during property creation, the property
        will utilize the docstring from the getter if available.
        """
        def getter1(self):
            return 1
        def getter2(self):
            """doc 2"""
            return 2
        def getter3(self):
            """doc 3"""
            return 3

        # Case-1: user-provided doc is preserved in copies
        #         of property with undocumented getter
        p = property(getter1, None, None, "doc-A")

        p2 = p.getter(getter2)
        self.assertEqual(p.__doc__, "doc-A")
        self.assertEqual(p2.__doc__, "doc-A")

        # Case-2: user-provided doc is preserved in copies
        #         of property with documented getter
        p = property(getter2, None, None, "doc-A")

        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "doc-A")
        self.assertEqual(p2.__doc__, "doc-A")

        # Case-3: with no user-provided doc new getter doc
        #         takes precedence
        p = property(getter2, None, None, None)

        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "doc 2")
        self.assertEqual(p2.__doc__, "doc 3")

        # Case-4: A user-provided doc is assigned after property construction
        #         with documented getter. The doc IS NOT preserved.
        #         It's an odd behaviour, but it's a strange enough
        #         use case with no easy solution.
        p = property(getter2, None, None, None)
        p.__doc__ = "user"
        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "user")
        self.assertEqual(p2.__doc__, "doc 3")

        # Case-5: A user-provided doc is assigned after property construction
        #         with UNdocumented getter. The doc IS preserved.
        p = property(getter1, None, None, None)
        p.__doc__ = "user"
        p2 = p.getter(getter2)
        self.assertEqual(p.__doc__, "user")
        self.assertEqual(p2.__doc__, "user")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_prefer_explicit_doc(self):
        # Issue 25757: subclasses of property lose docstring
        self.assertEqual(property(doc="explicit doc").__doc__, "explicit doc")
        self.assertEqual(PropertySub(doc="explicit doc").__doc__, "explicit doc")

        class Foo:
            spam = PropertySub(doc="spam explicit doc")

            @spam.getter
            def spam(self):
                """ignored as doc already set"""
                return 1

            def _stuff_getter(self):
                """ignored as doc set directly"""
            stuff = PropertySub(doc="stuff doc argument", fget=_stuff_getter)

        #self.assertEqual(Foo.spam.__doc__, "spam explicit doc")
        self.assertEqual(Foo.stuff.__doc__, "stuff doc argument")

    def test_property_no_doc_on_getter(self):
        # If a property's getter has no __doc__ then the property's doc should
        # be None; test that this is consistent with subclasses as well; see
        # GH-2487
        class NoDoc:
            @property
            def __doc__(self):
                raise AttributeError

        self.assertEqual(property(NoDoc()).__doc__, None)
        self.assertEqual(PropertySub(NoDoc()).__doc__, None)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_setter_copies_getter_docstring(self):
        class Foo(object):
            def __init__(self): self._spam = 1
            @PropertySub
            def spam(self):
                """spam wrapped in property subclass"""
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
            Foo.spam.__doc__,
            "spam wrapped in property subclass")
        class FooSub(Foo):
            @Foo.spam.setter
            def spam(self, value):
                """another ignored docstring"""
                self._spam = 'eggs'
        foosub = FooSub()
        self.assertEqual(foosub.spam, 1)
        foosub.spam = 7
        self.assertEqual(foosub.spam, 'eggs')
        self.assertEqual(
            FooSub.spam.__doc__,
            "spam wrapped in property subclass")

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
        self.assertEqual(Foo.spam.__doc__, "a new docstring")
        class FooBase(object):
            @PropertySub
            def spam(self):
                """a docstring"""
                return 1
        class Foo2(FooBase):
            @FooBase.spam.getter
            def spam(self):
                """a new docstring"""
                return 2
        self.assertEqual(Foo.spam.__doc__, "a new docstring")


class _PropertyUnreachableAttribute:
    msg_format = None
    obj = None
    cls = None

    def _format_exc_msg(self, msg):
        return self.msg_format.format(msg)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.obj = cls.cls()

    def test_get_property(self):
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no getter")):
            self.obj.foo

    def test_set_property(self):
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no setter")):
            self.obj.foo = None

    def test_del_property(self):
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no deleter")):
            del self.obj.foo


class PropertyUnreachableAttributeWithName(_PropertyUnreachableAttribute, CPythonTestCase):
    msg_format = r"^property 'foo' of 'PropertyUnreachableAttributeWithName\.cls' object {}$"

    class cls:
        foo = property()


class PropertyUnreachableAttributeNoName(_PropertyUnreachableAttribute, CPythonTestCase):
    msg_format = r"^property of 'PropertyUnreachableAttributeNoName\.cls' object {}$"

    class cls:
        pass

    cls.foo = property()


if __name__ == '__main__':
    run_tests()
