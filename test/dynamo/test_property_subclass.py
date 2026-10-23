# Owner(s): ["module: dynamo"]

import abc
import operator
import sys
import unittest

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class PropertyWithDoc(property):
    """Class docstring."""


class PropertyWithoutDoc(property):
    pass


@instantiate_parametrized_tests
class PropertySubclassTests(TestCase):
    @parametrize("cls", (PropertyWithDoc, PropertyWithoutDoc))
    @parametrize("getter_doc", (None, "Getter docstring."))
    @parametrize("doc", (None, "Explicit docstring.", ""))
    @parametrize("keyword", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_doc(self, cls, getter_doc, doc, keyword):
        def getter(obj):
            return obj.value

        getter.__doc__ = getter_doc

        def fn(x):
            if keyword:
                p = cls(fget=getter, doc=doc)
            else:
                p = cls(getter, None, None, doc)
            return x + 1, p.__doc__, p.fget is getter, p.fset, p.fdel, type(p) is cls

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(
            cls.__doc__, "Class docstring." if cls is PropertyWithDoc else None
        )

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_empty(self):
        def fn(x):
            p = PropertyWithDoc()
            return x + 1, p.__doc__, p.fget, p.fset, p.fdel

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_nested_getter_and_return(self):
        def fn(x):
            def getter(obj):
                """Nested getter."""
                return obj + x

            p = PropertyWithDoc(getter)
            p.tag = x + 1
            return x + 2, p, p.fget is getter

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for val in (x, x * 3):
            expected = fn(val)
            actual = opt_fn(val)
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1].__doc__, "Nested getter.")
            self.assertEqual(actual[1].tag, expected[1].tag)
            self.assertEqual(
                actual[1].__get__(val, type(val)), expected[1].__get__(val, type(val))
            )
            self.assertTrue(actual[2])
        self.assertIsNot(opt_fn(x)[1], opt_fn(x)[1])

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_graph_break(self):
        def fn(x):
            def getter(obj):
                """Original doc."""
                return obj + x

            p = PropertyWithoutDoc(getter)
            p.__doc__ = "Changed doc."
            y = x + 1
            torch._dynamo.graph_break()
            return y + 1, p.__doc__, p.__get__(x, type(x)), p

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[:3], expected[:3])
        self.assertEqual(actual[3].__doc__, "Changed doc.")
        self.assertEqual(actual[3].__get__(x, type(x)), expected[3].__get__(x, type(x)))

    @unittest.skipIf(
        sys.version_info < (3, 13), "property.__name__ requires Python 3.13+"
    )
    def test_getter_metadata(self):
        def fn(x):
            def getter(obj):
                return obj

            getter.__isabstractmethod__ = True
            p = PropertyWithDoc(getter)
            return x + 1, p.__name__, p.__isabstractmethod__, p.fget is getter

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("case", ("duplicate", "unknown", "too_many"))
    def test_invalid_args(self, case):
        def fn(x):
            try:
                if case == "duplicate":
                    PropertyWithDoc(None, fget=None)
                elif case == "unknown":
                    PropertyWithDoc(unknown=None)
                else:
                    PropertyWithDoc(None, None, None, None, None)
            except TypeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_readonly_accessors(self):
        p = PropertyWithDoc()

        def fn(x):
            try:
                p.fget = None
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("doc", (None, "Explicit doc.", ""))
    @parametrize("getter_doc", (None, "Getter doc."))
    @parametrize("doc_slot", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_slots(self, doc, getter_doc, doc_slot):
        class SlottedProperty(property):
            __slots__ = ("__doc__",) if doc_slot else ()

        def getter(obj):
            return obj

        getter.__doc__ = getter_doc

        def fn(x):
            try:
                p = SlottedProperty(getter, doc=doc)
                return x + 1, p.__doc__
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_existing_property(self):
        def getter(obj):
            """Original doc."""
            return obj

        p = PropertyWithDoc(getter)

        def fn(x):
            p.__doc__ = "Changed doc."
            return x + 1, p.__doc__, p.fget is getter, p.__get__(x, type(x))

        x = torch.ones(2)
        expected = fn(x)
        p.__doc__ = "Original doc."
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(p.__doc__, "Changed doc.")

    def test_descriptor_methods(self):
        class Holder:
            pass

        def getter(obj):
            return obj.value

        def setter(obj, value):
            obj.value = value

        def deleter(obj):
            del obj.value

        p = PropertyWithDoc(getter, setter, deleter)

        def fn(x):
            obj = Holder()
            p.__set__(obj, x + 1)
            y = p.__get__(obj, Holder)
            p.__delete__(obj)
            return y, hasattr(obj, "value"), p.fset is setter, p.fdel is deleter

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("getter", (None, 1))
    @unittest.skipIf(
        sys.version_info < (3, 13), "property.__name__ requires Python 3.13+"
    )
    def test_missing_name(self, getter):
        def fn(x):
            p = PropertyWithoutDoc(getter)
            try:
                return x + 1, p.__name__
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_getter_doc_mutation_guard(self):
        def getter(obj):
            """Original doc."""
            return obj

        def fn(x):
            p = PropertyWithoutDoc(getter)
            return x + 1, p.__doc__

        x = torch.ones(2)
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(counter.frame_count, 1)
        getter.__doc__ = "Changed doc."
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(counter.frame_count, 2)

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_pending_getter_doc(self):
        def fn(x):
            def getter(obj):
                """Original doc."""
                return obj + x

            getter.__doc__ = "Changed doc."
            p = PropertyWithoutDoc(getter)
            return x + 1, p.__doc__, p

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[1], "Changed doc.")
        self.assertEqual(actual[2].__doc__, "Changed doc.")
        self.assertEqual(actual[2].fget.__doc__, "Changed doc.")

    @parametrize("kind", ("tensor", "list"))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_nonstring_doc(self, kind):
        def fn(x):
            doc = x + 1
            if kind == "list":
                doc = [doc]
            p = PropertyWithoutDoc(doc=doc)
            return x + 2, p, doc, p.__doc__ is doc

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 2)
        self.assertIs(actual[1].__doc__, actual[2])
        self.assertTrue(actual[3])

    @parametrize("cls", (PropertyWithoutDoc, abc.abstractproperty))
    def test_existing_metadata(self, cls):
        def getter(obj):
            return obj

        p = cls(getter)

        def fn(x):
            return x + 1, p.__isabstractmethod__

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("abstract", (False, True))
    def test_shadowed_metadata(self, abstract):
        class NamedProperty(property):
            __name__ = "class name"
            __isabstractmethod__ = abstract

        def getter(obj):
            return obj

        getter.__isabstractmethod__ = not abstract
        p = NamedProperty(getter)

        def fn(x):
            return x + 1, p.__name__, p.__isabstractmethod__

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("same", (False, True))
    def test_existing_comparison(self, same):
        p = PropertyWithoutDoc()
        q = p if same else PropertyWithoutDoc()

        def fn(x):
            return x + 1, p == q, p != q, operator.ne(p, None), p in [q]

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_custom_comparison(self):
        class ComparableProperty(property):
            def __eq__(self, other):
                return True

        p, q = ComparableProperty(), ComparableProperty()

        def fn(x):
            return x + 1, p == q, p != q

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("direct", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_custom_init(self, direct):
        class InitializedProperty(property):
            def __init__(self, getter):
                if direct:
                    property.__init__(self, getter)
                else:
                    super().__init__(getter)
                self.tag = "initialized"

        def getter(obj):
            """Getter doc."""
            return obj + 1

        def fn(x):
            p = InitializedProperty(getter)
            return x + 2, p.__doc__, p.tag, p.__get__(x, type(x)), p

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:4], expected[:4])
        self.assertIs(actual[4].fget, getter)
        self.assertEqual(actual[4].__doc__, expected[4].__doc__)

    @parametrize("override", (False, True, "super"))
    def test_decorator_descriptor(self, override):
        class DecoratorProperty(property):
            if override:

                def __get__(self, obj, owner=None):
                    if obj is None:
                        return self
                    if override == "super":
                        return super().__get__(obj, owner) + 2
                    return obj.value + 2

                def __set__(self, obj, value):
                    if override == "super":
                        return super().__set__(obj, value + 3)
                    obj.value = value + 3

        def setter(obj, value):
            obj.value = value

        class Holder:
            @DecoratorProperty
            def prop(self):
                return self.value + 1

        Holder.prop = DecoratorProperty(Holder.__dict__["prop"].fget, setter)

        def fn(x):
            obj = Holder()
            obj.prop = x
            return obj.prop, obj.value

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("cls", (PropertyWithoutDoc, abc.abstractproperty))
    @unittest.skipIf(sys.version_info >= (3, 12), "only legacy constructor fallback")
    def test_legacy_construction_graph_break(self, cls):
        def getter(obj):
            return obj

        def fn(x):
            p = cls(getter)
            return x + 1, p.fget is getter

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "property subclass initialization"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    def test_existing_reinitialization_graph_break(self):
        def getter(obj):
            return obj + 1

        p = PropertyWithoutDoc()

        def fn(x):
            p.__init__(getter)
            return x + 1, p.__get__(x, type(x))

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "property subclass initialization"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        self.assertIs(p.fget, getter)

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_deleted_doc_on_return(self):
        def fn(x):
            p = PropertyWithoutDoc(doc="deleted doc")
            del p.__doc__
            return x + 1, p

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 1)
        self.assertIsNone(actual[1].__doc__)
        self.assertNotIn("__doc__", actual[1].__dict__)

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ docstrings")
    def test_pending_getter_doc_without_doc_slot(self):
        class SlottedProperty(property):
            __slots__ = ()

        def getter(obj):
            """Original doc."""
            return obj

        def fn(x):
            getter.__doc__ = None
            return x + 1, SlottedProperty(getter)

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 1)
        self.assertIs(actual[1].fget, getter)
        self.assertIsNone(getter.__doc__)

    @parametrize("replay", (False, True))
    def test_custom_setattr_graph_break(self, replay):
        events = []

        class RecordingProperty(property):
            def __setattr__(self, name, value):
                events.append(name)
                object.__setattr__(self, name, value)

        def getter(obj):
            """doc"""
            return obj

        def fn(x):
            return x + 1, RecordingProperty(getter)

        x = torch.ones(2)
        with torch._dynamo.config.patch(replay_side_effects=replay):
            with self.assertRaisesRegex(
                Unsupported, "property subclass initialization"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(events, [])
            actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[0], x + 1)
        self.assertEqual(actual[1].__doc__, "doc")
        self.assertEqual(events, ["__doc__"])

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_reinitialize_with_later_object(self):
        class Getter:
            def __call__(self, obj):
                return obj + self.value

        def fn(x):
            p = PropertyWithoutDoc()
            getter = Getter()
            getter.value = x + 1
            p.__init__(getter)
            return x + 2, p, getter

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 2)
        self.assertIs(actual[1].fget, actual[2])
        self.assertEqual(actual[1].__get__(x, type(x)), 2 * x + 1)

    @unittest.skipIf(
        sys.version_info < (3, 13), "property.__name__ requires Python 3.13+"
    )
    def test_reinitialization_resets_name(self):
        def getter(obj):
            return obj

        def fn(x):
            p = PropertyWithoutDoc()
            p.__name__ = "old name"
            p.__init__(getter)
            return x + 1, p.__name__, p

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:2], (x + 1, "getter"))
        self.assertEqual(actual[2].__name__, "getter")

    def test_custom_doc_setter_graph_break(self):
        events = []

        class RecordingDocProperty(property):
            @property
            def __doc__(self):
                return self.doc

            @__doc__.setter
            def __doc__(self, value):
                events.append(value)
                self.doc = value

        def getter(obj):
            """doc"""
            return obj

        def fn(x):
            return x + 1, RecordingDocProperty(getter)

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "property subclass initialization"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(events, [])
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[0], x + 1)
        self.assertEqual(actual[1].__doc__, "doc")
        self.assertEqual(events, ["doc"])


if __name__ == "__main__":
    run_tests()
