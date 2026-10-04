# Owner(s): ["module: dynamo"]

import sys
import unittest

import torch
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class PropertyWithDoc(property):
    """Class docstring."""


class PropertyWithoutDoc(property):
    pass


@unittest.skipIf(
    sys.version_info < (3, 12), "property subclass docstrings require Python 3.12+"
)
@instantiate_parametrized_tests
class PropertySubclassTests(TestCase):
    @parametrize("cls", (PropertyWithDoc, PropertyWithoutDoc))
    @parametrize("getter_doc", (None, "Getter docstring."))
    @parametrize("doc", (None, "Explicit docstring.", ""))
    @parametrize("keyword", (False, True))
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

    def test_empty(self):
        def fn(x):
            p = PropertyWithDoc()
            return x + 1, p.__doc__, p.fget, p.fset, p.fdel

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

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
        def fn(x):
            p = PropertyWithDoc()
            try:
                p.fget = None
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("doc", (None, "Explicit doc.", ""))
    @parametrize("getter_doc", (None, "Getter doc."))
    @parametrize("doc_slot", (False, True))
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

        def fn(x):
            p = PropertyWithDoc(getter, setter, deleter)
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

    def test_getter_doc_mutation_guard(self):
        def getter(obj):
            """Original doc."""
            return obj

        def fn(x):
            p = PropertyWithoutDoc(getter)
            return x + 1, p.__doc__

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))
        getter.__doc__ = "Changed doc."
        self.assertEqual(opt_fn(x), fn(x))

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


if __name__ == "__main__":
    run_tests()
