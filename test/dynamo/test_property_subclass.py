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

    @parametrize("shadow_name", (False, True))
    @parametrize("via_dict", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_reinitialization_preserves_instance_name(self, shadow_name, via_dict):
        class NamedProperty(property):
            __name__ = "class name"

        cls = NamedProperty if shadow_name else PropertyWithoutDoc

        def getter(obj):
            return obj

        def fn(x):
            p = cls()
            if via_dict:
                p.__dict__["__name__"] = "custom"
            else:
                p.__name__ = "custom"
            p.__init__(getter)
            return x + 1, p.__name__, "__name__" in p.__dict__, p

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:3], expected[:3])
        self.assertEqual(actual[3].__name__, expected[3].__name__)
        self.assertEqual(actual[3].__dict__, expected[3].__dict__)

    @parametrize("slotted", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_cond_reinitialization_rejects_outer_mutation(self, slotted):
        class SlottedProperty(property):
            __slots__ = ()

        cls = SlottedProperty if slotted else PropertyWithoutDoc

        def getter1(obj):
            return obj + 1

        def getter2(obj):
            return obj - 1

        def fn(pred, x):
            p = cls()

            def true_fn(x):
                p.__init__(getter1)
                return x + 1

            def false_fn(x):
                p.__init__(getter2)
                return x - 1

            return torch.cond(pred, true_fn, false_fn, (x,)), p

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError, "HOP: Unsafe side effect"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.tensor(True), torch.ones(2)
            )

    @parametrize("pred", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_cond_branch_local_initialization(self, pred):
        class SlottedProperty(property):
            __slots__ = ()

        def getter1(obj):
            return obj + 1

        def getter2(obj):
            return obj - 1

        def fn(pred, x):
            def true_fn(x):
                p = SlottedProperty(getter1)
                return p.__get__(x, type(x))

            def false_fn(x):
                p = SlottedProperty(getter2)
                return p.__get__(x, type(x))

            return torch.cond(pred, true_fn, false_fn, (x,))

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(pred), x
        )
        self.assertEqual(actual, x + 1 if pred else x - 1)

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

    @parametrize("action", ("set", "delete"))
    @parametrize("shadow_name", (False, True))
    @parametrize("explicit", (False, True))
    def test_missing_accessor_message(self, action, shadow_name, explicit):
        class NamedProperty(property):
            __name__ = "shadowed name"

        cls = NamedProperty if shadow_name else PropertyWithoutDoc

        class Holder:
            prop = cls(lambda obj: obj.value)

        p = Holder.__dict__["prop"]

        def fn(x):
            obj = Holder()
            try:
                if action == "set":
                    if explicit:
                        p.__set__(obj, x)
                    else:
                        obj.prop = x
                elif explicit:
                    p.__delete__(obj)
                else:
                    del obj.prop
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        if sys.version_info < (3, 13):
            with self.assertRaisesRegex(Unsupported, "legacy property error message"):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        else:
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @parametrize("action", ("set", "delete"))
    @parametrize(
        "name_case",
        ("default", "renamed", "shadowed", "shadowed_store", "quoted", "none"),
    )
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_new_missing_accessor_message(self, action, name_case):
        class NamedProperty(property):
            __name__ = "shadowed name"

        class Holder:
            pass

        cls = NamedProperty if name_case.startswith("shadowed") else PropertyWithoutDoc

        def getter(obj):
            return obj

        def fn(x):
            p = cls(getter)
            if name_case in ("renamed", "shadowed_store"):
                p.__name__ = "custom name"
            elif name_case == "quoted":
                p.__name__ = "name's quote"
            elif name_case == "none":
                p.__name__ = None
            try:
                if action == "set":
                    p.__set__(Holder(), x)
                else:
                    p.__delete__(Holder())
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("shadow_name", (False, True))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_name_mutation_guard(self, shadow_name):
        class NamedProperty(property):
            __name__ = "shadowed name"

        cls = NamedProperty if shadow_name else PropertyWithoutDoc

        class Holder:
            prop = cls()

        p = Holder.__dict__["prop"]

        def fn(x):
            try:
                p.__set__(Holder(), x)
            except AttributeError as exc:
                return x + 1, str(exc)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        property.__dict__["__name__"].__set__(p, "renamed property")
        self.assertEqual(compiled(x + 2), fn(x + 2))
        self.assertEqual(compiled(x + 3), fn(x + 3))
        self.assertEqual(counter.frame_count, 2)
        property.__dict__["__name__"].__delete__(p)
        self.assertEqual(compiled(x + 4), fn(x + 4))
        self.assertEqual(compiled(x + 5), fn(x + 5))
        self.assertEqual(counter.frame_count, 3)
        property.__dict__["__name__"].__set__(p, None)
        self.assertEqual(compiled(x + 6), fn(x + 6))
        self.assertEqual(compiled(x + 7), fn(x + 7))
        self.assertEqual(counter.frame_count, 4)

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_name_source_reconstruction(self):
        p = PropertyWithoutDoc()
        name = []
        p.__name__ = name

        def fn(x):
            return x + 1, p.__name__

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 1)
        self.assertIs(actual[1], name)

    @parametrize("reinitialize", (False, True))
    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_getter_missing_doc(self, reinitialize):
        class Getter:
            __slots__ = ("__doc__",)

            def __call__(self, obj):
                return obj

        getter = Getter()

        def fn(x):
            if reinitialize:
                p = PropertyWithDoc(doc="original doc")
                p.__init__(getter)
            else:
                p = PropertyWithDoc(getter)
            return x + 1, p.__doc__, p

        x = torch.ones(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(actual[2].__dict__, expected[2].__dict__)
        self.assertIs(actual[2].fget, getter)

    @parametrize("change", ("rename", "shadow", "insert_shadow", "swap"))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_name_mro_guard(self, change):
        class BaseProperty(property):
            pass

        class NamedProperty(property):
            __name__ = "shadowed name"

        class SubProperty(BaseProperty):
            pass

        class NamedMixin:
            __slots__ = ()
            __name__ = "shadowed name"

        if change == "swap":

            class SubProperty(property, NamedMixin):
                pass

        def getter(obj):
            return obj

        def fn(x):
            return x + 1, SubProperty(getter).__name__

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        if change == "rename":
            SubProperty.__name__ = "RenamedProperty"
        elif change == "shadow":
            SubProperty.__bases__ = (NamedProperty,)
        elif change == "insert_shadow":
            BaseProperty.__bases__ = (NamedMixin, property)
        else:
            SubProperty.__bases__ = (NamedMixin, property)
        self.assertEqual(compiled(x + 2), fn(x + 2))
        self.assertEqual(compiled(x + 3), fn(x + 3))
        self.assertEqual(counter.frame_count, 1 if change == "rename" else 2)

    @parametrize("action", ("get", "set", "class_get"))
    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_sourceless_custom_descriptor_graph_break(self, action):
        class CustomProperty(property):
            def __get__(self, obj, owner=None):
                return owner.marker if obj is None else obj.value + 1

            def __set__(self, obj, value):
                obj.value = value + 2

        descriptor = CustomProperty()

        def fn(x):
            class Holder:
                marker = "class access"
                prop = descriptor

            if action == "class_get":
                return x + 1, Holder.prop
            obj = Holder()
            obj.value = x
            if action == "set":
                obj.prop = x
                return obj.value
            return obj.prop

        kind = "setter" if action == "set" else "getter"
        with self.assertRaisesRegex(
            Unsupported, f"Property subclass {kind} without a source"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
        self.assertEqual(
            torch.compile(fn, backend="eager")(torch.ones(2)), fn(torch.ones(2))
        )

    @parametrize("inherited", (False, True))
    def test_plain_property_replacement_guard(self, inherited):
        class Base:
            pass

        class Holder(Base):
            pass

        owner = Base if inherited else Holder
        owner.prop = property(lambda obj: obj.value + 1)

        def fn(x):
            obj = Holder()
            obj.value = x
            return obj.prop

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        owner.prop = property(lambda obj: obj.value * 2)
        self.assertEqual(compiled(x + 2), fn(x + 2))
        self.assertEqual(compiled(x + 3), fn(x + 3))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("inherited", (False, True))
    @parametrize("mode", ("inherited", "super", "custom"))
    def test_class_descriptor(self, inherited, mode):
        class DecoratorProperty(property):
            if mode != "inherited":

                def __get__(self, obj, owner=None):
                    if mode == "custom" and obj is None:
                        return owner.marker
                    return super().__get__(obj, owner)

        def getter(obj):
            return obj.value

        descriptor = DecoratorProperty(getter)
        descriptor.tag = "descriptor tag"

        class Base:
            marker = "class access"

        class Holder(Base):
            pass

        owner = Base if inherited else Holder
        owner.prop = descriptor

        def fn(x):
            p = Holder.prop
            if mode == "custom":
                return x + 1, p
            return x + 1, type(p) is DecoratorProperty, p.tag, p.fget is getter

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_sourceless_class_descriptor(self):
        descriptor = PropertyWithDoc()
        descriptor.tag = "descriptor tag"

        def fn(x):
            class Holder:
                prop = descriptor

            p = Holder.prop
            return x + 1, type(p) is PropertyWithDoc, p.tag, p.__doc__

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("cls", (property, PropertyWithoutDoc))
    def test_descriptor_get_none(self, cls):
        p = cls(lambda obj: obj.value)

        class Holder:
            pass

        def fn(x):
            return x + 1, p.__get__(None, Holder) is p

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("receiver", ("none", "int", "object", "tensor", "missing"))
    @parametrize("keyword", (False, True))
    def test_property_init_invalid_receiver(self, receiver, keyword):
        values = {"none": None, "int": 1, "object": object(), "tensor": torch.ones(1)}
        value = values.get(receiver)

        def fn(x, value):
            try:
                if receiver == "missing":
                    if keyword:
                        property.__init__(fget=None)
                    else:
                        property.__init__()
                elif keyword:
                    property.__init__(value, fget=None)
                else:
                    property.__init__(value, None)
            except TypeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, value), fn(x, value)
        )

    def test_exact_property_initialization_graph_break(self):
        p = property()

        def getter(obj):
            return obj

        def fn(x):
            p.__init__(getter)
            return x + 1, p.fget is getter

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "Unsupported property initialization"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIsNone(p.fget)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @parametrize("method", ("getter", "setter", "deleter", "__set_name__"))
    def test_custom_property_methods(self, method):
        class CustomProperty(property):
            def getter(self, value):
                return value + 1

            def setter(self, value):
                return value + 2

            def deleter(self, value):
                return value + 3

            def __set_name__(self, owner, value):
                return value + 4

        p = CustomProperty()

        def fn(x):
            if method == "__set_name__":
                return p.__set_name__(None, x)
            return getattr(p, method)(x)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "attr", ("fget", "fset", "fdel", "__name__", "__isabstractmethod__", "__doc__")
    )
    @parametrize("new", (False, True))
    def test_custom_metadata_getattribute(self, attr, new):
        if new and sys.version_info < (3, 12):
            self.skipTest("requires Python 3.12+ construction")

        class InterceptedProperty(property):
            def __getattribute__(self, name):
                if name == attr:
                    return "intercepted"
                return super().__getattribute__(name)

        def getter(obj):
            return obj

        existing = InterceptedProperty(getter)

        def fn(x):
            p = InterceptedProperty(getter) if new else existing
            return x + 1, getattr(p, attr)

        x = torch.ones(2)
        if new:
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        else:
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @parametrize("new", (False, True))
    @parametrize("explicit", (False, True))
    def test_delegated_metadata_getattribute(self, new, explicit):
        if new and sys.version_info < (3, 12):
            self.skipTest("requires Python 3.12+ construction")

        class DelegatedProperty(property):
            def __getattribute__(self, name):
                return super().__getattribute__(name)

        def getter(obj):
            return obj

        existing = DelegatedProperty(getter)
        raw_fget = property.fget

        def fn(x):
            p = DelegatedProperty(getter) if new else existing
            fget = raw_fget.__get__(p) if explicit else p.fget
            return x + 1, fget is getter

        x = torch.ones(2)
        if new:
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        else:
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @parametrize("action", ("get", "set", "delete"))
    @parametrize("override", ("getattribute", "class_attribute"))
    def test_descriptor_uses_c_accessors(self, action, override):
        def fake(obj, value=None):
            obj.value = obj.value + 100
            return obj.value

        class OverriddenProperty(property):
            if override == "getattribute":

                def __getattribute__(self, name):
                    if name in ("fget", "fset", "fdel"):
                        return fake
                    return super().__getattribute__(name)
            else:
                fget = fset = fdel = fake

        def getter(obj):
            return obj.value + 1

        def setter(obj, value):
            obj.value = value + 2

        def deleter(obj):
            obj.value = obj.value + 3

        class Holder:
            prop = OverriddenProperty(getter, setter, deleter)

        def fn(x):
            obj = Holder()
            obj.value = x
            if action == "get":
                return obj.prop
            if action == "set":
                obj.prop = x
            else:
                del obj.prop
            return obj.value

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("accessor", ("fget", "fset", "fdel"))
    def test_abstractness_uses_c_accessors(self, accessor):
        class ShadowedProperty(property):
            fget = fset = fdel = None

        def fn_accessor(*args):
            return None

        fn_accessor.__isabstractmethod__ = True
        p = ShadowedProperty(**{accessor: fn_accessor})

        def fn(x):
            return x + 1, p.__isabstractmethod__

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("action", ("set", "delete"))
    @parametrize("failure", (None, "attribute", "value", "non_string"))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_traces_name_repr(self, action, failure):
        events = []

        class Name:
            def __repr__(self):
                events.append("repr")
                if failure == "attribute":
                    raise AttributeError("name repr failed")
                if failure == "value":
                    raise ValueError("name repr failed")
                if failure == "non_string":
                    return 123
                return "custom property name"

        class Holder:
            pass

        p = PropertyWithoutDoc()
        p.__name__ = Name()

        def fn(x):
            try:
                if action == "set":
                    p.__set__(Holder(), x)
                else:
                    p.__delete__(Holder())
            except (AttributeError, ValueError, TypeError) as exc:
                return x + 1, type(exc), str(exc)

        x = torch.ones(2)
        expected = fn(x)
        self.assertEqual(events, ["repr"])
        events.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, ["repr"])
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, ["repr", "repr"])

    @parametrize("action", ("set", "delete"))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_tensor_name_graph_break(self, action):
        class Holder:
            pass

        def fn(x):
            p = PropertyWithoutDoc()
            p.__name__ = x
            try:
                if action == "set":
                    p.__set__(Holder(), x)
                else:
                    p.__delete__(Holder())
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_existing_name_getter_graph_break(self):
        events = []

        class Getter:
            def __call__(self, obj):
                return obj

            def __getattribute__(self, name):
                if name == "__name__":
                    events.append("name")
                    return str(len(events))
                return super().__getattribute__(name)

        p = PropertyWithoutDoc(Getter())

        def fn(x):
            return x + 1, p.__name__

        x = torch.ones(2)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(events, [])
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(compiled(x), (x + 1, "1"))
        self.assertEqual(events, ["name"])
        self.assertEqual(compiled(x), (x + 1, "2"))
        self.assertEqual(events, ["name", "name"])

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_existing_name_getter_mutation_graph_break(self):
        def getter(obj):
            return obj

        p = PropertyWithoutDoc(getter)

        def fn(x):
            getter.__name__ = "renamed getter"
            return x + 1, p.__name__

        x = torch.ones(2)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(getter.__name__, "getter")
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @parametrize("dictionary_first", (False, True))
    @parametrize("reinitialize", (False, True))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_mixed_name_stores_graph_break(self, dictionary_first, reinitialize):
        def getter(obj):
            return obj

        def fn(x):
            p = PropertyWithoutDoc(getter)
            if dictionary_first:
                p.__dict__["__name__"] = "dict name"
                p.__name__ = "slot name"
            else:
                p.__name__ = "slot name"
                p.__dict__["__name__"] = "dict name"
            if reinitialize:
                p.__init__(getter)
            return x + 1, p.__name__, p.__dict__["__name__"], p

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "property.*name|Property.*name"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[:3], expected[:3])
        self.assertEqual(actual[3].__name__, expected[3].__name__)
        self.assertEqual(actual[3].__dict__, expected[3].__dict__)

    @parametrize("new", (False, True))
    @parametrize("attr", ("fget", "fset", "fdel", "__isabstractmethod__", "__doc__"))
    def test_raw_descriptor_ignores_shadow(self, new, attr):
        if new and sys.version_info < (3, 12):
            self.skipTest("requires Python 3.12+ construction")

        class ShadowProperty(property):
            fget = fset = fdel = "shadowed"
            __isabstractmethod__ = "shadowed"

        def getter(obj):
            """Getter docstring."""
            return obj

        existing = ShadowProperty(getter)
        descriptor = property.__dict__[attr]

        def fn(x):
            p = ShadowProperty(getter) if new else existing
            return x + 1, descriptor.__get__(p)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2)),
            fn(torch.ones(2)),
        )

    @parametrize("shadow", (False, True))
    @parametrize("delete", (False, True))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_raw_name_setter_graph_break(self, shadow, delete):
        events = []

        class CustomProperty(property):
            if shadow:
                __name__ = "shadowed"
            else:

                def __setattr__(self, name, value):
                    events.append(name)
                    super().__setattr__(name, value)

        descriptor = property.__dict__["__name__"]
        p = CustomProperty(lambda obj: obj)
        p.__name__ = "original"
        events.clear()

        def fn(x):
            if delete:
                descriptor.__delete__(p)
            else:
                descriptor.__set__(p, "changed")
            return x + 1

        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
        self.assertEqual(events, [])
        torch.compile(fn, backend="eager")(torch.ones(2))
        self.assertEqual(events, [])
        expected = "<lambda>" if delete else "changed"
        self.assertEqual(descriptor.__get__(p), expected)

    @parametrize("implicit", (False, True))
    def test_missing_getter(self, implicit):
        class Holder:
            prop = PropertyWithoutDoc()

        p = Holder.__dict__["prop"]

        def fn(x):
            try:
                obj = Holder()
                result = obj.prop if implicit else p.__get__(obj, Holder)
                return x + result
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        if sys.version_info < (3, 13):
            with self.assertRaises(Unsupported):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        else:
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @parametrize("existing_owner", (False, True))
    @parametrize("mutate_inside", (False, True))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_owner_name(self, mutate_inside, existing_owner):
        class Holder:
            pass

        p = PropertyWithoutDoc(lambda obj: obj)
        existing = Holder()

        def fn(x):
            if mutate_inside:
                Holder.__qualname__ = "Changed"
            try:
                p.__set__(existing if existing_owner else Holder(), x)
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        self.assertEqual(compiled(x), fn(x))
        if not mutate_inside:
            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(counter.frame_count, 1)
            Holder.__qualname__ = "Changed"
            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(counter.frame_count, 2)

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_owner_name_repr(self):
        events = []

        class Name(str):
            __slots__ = ()

            def __repr__(self):
                events.append("repr")
                return "custom owner"

        class Holder:
            pass

        Holder.__qualname__ = Name("Holder")
        p = PropertyWithoutDoc(lambda obj: obj)

        def fn(x):
            try:
                p.__set__(Holder(), x)
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        expected = fn(x)
        events.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, ["repr"])
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, ["repr", "repr"])

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_missing_accessor_custom_metaclass_graph_break(self):
        events = []

        class Meta(type):
            def __getattribute__(cls, name):
                if name == "__qualname__":
                    events.append(name)
                    return "fake owner"
                return super().__getattribute__(name)

        class Holder(metaclass=Meta):
            pass

        p = PropertyWithoutDoc(lambda obj: obj)

        def fn(x):
            try:
                p.__set__(Holder(), x)
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        expected = fn(x)
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(events, [])
        self.assertEqual(torch.compile(fn, backend="eager")(x), expected)
        self.assertEqual(events, [])

    @parametrize("manual_name", (False, True))
    def test_legacy_missing_accessor_name_graph_break(self, manual_name):
        if sys.version_info >= (3, 13):
            self.skipTest("legacy names are not exposed before Python 3.13")

        class Holder:
            prop = PropertyWithoutDoc()
            alias = prop

        if manual_name:
            Holder.prop.__set_name__(Holder, "renamed")

        def fn(x):
            try:
                Holder().prop = x
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "legacy property error message"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @parametrize("new", (False, True))
    @parametrize("explicit", (False, True))
    @parametrize("getter_name", (True, False))
    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_deleted_name_uses_getter_name(self, new, explicit, getter_name):
        def getter(obj):
            return obj

        getter = getter if getter_name else None
        existing = PropertyWithoutDoc(getter)
        descriptor = property.__dict__["__name__"]

        def fn(x):
            p = PropertyWithoutDoc(getter) if new else existing
            p.__name__ = "custom"
            if explicit:
                descriptor.__delete__(p)
            else:
                del p.__name__
            return x + 1, getattr(p, "__name__", "missing"), p

        x = torch.ones(2)
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for value in (x, x + 1):
            result = compiled(value)
            self.assertEqual(result[0], value + 1)
            self.assertEqual(result[1], expected[1])
            self.assertEqual(getattr(result[2], "__name__", "missing"), expected[1])

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_name_repr_string_subclass(self):
        events = []

        class Text(str):
            __slots__ = ()

            def __format__(self, spec):
                events.append("format")
                return "wrong"

        class Name:
            def __repr__(self):
                return Text("custom name")

        class Holder:
            pass

        p = PropertyWithoutDoc()
        p.__name__ = Name()

        def fn(x):
            try:
                p.__set__(Holder(), x)
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))
        self.assertEqual(events, [])

    @parametrize("cls", (property, PropertyWithoutDoc))
    @parametrize("action", ("get", "set", "delete"))
    @parametrize("explicit", (False, True))
    @parametrize("name_case", ("original", "alias", "renamed"))
    def test_missing_accessor_stored_name(self, cls, action, explicit, name_case):
        class Holder:
            prop = cls()
            if name_case == "alias":
                alias = prop

        p = Holder.__dict__["prop"]
        if name_case == "renamed":
            p.__set_name__(Holder, "renamed's property")

        def fn(x):
            obj = Holder()
            try:
                if action == "get":
                    if explicit:
                        p.__get__(obj, Holder)
                    else:
                        obj.prop
                elif action == "set":
                    if explicit:
                        p.__set__(obj, x)
                    else:
                        obj.prop = x
                elif explicit:
                    p.__delete__(obj)
                else:
                    del obj.prop
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(2)
        if sys.version_info < (3, 13):
            with self.assertRaisesRegex(Unsupported, "legacy property error message"):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
            compiled = torch.compile(fn, backend="eager")
        else:
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))


if __name__ == "__main__":
    run_tests()
