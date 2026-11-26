# Owner(s): ["module: dynamo"]

import abc
import sys
import unittest

import torch
from torch._dynamo.exc import Unsupported
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class PropertyWithDoc(property):
    """Class docstring."""


class PropertyWithoutDoc(property):
    pass


requires_property_init = unittest.skipIf(
    sys.version_info < (3, 12), "requires Python 3.12+ property initialization"
)


@instantiate_parametrized_tests
class PropertySubclassTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("cls", (PropertyWithDoc, PropertyWithoutDoc))
    @parametrize("getter_doc", (None, "Getter docstring."))
    @parametrize("doc", (None, "Explicit docstring.", ""))
    @parametrize("keyword", (False, True))
    @requires_property_init
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

    @parametrize("graph_break", (False, True))
    @requires_property_init
    def test_nested_getter_and_return(self, graph_break):
        def fn(x):
            def getter(obj):
                """Nested getter."""
                return obj + x

            p = PropertyWithDoc(getter)
            p.tag = x + 1
            if graph_break:
                p.__doc__ = "Changed doc."
                torch._dynamo.graph_break()
            return p.tag + 1, p, p.fget is getter, p.__get__(x)

        x = torch.ones(2)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=not graph_break)
        if graph_break:
            # Check before running the eager reference.
            compiled(x)
            self.assertEqual(counter.frame_count, 2)
            self.assertEqual(counter.op_count, 3)
            self.assertEqual(len(counters["graph_break"]), 1)
            self.assertIn(
                "Call to `torch._dynamo.graph_break()`",
                next(iter(counters["graph_break"])),
            )
        for val in (x, x * 3):
            expected = fn(val)
            actual = compiled(val)
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1].__doc__, expected[1].__doc__)
            self.assertEqual(actual[1].tag, expected[1].tag)
            self.assertEqual(
                actual[1].__get__(val, type(val)), expected[1].__get__(val, type(val))
            )
            self.assertTrue(actual[2])
            self.assertEqual(actual[3], expected[3])
        self.assertIsNot(compiled(x)[1], compiled(x)[1])

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

    @parametrize("doc", (None, "Explicit doc.", ""))
    @parametrize("getter_doc", (None, "Getter doc."))
    @parametrize("doc_slot", (False, True))
    @parametrize("delete_doc", (False, True))
    @requires_property_init
    def test_slots(self, doc, getter_doc, doc_slot, delete_doc):
        class SlottedProperty(property):
            __slots__ = ("__doc__",) if doc_slot else ()

        def getter(obj):
            return obj + 1

        getter.__doc__ = getter_doc

        def fn(x):
            try:
                p = SlottedProperty(getter, doc=doc)
                read_doc = p.__doc__ if doc_slot else getattr(p, "__doc__", None)
                if doc_slot and delete_doc:
                    del p.__doc__
                return x + 1, read_doc, p
            except AttributeError as exc:
                return x + 1, str(exc), None

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for value in (x, x + 3):
            expected = fn(value)
            actual = compiled(value)
            self.assertEqual(actual[:2], expected[:2])
            prop = actual[2]
            if expected[2] is None:
                self.assertIsNone(prop)
                continue
            self.assertIs(type(prop), SlottedProperty)
            self.assertIs(prop.fget, getter)
            self.assertEqual(prop.__get__(value), expected[2].__get__(value))
            sentinel = object()
            self.assertEqual(
                getattr(prop, "__doc__", sentinel),
                getattr(expected[2], "__doc__", sentinel),
            )

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_foreign_doc_slot_graph_break(self):
        events = []

        class Meta(type):
            def __subclasscheck__(cls, other):
                events.append("subclass check")
                return True

        class Other(metaclass=Meta):
            __slots__ = ("__doc__",)

        class ForeignDocProperty(property):
            __doc__ = Other.__dict__["__doc__"]

            def __init__(self, getter):
                try:
                    super().__init__(getter)
                except TypeError:
                    pass

        def getter(obj):
            return obj + 1

        def fn(x):
            return x + 1, ForeignDocProperty(getter)

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "custom doc setter"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(events, [])
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[0], fn(x)[0])
        self.assertIs(type(actual[1]), ForeignDocProperty)
        self.assertIs(actual[1].fget, getter)
        self.assertEqual(actual[1].__get__(x), x + 1)
        self.assertEqual(events, [])

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_doc_slot_uses_real_mro(self):
        class Meta(type):
            @property
            def __mro__(cls):
                return (property, object)

        class SlottedProperty(property, metaclass=Meta):
            __slots__ = ("__doc__",)

        def getter(obj):
            """Getter doc."""
            return obj

        def fn(x):
            return x + 1, SlottedProperty(getter)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for value in (torch.ones(2), torch.zeros(2)):
            actual = compiled(value)
            self.assertEqual(actual[0], fn(value)[0])
            self.assertIs(type(actual[1]), SlottedProperty)
            self.assertIs(actual[1].fget, getter)
            self.assertEqual(actual[1].__doc__, "Getter doc.")
        self.assertEqual(counter.frame_count, 1)

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

    @parametrize("cls", (PropertyWithoutDoc, abc.abstractproperty))
    @unittest.skipIf(sys.version_info >= (3, 12), "only legacy constructor fallback")
    def test_legacy_construction_graph_break(self, cls):
        def getter(obj):
            return obj

        def fn(x):
            p = cls(getter)
            return x + 1, p.fget is getter

        x = torch.ones(2)
        with self.assertRaisesRegex(
            Unsupported, r"^Unsupported property subclass initialization\n"
        ):
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
        with self.assertRaisesRegex(
            Unsupported, r"^Unsupported property subclass initialization\n"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))
        self.assertIs(p.fget, getter)

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
    @requires_property_init
    def test_custom_setattr_graph_break(self, replay):
        events = []

        class RecordingProperty(property):
            pass

        def set_attr(obj, name, value):
            events.append((name, value))
            object.__setattr__(obj, name, value)

        def getter(obj):
            """doc"""
            return obj

        def fn(x):
            return x + 1, RecordingProperty(getter)

        x = torch.ones(2)
        with torch._dynamo.config.patch(replay_side_effects=replay):
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(compiled(x)[1].__doc__, "doc")
            RecordingProperty.__setattr__ = set_attr
            with self.assertRaisesRegex(
                Unsupported, r"^Unsupported property subclass initialization\n"
            ):
                compiled(x)
            self.assertEqual(events, [])
            actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[0], x + 1)
        self.assertEqual(actual[1].__doc__, "doc")
        self.assertEqual(events, [("__doc__", "doc")])

    @unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+ construction")
    def test_reinitialize_with_later_object(self):
        class Getter:
            def __call__(self, obj):
                return obj + self.value

        def fn(x):
            p = PropertyWithoutDoc()
            getter = Getter()
            getter.value = x + 1
            p.__dict__["__name__"] = "instance name"
            p.__init__(getter)
            return x + 2, p, getter

        x = torch.ones(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual[0], x + 2)
        self.assertIs(actual[1].fget, actual[2])
        self.assertEqual(actual[1].__dict__["__name__"], "instance name")
        self.assertEqual(actual[1].__get__(x, type(x)), 2 * x + 1)

    @unittest.skipIf(sys.version_info < (3, 13), "requires the property name getset")
    def test_c_name_write_reinitialization_falls_back(self):
        def getter(obj):
            return obj

        def fn(x):
            p = PropertyWithoutDoc()
            p.__name__ = "old name"
            p.__init__(getter)
            return x + 1, p.__name__, p

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "Write to unmodeled getset/member"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[:2], fn(x)[:2])
        self.assertEqual(actual[2].__name__, "getter")

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
        if sys.version_info < (3, 12, 5):
            expected_dict = {"__doc__": "original doc"} if reinitialize else {}
        else:
            expected_dict = {"__doc__": None}
        self.assertEqual(actual[2].__dict__, expected_dict)


if __name__ == "__main__":
    run_tests()
