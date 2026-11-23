# Owner(s): ["module: dynamo"]

"""Regression tests for TorchDynamo property subclass support."""

import sys
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.test_case import run_tests
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class PropertySub(property):
    """This is a subclass of property"""


@instantiate_parametrized_tests
class PropertySubclassTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @unittest.skipIf(
        sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above"
    )
    def test_issue41287(self):
        def fn():
            cls_doc = PropertySub.__doc__
            inst_doc = PropertySub(None, None, None, "issue 41287 is fixed").__doc__
            return cls_doc, inst_doc

        cls_doc, inst_doc = fn()
        self.assertEqual(cls_doc, "This is a subclass of property")
        self.assertEqual(inst_doc, "issue 41287 is fixed")

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        cls_doc2, inst_doc2 = opt_fn()
        self.assertEqual(cls_doc2, "This is a subclass of property")
        self.assertEqual(inst_doc2, "issue 41287 is fixed")

    @unittest.skipIf(
        sys.version_info < (3, 13), "Requires Python 3.13 property semantics"
    )
    def test_property_with_slots_no_docstring(self):
        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop(lambda self: 1)
            return getattr(p, "__doc__", None)

        self.assertIsNone(fn())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNone(opt_fn())

    @unittest.skipIf(
        sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above"
    )
    def test_property_with_slots_docstring_silently_dropped(self):
        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop(doc="what's up")
            return p.__doc__

        self.assertIsNone(fn())
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNone(opt_fn())

    @unittest.skipIf(
        sys.flags.optimize >= 2, "Docstrings are omitted with -O2 and above"
    )
    @unittest.skipIf(
        sys.version_info < (3, 13), "Requires Python 3.13 property semantics"
    )
    def test_property_with_slots_and_doc_slot_docstring_present(self):
        class slotted_prop(property):
            __slots__ = ("foo", "__doc__")

        def fn_kwarg():
            p = slotted_prop(doc="what's up")
            return p.__doc__

        self.assertEqual(fn_kwarg(), "what's up")
        opt_fn = torch.compile(fn_kwarg, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "what's up")

        def documented_getter():
            """what's up getter doc?"""
            return 4

        def fn_getter():
            p = slotted_prop(documented_getter)
            return p.__doc__

        self.assertEqual(fn_getter(), "what's up getter doc?")
        opt_fn2 = torch.compile(fn_getter, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn2(), "what's up getter doc?")

    def test_property_subclass_getter_chaining(self):
        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p2 = p.getter(lambda self: 2)
            return type(p2).__name__, p2.fget(None)

        name, val = fn()
        self.assertEqual(name, "Sub")
        self.assertEqual(val, 2)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        name2, val2 = opt_fn()
        self.assertEqual(name2, "Sub")
        self.assertEqual(val2, 2)

    def test_property_subclass_setter_chaining(self):
        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)

            def setter(self, v):
                return v + 1

            p2 = p.setter(setter)
            return (
                type(p2).__name__,
                p2.fget is p.fget,
                p2.fset is setter,
                p2.fset(None, 2),
            )

        self.assertEqual(fn(), ("Sub", True, True, 3))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), fn())

    def test_property_subclass_deleter_chaining(self):
        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)

            def deleter(self):
                return 3

            p2 = p.deleter(deleter)
            return (
                type(p2).__name__,
                p2.fget is p.fget,
                p2.fdel is deleter,
                p2.fdel(None),
            )

        self.assertEqual(fn(), ("Sub", True, True, 3))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), fn())

    def test_property_subclass_instance_dict_attr(self):
        class Tagged(property):
            pass

        def fn():
            p = Tagged(lambda self: 99)
            p.tag = "mytag"
            return p.tag

        self.assertEqual(fn(), "mytag")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "mytag")

    def test_property_doc_write_base(self):
        def fn():
            p = property(lambda self: 1)
            p.__doc__ = "new doc"
            return p.__doc__

        self.assertEqual(fn(), "new doc")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "new doc")

    def test_property_subclass_doc_write_with_dict(self):
        class Sub(property):
            pass

        def fn():
            p = Sub(lambda self: 1)
            p.__doc__ = "written"
            return p.__doc__

        self.assertEqual(fn(), "written")
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), "written")

    def test_property_subclass_doc_write_readonly(self):
        class slotted_prop(property):
            __slots__ = ("foo",)

        def fn():
            p = slotted_prop()
            try:
                p.__doc__ = "nope"
            except AttributeError as exc:
                return str(exc)
            return None

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNotNone(fn())
        self.assertEqual(opt_fn(), fn())

    @parametrize("construct", [False, True])
    def test_descriptor_override(self, construct):
        class Sub(property):
            def __get__(self, obj, owner=None):
                return obj + 2

        prop = Sub(lambda obj: obj + 1)

        def fn(x):
            p = Sub(lambda obj: obj + 1) if construct else prop
            return p.__get__(x)

        x = torch.ones(2)
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter, fullgraph=not construct)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertGreater(counter.frame_count, 0)

    def test_closure_rebinding(self):
        def fn(x):
            value = 1

            def getter(obj):
                return obj + value

            prop = PropertySub(getter)
            value = 2
            return prop.__get__(x), prop.fget is getter

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    @parametrize("cls", [property, PropertySub])
    def test_return_property(self, cls):
        def fn(x):
            prop = cls(lambda obj: obj + x)
            return x + 1, prop

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        first_result, first = opt_fn(x)
        second_result, second = opt_fn(x)
        self.assertEqual(first_result, x + 1)
        self.assertEqual(second_result, x + 1)
        self.assertIsInstance(first, cls)
        self.assertIsInstance(second, cls)
        self.assertIsNot(first, second)
        self.assertEqual(first.fget(x), x + x)
        self.assertEqual(second.fget(x), x + x)

    @parametrize("cls", [property, PropertySub])
    def test_new_only_allocates(self, cls):
        def fn(x):
            prop = property.__new__(cls, lambda obj: obj)
            return x + 1, prop.fget, prop.fset, prop.fdel

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_subclass_constructor_side_effects(self):
        class Sub(property):
            def __new__(cls, getter, events):
                events.append("new")
                return property.__new__(cls)

            def __init__(self, getter, events):
                events.append("init")
                super().__init__(getter)

        def fn(x, events):
            prop = Sub(lambda obj: obj + 1, events)
            return prop.__get__(x)

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for _ in range(2):
            events = []
            self.assertEqual(opt_fn(x, events), x + 1)
            self.assertEqual(events, ["new", "init"])

    @parametrize("cls", [property, PropertySub])
    def test_undocumented_getter_doc(self, cls):
        def getter(obj):
            return obj

        def fn():
            return cls(getter).__doc__

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertIsNone(fn())
        self.assertEqual(opt_fn(), fn())

    @parametrize("cls", [property, PropertySub])
    def test_doc_write_then_replace_undocumented_getter(self, cls):
        def getter(obj):
            return obj

        def replacement(obj):
            """replacement doc"""
            return obj

        def fn():
            prop = cls(getter)
            prop.__doc__ = "updated doc"
            return prop.getter(replacement).__doc__

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), fn())

    @parametrize("cls", [property, PropertySub])
    def test_doc_write_then_getter(self, cls):
        def getter(obj):
            """original getter"""
            return obj + 1

        def replacement(obj):
            """replacement getter"""
            return obj + 2

        def fn(x):
            prop = cls(getter, doc="explicit doc")
            prop.__doc__ = "updated doc"
            copied = prop.getter(replacement)
            return copied.__get__(x), copied.__doc__

        x = torch.ones(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


if __name__ == "__main__":
    run_tests()
