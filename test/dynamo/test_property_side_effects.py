# Owner(s): ["module: dynamo"]

import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class PropertyWithName(property):
    pass


@instantiate_parametrized_tests
class PropertySideEffectsTests(torch._dynamo.test_case.TestCase):
    @parametrize(
        "storage",
        ("instance", "slot", "descriptor", "property_slot", "property_name_slot"),
    )
    def test_deleted_attribute_then_dictionary_store_order(self, storage):
        class Ordinary:
            pass

        class Slotted:
            __slots__ = ("value", "__dict__")

        class Descriptor:
            def __get__(self, obj, owner=None):
                return obj._value

            def __set__(self, obj, value):
                obj._value = value

            def __delete__(self, obj):
                del obj._value

        class Described:
            value = Descriptor()

        class SlottedProperty(property):
            __slots__ = ("value", "__dict__")

        class NamedSlottedProperty(property):
            __slots__ = ("__name__", "__dict__")

        cls = {
            "instance": Ordinary,
            "slot": Slotted,
            "descriptor": Described,
            "property_slot": SlottedProperty,
            "property_name_slot": NamedSlottedProperty,
        }[storage]
        attr = "__name__" if storage == "property_name_slot" else "value"

        def fn(x):
            obj = cls()
            setattr(obj, attr, 1)
            delattr(obj, attr)
            obj.middle = 2
            obj.__dict__[attr] = 3
            return x + 1, list(obj.__dict__), obj

        x = torch.ones(2)
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        if storage.startswith("property_") and sys.version_info < (3, 12):
            with self.assertRaisesRegex(
                Unsupported, "Unsupported property subclass initialization"
            ):
                compiled(x)
            compiled = torch.compile(fn, backend="eager")
        actual = compiled(x)
        self.assertEqual(actual[:2], expected[:2])
        self.assertEqual(
            list(actual[2].__dict__.items()), list(expected[2].__dict__.items())
        )
        self.assertEqual(hasattr(actual[2], attr), hasattr(expected[2], attr))

    @parametrize("attr", ("__doc__", "__name__", "fget"))
    def test_custom_getattribute_dictionary_mutation(self, attr):
        class InterceptedProperty(property):
            fget = "class shadow"

            def __getattribute__(self, name):
                if name in ("__doc__", "__name__", "fget"):
                    return "intercepted"
                return super().__getattribute__(name)

        p = InterceptedProperty()
        original_dict = p.__dict__.copy()

        def fn(x):
            p.__dict__[attr] = "stored"
            return x + 1, getattr(p, attr)

        x = torch.ones(2)
        with self.assertRaisesRegex(
            Unsupported, "Untracked property subclass mutation"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(p.__dict__, original_dict)

        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual, fn(x))
        self.assertEqual(actual[1], "intercepted")
        self.assertEqual(p.__dict__[attr], "stored")

    @unittest.skipIf(sys.version_info < (3, 13), "requires Python 3.13+ property names")
    def test_deleted_name_then_instance_dictionary_store(self):
        def getter(obj):
            return obj

        p = PropertyWithName(getter)
        p.__name__ = "old name"

        def fn(x):
            del p.__name__
            p.__dict__["__name__"] = "dict name"
            return x + 1, p

        x = torch.ones(2)
        with self.assertRaisesRegex(Unsupported, "Mixed property name"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(p.__name__, "old name")
        self.assertNotIn("__name__", p.__dict__)

        expected = fn(x)
        expected_name = p.__name__
        expected_dict = p.__dict__.copy()
        p.__name__ = "old name"
        del p.__dict__["__name__"]
        actual = torch.compile(fn, backend="eager")(x)
        self.assertEqual(actual[0], expected[0])
        self.assertIs(actual[1], p)
        self.assertEqual(p.__name__, expected_name)
        self.assertEqual(p.__dict__, expected_dict)


if __name__ == "__main__":
    run_tests()
