# Owner(s): ["module: dynamo"]

import sys
from operator import ne

import torch
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


@instantiate_parametrized_tests
class PropertyDescriptorDispatchTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_subclass_dispatch_and_identity(self):
        class CustomProperty(property):
            def __getattribute__(self, name):
                if name == "fget":
                    return "custom"
                return super().__getattribute__(name)

            def __get__(self, obj, owner=None):
                if obj is None:
                    return self.label
                return super().__get__(obj, owner) + 1

            def __set__(self, obj, value):
                obj.value = value + 2

        class Holder:
            @CustomProperty
            def prop(self):
                return self.value

        prop = Holder.__dict__["prop"]
        prop.label = "descriptor"

        def fn(x):
            obj = Holder()
            obj.prop = x
            return obj.prop, Holder.prop, prop.fget, prop == prop, ne(prop, None)

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("raw", (False, True))
    @parametrize("name", ("fget", "fset", "fdel", "__isabstractmethod__", "__name__"))
    def test_property_metadata_mro_guard(self, raw, name):
        if name == "__name__" and sys.version_info < (3, 13):
            self.skipTest("property.__name__ requires Python 3.13+")
        mixin = type("Mixin", (), {"__slots__": (), name: "shadowed"})

        class SubProperty(property, mixin):
            pass

        def getter(obj):
            return obj

        prop = SubProperty(getter, getter, getter)
        descriptor = property.__dict__[name]

        def fn(x):
            result = (
                descriptor.__get__(prop, type(prop)) if raw else getattr(prop, name)
            )
            return x + 1, result

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        SubProperty.__bases__ = (mixin, property)
        self.assertEqual(compiled(x + 2), fn(x + 2))
        self.assertEqual(compiled(x + 3), fn(x + 3))
        self.assertEqual(counter.frame_count, 1 if raw else 2)

    @torch._dynamo.disable(recursive=False)
    def test_metaclass_descriptor_replacement(self):
        class CustomProperty(property):
            def __get__(self, obj, owner=None):
                return super().__get__(obj, owner) + 100

        class Meta(type):
            prop = CustomProperty(lambda obj: 1)

        class Holder(metaclass=Meta):
            pass

        def fn(x):
            return x + Holder.prop

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        Meta.prop = CustomProperty(lambda obj: 2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 2)

    def test_explicit_property_slots_bypass_subclass_overrides(self):
        class CustomProperty(property):
            def __get__(self, obj, owner=None):
                raise AssertionError("subclass getter")

            def __set__(self, obj, value):
                raise AssertionError("subclass setter")

            def __delete__(self, obj):
                raise AssertionError("subclass deleter")

        class Holder:
            def __init__(self, value):
                self.value = value

        def setter(obj, value):
            obj.value = value

        def deleter(obj):
            del obj.value

        prop = CustomProperty(lambda obj: obj.value, setter, deleter)

        def fn(x):
            holder = Holder(x)
            property.__set__(prop, holder, x + 1)
            value = property.__get__(prop, holder, Holder)
            property.__delete__(prop, holder)
            return value, hasattr(holder, "value")

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))


if __name__ == "__main__":
    run_tests()
