# Owner(s): ["module: dynamo"]

from collections.abc import MutableMapping

import torch
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


class GlobalProperty(property):
    pass


def global_getter(obj):
    return obj.value + 1


global_property = GlobalProperty(global_getter)


@instantiate_parametrized_tests
class PropertyDescriptorDispatchTests(TestCase):
    @parametrize("custom_get", (False, True))
    def test_metaclass_descriptor_side_effects(self, custom_get):
        events = []

        def getter(obj):
            events.append("get")
            return obj.value + 1

        class CustomProperty(property):
            def __get__(self, obj, owner=None):
                events.append("custom get")
                return obj.value + 100

        class Meta(type):
            prop = CustomProperty(getter) if custom_get else property(getter)

        class Holder(metaclass=Meta):
            value = 1

        def fn(x):
            return x + Holder.prop

        x = torch.ones(2)
        expected = fn(x)
        expected_events = events[:]
        events.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, expected_events)
        self.assertEqual(compiled(x), expected)
        self.assertEqual(events, expected_events * 2)

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

    def test_tensor_subclass_descriptor(self):
        class CustomProperty(property):
            def __get__(self, obj, owner=None):
                if obj is None:
                    return self
                return obj + 100

        class TensorHolder(torch.Tensor):
            prop = CustomProperty(lambda obj: obj + 1)

        def fn(x):
            return x.prop

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2).as_subclass(TensorHolder)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        TensorHolder.prop = property(lambda obj: obj + 2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("delete", (False, True))
    def test_mutable_mapping_descriptor(self, delete):
        class CustomProperty(property):
            def __set__(self, obj, value):
                obj.value = value + 100

            def __delete__(self, obj):
                obj.value = obj.value + 200

        def setter(obj, value):
            obj.value = value + 1

        def deleter(obj):
            obj.value = obj.value + 2

        class Holder(MutableMapping):
            prop = CustomProperty(fset=setter, fdel=deleter)

            def __getitem__(self, key):
                return self.data[key]

            def __setitem__(self, key, value):
                self.data[key] = value

            def __delitem__(self, key):
                del self.data[key]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        def fn(x):
            obj = Holder()
            obj.prop = x
            if delete:
                del obj.prop
            return obj.value

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    @parametrize("custom_get", (False, True))
    def test_accessor_shadow_does_not_change_descriptor_dispatch(self, custom_get):
        class CustomProperty(property):
            def __getattribute__(self, name):
                if name == "fget":
                    return len
                return super().__getattribute__(name)

        class OverrideProperty(CustomProperty):
            def __get__(self, obj, owner=None):
                return obj.value + 100

        cls = OverrideProperty if custom_get else CustomProperty

        class Holder:
            prop = cls(lambda obj: obj.value + 1)

        def fn(x):
            obj = Holder()
            obj.value = x
            return obj.prop

        x = torch.ones(2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    @parametrize("from_global", (False, True))
    @torch._dynamo.config.patch(enable_trace_load_build_class=True)
    def test_sourceless_owner_preserves_descriptor_guard(self, from_global):
        class CustomProperty(property):
            pass

        def first(obj):
            return obj.value + 1

        def second(obj):
            return obj.value + 100

        descriptor = CustomProperty(first)
        selected = global_property if from_global else descriptor
        self.addCleanup(selected.__init__, global_getter if from_global else first)

        def fn(x):
            class Holder:
                prop = global_property if from_global else descriptor

            obj = Holder()
            obj.value = x
            return obj.prop

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 1)
        selected.__init__(second)
        self.assertEqual(compiled(x), fn(x))
        self.assertEqual(compiled(x + 1), fn(x + 1))
        self.assertEqual(counter.frame_count, 2)


if __name__ == "__main__":
    run_tests()
