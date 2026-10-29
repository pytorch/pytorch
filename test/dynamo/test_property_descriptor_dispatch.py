# Owner(s): ["module: dynamo"]

from collections.abc import MutableMapping

import torch
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch.overrides import TorchFunctionMode
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

    @parametrize("descriptor_type", (property, GlobalProperty))
    @parametrize("inherited", (False, True))
    def test_mode_property_source_and_replacement(self, descriptor_type, inherited):
        class BaseMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        class Mode(BaseMode):
            pass

        owner = BaseMode if inherited else Mode
        owner.offset = descriptor_type(lambda obj: 1)
        mode = Mode()

        def fn(x, mode):
            return x + mode.offset

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x, mode), fn(x, mode))
        self.assertEqual(compiled(x + 1, mode), fn(x + 1, mode))
        self.assertEqual(counter.frame_count, 1)
        Mode.offset = descriptor_type(lambda obj: 2)
        self.assertEqual(compiled(x, mode), fn(x, mode))
        self.assertEqual(compiled(x + 1, mode), fn(x + 1, mode))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("descriptor_type", (property, GlobalProperty))
    @parametrize("inherited", (False, True))
    @parametrize("delete", (False, True))
    def test_mode_property_mutation_replacement(
        self, descriptor_type, inherited, delete
    ):
        class BaseMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        class Mode(BaseMode):
            pass

        def first_setter(obj, value):
            obj.value = value + 1

        def second_setter(obj, value):
            obj.value = value + 100

        def first_deleter(obj):
            obj.value = obj.value + 1

        def second_deleter(obj):
            obj.value = obj.value + 100

        owner = BaseMode if inherited else Mode
        owner.offset = descriptor_type(fset=first_setter, fdel=first_deleter)
        mode = Mode()

        def fn(x, mode):
            if delete:
                mode.value = x
                del mode.offset
            else:
                mode.offset = x
            return mode.value

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x, mode), fn(x, mode))
        self.assertEqual(compiled(x + 1, mode), fn(x + 1, mode))
        self.assertEqual(counter.frame_count, 1)
        Mode.offset = descriptor_type(fset=second_setter, fdel=second_deleter)
        self.assertEqual(compiled(x, mode), fn(x, mode))
        self.assertEqual(compiled(x + 1, mode), fn(x + 1, mode))
        self.assertEqual(counter.frame_count, 2)

    def test_mode_property_descriptor_reconstruction(self):
        events = []

        class ReturningProperty(property):
            def __get__(self, obj, owner=None):
                if obj is not None:
                    events.append("get")
                return self

        class Mode(TorchFunctionMode):
            prop = ReturningProperty()

            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        mode = Mode()
        descriptor = Mode.__dict__["prop"]

        def fn(x, mode):
            return x + 1, mode.prop

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for value in (x, x + 1):
            result, prop = compiled(value, mode)
            self.assertEqual(result, value + 1)
            self.assertIs(prop, descriptor)
            self.assertEqual(events, ["get"])
            events.clear()
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    run_tests()
