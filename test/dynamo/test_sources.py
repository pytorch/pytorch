# Owner(s): ["module: dynamo"]

from unittest import mock

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.nn as nn
from torch._dynamo.source import (
    AttrSource,
    DictKeysSource,
    FrozenDictItemsSource,
    GetItemSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
    TypeDictSource,
)
from torch._dynamo.testing import CompileCounter


class CausalLMOutputWithPast:
    value = 5


class SourceTests(torch._dynamo.test_case.TestCase):
    def test_constant_source_property_setter_falls_back(self):
        class Box:
            def __init__(self):
                self._value = 2

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, value):
                self._value = value

        box = Box()

        @torch._dynamo.assume_constant_result
        def get_box():
            return box

        def fn(x):
            target = get_box()
            target.value = 3
            return x + target.value

        x = torch.ones(1)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Descriptor mutation on a constant-source object",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(box.value, 2)
        self.assertEqual(torch.compile(fn, backend="eager")(x), x + 3)
        self.assertEqual(box.value, 3)

    def test_property_setter_guards_replacement(self):
        class Box:
            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, value):
                self._value = value

        def fn(x, box):
            box.value = 3
            return x + box.value

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x, box = torch.ones(1), Box()
        self.assertEqual(compiled(x, box), x + 3)
        self.assertEqual(box.value, 3)
        self.assertEqual(counter.frame_count, 1)

        def replacement(self, value):
            self._value = value + 4

        Box.value = Box.value.setter(replacement)
        self.assertEqual(compiled(x, box), x + 7)
        self.assertEqual(box.value, 7)
        self.assertEqual(counter.frame_count, 2)

    def test_dict_keys_snapshot_cannot_create_guards(self):
        source = DictKeysSource(LocalSource("mapping"))
        with self.assertRaisesRegex(NotImplementedError, "codegen-only"):
            source.make_guard(lambda _: None)

    def test_type_dict_source_bypasses_metaclass_getattribute(self):
        class Meta(type):
            def __getattribute__(cls, name):
                if name == "__dict__":
                    raise AssertionError("type dictionary lookup must bypass metaclass")
                return type.__getattribute__(cls, name)

        class Key(metaclass=Meta):
            value = object()

        source = TypeDictSource(LocalSource("cls"))
        cache = {}
        scope = {"L": {"cls": Key}}
        namespace = source.get_value(scope, {}, cache)
        self.assertIs(type(namespace), dict)
        self.assertIs(namespace["value"], Key.value)
        self.assertIs(source.get_value(scope, {}, cache), namespace)

    def test_frozendict_items_source(self):
        mapping, key, value = object(), object(), torch.randn(2)
        source = FrozenDictItemsSource(LocalSource("mapping"))
        globals = {"L": {"mapping": mapping}}
        cache = {}
        # Exercise source resolution on Python versions without frozendict.
        with mock.patch(
            "torch._dynamo.source.utils.frozendict_items", return_value=[(key, value)]
        ):
            item_source = GetItemSource(source, 0)
            self.assertIs(
                GetItemSource(item_source, 0).get_value(globals, {}, cache), key
            )
            self.assertIs(
                GetItemSource(item_source, 1).get_value(globals, {}, cache), value
            )
            self.assertIs(source.get_value(globals, {}, cache), cache[source])

    def test_is_local(self):
        x_src = LocalSource("x")
        y_src = GlobalSource("y")

        attr_x_a = AttrSource(x_src, "a")
        attr_y_b = AttrSource(y_src, "b")

        self.assertTrue(is_from_local_source(attr_x_a))
        self.assertEqual(is_from_local_source(attr_y_b), False)

    def test_property_closure(self):
        def external_property():
            closed_value = 7

            def internal_function(self):
                return closed_value

            return internal_function

        class Elements:
            myprop = property(external_property())

        def func(elements):
            if not elements.myprop:
                return torch.tensor([1, 2, 3])
            else:
                return torch.tensor([4, 5, 6])

        e = Elements()
        a = func(e)
        b = torch.compile(func, backend="eager", fullgraph=True)(e)
        self.assertEqual(a, b)

    def test_supported_nodes(self):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.randn(10, 10)

            def forward(self):
                if (
                    torch.utils._pytree.SUPPORTED_NODES[CausalLMOutputWithPast].type
                    is int
                ):
                    x = torch.sin(self.x)
                else:
                    x = torch.cos(self.x)
                return x

        torch.utils._pytree.register_pytree_node(
            CausalLMOutputWithPast,
            lambda x: ((), None),
            lambda x, _: CausalLMOutputWithPast(),
        )

        torch.export.export(Model(), (), strict=True)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
