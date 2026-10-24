# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn as nn
from torch._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)


class CausalLMOutputWithPast:
    value = 5


class SourceTests(torch._dynamo.test_case.TestCase):
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

    def test_property_accessor_guard_and_reconstruction(self):
        accesses = []

        class Property(property):
            fget = None

            def __getattribute__(self, name):
                if name == "fget":
                    accesses.append(name)
                    return None
                return super().__getattribute__(name)

        class Getter:
            def __init__(self, offset):
                self.offset = offset

            def __call__(self, value):
                return value + self.offset, self

        first = Getter(1)
        second = Getter(2)
        prop = Property(first)

        def fn(x):
            return prop.__get__(x)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for _ in range(2):
            result, accessor = compiled(x)
            self.assertEqual(result, x + 1)
            self.assertIs(accessor, first)
        self.assertEqual(counter.frame_count, 1)
        property.__init__(prop, second)
        for _ in range(2):
            result, accessor = compiled(x)
            self.assertEqual(result, x + 2)
            self.assertIs(accessor, second)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(accesses, [])

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
