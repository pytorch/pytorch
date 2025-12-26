# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.nn as nn
from torch._dynamo.source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
    NegateSource,
    TensorPropertySource,
    TensorProperty,
    TypeSource,
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

    def test_source_name_is_cached(self):
        """Verify that .name property is cached as an instance attribute."""
        # Test base Source subclass (LocalSource)
        local_src = LocalSource("x")
        name1 = local_src.name
        name2 = local_src.name
        # Verify it's cached in __dict__
        self.assertIn("name", local_src.__dict__)
        # Verify subsequent accesses return the same cached string object
        self.assertIs(name1, name2)
        self.assertEqual(name1, "L['x']")

        # Test ChainedSource subclass with @functools.cached_property _name_template (AttrSource)
        attr_src = AttrSource(local_src, "foo")
        attr_name1 = attr_src.name
        attr_name2 = attr_src.name
        self.assertIn("name", attr_src.__dict__)
        self.assertIs(attr_name1, attr_name2)
        self.assertEqual(attr_name1, "L['x'].foo")

        # Test ChainedSource subclass with @property _name_template (TypeSource)
        type_src = TypeSource(base=local_src)
        type_name1 = type_src.name
        type_name2 = type_src.name
        self.assertIn("name", type_src.__dict__)
        self.assertIs(type_name1, type_name2)
        self.assertEqual(type_name1, "type(L['x'])")

        # Test deeply nested ChainedSource
        nested_src = GetItemSource(AttrSource(AttrSource(local_src, "a"), "b"), 0)
        nested_name1 = nested_src.name
        nested_name2 = nested_src.name
        self.assertIn("name", nested_src.__dict__)
        self.assertIs(nested_name1, nested_name2)
        self.assertEqual(nested_name1, "L['x'].a.b[0]")

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
