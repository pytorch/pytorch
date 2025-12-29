# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch.nn as nn
from torch._dynamo.source import (
    _esc_str,
    _esc_str_cached,
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)


class CausalLMOutputWithPast:
    value = 5


class SourceTests(torch._dynamo.test_case.TestCase):
    def test_esc_str_caching(self):
        _esc_str_cached.cache_clear()

        s = "test{value}"
        self.assertEqual(_esc_str(s), "test{{value}}")
        self.assertEqual(_esc_str(s), "test{{value}}")
        self.assertIs(_esc_str_cached(s), _esc_str_cached(s))
        self.assertGreater(_esc_str_cached.cache_info().hits, 0)

        self.assertEqual(_esc_str(s, apply_repr=True), "'test{{value}}'")

        self.assertEqual(_esc_str("{}"), "{{}}")
        self.assertEqual(_esc_str("{a}{b}"), "{{a}}{{b}}")
        self.assertEqual(_esc_str("no braces"), "no braces")

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
