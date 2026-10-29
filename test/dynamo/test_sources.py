# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn as nn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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

    @parametrize("subclass", (False, True))
    def test_property_accessor_guard_and_reconstruction(self, subclass):
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
        prop = Property(first) if subclass else property(first)

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

    @parametrize("subclass", (False, True))
    @parametrize("member", ("fget", "fset", "fdel"))
    def test_property_accessor_guard_structure(self, subclass, member):
        class Property(property):
            def __getattribute__(self, name):
                if name in ("fget", "fset", "fdel"):
                    raise AssertionError("Public accessor lookup must be bypassed")
                return super().__getattribute__(name)

        def first(x):
            return x + 1

        def second(x):
            return x + 2

        cls = Property if subclass else property
        prop = cls(**{member: first})

        def fn(x, p):
            return property.__dict__[member].__get__(p)(x)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x, prop), x + 1)
        self.assertEqual(compiled(x, prop), x + 1)
        self.assertEqual(counter.frame_count, 1)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 1)
        guard_tree = str(entries[0].guard_manager)
        if subclass:
            self.assertIn(
                f"source=___get_property_accessor(L['p'], {member!r}), accessed_by=PythonLambdaGuardAccessor",
                guard_tree,
            )
        else:
            self.assertIn(
                f"source=L['p'].{member}, accessed_by=GetAttrGuardAccessor({member})",
                guard_tree,
            )
            self.assertNotIn("___get_property_accessor", guard_tree)
        property.__init__(prop, **{member: second})
        self.assertEqual(compiled(x, prop), x + 2)
        self.assertEqual(compiled(x, prop), x + 2)
        self.assertEqual(counter.frame_count, 2)

    def test_constructed_object_class_attribute_source(self):
        class Holder:
            offset = 1

        def fn(x):
            return x + Holder().offset

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        self.assertEqual(compiled(x), x + 1)
        self.assertEqual(compiled(x + 1), x + 2)
        self.assertEqual(counter.frame_count, 1)
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(entries), 1)
        self.assertIn(
            "source=L['Holder'].offset, accessed_by=GetAttrGuardAccessor(offset)",
            str(entries[0].guard_manager),
        )
        Holder.offset = 2
        self.assertEqual(compiled(x), x + 2)
        self.assertEqual(compiled(x + 1), x + 3)
        self.assertEqual(counter.frame_count, 2)

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


instantiate_parametrized_tests(SourceTests)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
