# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo
import torch._dynamo.test_case
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


class GuardProvenanceTests(torch._dynamo.test_case.TestCase):
    # See Note [Guard provenance] in torch/_guards.py.

    def test_every_source_declares_provenance(self):
        # Totality: every ROOT Source class must declare _provenance, so a new
        # source cannot land unclassified and silently default in a consumer.
        # ChainedSources delegate to their root and are exempt.
        from torch._guards import ChainedSource, GuardProvenance, Source

        roots = []

        def walk(cls):
            for sub in cls.__subclasses__():
                if not issubclass(sub, ChainedSource) and sub is not ChainedSource:
                    roots.append(sub)
                walk(sub)

        walk(Source)
        self.assertGreater(len(roots), 10)  # sanity: the walk found the roots
        for cls in roots:
            self.assertIsInstance(
                cls._provenance,
                GuardProvenance,
                f"{cls.__module__}.{cls.__name__} must declare _provenance "
                "(see Note [Guard provenance] in torch/_guards.py)",
            )

    def test_provenance_classifies_by_root(self):
        from torch._dynamo.source import (
            GlobalStateSource,
            NumpyTensorSource,
            ShapeEnvSource,
            TupleIteratorGetItemSource,
            TypeSource,
        )
        from torch._guards import GuardProvenance

        local = LocalSource("x")
        # Call-wrapped input roots are exactly the shapes a rendered-name
        # prefix test misfiles as environment guards.
        self.assertEqual(
            TupleIteratorGetItemSource(local, index=0).provenance,
            GuardProvenance.INPUT,
        )
        self.assertEqual(NumpyTensorSource(local).provenance, GuardProvenance.INPUT)
        self.assertEqual(TypeSource(local).provenance, GuardProvenance.INPUT)
        self.assertEqual(
            AttrSource(GlobalSource("g"), "attr").provenance,
            GuardProvenance.GLOBAL,
        )
        self.assertEqual(GlobalStateSource().provenance, GuardProvenance.AMBIENT)
        self.assertEqual(ShapeEnvSource().provenance, GuardProvenance.AMBIENT)

    def test_unclassified_root_fails_closed(self):
        from torch._guards import Source

        class UnclassifiedSource(Source):
            pass

        with self.assertRaisesRegex(NotImplementedError, "Guard provenance"):
            _ = UnclassifiedSource().provenance

    def test_guard_filter_entries_carry_provenance(self):
        # End to end: a compiled fn's guard filter sees a typed provenance on
        # every entry, and the tensor-argument guard is classified INPUT.
        from torch._guards import GuardProvenance

        seen = []

        def filter_fn(entries):
            seen.extend(entries)
            return [True] * len(entries)

        @torch.compile(backend="eager", options={"guard_filter_fn": filter_fn})
        def fn(x):
            return x * 2

        fn(torch.randn(3))
        self.assertTrue(seen)
        for entry in seen:
            self.assertIsInstance(entry.provenance, GuardProvenance)
        input_entries = [e for e in seen if e.name == "x"]
        self.assertTrue(input_entries)
        self.assertEqual(input_entries[0].provenance, GuardProvenance.INPUT)
        ambient = [e for e in seen if e.guard_type == "GRAD_MODE"]
        self.assertTrue(ambient)
        self.assertEqual(ambient[0].provenance, GuardProvenance.AMBIENT)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
