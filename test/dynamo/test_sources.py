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
        chained = []

        def walk(cls):
            for sub in cls.__subclasses__():
                if not issubclass(sub, ChainedSource):
                    roots.append(sub)
                elif sub is not ChainedSource:
                    chained.append(sub)
                walk(sub)

        walk(Source)

        def is_torch_own(cls):
            # Test fixtures (e.g. the fail-closed test below) linger in
            # __subclasses__ until cyclic GC runs, so a same-process test
            # ordering must not flip this walk's verdict; only torch's own
            # sources are enforced. Match "torch" and "torch.*" exactly, not
            # torch*-prefixed third-party packages (torchvision, torch_xla).
            mod = cls.__module__
            return mod == "torch" or mod.startswith("torch.")

        self.assertGreater(len(roots), 10)  # sanity: the walk found the roots
        for cls in roots:
            if not is_torch_own(cls):
                continue
            self.assertIsInstance(
                cls._provenance,
                GuardProvenance,
                f"{cls.__module__}.{cls.__name__} must declare _provenance "
                "(see Note [Guard provenance] in torch/_guards.py)",
            )
        for cls in chained:
            if not is_torch_own(cls):
                continue
            # ChainedSource.provenance delegates to the root unconditionally,
            # so a _provenance declared on a chained class is silently ignored;
            # reject the ambiguous declaration instead.
            self.assertNotIn(
                "_provenance",
                vars(cls),
                f"{cls.__module__}.{cls.__name__} is a ChainedSource and must "
                "not declare _provenance; classification always follows the "
                "root source (see Note [Guard provenance] in torch/_guards.py)",
            )

    def test_provenance_classifies_by_root(self):
        from torch._dynamo.source import (
            ConstDictKeySource,
            DictGetItemSource,
            GetItemSource,
            GlobalStateSource,
            GlobalWeakRefSource,
            ImportSource,
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
        # Deeper chains and dict-key indices still classify by the root.
        self.assertEqual(
            GetItemSource(AttrSource(local, "a"), 0).provenance, GuardProvenance.INPUT
        )
        self.assertEqual(
            DictGetItemSource(local, ConstDictKeySource(local, 0)).provenance,
            GuardProvenance.INPUT,
        )
        self.assertEqual(
            AttrSource(GlobalSource("g"), "attr").provenance,
            GuardProvenance.GLOBAL,
        )
        self.assertEqual(
            AttrSource(ImportSource("torch"), "utils").provenance,
            GuardProvenance.GLOBAL,
        )
        self.assertEqual(GlobalStateSource().provenance, GuardProvenance.AMBIENT)
        # SHAPE_ENV is the dynamic-shape dispatch guard over input sizes.
        self.assertEqual(ShapeEnvSource().provenance, GuardProvenance.INPUT)
        # A liveness guard on a compile-time-bound object the compiled bytecode
        # dereferences at runtime: kept conservatively, never droppable GLOBAL.
        self.assertEqual(
            GlobalWeakRefSource("__optimizer_1").provenance, GuardProvenance.INPUT
        )

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

    def test_optimizer_weakref_guards_classify_as_input(self):
        # Compiling an optimizer step installs weakrefs to its params
        # (G['__optimizer_...']) that the compiled bytecode dereferences, and
        # guards their liveness with WEAKREF_ALIVE. Dropping that guard as an
        # environment guard would run the code against a dead weakref, so it
        # must classify INPUT.
        from torch._guards import GuardProvenance

        seen = []

        def filter_fn(entries):
            seen.extend(entries)
            return [True] * len(entries)

        p = torch.nn.Parameter(torch.randn(4))
        opt = torch.optim.Adam([p])
        p.grad = torch.randn(4)

        @torch.compile(backend="eager", options={"guard_filter_fn": filter_fn})
        def step(o):
            o.step()

        step(opt)
        weakref_entries = [e for e in seen if e.guard_type == "WEAKREF_ALIVE"]
        self.assertTrue(weakref_entries)
        for entry in weakref_entries:
            self.assertEqual(entry.provenance, GuardProvenance.INPUT)

    def test_input_only_filter_keeps_shape_env_guard(self):
        # A filter that keeps only INPUT guards (the environment-drop contract
        # of the Note) must keep SHAPE_ENV: it is the dynamic-shape dispatch
        # guard, and without it a size-2 call is served the size-5 branch.
        from torch._guards import GuardProvenance

        kept_types = []

        def keep_input_guards(entries):
            keep = [e.provenance is GuardProvenance.INPUT for e in entries]
            kept_types.extend(e.guard_type for e, k in zip(entries, keep) if k)
            return keep

        opts = {"guard_filter_fn": keep_input_guards}

        @torch.compile(backend="eager", dynamic=True, options=opts)
        def fn(x):
            return x + 1 if x.size(0) > 3 else x - 1

        self.assertEqual(fn(torch.zeros(5)), torch.ones(5))
        self.assertIn("SHAPE_ENV", kept_types)
        self.assertEqual(fn(torch.zeros(2)), -torch.ones(2))

    def test_guard_filter_entry_provenance_is_lazy(self):
        # The base seven-field positional constructor still works, and
        # provenance is derived from orig_guard on access, so an unclassified
        # out-of-tree Source only fails the filter that asks for it.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.types import GuardFilterEntry
        from torch._guards import Guard, GuardProvenance, Source

        class UnclassifiedSource(Source):
            pass

        guard = Guard(UnclassifiedSource(), GuardBuilder.TYPE_MATCH)
        entry = GuardFilterEntry("x", False, None, "TYPE_MATCH", (), False, guard)
        self.assertIs(entry.orig_guard, guard)
        with self.assertRaisesRegex(NotImplementedError, "Guard provenance"):
            _ = entry.provenance

        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        entry = GuardFilterEntry("x", False, None, "TYPE_MATCH", (), False, guard)
        self.assertIs(entry.provenance, entry.orig_guard.provenance)
        self.assertIs(entry.provenance, GuardProvenance.INPUT)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
