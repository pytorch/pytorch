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
        import torch._dynamo.source  # noqa: F401  register all root sources
        from torch._guards import ChainedSource, GuardProvenance, Source

        roots = []
        chained = []

        def walk(cls):
            for sub in cls.__subclasses__():
                # ChainedSource is itself a Source subclass, so it lands here
                # and is collected as chained (never as a root).
                if issubclass(sub, ChainedSource):
                    chained.append(sub)
                else:
                    roots.append(sub)
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

        # The walk sees a root class only once its defining module is imported,
        # so its coverage is exactly the modules imported by this test. This
        # assertion cannot see a root defined in an unimported module either; it
        # pins the convention that roots live in the two modules imported above
        # by rejecting any root the walk DOES find elsewhere, so a root added to
        # a module this test happens to import transitively fails here with the
        # placement rule instead of being silently checked or skipped.
        root_source_modules = {"torch._dynamo.source", "torch._guards"}
        for cls in roots:
            if not is_torch_own(cls):
                continue
            self.assertIn(
                cls.__module__,
                root_source_modules,
                f"{cls.__module__}.{cls.__name__} is a root Source defined "
                f"outside {sorted(root_source_modules)}; move it there so "
                "test_every_source_declares_provenance can see and enforce its "
                "classification.",
            )

        for cls in roots:
            if not is_torch_own(cls):
                continue
            # Require a direct declaration, not an inherited one: a root that
            # subclasses another root (e.g. class Foo(GlobalSource)) would
            # otherwise silently inherit the base's provenance even when it
            # should differ, so force every root to redeclare.
            self.assertIn(
                "_provenance",
                vars(cls),
                f"{cls.__module__}.{cls.__name__} must declare _provenance "
                "directly (see Note [Guard provenance] in torch/_guards.py)",
            )
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

    def test_guard_provenance_is_public(self):
        # The enum is re-exported from torch.compiler for guard_filter_fn
        # authors; the re-export is the same object and reports the public
        # module (test_public_bindings checks the latter, not the former).
        import torch.compiler
        from torch._guards import GuardProvenance

        self.assertIs(torch.compiler.GuardProvenance, GuardProvenance)
        self.assertEqual(torch.compiler.GuardProvenance.__module__, "torch.compiler")
        self.assertIn("GuardProvenance", torch.compiler.__all__)

    def test_provenance_classifies_by_root(self):
        from torch._dynamo.source import (
            BackwardStateSource,
            ContextVarGetSource,
            CurrentStreamSource,
            GlobalStateSource,
            GlobalWeakRefSource,
            ImportSource,
            LocalCellSource,
            NumpyTensorSource,
            ShapeEnvSource,
            TorchFunctionModeStackSource,
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
        # The SHAPE_ENV guard rooted here is derived from input shapes, so it is
        # dispatch-relevant (INPUT), not droppable environment state.
        self.assertEqual(ShapeEnvSource().provenance, GuardProvenance.INPUT)
        # BackwardState is a Dynamo-installed container, not ambient state.
        self.assertEqual(BackwardStateSource().provenance, GuardProvenance.SYNTHETIC)
        # Dynamo-installed weakref proxies stand in for traced-value identity;
        # an environment-drop policy must never see them as droppable GLOBAL.
        self.assertEqual(
            GlobalWeakRefSource("__optimizer_1").provenance, GuardProvenance.INPUT
        )
        # ImportSource is GLOBAL even though is_global() is False for it (that
        # predicate is true only for chains rooted at exactly GlobalSource).
        self.assertEqual(ImportSource("torch").provenance, GuardProvenance.GLOBAL)
        self.assertEqual(LocalCellSource("c").provenance, GuardProvenance.INPUT)
        self.assertEqual(
            CurrentStreamSource(torch.device("cpu")).provenance,
            GuardProvenance.AMBIENT,
        )
        self.assertEqual(
            TorchFunctionModeStackSource(0).provenance, GuardProvenance.AMBIENT
        )
        # Values read through a global at check time stay GLOBAL: the
        # environment-invariant contract must cover them (see the enum docs).
        self.assertEqual(
            ContextVarGetSource(GlobalSource("cv")).provenance,
            GuardProvenance.GLOBAL,
        )

    def test_unclassified_root_fails_closed(self):
        from torch._guards import Source

        class UnclassifiedSource(Source):
            pass

        with self.assertRaisesRegex(NotImplementedError, "Guard provenance"):
            _ = UnclassifiedSource().provenance

    @staticmethod
    def _collecting_filter():
        # Returns (seen, filter_fn): filter_fn keeps every guard and records the
        # entries it was handed, so a test can inspect their provenance.
        seen = []

        def filter_fn(entries):
            seen.extend(entries)
            return [True] * len(entries)

        return seen, filter_fn

    def test_guard_filter_entries_carry_provenance(self):
        # End to end: a compiled fn's guard filter sees a typed provenance on
        # every entry, and the tensor-argument guard is classified INPUT.
        from torch._guards import GuardProvenance

        seen, filter_fn = self._collecting_filter()

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
        # Compiling an optimizer step installs weakref proxies for the
        # optimizer object and its params (G['__optimizer_...']) and guards
        # their liveness with WEAKREF_ALIVE; those guards distinguish
        # optimizer instances, so they must classify INPUT, not as droppable
        # environment guards.
        from torch._guards import GuardProvenance

        seen, filter_fn = self._collecting_filter()

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


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
