# Owner(s): ["module: dynamo"]

import functools

import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.guards import GuardBuilder
from torch._dynamo.source import AttrSource, LocalSource
from torch._dynamo.testing import CompileCounter
from torch._dynamo.variables.lazy import LazyConstantVariable
from torch._guards import Guard, GuardsContext, GuardSource, GuardsSet
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._ordered_set import OrderedSet


class GuardsSetTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_empty_guard_labels(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        for guards in (GuardsSet(), GuardsSet(OrderedSet([guard]))):
            self.assertEqual(guards.guard_labels, {})
            self.assertEqual(
                guards._get_guard_labels(torch._guards.GuardKey.from_guard(guard)),
                OrderedSet(),
            )
            self.assertEqual(guards.guard_labels, {})

    def test_existing_guard_set_operations(self):
        source = LocalSource("x")
        guard = Guard(source, GuardBuilder.TYPE_MATCH)
        attr_guard = Guard(AttrSource(source, "value"), GuardBuilder.ID_MATCH)
        other_guard = Guard(LocalSource("y"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        self.assertFalse(guards)
        guards.add(guard, collect_debug_stack=False)
        guards.add(guard, collect_debug_stack=False)
        guards.update(OrderedSet([attr_guard]), OrderedSet([other_guard]))
        self.assertEqual(list(guards), [guard, attr_guard, other_guard])
        self.assertEqual(len(guards), 3)
        self.assertTrue(guards)
        self.assertEqual(guards.get_guards_for_source(source), [guard])
        self.assertEqual(guards.guard_labels, {})
        for item in guards:
            self.assertEqual(
                guards._get_guard_labels(torch._guards.GuardKey.from_guard(item)),
                OrderedSet(),
            )
        self.assertEqual(guards.guard_labels, {})

        difference = guards - GuardsSet(OrderedSet([guard]))
        self.assertEqual(list(difference), [attr_guard, other_guard])
        self.assertEqual(difference.guard_labels, {})
        self.assertEqual(list(guards), [guard, attr_guard, other_guard])

        guards.remove_guards_with_source(source)
        self.assertEqual(list(guards), [other_guard])
        self.assertEqual(guards.get_guards_for_source(source), [])
        self.assertEqual(guards.guard_labels, {})
        guards.clear()
        self.assertFalse(guards)
        self.assertEqual(list(guards), [])
        self.assertEqual(
            guards.get_guards_for_source(other_guard.originating_source), []
        )
        self.assertEqual(guards.guard_labels, {})

    def test_guard_checkpoint_operations(self):
        context = GuardsContext()
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        context.dynamo_guards.add(guard, collect_debug_stack=False)
        checkpoint = context.copy_graphstate()
        context.dynamo_guards.clear()
        context.restore_graphstate(checkpoint)
        self.assertEqual(list(context.dynamo_guards), [guard])
        self.assertEqual(context.dynamo_guards.guard_labels, {})

    def test_guard_key_uses_hash_inputs(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        key = torch._guards.GuardKey.from_guard(guard)
        self.assertEqual(hash(key), hash(guard))
        equivalent = Guard(LocalSource("x", is_input=True), GuardBuilder.TYPE_MATCH)
        self.assertEqual(key, torch._guards.GuardKey.from_guard(equivalent))

        guard.guard_types = ["TYPE_MATCH"]
        guard.code_list = ["check_type_id(L['x'], 1)"]
        guard._unserializable = type(self)
        guard._force_dict_keys_match = True
        self.assertEqual(key, torch._guards.GuardKey.from_guard(guard))
        self.assertEqual(hash(key), hash(guard))
        self.assertNotEqual(
            key, torch._guards.GuardKey("L['y']", key.source, key.create_fn_id)
        )
        self.assertNotEqual(
            key, torch._guards.GuardKey(key.name, GuardSource.GLOBAL, key.create_fn_id)
        )

        first = functools.partial(GuardBuilder.TYPE_MATCH)
        second = functools.partial(GuardBuilder.TYPE_MATCH)
        self.assertNotEqual(
            torch._guards.GuardKey.from_guard(Guard(LocalSource("x"), first)),
            torch._guards.GuardKey.from_guard(Guard(LocalSource("x"), second)),
        )

    def test_guard_label_helpers(self):
        guards = GuardsSet()
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        key = torch._guards.GuardKey.from_guard(guard)
        missing_key = torch._guards.GuardKey.from_guard(
            Guard(LocalSource("y"), GuardBuilder.TYPE_MATCH)
        )
        guards._add_guard_labels(key, [])
        self.assertEqual(guards.guard_labels, {})
        guards._add_guard_labels(key, ["outer", "inner"])
        guards._add_guard_labels(key, ["inner", "other"])
        self.assertEqual(
            list(guards._get_guard_labels(key)), ["outer", "inner", "other"]
        )
        labels = guards._get_guard_labels(key)
        labels.add("unattached")
        self.assertNotIn("unattached", guards._get_guard_labels(key))
        self.assertEqual(guards._get_guard_labels(missing_key), OrderedSet())
        self.assertNotIn(missing_key, guards.guard_labels)
        guards._remove_guard_labels(missing_key)
        guards._remove_guard_labels(key)
        self.assertEqual(guards.guard_labels, {})
        guards._add_guard_labels(key, ["outer"])
        guards._clear_guard_labels()
        self.assertEqual(guards.guard_labels, {})

    def test_remove_preserves_labels_for_semantic_duplicates(self):
        source = LocalSource("x")
        guard = Guard(source, GuardBuilder.TYPE_MATCH)
        duplicate = Guard(source, GuardBuilder.TYPE_MATCH, code_list=["different"])
        guards = GuardsSet(OrderedSet([guard, duplicate]))
        key = torch._guards.GuardKey.from_guard(guard)
        guards._add_guard_labels(key, ["outer", "inner"])

        guards.remove(guard)
        self.assertEqual(list(guards), [duplicate])
        self.assertEqual(guards.get_guards_for_source(source), [duplicate])
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer", "inner"])
        guards.remove(guard)
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer", "inner"])
        guards.remove(duplicate)
        self.assertFalse(guards)
        self.assertEqual(guards.get_guards_for_source(source), [])
        self.assertEqual(guards.guard_labels, {})

    def test_clear_removes_labels_and_source_index(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet(OrderedSet([guard]))
        guards._add_guard_labels(torch._guards.GuardKey.from_guard(guard), ["outer"])
        guards.clear()
        self.assertFalse(guards)
        self.assertEqual(guards.guard_labels, {})
        self.assertEqual(guards.get_guards_for_source(guard.originating_source), [])

    def test_copy_preserves_labels_without_aliasing(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        key = torch._guards.GuardKey.from_guard(guard)
        inner = OrderedSet([guard])
        labels = {key: OrderedSet(["outer"])}
        guards = GuardsSet(inner, guard_labels=labels)
        inner.clear()
        labels[key].add("unattached")
        self.assertEqual(list(guards), [guard])
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer"])

        copied = guards.copy()
        copied._add_guard_labels(key, ["inner"])
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer"])
        self.assertEqual(list(copied._get_guard_labels(key)), ["outer", "inner"])
        guards.clear()
        self.assertEqual(list(copied), [guard])
        self.assertEqual(
            copied.get_guards_for_source(guard.originating_source), [guard]
        )

    def test_subtraction_preserves_surviving_semantic_labels(self):
        source = LocalSource("x")
        guard = Guard(source, GuardBuilder.TYPE_MATCH)
        duplicate = Guard(source, GuardBuilder.TYPE_MATCH, code_list=["different"])
        other = Guard(LocalSource("y"), GuardBuilder.ID_MATCH)
        guards = GuardsSet(OrderedSet([guard, duplicate, other]))
        key = torch._guards.GuardKey.from_guard(guard)
        other_key = torch._guards.GuardKey.from_guard(other)
        guards._add_guard_labels(key, ["outer"])
        guards._add_guard_labels(other_key, ["other"])

        difference = guards - GuardsSet(OrderedSet([guard, other]))
        self.assertEqual(list(difference), [duplicate])
        self.assertEqual(difference.get_guards_for_source(source), [duplicate])
        self.assertEqual(difference.guard_labels, {key: OrderedSet(["outer"])})
        difference._add_guard_labels(key, ["inner"])
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer"])
        self.assertEqual(list(guards), [guard, duplicate, other])

    def test_update_unions_labels_from_guard_sets(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        duplicate = Guard(guard.originating_source, GuardBuilder.TYPE_MATCH)
        other = Guard(LocalSource("y"), GuardBuilder.ID_MATCH)
        key = torch._guards.GuardKey.from_guard(guard)
        guards = GuardsSet(OrderedSet([guard]))
        incoming = GuardsSet(OrderedSet([duplicate, other]))
        guards._add_guard_labels(key, ["outer"])
        incoming._add_guard_labels(key, ["inner", "outer"])
        other_key = torch._guards.GuardKey.from_guard(other)
        incoming._add_guard_labels(other_key, ["other"])

        guards.update(incoming, (item for item in [guard]))
        self.assertEqual(list(guards), [guard, other])
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer", "inner"])
        self.assertEqual(list(guards._get_guard_labels(other_key)), ["other"])
        self.assertEqual(
            guards.get_guards_for_source(guard.originating_source), [guard]
        )
        guards.update(guards)
        incoming.clear()
        self.assertEqual(list(guards._get_guard_labels(key)), ["outer", "inner"])
        self.assertEqual(list(guards._get_guard_labels(other_key)), ["other"])

    def test_source_removal_preserves_retained_semantic_labels(self):
        source = LocalSource("x")
        other_source = LocalSource("x", is_input=True)
        guard = Guard(source, GuardBuilder.TYPE_MATCH)
        duplicate = Guard(other_source, GuardBuilder.TYPE_MATCH)
        attr_source = AttrSource(source, "value")
        attr_guard = Guard(attr_source, GuardBuilder.ID_MATCH)
        guards = GuardsSet(OrderedSet([guard, attr_guard, duplicate]))
        key = torch._guards.GuardKey.from_guard(guard)
        guards._add_guard_labels(key, ["outer"])
        guards._add_guard_labels(
            torch._guards.GuardKey.from_guard(attr_guard), ["inner"]
        )

        guards.remove_guards_with_source(source)
        self.assertEqual(list(guards), [duplicate])
        self.assertEqual(guards.get_guards_for_source(source), [])
        self.assertEqual(guards.get_guards_for_source(attr_source), [])
        self.assertEqual(guards.get_guards_for_source(other_source), [duplicate])
        self.assertEqual(guards.guard_labels, {key: OrderedSet(["outer"])})
        guards.remove_guards_with_source(other_source)
        self.assertFalse(guards)
        self.assertEqual(guards.guard_labels, {})

    def test_checkpoint_preserves_labels_without_aliasing(self):
        context = GuardsContext()
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        key = torch._guards.GuardKey.from_guard(guard)
        context.dynamo_guards.add(guard, collect_debug_stack=False)
        context.dynamo_guards._add_guard_labels(key, ["outer"])
        checkpoint = context.copy_graphstate()
        context.dynamo_guards._add_guard_labels(key, ["after_checkpoint"])
        context.dynamo_guards.clear()
        context.restore_graphstate(checkpoint)
        self.assertEqual(list(context.dynamo_guards), [guard])
        self.assertEqual(
            context.dynamo_guards.get_guards_for_source(guard.originating_source),
            [guard],
        )
        self.assertEqual(list(context.dynamo_guards._get_guard_labels(key)), ["outer"])
        context.dynamo_guards._add_guard_labels(key, ["after_restore"])
        context.dynamo_guards.remove(guard)
        context.restore_graphstate(checkpoint)
        self.assertEqual(list(context.dynamo_guards), [guard])
        self.assertEqual(list(context.dynamo_guards._get_guard_labels(key)), ["outer"])


class LazyGuardRemovalTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_realization_removes_type_guard_metadata(self):
        def check_realization(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            value = tx.symbolic_locals["value"]
            self.assertIs(type(value), LazyConstantVariable)
            guards = tx.output.guards
            source = value.source
            type_guard = value._type_guard_ref
            self.assertIsNotNone(type_guard)
            key = torch._guards.GuardKey.from_guard(type_guard)
            guards._add_guard_labels(key, ["synthetic"])
            value.realize()
            self.assertNotIn(type_guard, guards)
            self.assertNotIn(type_guard, guards.get_guards_for_source(source))
            self.assertNotIn(key, guards.guard_labels)
            self.assertTrue(
                any(
                    guard.create_fn is GuardBuilder.CONSTANT_MATCH
                    for guard in guards.get_guards_for_source(source)
                )
            )

        def fn(x, value):
            if type(value) is str:
                comptime(check_realization)
                return x + len(value)
            return x - 1

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for value in ("first", "second", 3):
            self.assertEqual(compiled(x, value), fn(x, value))
            frames = counter.frame_count
            self.assertEqual(compiled(x, value), fn(x, value))
            self.assertEqual(counter.frame_count, frames)


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
