# Owner(s): ["module: dynamo"]

from types import SimpleNamespace
from unittest.mock import Mock, patch

from torch._dynamo.side_effects import (
    _side_effect_replay_registry,
    _SideEffectReplayRegistry,
    SideEffectReplayContext,
    SideEffectReplayHandler,
    SideEffects,
)
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.variables import (
    CellVariable,
    ConstantVariable,
    DeletedVariable,
    UserDefinedObjectVariable,
)
from torch._dynamo.variables.base import (
    AttributeMutation,
    AttributeMutationNew,
    AttrMutationKind,
    VariableTracker,
)
from torch._dynamo.variables.dicts import SideEffectsProxyDict


class SideEffectReplayRegistryTests(TestCase):
    def test_attribute_mutation_overlaps_clone(self) -> None:
        class Object:
            __slots__ = ("name", "__dict__")

        output = Mock()
        side_effects = SideEffects(output)
        obj = Object()
        var = UserDefinedObjectVariable(obj)
        side_effects.track_object_existing(obj, var)
        side_effects.store_attr(var, "name", ConstantVariable.create("slot"))
        side_effects.store_instance_dict_attr(
            var, "name", ConstantVariable.create("dict")
        )
        saved = side_effects.clone()
        side_effects.store_attr(var, "name", ConstantVariable.create("changed slot"))
        side_effects.store_instance_dict_attr(
            var, "name", ConstantVariable.create("changed dict")
        )
        self.assertEqual(
            saved.load_attr(
                var, "name", mutation_kind=AttrMutationKind.GENERIC_SETATTR
            ).as_python_constant(),
            "slot",
        )
        self.assertEqual(
            saved.load_attr(
                var, "name", mutation_kind=AttrMutationKind.INSTANCE_DICT
            ).as_python_constant(),
            "dict",
        )
        self.assertNotEqual(side_effects, saved)
        self.assertEqual(saved, saved.clone())

    def test_instance_dict_membership_ignores_slot_mutations(self) -> None:
        class Object:
            __slots__ = ("name", "__dict__")

        output = Mock()
        side_effects = output.side_effects = SideEffects(output)
        tx = Mock(output=output)
        obj = Object()
        var = UserDefinedObjectVariable(obj)
        side_effects.track_object_existing(obj, var)
        side_effects.store_instance_dict_attr(
            var, "name", ConstantVariable.create("dict")
        )
        side_effects.store_attr(var, "name", DeletedVariable())
        self.assertTrue(var.has_key_in_generic_dict(tx, "name"))
        side_effects.store_instance_dict_attr(var, "name", DeletedVariable())
        side_effects.store_attr(var, "name", ConstantVariable.create("slot"))
        self.assertFalse(var.has_key_in_generic_dict(tx, "name"))

    def test_instance_dict_deletion_restores_with_side_effects(self) -> None:
        output = Mock()
        output.side_effects = SideEffects(output)
        tx = Mock(output=output)
        obj = SimpleNamespace(a=1, b=2)
        var = UserDefinedObjectVariable(obj)
        output.side_effects.track_object_existing(obj, var)
        proxy = SideEffectsProxyDict(var, tx)
        saved = output.side_effects.clone()
        del proxy["a"]
        proxy["a"] = ConstantVariable.create(3)
        self.assertNotEqual(output.side_effects, saved)
        self.assertIsNotNone(output.side_effects.diff(saved))
        output.side_effects = saved
        with patch.object(InstructionTranslator, "current_tx", return_value=tx):
            self.assertEqual([key.vt.as_python_constant() for key in proxy], ["a", "b"])
        self.assertEqual(proxy["a"].as_python_constant(), 1)
        self.assertEqual(saved, saved.clone())

    @staticmethod
    def make_context(var: VariableTracker | None = None) -> SideEffectReplayContext:
        side_effects = Mock(spec=SideEffects)
        side_effects.is_attribute_mutation.side_effect = lambda item: isinstance(
            item.mutation_type, AttributeMutation
        )
        return SideEffectReplayContext(
            side_effects=side_effects,
            codegen=Mock(),
            var=var if var is not None else VariableTracker(),
            suffixes=[],
            log=Mock(),
        )

    def test_priority_wins_independent_of_registration_order(self) -> None:
        registry = _SideEffectReplayRegistry()
        ctx = self.make_context()
        lower = SideEffectReplayHandler("lower", lambda _: True, Mock(), priority=1)
        higher = SideEffectReplayHandler("higher", lambda _: True, Mock(), priority=2)

        registry.register_handler(lower)
        higher_handle = registry.register_handler(higher)
        self.assertIs(registry.lookup_handler(ctx), higher)

        higher_handle.destroy()
        self.assertIs(registry.lookup_handler(ctx), lower)

    def test_equal_priority_matches_are_ambiguous(self) -> None:
        registry = _SideEffectReplayRegistry()
        ctx = self.make_context()
        registry.register_handler(
            SideEffectReplayHandler("first", lambda _: True, Mock(), priority=1)
        )
        registry.register_handler(
            SideEffectReplayHandler("second", lambda _: True, Mock(), priority=1)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Ambiguous side effect replay handler for VariableTracker: "
            r"\['first', 'second'\]",
        ):
            registry.lookup_handler(ctx)

    def test_builtin_priorities_preserve_legacy_dispatch_order(self) -> None:
        self.assertEqual(
            [
                (handler.name, handler.priority)
                for handler in _side_effect_replay_registry.handlers
            ],
            [
                ("list_mutation", 90),
                ("bytearray_mutation", 85),
                ("deque_mutation", 80),
                ("const_dict_or_set_mutation", 70),
                ("torch_function_mode_stack_mutation", 60),
                ("cell_mutation", 50),
                ("attribute_mutation", 40),
                ("list_iterator_mutation", 30),
                ("count_iterator_mutation", 20),
                ("random_mutation", 10),
            ],
        )

    def test_cell_handler_wins_attribute_mutation_overlap(self) -> None:
        cell = CellVariable(mutation_type=AttributeMutationNew())
        cell.local_name = "cell"

        handler = _side_effect_replay_registry.lookup_handler(self.make_context(cell))

        self.assertIsNotNone(handler)
        self.assertEqual(handler.name, "cell_mutation")


if __name__ == "__main__":
    run_tests()
