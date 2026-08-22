# Owner(s): ["module: dynamo"]

from unittest.mock import Mock

from torch._dynamo.side_effects import (
    _side_effect_replay_registry,
    _SideEffectReplayRegistry,
    SideEffectReplayContext,
    SideEffectReplayHandler,
    SideEffects,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.variables import CellVariable
from torch._dynamo.variables.base import (
    AttributeMutation,
    AttributeMutationNew,
    VariableTracker,
)


class SideEffectReplayRegistryTests(TestCase):
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
                ("deque_mutation", 80),
                ("const_dict_or_set_mutation", 70),
                ("torch_function_mode_stack_mutation", 60),
                ("cell_mutation", 50),
                ("attribute_mutation", 40),
                ("list_iterator_mutation", 30),
                ("count_iterator_mutation", 20),
                ("random_mutation", 10),
                ("contextvar_mutation", 5),
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
