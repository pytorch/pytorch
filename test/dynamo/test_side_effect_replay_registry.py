# Owner(s): ["module: dynamo"]

import enum
import types
import typing
from unittest.mock import Mock

from torch._dynamo import config
from torch._dynamo.exc import Unsupported
from torch._dynamo.side_effects import (
    _side_effect_replay_registry,
    _SideEffectReplayRegistry,
    SideEffectReplayContext,
    SideEffectReplayHandler,
    SideEffects,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.variables import (
    CellVariable,
    DeletedVariable,
    UserDefinedClassVariable,
)
from torch._dynamo.variables.base import (
    AttributeMutation,
    AttributeMutationNew,
    VariableTracker,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class SideEffectReplayRegistryTests(TestCase):
    @parametrize("kind", ("generic", "typing", "union", "slice", "enum", "typevar"))
    @config.patch(side_effect_replay_policy="silent")
    def test_materialized_cell_in_constant_wrapper(self, kind) -> None:
        class Owner:
            pass

        graph = Mock()
        side_effects = SideEffects(graph)
        cell = side_effects.track_cell_new(
            cell=types.CellType(Owner),
            contents=UserDefinedClassVariable(Owner),
            class_owner=Owner,
        )
        wrapped = {
            "generic": list[Owner],
            "typing": typing.__dict__["List"][Owner],
            "union": Owner | None,
            "slice": slice(None, Owner),
            "enum": enum.Enum("Wrapped", {"item": Owner}).item,
            "typevar": typing.TypeVar("Wrapped", bound=Owner),
        }[kind]
        if kind == "typing":
            self.assertEqual(type(wrapped).__module__, "typing")
        side_effects.store_cell(cell, DeletedVariable())
        with self.assertRaises(Unsupported):
            side_effects.check_materialized_class_escape(wrapped)

    @config.patch(side_effect_replay_policy="silent")
    def test_materialized_cell_clone_and_cleanup(self) -> None:
        graph = Mock()
        side_effects = SideEffects(graph)

        class Owner:
            pass

        initial = UserDefinedClassVariable(Owner)
        cell = types.CellType(Owner)
        tracked = side_effects.track_cell_new(
            cell=cell, contents=initial, class_owner=Owner
        )
        side_effects.check_materialized_class_escape(Owner, emitted=True)
        clone = side_effects.clone()
        self.assertIs(clone[cell], tracked)
        self.assertEqual(side_effects, clone)
        clone.store_cell(tracked, DeletedVariable())
        self.assertIsNone(
            tracked._current_contents(Mock(output=Mock(side_effects=clone)))
        )
        self.assertIs(side_effects.load_cell(tracked), initial)
        self.assertIsNotNone(side_effects.diff(clone))
        with self.assertRaises(Unsupported):
            clone.check_emitted_class_cells()
        clone.clear()
        self.assertNotIn(cell, clone)
        self.assertIn(cell, side_effects)

        prior = side_effects.clone()
        side_effects.before_opaque_class_body()
        with self.assertRaises(Unsupported):
            prior.store_cell(tracked, DeletedVariable())
        self.assertIs(cell.cell_contents, Owner)

    def test_materialized_scanner_does_not_call_container_hooks(self) -> None:
        class TrapTuple(tuple):
            __slots__ = ()

            def __iter__(self):
                raise AssertionError("unexpected user iterator")

            def __bool__(self):
                raise AssertionError("unexpected user truth test")

        class TrapDict(dict):
            def values(self):
                raise AssertionError("unexpected user values call")

            def __bool__(self):
                raise AssertionError("unexpected user truth test")

        class Owner:
            pass

        graph = Mock()
        side_effects = SideEffects(graph)
        side_effects.track_cell_new(
            cell=types.CellType(Owner),
            contents=UserDefinedClassVariable(Owner),
            class_owner=Owner,
        )
        captured = None

        def source(arg=None, *, option=None):
            return captured

        fn = types.FunctionType(
            source.__code__, globals(), closure=TrapTuple(source.__closure__)
        )
        fn.__defaults__ = TrapTuple((None,))
        fn.__kwdefaults__ = TrapDict(option=None)
        fn.__dict__ = TrapDict()
        side_effects.check_materialized_class_escape(fn, emitted=True)
        code = source.__code__.replace(co_consts=TrapTuple(source.__code__.co_consts))
        side_effects.check_materialized_class_escape(code, emitted=True)

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


instantiate_parametrized_tests(SideEffectReplayRegistryTests)


if __name__ == "__main__":
    run_tests()
