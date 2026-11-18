# Owner(s): ["module: dynamo"]

import dataclasses
import functools
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import Mock, patch

import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.guards import GuardBuilder, install_guard
from torch._dynamo.source import (
    AttrSource,
    ConstantSource,
    EphemeralSource,
    LocalSource,
    SkipGuardSource,
)
from torch._dynamo.testing import CompileCounter
from torch._dynamo.variables.ctx_manager import GradModeVariable
from torch._dynamo.variables.lazy import LazyConstantVariable
from torch._guards import (
    Guard,
    GuardsContext,
    GuardSource,
    GuardsSet,
    tracing,
    TracingContext,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet


class GuardsSetTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_add_labels_before_duplicate_return(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="one", collect_debug_stack=False)
        guards.add(guard, label="two", collect_debug_stack=False)
        # Existing per-build fields still distinguish objects in the set.
        variant = dataclasses.replace(guard, code_list=["different debug info"])
        guards.add(variant, label="three", collect_debug_stack=False)
        self.assertEqual(len(guards), 2)
        key = torch._guards.GuardKey.from_guard(guard)
        self.assertEqual(list(guards._get_guard_labels(key)), ["one", "two", "three"])
        guards.remove(guard)
        self.assertEqual(list(guards._get_guard_labels(key)), ["one", "two", "three"])
        guards.remove(variant)
        self.assertEqual(guards.guard_labels, {})

    def test_creation_label_preserves_guard_and_source_semantics(self):
        source = AttrSource(LocalSource("x"), "value")
        context = TracingContext(None)
        with tracing(context):
            with context.guards_context.guard_label("one"):
                guard = source.make_guard(GuardBuilder.TYPE_MATCH)
            with context.guards_context.guard_label("two"):
                duplicate = source.make_guard(GuardBuilder.TYPE_MATCH)
        self.assertEqual(guard.creation_label, "one")
        self.assertEqual(duplicate.creation_label, "two")
        self.assertEqual(guard, duplicate)
        self.assertEqual(hash(guard), hash(duplicate))
        self.assertEqual(str(guard), str(duplicate))
        self.assertEqual(repr(guard), repr(duplicate))
        self.assertEqual(source, AttrSource(LocalSource("x"), "value"))
        self.assertEqual(hash(source), hash(AttrSource(LocalSource("x"), "value")))
        value = object()
        obj = type("Input", (), {"value": value})()
        self.assertIs(source.get_value({}, {"L": {"x": obj}}, {}), value)
        with self.assertRaises(AttributeError):
            source.get_value({}, {"L": {"x": object()}}, {})
        first_codegen, second_codegen = Mock(), Mock()
        first_codegen.create_load_attrs.return_value = ("LOAD_ATTR", "value")
        second_codegen.create_load_attrs.return_value = ("LOAD_ATTR", "value")
        with tracing(context):
            with context.guards_context.guard_label("one"):
                source.reconstruct(first_codegen)
            with context.guards_context.guard_label("two"):
                source.reconstruct(second_codegen)
        self.assertEqual(first_codegen.mock_calls, second_codegen.mock_calls)
        for unguardable in (ConstantSource("constant"), EphemeralSource()):
            with self.assertRaises(NotImplementedError):
                unguardable.make_guard(GuardBuilder.TYPE_MATCH)

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


@instantiate_parametrized_tests
class GuardContextTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_skipped_install_does_not_attach_labels(self):
        context = TracingContext(None)
        guards_context = context.guards_context
        with tracing(context), guards_context.guard_label("one"):
            skipped = SkipGuardSource(LocalSource("x")).make_guard(
                GuardBuilder.TYPE_MATCH
            )
            self.assertEqual(skipped.creation_label, "one")
            install_guard(skipped)
            with guards_context.skip_guard_install():
                install_guard(LocalSource("x").make_guard(GuardBuilder.TYPE_MATCH))
        self.assertFalse(guards_context.dynamo_guards)
        self.assertEqual(guards_context.dynamo_guards.guard_labels, {})

    def test_creation_label_and_install_fallback(self):
        context = TracingContext(None)
        guards_context = context.guards_context
        source = LocalSource("x")
        direct = Guard(source, GuardBuilder.GRAD_MODE)
        singleton = GradModeVariable._guards_singleton
        unlabeled = source.make_guard(GuardBuilder.TYPE_MATCH)
        with torch.compiler.guard_ctx("pending"):
            # Only the active trace is authoritative, even when it masks pending.
            with tracing(context):
                self.assertIsNone(guards_context.current_guard_label)
                self.assertIsNone(
                    source.make_guard(GuardBuilder.TYPE_MATCH).creation_label
                )
                install_guard(unlabeled)
                self.assertEqual(guards_context.dynamo_guards.guard_labels, {})
                with guards_context.guard_label("one"):
                    first = source.make_guard(GuardBuilder.CONSTANT_MATCH)
                    with guards_context.guard_label("inner"):
                        inner = source.make_guard(GuardBuilder.ID_MATCH)
                with guards_context.guard_label("two"):
                    install_guard(first, inner, direct, unlabeled, singleton)
                with guards_context.guard_label("three"):
                    install_guard(direct, singleton)
                    second = source.make_guard(GuardBuilder.CONSTANT_MATCH)
                    install_guard(second)
                nested = TracingContext(None)
                with tracing(nested):
                    nested_guard = source.make_guard(GuardBuilder.TYPE_MATCH)
                    install_guard(nested_guard)
                    self.assertIsNone(nested_guard.creation_label)
                    self.assertEqual(
                        nested.guards_context.dynamo_guards.guard_labels, {}
                    )
        guards = guards_context.dynamo_guards
        for guard, labels in (
            (first, ["one", "three"]),
            (inner, ["inner"]),
            (direct, ["two", "three"]),
            (singleton, ["two", "three"]),
            (unlabeled, ["two"]),
        ):
            self.assertEqual(
                list(
                    guards._get_guard_labels(torch._guards.GuardKey.from_guard(guard))
                ),
                labels,
            )
        self.assertEqual(first.creation_label, "one")
        self.assertIsNone(direct.creation_label)
        self.assertIsNone(unlabeled.creation_label)

    def test_installed_labels_survive_set_operations_and_checkpoint(self):
        context = TracingContext(None)
        guards_context = context.guards_context
        source = LocalSource("x")
        with tracing(context), guards_context.guard_label("one"):
            first = source.make_guard(GuardBuilder.TYPE_MATCH)
            second = AttrSource(source, "value").make_guard(GuardBuilder.ID_MATCH)
            install_guard(first, second)
            checkpoint = guards_context.copy_graphstate()
        key = torch._guards.GuardKey.from_guard(first)
        copied = guards_context.dynamo_guards.copy()
        difference = copied - GuardsSet(OrderedSet([second]))
        self.assertEqual(difference.guard_labels, {key: OrderedSet(["one"])})
        updated = GuardsSet()
        updated.update(difference)
        self.assertEqual(updated.guard_labels, difference.guard_labels)
        updated.remove(first)
        self.assertEqual(updated.guard_labels, {})
        copied.remove_guards_with_source(source)
        self.assertEqual(copied.guard_labels, {})
        guards_context.dynamo_guards.clear()
        guards_context.restore_graphstate(checkpoint)
        self.assertEqual(guards_context.current_guard_label, "one")
        self.assertEqual(
            guards_context.dynamo_guards.guard_labels, checkpoint.guard_labels
        )
        self.assertEqual(
            list(guards_context.dynamo_guards._get_guard_labels(key)), ["one"]
        )

    @parametrize("graph_break", [False, True])
    @parametrize("nested_graph_breaks", [False, True])
    def test_region_guard_creation_across_break(self, graph_break, nested_graph_breaks):
        labels = {}
        expected = {}
        graphs = []

        def record_source(ctx, name, label):
            value = ctx.get_local(
                name
            )._i_will_not_complain_if_bc_breaks_VariableTracker()
            expected[value.source.name] = {label}

        def record_first(ctx):
            record_source(ctx, "first", "inner")

        def record_second(ctx):
            record_source(ctx, "second", "inner")

        def record_outer(ctx):
            record_source(ctx, "last", "outer")

        def backend(graph, inputs):
            graphs.append(graph)
            guards = TracingContext.get().guards_context.dynamo_guards
            for guard in guards:
                if guard.create_fn is GuardBuilder.CONSTANT_MATCH:
                    key = torch._guards.GuardKey.from_guard(guard)
                    labels.setdefault(guard.name, set()).update(
                        guards._get_guard_labels(key)
                    )
            return graph.forward

        def inner(x, values):
            with torch.compiler.guard_ctx("inner"):
                first = values[0]
                x = x + len(first)
                comptime(record_first)
                if graph_break:
                    torch._dynamo.graph_break()
                second = values[1]
                x = x + len(second)
                comptime(record_second)
                return x

        def fn(x, values):
            with torch.compiler.guard_ctx("outer"):
                x = inner(x, values)
                last = values[2]
                x = x + len(last)
                comptime(record_outer)
                return x

        compiled = torch.compile(fn, backend=backend)
        x = torch.ones(2)
        with torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks):
            for values in (["a", "bb", "ccc"], ["longer", "bb", "changed"]):
                self.assertEqual(compiled(x, values), x + sum(map(len, values)))
                snapshot = {name: set(value) for name, value in labels.items()}
                frames = len(graphs)
                self.assertEqual(compiled(x, values), x + sum(map(len, values)))
                self.assertEqual(labels, snapshot)
                self.assertEqual(len(graphs), frames)
        self.assertGreaterEqual(len(expected), 3)
        for name, expected_labels in expected.items():
            self.assertEqual(labels[name], expected_labels)
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("fullgraph", [False, True])
    @parametrize("prebuilt", [False, True])
    def test_context_inside_compiled_function(self, fullgraph, prebuilt):
        labels = []
        graphs = []
        manager = torch.compiler.guard_ctx("outer")

        def check_label(ctx):
            context = TracingContext.get().guards_context
            labels.append(context.current_guard_label)
            callback = next(
                g for g in context.dynamo_guards if g.name == "L['check_label']"
            )
            self.assertEqual(
                context.dynamo_guards._get_guard_labels(
                    torch._guards.GuardKey.from_guard(callback)
                ),
                OrderedSet(["ambient"]),
            )
            nested = TracingContext(None)
            self.assertIsNone(nested.guards_context.current_guard_label)

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        def fn(x):
            comptime(check_label)
            cm = manager if prebuilt else torch.compiler.guard_ctx(label="outer")
            alias = cm
            with cm as entered:
                assert entered is None  # noqa: S101
                comptime(check_label)
                with torch.compiler.guard_ctx("inner"):
                    comptime(check_label)
                    with alias:
                        comptime(check_label)
                        x = x.sin()
                    comptime(check_label)
                comptime(check_label)
            comptime(check_label)
            return x.cos()

        compiled = torch.compile(fn, backend=backend, fullgraph=fullgraph)
        x = torch.ones(2)
        with torch.compiler.guard_ctx("ambient"):
            with patch.object(torch._guards._GuardContext, "__enter__") as enter:
                with patch.object(torch._guards._GuardContext, "__exit__") as exit:
                    self.assertEqual(compiled(x), x.sin().cos())
                    self.assertEqual(compiled(x), x.sin().cos())
                    enter.assert_not_called()
                    exit.assert_not_called()
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
        self.assertEqual(
            labels, ["ambient", "outer", "inner", "outer", "inner", "outer", "ambient"]
        )
        self.assertEqual(len(graphs), 1)
        self.assertEqual(
            [(node.op, node.target) for node in graphs[0].graph.nodes],
            [
                ("placeholder", "L_x_"),
                ("call_method", "sin"),
                ("call_method", "cos"),
                ("output", "output"),
            ],
        )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("inside", [False, True])
    def test_decorator_inside_compilation(self, inside):
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        @torch.compiler.guard_ctx("decorated")
        def decorated(x):
            comptime(check_label)
            return x.sin()

        def fn(x):
            if inside:

                @torch.compiler.guard_ctx("local")
                def local(x):
                    comptime(check_label)
                    return decorated(x).cos()

                return local(x)
            return decorated(x)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(2)
        expected = x.sin().cos() if inside else x.sin()
        self.assertEqual(compiled(x), expected)
        self.assertEqual(compiled(x), expected)
        self.assertEqual(labels, ["local", "decorated"] if inside else ["decorated"])
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("nested_break", [False, True])
    @parametrize("nested_graph_breaks", [False, True])
    def test_internal_context_graph_break_resume(
        self, nested_break, nested_graph_breaks
    ):
        labels = []

        def check_label(ctx):
            labels.append(
                (
                    ctx.get_local("_phase").as_python_constant(),
                    TracingContext.get().guards_context.current_guard_label,
                )
            )

        def inner(x):
            with torch.compiler.guard_ctx("inner"):
                _phase = "inner_entry"
                comptime(check_label)
                x = x.sin()
                torch._dynamo.graph_break()
                _phase = "inner_resume"
                comptime(check_label)
                x = x.cos()
            _phase = "inner_exit"
            comptime(check_label)
            return x

        def fn(x):
            with torch.compiler.guard_ctx("outer"):
                _phase = "outer_entry"
                comptime(check_label)
                x = x + 1
                if nested_break:
                    x = inner(x)
                else:
                    with torch.compiler.guard_ctx("inner"):
                        _phase = "inner_entry"
                        comptime(check_label)
                        x = x.sin()
                        torch._dynamo.graph_break()
                        _phase = "inner_resume"
                        comptime(check_label)
                        x = x.cos()
                _phase = "outer_resume"
                comptime(check_label)
                torch._dynamo.graph_break()
                _phase = "outer_second_resume"
                comptime(check_label)
                x = x + 2
            _phase = "outer_exit"
            comptime(check_label)
            return x + 3

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        x = torch.ones(2)
        with (
            torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks),
            torch.compiler.guard_ctx("ambient"),
        ):
            self.assertEqual(compiled(x), (x + 1).sin().cos() + 5)
            expected = {
                ("outer_entry", "outer"),
                ("inner_entry", "inner"),
                ("inner_resume", "inner"),
                ("outer_resume", "outer"),
                ("outer_second_resume", "outer"),
                ("outer_exit", "ambient"),
            }
            if nested_break:
                expected.add(("inner_exit", "outer"))
            self.assertEqual(set(labels), expected)
            observed = labels.copy()
            frames = counter.frame_count
            self.assertGreater(frames, 1)
            self.assertEqual(compiled(x), (x + 1).sin().cos() + 5)
            self.assertEqual(labels, observed)
            self.assertEqual(counter.frame_count, frames)
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("graph_break", [False, True])
    def test_internal_context_exception_cleanup(self, graph_break):
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        def fn(x):
            with torch.compiler.guard_ctx("outer"):
                if graph_break:
                    with torch.compiler.guard_ctx("inner"):
                        comptime(check_label)
                        x = x.sin()
                        torch._dynamo.graph_break()
                        comptime(check_label)
                        raise RuntimeError("internal failure")
                try:
                    with torch.compiler.guard_ctx("inner"):
                        comptime(check_label)
                        x = x.sin()
                        comptime(check_label)
                        raise RuntimeError("internal failure")
                except RuntimeError:
                    comptime(check_label)
                    x = x.cos()
            comptime(check_label)
            return x

        compiled = torch.compile(fn, backend="eager", fullgraph=not graph_break)
        with torch.compiler.guard_ctx("ambient"):
            if graph_break:
                with self.assertRaisesRegex(RuntimeError, "internal failure"):
                    compiled(torch.ones(2))
                self.assertEqual(set(labels), {"inner"})
                self.assertGreaterEqual(len(labels), 2)
            else:
                self.assertEqual(compiled(torch.ones(2)), torch.ones(2).sin().cos())
                self.assertEqual(set(labels), {"inner", "outer", "ambient"})
                self.assertEqual(labels[-2:], ["outer", "ambient"])
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("fullgraph", [False, True])
    def test_compiled_factory_validation(self, fullgraph):
        def fn(x, label):
            try:
                with torch.compiler.guard_ctx(label):
                    return x.sin()
            except TypeError as error:
                assert str(error) == "guard_ctx label must be a non-empty string"  # noqa: S101
                return x.cos()
            except ValueError as error:
                assert str(error) == "guard_ctx label must be a non-empty string"  # noqa: S101
                return x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph)
        x = torch.ones(2)
        for label in (None, 1, [], False, "", "valid"):
            with self.subTest(label=label):
                self.assertEqual(compiled(x, label), fn(x, label))

    @parametrize("fullgraph", [False, True])
    def test_factory_str_subclass_validation_containment(self, fullgraph):
        events = []

        class Label(str):
            __slots__ = ()

            def __bool__(self):
                events.append("validate")
                return True

        def fn(x, label):
            with torch.compiler.guard_ctx(label):
                return x.sin()

        x = torch.ones(2)
        label = Label("region")
        for _ in range(2):
            self.assertEqual(fn(x, label), x.sin())
        self.assertEqual(events, ["validate", "validate"])
        events.clear()
        compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph)
        if fullgraph:
            with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "str subclass"):
                compiled(x, label)
            self.assertEqual(events, [])
        else:
            for _ in range(2):
                self.assertEqual(compiled(x, label), x.sin())
            self.assertEqual(events, ["validate", "validate"])

    @parametrize("decorator", [False, True])
    @parametrize("fullgraph", [False, True])
    def test_prebuilt_str_subclass_context_containment(self, decorator, fullgraph):
        events = []

        class Label(str):
            __slots__ = ()

            def __bool__(self):
                events.append("validate")
                return True

        manager = torch.compiler.guard_ctx(Label("region"))
        self.assertEqual(events, ["validate"])
        events.clear()

        def body(x):
            x = x.sin()
            torch._dynamo.graph_break()
            return x.cos()

        def with_context(x):
            with manager:
                return body(x)

        fn = manager(body) if decorator else with_context
        x = torch.ones(2)
        expected = fn(x)
        self.assertEqual(events, [])
        compiled = torch.compile(fn, backend="eager", fullgraph=fullgraph)
        with (
            torch._dynamo.config.patch(nested_graph_breaks=True),
            torch.compiler.guard_ctx("ambient"),
        ):
            if fullgraph:
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, "str subclass"
                ):
                    compiled(x)
            else:
                for _ in range(2):
                    self.assertEqual(compiled(x), expected)
            self.assertEqual(events, [])
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_compile_decorated_function(self):
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        @torch.compile(backend="eager", fullgraph=True)
        @torch.compiler.guard_ctx("decorated")
        def fn(x):
            comptime(check_label)
            return x.sin()

        self.assertEqual(fn(torch.ones(2)), torch.ones(2).sin())
        self.assertEqual(fn(torch.ones(2)), torch.ones(2).sin())
        self.assertEqual(labels, ["decorated"])

    def test_prebuilt_label_rebinding_and_decorator_clone(self):
        manager = torch.compiler.guard_ctx("first")
        clone = manager.clone()
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        @manager
        def decorated(x):
            comptime(check_label)
            return x.sin()

        def fn(x):
            with manager:
                comptime(check_label)
                x = x + 1
            with clone:
                comptime(check_label)
                x = x.cos()
            return decorated(x)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for _ in range(2):
            self.assertEqual(compiled(x), (x + 1).cos().sin())
        self.assertEqual(labels, ["first", "first", "first"])
        self.assertEqual(counter.frame_count, 1)
        manager.label = "second"
        for _ in range(2):
            self.assertEqual(compiled(x), (x + 1).cos().sin())
        self.assertEqual(
            labels, ["first", "first", "first", "second", "first", "second"]
        )
        self.assertEqual(counter.frame_count, 2)

    @parametrize("prebuilt", [False, True])
    def test_context_label_attribute_read_and_rebinding(self, prebuilt):
        manager = torch.compiler.guard_ctx("one")

        def fn(x, label):
            cm = manager if prebuilt else torch.compiler.guard_ctx(label)
            return x + (1 if cm.label == "one" else 2), cm.label

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(2)
        for index, label in enumerate(("one", "two"), start=1):
            manager.label = label
            argument = "unused" if prebuilt else label
            for _ in range(2):
                self.assertEqual(compiled(x, argument), (x + index, label))
                self.assertEqual(counter.frame_count, index)

    def test_private_context_label_mutation_containment(self):
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        def fn(x, manager):
            manager.label = "after"
            with manager:
                comptime(check_label)
                return x.sin(), manager.label, manager

        manager = torch.compiler.guard_ctx("before")
        x = torch.ones(2)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "setattr"):
            torch.compile(fn, backend="eager", fullgraph=True)(x, manager)
        self.assertEqual(manager.label, "before")
        compiled = torch.compile(fn, backend="eager")
        with torch.compiler.guard_ctx("ambient"):
            for _ in range(2):
                manager.label = "before"
                output, label, returned = compiled(x, manager)
                self.assertEqual(output, x.sin())
                self.assertEqual(label, "after")
                self.assertEqual(manager.label, "after")
                self.assertIs(returned, manager)
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, "ambient"
                )
        self.assertEqual(labels, ["after"])
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("nested_graph_breaks", [False, True])
    def test_reentrant_prebuilt_context_graph_break(self, nested_graph_breaks):
        manager = torch.compiler.guard_ctx("shared")
        labels = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        def fn(x):
            with manager:
                with torch.compiler.guard_ctx("inner"):
                    with manager:
                        x = x.sin()
                        torch._dynamo.graph_break()
                        comptime(check_label)
                    comptime(check_label)
                comptime(check_label)
            comptime(check_label)
            return x.cos()

        counter = CompileCounter()
        with torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks):
            compiled = torch.compile(fn, backend=counter)
            x = torch.ones(2)
            self.assertEqual(compiled(x), x.sin().cos())
            self.assertEqual(labels, ["shared", "inner", "shared", None])
            frames = counter.frame_count
            self.assertEqual(compiled(x), x.sin().cos())
            self.assertEqual(counter.frame_count, frames)
            self.assertEqual(labels, ["shared", "inner", "shared", None])
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_graph_break_eager_execution_uses_pending_bridge(self):
        labels = []

        @torch._dynamo.disable
        def observe_eager(x):
            self.assertIsNone(TracingContext.try_get())
            labels.append(TracingContext(None).guards_context.current_guard_label)
            return x.cos()

        def fn(x):
            with torch.compiler.guard_ctx("compiled"):
                x = x.sin()
                x = observe_eager(x)
            return x + 1

        compiled = torch.compile(fn, backend="eager")
        with torch.compiler.guard_ctx("ambient"):
            for _ in range(2):
                self.assertEqual(compiled(torch.ones(2)), torch.ones(2).sin().cos() + 1)
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, "ambient"
                )
        self.assertEqual(labels, ["compiled", "compiled"])
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_guard_label_nested_and_exception(self):
        context = GuardsContext()
        self.assertIsNone(context.current_guard_label)
        with context.guard_label("outer"):
            self.assertEqual(context.current_guard_label, "outer")
            with context.guard_label(None):
                self.assertIsNone(context.current_guard_label)
            with self.assertRaisesRegex(RuntimeError, "inner failure"):
                with context.guard_label("inner"):
                    self.assertEqual(context.current_guard_label, "inner")
                    raise RuntimeError("inner failure")
            self.assertEqual(context.current_guard_label, "outer")
        self.assertIsNone(context.current_guard_label)

    def test_guard_label_checkpoint_restore(self):
        context = GuardsContext()
        empty = context.copy_graphstate()
        self.assertIsNone(empty.current_guard_label)
        with context.guard_label("outer"):
            checkpoint = context.copy_graphstate()
            with context.guard_label("inner"):
                context.restore_graphstate(checkpoint)
                self.assertEqual(context.current_guard_label, "outer")
                context.restore_graphstate(empty)
                self.assertIsNone(context.current_guard_label)
                context.restore_graphstate(checkpoint)
                self.assertEqual(context.current_guard_label, "outer")
            self.assertEqual(checkpoint.current_guard_label, "outer")
        self.assertIsNone(context.current_guard_label)

    def test_public_guard_ctx_validation(self):
        self.assertIn("guard_ctx", torch.compiler.__all__)
        for label in (None, 1, [], False):
            with self.subTest(label=label):
                with self.assertRaisesRegex(TypeError, "non-empty string"):
                    torch.compiler.guard_ctx(label)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            torch.compiler.guard_ctx("")

    def test_pending_labels_nested_reentrant_and_exception(self):
        self.assertIsNone(TracingContext.try_get())
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)
        manager = torch.compiler.guard_ctx("outer")
        for _ in range(2):
            with manager:
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, "outer"
                )
                with torch.compiler.guard_ctx("inner"):
                    with manager:
                        self.assertEqual(
                            TracingContext(None).guards_context.current_guard_label,
                            "outer",
                        )
                    self.assertEqual(
                        TracingContext(None).guards_context.current_guard_label, "inner"
                    )
                with self.assertRaisesRegex(RuntimeError, "pending failure"):
                    with torch.compiler.guard_ctx("exception"):
                        raise RuntimeError("pending failure")
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, "outer"
                )
            self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_active_trace_is_authoritative_and_nested_trace_isolated(self):
        with torch.compiler.guard_ctx("pending"):
            outer = TracingContext(None)
            self.assertEqual(outer.guards_context.current_guard_label, "pending")
            with tracing(outer):
                manager = torch.compiler.guard_ctx("outer")
                with manager:
                    self.assertEqual(outer.guards_context.current_guard_label, "outer")
                    nested = TracingContext(None)
                    self.assertIsNone(nested.guards_context.current_guard_label)
                    with tracing(nested):
                        with torch.compiler.guard_ctx("inner"):
                            with manager:
                                self.assertEqual(
                                    nested.guards_context.current_guard_label, "outer"
                                )
                            self.assertEqual(
                                nested.guards_context.current_guard_label, "inner"
                            )
                            self.assertIsNone(
                                TracingContext(None).guards_context.current_guard_label
                            )
                        self.assertIsNone(nested.guards_context.current_guard_label)
                    self.assertEqual(outer.guards_context.current_guard_label, "outer")
                self.assertEqual(outer.guards_context.current_guard_label, "pending")
                with self.assertRaisesRegex(RuntimeError, "trace failure"):
                    with torch.compiler.guard_ctx("exception"):
                        raise RuntimeError("trace failure")
                self.assertEqual(outer.guards_context.current_guard_label, "pending")
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "pending"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_decorator_recursive_and_repeated(self):
        @torch.compiler.guard_ctx("decorated")
        def decorated(depth):
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "decorated"
            )
            if depth:
                with torch.compiler.guard_ctx("recursive"):
                    decorated(depth - 1)
                    self.assertEqual(
                        TracingContext(None).guards_context.current_guard_label,
                        "recursive",
                    )
            return depth

        for _ in range(2):
            with torch.compiler.guard_ctx("ambient"):
                self.assertEqual(decorated(2), 2)
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, "ambient"
                )
            self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("termination", ["exhaust", "throw", "close"])
    def test_generator_decorator_resume_and_cleanup(self, termination):
        labels = []
        backend_labels = []
        events = []

        def check_label(ctx):
            labels.append(TracingContext.get().guards_context.current_guard_label)

        def backend(graph, inputs):
            backend_labels.append(
                TracingContext.get().guards_context.current_guard_label
            )
            return graph.forward

        @torch.compile(backend=backend, fullgraph=True)
        def compiled(x):
            comptime(check_label)
            return x.sin()

        @torch.compiler.guard_ctx("generator")
        def generate(x):
            try:
                events.append(
                    ("entry", TracingContext(None).guards_context.current_guard_label)
                )
                increment = yield compiled(x)
                events.append(
                    ("send", TracingContext(None).guards_context.current_guard_label)
                )
                yield compiled(x + increment)
            finally:
                events.append(
                    ("finally", TracingContext(None).guards_context.current_guard_label)
                )

        x = torch.ones(2)
        iterator = generate(x)
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)
        with torch.compiler.guard_ctx("ambient"):
            self.assertEqual(next(iterator), x.sin())
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
            self.assertEqual(iterator.send(2), (x + 2).sin())
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
            if termination == "throw":
                with self.assertRaisesRegex(RuntimeError, "generator failure"):
                    iterator.throw(RuntimeError("generator failure"))
            elif termination == "close":
                iterator.close()
            else:
                with self.assertRaises(StopIteration):
                    next(iterator)
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "ambient"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)
        self.assertEqual(labels, ["generator"])
        self.assertEqual(backend_labels, ["generator"])
        self.assertEqual(
            events,
            [(phase, "generator") for phase in ("entry", "send", "finally")],
        )

    def test_masked_nested_trace_isolation_and_explicit_label(self):
        with torch.compiler.guard_ctx("pending"):
            outer = TracingContext(None)
            with tracing(outer), torch.compiler.guard_ctx("outer"):
                with self.assertRaisesRegex(RuntimeError, "masked failure"):
                    with tracing(None):
                        self.assertIsNone(TracingContext.try_get())
                        nested = TracingContext(None)
                        self.assertIsNone(nested.guards_context.current_guard_label)
                        with tracing(nested):
                            self.assertIsNone(nested.guards_context.current_guard_label)
                        with torch.compiler.guard_ctx("explicit"):
                            explicit = TracingContext(None)
                            self.assertEqual(
                                explicit.guards_context.current_guard_label, "explicit"
                            )
                            with tracing(explicit):
                                self.assertEqual(
                                    explicit.guards_context.current_guard_label,
                                    "explicit",
                                )
                        self.assertIsNone(
                            TracingContext(None).guards_context.current_guard_label
                        )
                        raise RuntimeError("masked failure")
                self.assertIs(TracingContext.get(), outer)
                self.assertEqual(outer.guards_context.current_guard_label, "outer")
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "pending"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    @parametrize("fullgraph", [False, True])
    @parametrize("decorator", [False, True])
    def test_external_context_seeds_compiled_call(self, fullgraph, decorator):
        labels = []
        backend_labels = []

        def check_label(ctx):
            context = TracingContext.get().guards_context
            labels.append(context.current_guard_label)
            callback = next(
                g for g in context.dynamo_guards if g.name == "L['check_label']"
            )
            self.assertEqual(
                context.dynamo_guards._get_guard_labels(
                    torch._guards.GuardKey.from_guard(callback)
                ),
                OrderedSet(["compiled"]),
            )
            ambient = [guard for guard in context.dynamo_guards if not guard.name]
            self.assertTrue(ambient)
            for guard in ambient:
                self.assertIsNone(guard.creation_label)
                self.assertEqual(
                    context.dynamo_guards._get_guard_labels(
                        torch._guards.GuardKey.from_guard(guard)
                    ),
                    OrderedSet(),
                )

        def backend(graph, inputs):
            backend_labels.append(
                TracingContext.get().guards_context.current_guard_label
            )
            return graph.forward

        def fn(x):
            comptime(check_label)
            return x.sin() + 1

        compiled = torch.compile(fn, backend=backend, fullgraph=fullgraph)
        manager = torch.compiler.guard_ctx("compiled")
        x = torch.ones(2)
        if decorator:
            run = manager(compiled)
            self.assertEqual(run(x), fn(x))
            self.assertEqual(run(x), fn(x))
        else:
            with manager:
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(compiled(x), fn(x))
        self.assertEqual(labels, ["compiled"])
        self.assertEqual(backend_labels, ["compiled"])
        self.assertIsNone(TracingContext.try_get())
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_external_context_survives_nested_graph_breaks(self):
        labels = []

        def check_label(ctx):
            labels.append(
                (
                    ctx.get_local("_phase").as_python_constant(),
                    TracingContext.get().guards_context.current_guard_label,
                )
            )

        def inner(x):
            _phase = "inner_entry"
            comptime(check_label)
            x = x.sin()
            torch._dynamo.graph_break()
            _phase = "inner_resume"
            comptime(check_label)
            return x.cos()

        def fn(x):
            _phase = "outer_entry"
            comptime(check_label)
            x = x + 1
            x = inner(x)
            _phase = "outer_after_inner"
            comptime(check_label)
            torch._dynamo.graph_break()
            _phase = "outer_resume"
            comptime(check_label)
            return x + 2

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        x = torch.ones(2)
        with torch.compiler.guard_ctx("outer"):
            with torch.compiler.guard_ctx("inner"):
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(
                    set(labels),
                    {
                        (phase, "inner")
                        for phase in (
                            "outer_entry",
                            "inner_entry",
                            "inner_resume",
                            "outer_after_inner",
                            "outer_resume",
                        )
                    },
                )
                observed = labels.copy()
                frames = counter.frame_count
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(labels, observed)
                self.assertEqual(counter.frame_count, frames)
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "outer"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_external_context_unwinds_compiled_exception(self):
        labels = []

        def check_label(ctx):
            labels.append(
                (
                    ctx.get_local("_phase").as_python_constant(),
                    TracingContext.get().guards_context.current_guard_label,
                )
            )

        def fn(x):
            _phase = "entry"
            comptime(check_label)
            x = x.sin()
            torch._dynamo.graph_break()
            _phase = "resume"
            comptime(check_label)
            raise RuntimeError("compiled failure")

        compiled = torch.compile(fn, backend="eager")
        with torch.compiler.guard_ctx("outer"):
            with self.assertRaisesRegex(RuntimeError, "compiled failure"):
                with torch.compiler.guard_ctx("inner"):
                    compiled(torch.ones(2))
            self.assertEqual(set(labels), {("entry", "inner"), ("resume", "inner")})
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "outer"
            )
        self.assertIsNone(TracingContext.try_get())
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

    def test_pending_labels_thread_isolation(self):
        barrier = Barrier(2)
        first_entered = Event()
        first_exited = Event()
        manager = torch.compiler.guard_ctx("shared")

        def worker(label):
            self.assertIsNone(TracingContext.try_get())
            self.assertIsNone(TracingContext(None).guards_context.current_guard_label)
            if label == "two":
                self.assertTrue(first_entered.wait(timeout=10))
            with torch.compiler.guard_ctx(label):
                with manager:
                    if label == "one":
                        first_entered.set()
                    barrier.wait(timeout=10)
                    if label == "two":
                        self.assertTrue(first_exited.wait(timeout=10))
                    self.assertEqual(
                        TracingContext(None).guards_context.current_guard_label,
                        "shared",
                    )
                self.assertEqual(
                    TracingContext(None).guards_context.current_guard_label, label
                )
                if label == "one":
                    first_exited.set()
            self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

        with torch.compiler.guard_ctx("main"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker, label) for label in ("one", "two")]
                for future in futures:
                    future.result(timeout=20)
            self.assertEqual(
                TracingContext(None).guards_context.current_guard_label, "main"
            )
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)


@instantiate_parametrized_tests
class LazyGuardRemovalTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("graph_break", [False, True])
    @parametrize("nested_graph_breaks", [False, True])
    def test_lazy_creation_and_delayed_guard_install(
        self, graph_break, nested_graph_breaks
    ):
        held = []
        observed = []

        def create_guard(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            value = tx.symbolic_locals["value"]
            self.assertIs(type(value), LazyConstantVariable)
            self.assertFalse(value.is_realized())
            guard = value.source.make_guard(GuardBuilder.CONSTANT_MATCH)
            self.assertEqual(guard.creation_label, "one")
            self.assertFalse(value.is_realized())
            held.append(guard)

        def install_delayed(ctx):
            guard = held[-1]
            self.assertEqual(
                TracingContext.get().guards_context.current_guard_label, "two"
            )
            install_guard(guard)
            guards = TracingContext.get().guards_context.dynamo_guards
            labels = guards._get_guard_labels(torch._guards.GuardKey.from_guard(guard))
            self.assertIn("one", labels)
            observed.append(True)

        def check_realization(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            value = tx.symbolic_locals["value"]
            value.realize()
            guards = tx.output.guards
            matching = [
                guard
                for guard in guards.get_guards_for_source(value.source)
                if guard.create_fn is GuardBuilder.CONSTANT_MATCH
            ]
            self.assertTrue(matching)
            # Without a break, realization creates a NEW guard in region two.
            # A break may already realize the value while reconstructing region one.
            if not graph_break:
                self.assertIn(
                    "two",
                    guards._get_guard_labels(
                        torch._guards.GuardKey.from_guard(matching[0])
                    ),
                )

        def fn(x, values):
            with torch.compiler.guard_ctx("one"):
                value = values[0]
                comptime(create_guard)
                x = x.sin()
                if graph_break:
                    torch._dynamo.graph_break()
            with torch.compiler.guard_ctx("two"):
                comptime(install_delayed)
                comptime(check_realization)
                return x + len(value) + len(values)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)
        x = torch.ones(2)
        with torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks):
            for value in ("first", "longer value"):
                self.assertEqual(compiled(x, [value]), x.sin() + len(value) + 1)
                frames = counter.frame_count
                callbacks = len(observed)
                self.assertEqual(compiled(x, [value]), x.sin() + len(value) + 1)
                self.assertEqual(counter.frame_count, frames)
                self.assertEqual(len(observed), callbacks)
        self.assertTrue(observed)
        self.assertIsNone(TracingContext(None).guards_context.current_guard_label)

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
