# Owner(s): ["module: dynamo"]

import functools
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.guards import GuardBuilder
from torch._dynamo.source import AttrSource, LocalSource
from torch._dynamo.testing import CompileCounter
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
            self.assertEqual(context.dynamo_guards.guard_labels, {})

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
