# Owner(s): ["module: dynamo"]

import dataclasses
import functools
import io
import os
import tempfile
import threading
from unittest.mock import Mock, patch

import torch
import torch._dynamo.test_case
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.guards import (
    _filter_labeled_guards,
    CheckFunctionManager,
    GuardBuilder,
    make_guard_filter_entry,
)
from torch._dynamo.output_graph import OutputGraphGuardsState
from torch._dynamo.source import DictGetItemSource, GlobalSource, LocalSource
from torch._dynamo.testing import CompileCounter
from torch._guards import (
    DuplicateInputs,
    Guard,
    GuardsSet,
    StorageOverlap,
    StorageOverlapPartition,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class GuardFilterTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        registry = patch.dict(torch.compiler._guard_filter_registry, clear=True)
        registry.start()
        self.addCleanup(registry.stop)

    def test_registration(self):
        def callback(entries):
            return [True] * len(entries)

        self.assertIs(
            torch.compiler.register_guard_filter("region")(callback), callback
        )
        with self.assertRaisesRegex(ValueError, "already registered"):
            torch.compiler.register_guard_filter("region")(callback)
        self.assertIs(torch.compiler._guard_filter_registry["region"], callback)
        for label in (None, 1, object()):
            with self.assertRaisesRegex(TypeError, "non-empty string"):
                torch.compiler.register_guard_filter(label)
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            torch.compiler.register_guard_filter("")
        with self.assertRaisesRegex(TypeError, "callable"):
            torch.compiler.register_guard_filter("bad")(None)
        self.assertNotIn("bad", torch.compiler._guard_filter_registry)

    def test_competing_registrations_are_atomic(self):
        first_checked = threading.Event()
        release_first = threading.Event()
        accepted, rejected, errors = [], [], []
        lock = threading.RLock()

        class Registry(dict):
            def __contains__(self, label):
                present = super().__contains__(label)
                if threading.current_thread() is first:
                    first_checked.set()
                    if not release_first.wait(timeout=5):
                        raise AssertionError("Competing registration did not run")
                else:
                    # Without synchronization both threads read the absent key
                    # before either stores its opposing filter policy.
                    release_first.set()
                return present

        class RegistrationLock:
            def __enter__(self):
                # With synchronization the competitor cannot reach __contains__
                # until the first registration finishes. Release its wait here.
                if threading.current_thread() is second:
                    release_first.set()
                lock.acquire()

            def __exit__(self, *exc):
                lock.release()

        def accept(entries):
            return [True] * len(entries)

        def reject(entries):
            return [False] * len(entries)

        def register(callback):
            try:
                accepted.append(torch.compiler.register_guard_filter("race")(callback))
            except ValueError as error:
                rejected.append(error)
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=register, args=(accept,), daemon=True)
        second = threading.Thread(target=register, args=(reject,), daemon=True)
        registry = Registry()
        with (
            patch.object(torch.compiler, "_guard_filter_registry", registry),
            patch.object(
                torch.compiler,
                "_guard_filter_registry_lock",
                RegistrationLock(),
                create=True,
            ),
        ):
            first.start()
            try:
                self.assertTrue(first_checked.wait(timeout=5))
                second.start()
            finally:
                if second.ident is None:
                    release_first.set()
                first.join(timeout=5)
                if second.ident is not None:
                    second.join(timeout=5)
                release_first.set()
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(accepted, [accept])
        self.assertEqual(len(rejected), 1)
        self.assertIs(registry["race"], accept)

    @parametrize("first_keep", [False, True])
    @parametrize("second_keep", [False, True])
    def test_two_matching_filters_or(self, first_keep, second_keep):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="first", collect_debug_stack=False)
        guards.add(guard, label="second", collect_debug_stack=False)
        callbacks = []

        @torch.compiler.register_guard_filter("first")
        def first(entries):
            callbacks.append("first")
            self.assertEqual([entry.orig_guard for entry in entries], [guard])
            return [first_keep]

        @torch.compiler.register_guard_filter("second")
        def second(entries):
            callbacks.append("second")
            self.assertEqual([entry.orig_guard for entry in entries], [guard])
            return [second_keep]

        self.assertEqual(
            _filter_labeled_guards([guard], guards),
            [guard] if first_keep or second_keep else [],
        )
        self.assertEqual(callbacks, ["first", "second"])

    def test_callbacks_run_outside_registry_lock(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="first", collect_debug_stack=False)
        guards.add(
            Guard(LocalSource("y"), GuardBuilder.TYPE_MATCH),
            label="later",
            collect_debug_stack=False,
        )
        registered = threading.Event()
        errors = []

        def register_later():
            try:
                torch.compiler.register_guard_filter("later")(
                    lambda entries: [False] * len(entries)
                )
            except Exception as error:
                errors.append(error)
            finally:
                registered.set()

        @torch.compiler.register_guard_filter("first")
        def first(entries):
            if not registered.is_set():
                worker = threading.Thread(target=register_later, daemon=True)
                worker.start()
                self.addCleanup(worker.join, 5)
                self.assertTrue(
                    registered.wait(timeout=5), "Callback held registry lock"
                )
            return [True] * len(entries)

        # New registrations affect the next finalization, not this snapshot.
        self.assertEqual(_filter_labeled_guards(list(guards), guards), list(guards))
        self.assertEqual(errors, [])
        self.assertEqual(_filter_labeled_guards(list(guards), guards), [guard])

    def test_builder_free_entry(self):
        guard = Guard(GlobalSource("value"), GuardBuilder.TYPE_MATCH)
        guard.guard_types = ["STALE_TYPE"]
        guard.code_list = ["stale code"]
        entry = make_guard_filter_entry(guard, labels=frozenset({"region"}))
        self.assertEqual(entry.name, "G['value']")
        self.assertEqual(entry.guard_type, "TYPE_MATCH")
        self.assertTrue(entry.is_global)
        self.assertFalse(entry.has_value)
        self.assertIs(entry.orig_guard, guard)
        self.assertEqual(entry.labels, frozenset({"region"}))
        self.assertEqual(entry.derived_guard_types, ())
        self.assertEqual(entry.code_parts, ())

    def test_per_label_batch_or_and_default_keep(self):
        guards = GuardsSet()
        by_name = {
            name: Guard(LocalSource(name), GuardBuilder.TYPE_MATCH)
            for name in ("reject", "union", "accept", "unknown", "unlabeled")
        }
        for name, labels in {
            "reject": ("no", "unregistered"),
            "union": ("no", "yes"),
            "accept": ("yes",),
            "unknown": ("unregistered",),
            "unlabeled": (),
        }.items():
            guards.add(by_name[name], collect_debug_stack=False)
            for label in labels:
                # Use semantic duplicates to exercise the side-table union.
                guards.add(
                    dataclasses.replace(by_name[name]),
                    label=label,
                    collect_debug_stack=False,
                )
        seen = {}

        def record(label, entries, keep):
            seen[label] = entries
            return [keep] * len(entries)

        torch.compiler.register_guard_filter("no")(
            functools.partial(record, "no", keep=False)
        )
        torch.compiler.register_guard_filter("yes")(
            functools.partial(record, "yes", keep=True)
        )
        torch.compiler.register_guard_filter("unused")(
            Mock(side_effect=AssertionError("unused"))
        )
        kept = _filter_labeled_guards(list(guards), guards)
        self.assertEqual(
            [guard.name for guard in kept],
            [
                by_name[name].name
                for name in ("union", "accept", "unknown", "unlabeled")
            ],
        )
        self.assertEqual([entry.name for entry in seen["no"]], ["reject", "union"])
        self.assertEqual([entry.name for entry in seen["yes"]], ["union", "accept"])
        self.assertEqual(seen["no"][1].labels, frozenset({"no", "yes"}))
        guards.clear()
        self.assertEqual(seen["no"][1].labels, frozenset({"no", "yes"}))

    def test_no_matching_filter_is_cheap(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="unknown", collect_debug_stack=False)
        ordered = [guard]
        with patch(
            "torch._dynamo.guards.make_guard_filter_entry",
            side_effect=AssertionError("entry"),
        ):
            self.assertIs(_filter_labeled_guards(ordered, guards), ordered)
            torch.compiler.register_guard_filter("unused")(
                Mock(side_effect=AssertionError("callback"))
            )
            self.assertEqual(_filter_labeled_guards(ordered, guards), ordered)
            guards._clear_guard_labels()
            self.assertIs(_filter_labeled_guards(ordered, guards), ordered)

    @parametrize("result", ["short", "long", "integer", "none", "tensor"])
    def test_result_validation(self, result):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="bad", collect_debug_stack=False)
        results = {
            "short": [],
            "long": [True, False],
            "integer": [1],
            "none": [None],
            "tensor": [torch.tensor(True)],
        }
        torch.compiler.register_guard_filter("bad")(lambda entries: results[result])
        with self.assertRaisesRegex(
            AssertionError, "Guard filter 'bad'.*(results|bool)"
        ):
            _filter_labeled_guards([guard], guards)

    def test_registry_lookup_at_finalization(self):
        seen = []

        def backend(gm, example_inputs):
            @torch.compiler.register_guard_filter("late")
            def late(entries):
                seen.extend(entries)
                return [False] * len(entries)

            return gm.forward

        def fn(x, value):
            return x + value

        with torch.compiler.guard_ctx("late"):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            self.assertEqual(compiled(torch.ones(2), 3), torch.full((2,), 4.0))
        self.assertTrue(seen)
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(torch.ones(2), 8), torch.full((2,), 4.0))

    def make_output(self, guards, locals, *, key_order=(), aot_guards=()):
        return OutputGraphGuardsState(
            local_scope=locals,
            global_scope={},
            torch_function_mode_stack=[],
            guard_on_key_order=set(key_order),
            input_source_to_sizes_strides={},
            dual_level=0,
            functorch_layers=[],
            current_device=None,
            global_state_guard=torch._C._dynamo.guards.GlobalStateGuard(),
            _guards=guards,
            _aotautograd_guards=list(aot_guards),
            skip_guards_check=True,
        )

    @staticmethod
    def frame(x, y, mapping):
        return x

    @parametrize("global_filter", [False, True])
    @torch._dynamo.config.patch(use_lamba_guard_for_object_aliasing=True)
    def test_no_cpp_construction_for_dropped_guards(self, global_filter):
        # These real guards would create a leaf, an accessor, and an epilogue.
        x, y = LocalSource("x"), LocalSource("y")
        guards = GuardsSet()
        guards.add(
            Guard(x, GuardBuilder.TYPE_MATCH), label="drop", collect_debug_stack=False
        )
        guards.add(
            Guard(x, functools.partial(GuardBuilder.DUPLICATE_INPUT, source_b=y)),
            label="drop",
            collect_debug_stack=False,
        )
        obj = object()
        output = self.make_output(guards, {"x": obj, "y": obj})
        build_inputs, created = [], []
        original_build = CheckFunctionManager.build_guards
        original_create = Guard.create

        def build(manager, sorted_guards, *args, **kwargs):
            build_inputs.append(list(sorted_guards))
            return original_build(manager, sorted_guards, *args, **kwargs)

        def create(guard, builder):
            created.append(guard)
            return original_create(guard, builder)

        @torch.compiler.register_guard_filter("drop")
        def drop(entries):
            return [False] * len(entries)

        global_callback = Mock(return_value=[])
        with (
            patch.object(CheckFunctionManager, "build_guards", build),
            patch.object(Guard, "create", create),
        ):
            manager = CheckFunctionManager(
                self.frame.__code__,
                output,
                guard_filter_fn=global_callback if global_filter else None,
            ).guard_manager
        self.assertEqual(created, [])
        self.assertEqual(build_inputs, [[], []] if global_filter else [[]])
        if global_filter:
            global_callback.assert_called_once_with([])
        self.assertEqual(manager.root.get_leaf_guards(), [])
        self.assertEqual(manager.root.get_accessors(), [])
        self.assertEqual(manager.root.get_child_managers(), [])
        self.assertEqual(manager.root.get_epilogue_lambda_guards(), [])

        # Control: the exact same input set builds the expected C++ components.
        torch.compiler._guard_filter_registry.clear()
        control = CheckFunctionManager(self.frame.__code__, output).guard_manager
        self.assertTrue(control.root.get_child_managers())
        self.assertTrue(control.root.get_accessors())
        self.assertTrue(control.root.get_child_managers()[0].get_leaf_guards())
        self.assertTrue(control.root.get_epilogue_lambda_guards())
        self.assertTrue(control.check(output.local_scope))

    def test_global_filter_receives_survivors_with_built_snapshots(self):
        regional, global_entries = [], []

        @torch.compiler.register_guard_filter("region")
        def region(entries):
            regional.extend(entries)
            return [entry.name != "drop" for entry in entries]

        def global_filter(entries):
            global_entries.extend(entries)
            return [entry.name != "keep" for entry in entries]

        def fn(x, drop, keep):
            return x + drop + keep

        with torch.compiler.guard_ctx("region"):
            compiled = torch.compile(
                fn,
                backend="eager",
                fullgraph=True,
                options={"guard_filter_fn": global_filter},
            )
            compiled(torch.ones(2), 3, 4)
        self.assertIn("drop", [entry.name for entry in regional])
        self.assertNotIn("drop", [entry.name for entry in global_entries])
        kept_entry = next(entry for entry in global_entries if entry.name == "keep")
        self.assertTrue(kept_entry.has_value)
        self.assertEqual(kept_entry.value, 4)
        self.assertEqual(kept_entry.labels, frozenset({"region"}))
        self.assertTrue(kept_entry.derived_guard_types)
        self.assertTrue(kept_entry.code_parts)
        # Global filtering can still drop a regional survivor.
        with torch.compiler.set_stance("fail_on_recompile"):
            self.assertEqual(compiled(torch.ones(2), 20, 30), torch.full((2,), 8.0))
        kept_entry.orig_guard.code_list = ["mutated after filtering"]
        self.assertNotEqual(kept_entry.code_parts, ("mutated after filtering",))
        ambient = [entry for entry in global_entries if entry.name == ""]
        self.assertTrue(ambient)
        self.assertTrue(all(entry.labels == frozenset() for entry in ambient))

    def test_key_order_manager_boundary(self):
        source = LocalSource("mapping")
        guards = GuardsSet()
        guards.add(
            Guard(source, GuardBuilder.DICT_KEYS_MATCH),
            label="dict",
            collect_debug_stack=False,
        )
        guards.add(
            Guard(DictGetItemSource(source, "a"), GuardBuilder.CONSTANT_MATCH),
            label="dict",
            collect_debug_stack=False,
        )
        output = self.make_output(
            guards, {"mapping": {"a": 1, "b": 2}}, key_order=[source]
        )
        entries_seen = []

        @torch.compiler.register_guard_filter("dict")
        def drop_dict_keys(entries):
            entries_seen.extend(entries)
            return [entry.guard_type != "DICT_KEYS_MATCH" for entry in entries]

        manager = CheckFunctionManager(self.frame.__code__, output).guard_manager
        self.assertEqual(
            {entry.guard_type for entry in entries_seen},
            {"DICT_KEYS_MATCH", "CONSTANT_MATCH"},
        )
        # A surviving child's source still requires a DictGuardManager. Regional
        # filtering does not attribute output_graph.guard_on_key_order to labels.
        (dict_manager,) = manager.root.get_child_managers()
        self.assertIsInstance(dict_manager, torch._C._dynamo.guards.DictGuardManager)
        self.assertTrue(manager.check({"mapping": {"a": 1, "b": 2}}))
        self.assertFalse(manager.check({"mapping": {"b": 2, "a": 1}}))
        torch.compiler._guard_filter_registry["dict"] = lambda entries: [False] * len(
            entries
        )
        manager = CheckFunctionManager(self.frame.__code__, output).guard_manager
        self.assertEqual(manager.root.get_accessors(), [])
        self.assertEqual(manager.root.get_child_managers(), [])
        self.assertTrue(manager.check({"mapping": {"b": 2, "a": 1}}))

    @parametrize("constraint", ["duplicate", "overlap", "partition"])
    def test_aot_relational_constraints_remain_unlabeled(self, constraint):
        x, y = LocalSource("x"), LocalSource("y")
        value = torch.ones(4)
        constraints = {
            "duplicate": DuplicateInputs(x, y),
            "overlap": StorageOverlap([x, y], []),
            "partition": StorageOverlapPartition([x, y], ((0, 1),)),
        }
        guards = GuardsSet()
        guards.add(
            Guard(x, GuardBuilder.TYPE_MATCH), label="drop", collect_debug_stack=False
        )
        seen = []

        @torch.compiler.register_guard_filter("drop")
        def drop(entries):
            seen.extend(entries)
            return [False] * len(entries)

        output = self.make_output(
            guards, {"x": value, "y": value}, aot_guards=[constraints[constraint]]
        )
        manager = CheckFunctionManager(self.frame.__code__, output).guard_manager
        self.assertEqual([entry.guard_type for entry in seen], ["TYPE_MATCH"])
        self.assertTrue(manager.root.get_accessors())
        self.assertTrue(manager.check({"x": value, "y": value}))
        self.assertFalse(manager.check({"x": value, "y": value.clone()}))

    def test_precompile_drop_capability_rejects_before_build(self):
        guard = Guard(LocalSource("x"), GuardBuilder.TYPE_MATCH)
        guards = GuardsSet()
        guards.add(guard, label="drop", collect_debug_stack=False)
        output = self.make_output(guards, {"x": object()})
        output.package = Mock(supports_regional_guard_filter_drops=False)
        torch.compiler.register_guard_filter("drop")(
            lambda entries: [False] * len(entries)
        )
        with patch.object(CheckFunctionManager, "build_guards") as build:
            with self.assertRaisesRegex(
                RuntimeError, "Regional guard filter drops.*precompile.capture"
            ):
                CheckFunctionManager(self.frame.__code__, output)
        build.assert_not_called()

    @parametrize("nested_graph_breaks", [False, True])
    @parametrize("graph_break", [False, True])
    def test_nested_regions_across_graph_break(self, nested_graph_breaks, graph_break):
        seen = {"outer": [], "inner": []}

        def regional_filter(label, entries):
            seen[label].extend(entries)
            return [True] * len(entries)

        for label in seen:
            torch.compiler.register_guard_filter(label)(
                functools.partial(regional_filter, label)
            )

        @torch.compiler.guard_ctx("outer")
        def fn(x, values):
            with torch.compiler.guard_ctx("inner"):
                y = x + values[0]
                if graph_break:
                    torch._dynamo.graph_break()
                y = y + values[1]
            return y + values[2]

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=not graph_break)
        with torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks):
            self.assertEqual(compiled(torch.ones(2), [2, 3, 4]), torch.full((2,), 10.0))
            frames = counter.frame_count
            self.assertEqual(compiled(torch.ones(2), [2, 3, 4]), torch.full((2,), 10.0))
            self.assertEqual(counter.frame_count, frames)
        for label, entries in seen.items():
            self.assertTrue(entries, label)
            self.assertTrue(all(label in entry.labels for entry in entries))

    @parametrize("caching_precompile", [False, True])
    def test_aot_save_load_keeps_filtered_survivors(self, caching_precompile):
        seen = []

        @torch.compiler.register_guard_filter("aot")
        def regional(entries):
            seen.extend(entries)
            return [entry.name != "value" for entry in entries]

        def fn(x, value):
            return x.sin() * value

        x = torch.ones(2, requires_grad=True)
        with (
            torch.compiler.guard_ctx("aot"),
            torch._dynamo.config.patch(caching_precompile=caching_precompile),
        ):
            compiled = torch.compile(
                fn, backend="aot_eager", fullgraph=True
            ).aot_compile(((x, 3), {}))
        self.assertTrue(any(entry.name == "value" for entry in seen))
        state = torch._dynamo.package.load_guards_state(
            compiled._artifacts.guards_state
        )
        self.assertNotIn(
            "L['value']", [guard.name for guard in state.output_graph.guards]
        )
        self.assertEqual(state.output_graph.guards.guard_labels, {})
        self.assertTrue(
            all(guard.creation_label is None for guard in state.output_graph.guards)
        )
        saved = AOTCompiledFunction.serialize(compiled).serialized_data
        torch._dynamo.reset()
        # Load must rebuild the saved survivors without consulting registrations.
        torch.compiler._guard_filter_registry["aot"] = Mock(
            side_effect=AssertionError("load-time filter")
        )
        loaded = torch.compiler.load_compiled_function(io.BytesIO(saved))
        for callable in (compiled, loaded):
            actual_input = torch.ones(2, requires_grad=True)
            expected_input = actual_input.detach().clone().requires_grad_()
            actual = callable(actual_input, 9)
            expected = fn(expected_input, 3)
            self.assertEqual(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            self.assertEqual(actual_input.grad, expected_input.grad)
            self.assertFalse(
                callable._artifacts.guard_manager.check(
                    {"x": torch.ones(5), "value": 9}
                )
            )

    @parametrize("keep", [False, True])
    def test_prototype_precompile_regional_filter_boundary(self, keep):
        calls = []

        @torch.compiler.register_guard_filter("capture")
        def regional(entries):
            calls.append(entries)
            return [keep] * len(entries)

        def fn(x, value):
            return x + value

        with tempfile.TemporaryDirectory() as directory:
            artifact = os.path.join(directory, "artifact.py")
            cache = os.path.join(directory, "cache.pt")
            capture = torch.compiler.precompile.capture(
                fn, artifact_path=artifact, cache_path=cache, backend="eager"
            )
            if keep:
                with capture as captured, torch.compiler.guard_ctx("capture"):
                    self.assertEqual(captured(torch.ones(2), 3), torch.full((2,), 4.0))
                self.assertTrue(os.path.isfile(artifact))
                self.assertTrue(os.path.isfile(cache))
            else:
                with self.assertRaisesRegex(
                    RuntimeError, "Regional guard filter drops.*precompile.capture"
                ):
                    with capture as captured, torch.compiler.guard_ctx("capture"):
                        captured(torch.ones(2), 3)
                self.assertFalse(os.path.exists(artifact))
                self.assertFalse(os.path.exists(cache))
        self.assertTrue(calls)

    def test_registered_filter_runs_before_construction(self):
        entries_seen = []

        @torch.compiler.register_guard_filter("drop")
        def drop(entries):
            entries_seen.extend(entries)
            return [False] * len(entries)

        self.addCleanup(torch.compiler._guard_filter_registry.pop, "drop")
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, value):
            return x + value

        with torch.compiler.guard_ctx("drop"):
            self.assertEqual(fn(torch.ones(2), 3), torch.full((2,), 4.0))
        self.assertTrue(entries_seen)
        self.assertTrue(
            all(entry.labels == frozenset({"drop"}) for entry in entries_seen)
        )
        self.assertTrue(all(entry.derived_guard_types == () for entry in entries_seen))
        self.assertTrue(all(entry.code_parts == () for entry in entries_seen))
        with torch.compiler.set_stance("fail_on_recompile"):
            # The user explicitly takes responsibility for the specialization.
            self.assertEqual(fn(torch.ones(2), 9), torch.full((2,), 4.0))
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
