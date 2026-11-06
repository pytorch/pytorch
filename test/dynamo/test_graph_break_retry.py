# Owner(s): ["module: dynamo"]

import collections
import gc
import inspect
import operator
import subprocess
import sys
import textwrap
import types
import weakref
from unittest import mock

import torch
from torch._dynamo import compiled_autograd
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


class InitializationState:
    def __init__(self):
        self.initialized = False
        self.calls = 0
        self.unrelated = False


global_state = None
global_initialized = False


def global_initialize(x):
    if not global_state.initialized:
        torch._dynamo.graph_break()
        global_state.initialized = True
        global_state.calls += 1
    return torch.cos(torch.sin(x))


def global_bool_body(x):
    if not global_initialized:
        torch._dynamo.graph_break()
    return torch.cos(torch.sin(x))


@torch.compiler.disable
def disabled_initialization(state):
    state.initialized = True
    state.calls += 1


def explicit_break():
    torch._dynamo.graph_break()


def read_initialized(state):
    return state.initialized


def read_state(state):
    return state


def unfiltered_backward_hooks(module):
    return list(module._backward_hooks.values()), []


def empty_backward_pre_hooks(module):
    return []


def initialize(x, state):
    if not state.initialized:
        torch._dynamo.graph_break()
        state.initialized = True
        state.calls += 1
    return torch.cos(torch.sin(x))


def initialize_outer(x, state):
    return initialize(x + 1, state) * 2


class HookModule(torch.nn.Module):
    def forward(self, x):
        return torch.cos(torch.sin(x))


def hook_outer(x, module):
    return module(x + 1) * 2


def checkpoint_captured(counter):
    return any(
        node.target is tag_activation_checkpoint
        for graph in counter.graphs
        for node in graph.graph.nodes
    )


def whole_region_captured(counter):
    return any(
        {operator.add, torch.sin, operator.mul}.issubset(
            {node.target for node in graph.graph.nodes}
        )
        for graph in counter.graphs
    )


class GraphBreakRetryTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        nn_module = torch.nn.modules.module
        self.enterContext(
            mock.patch.object(
                nn_module,
                "_global_is_full_backward_hook",
                nn_module._global_is_full_backward_hook,
            )
        )

    def compare(self, fn, compiled, device, args=(), eager_args=None, size=8):
        if eager_args is None:
            eager_args = args
        x = torch.randn(size, dtype=torch.double, device=device, requires_grad=True)
        eager_x = x.detach().clone().requires_grad_()
        actual = compiled(x, *args)
        expected = fn(eager_x, *eager_args)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, 2 * torch.cos(torch.sin(x.detach().double() + 1)))
        self.assertEqual(
            torch.autograd.grad(actual.sum(), x)[0],
            torch.autograd.grad(expected.sum(), eager_x)[0],
        )

    def compare_effects(
        self, fn, compiled, device, calls, args=(), size=8, before_call=None
    ):
        x = torch.randn(size, dtype=torch.double, device=device, requires_grad=True)
        eager_x = x.detach().clone().requires_grad_()
        if before_call is not None:
            before_call()
        calls.clear()
        actual = compiled(x, *args)
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        actual_calls = list(calls)
        if before_call is not None:
            before_call()
        calls.clear()
        expected = fn(eager_x, *args)
        expected_grad = torch.autograd.grad(expected.sum(), eager_x)[0]
        self.assertEqual(actual, expected)
        self.assertEqual(actual_grad, expected_grad)
        self.assertEqual(actual_calls, calls)
        return x

    def native_guards(self, fn):
        entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertTrue(entries)
        return entries[0].guard_manager

    def frame_locals(self, fn, **kwargs):
        result = dict(
            zip(
                fn.__code__.co_freevars,
                (cell.cell_contents for cell in fn.__closure__ or ()),
            )
        )
        result.update(kwargs)
        return result

    @parametrize(
        "scenario",
        (
            "explicit_checkpoint",
            "disabled_checkpoint",
            "explicit_no_nested",
            "disabled_no_nested",
            "helper_checkpoint",
            "early_return_checkpoint",
        ),
    )
    def test_initialization_recovers(self, device, scenario):
        use_checkpoint = "checkpoint" in scenario

        def body(x, state):
            if scenario == "early_return_checkpoint" and state.initialized:
                return torch.cos(torch.sin(x))
            if not state.initialized:
                if scenario.startswith("disabled"):
                    disabled_initialization(state)
                else:
                    if scenario == "helper_checkpoint":
                        explicit_break()
                    else:
                        torch._dynamo.graph_break()
                    state.initialized = True
                    state.calls += 1
            return torch.cos(torch.sin(x))

        def fn(x, state):
            if use_checkpoint:
                return checkpoint(body, x + 1, state, use_reentrant=False) * 2
            return body(x + 1, state) * 2

        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        with torch._dynamo.config.patch(nested_graph_breaks=use_checkpoint):
            compiled = torch.compile(fn, backend=counter, dynamic=True)
            frames = []
            for index, size in enumerate((8, 8, 11, 15, 8)):
                self.compare(fn, compiled, device, (state,), (eager_state,), size)
                self.assertEqual(state.calls, 1)
                self.assertEqual(state.calls, eager_state.calls)
                if use_checkpoint:
                    self.assertEqual(checkpoint_captured(counter), index != 0)
                else:
                    self.assertEqual(whole_region_captured(counter), index != 0)
                frames.append(counter.frame_count)
            self.assertEqual(
                frames, [2, 3, 3, 3, 3] if use_checkpoint else [3, 4, 4, 4, 4]
            )

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_direct_bool_argument(self, device):
        def body(x, initialized, state):
            if not initialized:
                torch._dynamo.graph_break()
                state.calls += 1
            return torch.cos(torch.sin(x))

        def fn(x, initialized, state):
            return checkpoint(body, x + 1, initialized, state, use_reentrant=False) * 2

        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for index in range(4):
            initialized = index != 0
            self.compare(
                fn, compiled, device, (initialized, state), (initialized, eager_state)
            )
            frames.append(counter.frame_count)
            self.assertEqual(checkpoint_captured(counter), initialized)
            self.assertEqual(state.calls, 2)
            self.assertEqual(state.calls, eager_state.calls)
        self.assertEqual(frames, [2, 3, 3, 3])

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_parent_child_bool_members_remain_distinct(self, device):
        class State:
            def __init__(self):
                self.first = False
                self.second = False

        def child(x, state):
            if not state.second:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def body(x, state):
            if not state.first:
                return child(x, state)
            return torch.cos(torch.sin(x)) + 3

        def fn(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        graphs, executed = [], []

        def backend(graph, inputs):
            graphs.append(graph)
            captured = any(
                node.target is tag_activation_checkpoint for node in graph.graph.nodes
            )

            def forward(*args):
                executed.append(captured)
                return graph.forward(*args)

            return forward

        state = State()
        compiled = torch.compile(fn, backend=backend, dynamic=True)
        frames = []
        for first, second, captured in (
            (False, False, False),
            (False, True, True),
            (False, False, False),
            (True, False, True),
            (True, True, True),
        ):
            state.first, state.second = first, second
            x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
            eager_x = x.detach().clone().requires_grad_()
            executed.clear()
            actual = compiled(x, state)
            actual_grad = torch.autograd.grad(actual.sum(), x)[0]
            self.assertEqual(any(executed), captured)
            expected = fn(eager_x, state)
            self.assertEqual(actual, expected)
            self.assertEqual(
                actual, 2 * (torch.cos(torch.sin(x.detach() + 1)) + (3 if first else 0))
            )
            self.assertEqual(
                actual_grad, torch.autograd.grad(expected.sum(), eager_x)[0]
            )
            frames.append(len(graphs))
            if len(frames) == 1:
                root = self.native_guards(fn)
                scope = self.frame_locals(fn, x=x, state=state)
                self.assertTrue(root.check(scope))
                state.first = True
                self.assertFalse(root.check(scope))
                state.first, state.second = False, True
                self.assertFalse(root.check(scope))
                state.second = False
                self.assertTrue(root.check(scope))
                text = str(root)
                self.assertIn("'first'", text)
                self.assertIn("'second'", text)
        self.assertEqual(frames, [2, 3, 3, 4, 4])

    @parametrize("disabled", (False, True))
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_nested_direct_remains_supported(self, device, disabled):
        def body(x, state):
            if not state.initialized:
                if disabled:
                    disabled_initialization(state)
                else:
                    torch._dynamo.graph_break()
                    state.initialized = True
                    state.calls += 1
            return torch.cos(torch.sin(x))

        def fn(x, state):
            return body(x + 1, state) * 2

        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for _ in range(4):
            self.compare(fn, compiled, device, (state,), (eager_state,))
            frames.append(counter.frame_count)
        self.assertEqual(frames, [2, 3, 3, 3])
        self.assertTrue(whole_region_captured(counter))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_raw_flag_rebinding_to_unsafe_class(self, device):
        class State:
            def __init__(self):
                self.initialized = False

        class NoRetryState:
            __slots__ = ("__dict__",)

            def __init__(self):
                self.initialized = False

        def body(x, state):
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        def control(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        state, calls = State(), []
        original_state = state
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        x = self.compare_effects(fn, compiled, device, calls, (state,))
        root = self.native_guards(fn)
        self.assertIn("initialized", str(root))
        steady = counter.frame_count
        control_counter = CompileCounterWithBackend("eager")
        compiled_control = torch.compile(control, backend=control_counter, dynamic=True)
        no_retry = NoRetryState()
        self.compare_effects(control, compiled_control, device, calls, (no_retry,))
        self.assertNotIn("___instance_dict_bool", str(self.native_guards(control)))

        class Name(str):
            __slots__ = ()

            def __hash__(self):
                return hash("__dict__")

            def __eq__(self, other):
                calls.append("eq")
                return False

        key = Name("foreign_key")
        namespace = gc.get_referents(type.__dict__["__dict__"].__get__(State))[0]
        namespace[key] = None
        State.unrelated = 1
        scope = self.frame_locals(fn, x=x, state=state)
        for initialized in (False, True, False):
            state.initialized = initialized
            calls.clear()
            self.assertTrue(root.check(scope))
            self.assertEqual(calls, [])
            self.compare_effects(fn, compiled, device, calls, (state,))
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady)
        del namespace[key]
        State.unrelated = 2
        OtherState = type("OtherState", (), {key: None})
        state = OtherState()
        state.initialized = False
        scope = self.frame_locals(fn, x=x, state=state)
        calls.clear()
        self.assertFalse(root.check(scope))
        self.assertEqual(calls, [])
        self.assertIn("___check_type_id", str(root.check_verbose(scope)))
        OtherState.unrelated = object()
        calls.clear()
        control_x = x.detach().clone().requires_grad_()
        control_result = compiled_control(control_x, state)
        control_grad = torch.autograd.grad(control_result.sum(), control_x)[0]
        control_calls = list(calls)
        OtherState.unrelated = object()
        calls.clear()
        result = compiled(x, state)
        result_grad = torch.autograd.grad(result.sum(), x)[0]
        self.assertEqual(result, control_result)
        self.assertEqual(result_grad, control_grad)
        self.assertLessEqual(len(calls), len(control_calls))
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(calls, [])
        steady = counter.frame_count
        fallback_root = self.native_guards(fn)
        self.assertNotIn("___instance_dict_bool", str(fallback_root))
        for initialized, size in ((False, 8), (True, 11), (False, 15), (True, 8)):
            state.initialized = initialized
            calls.clear()
            self.assertTrue(fallback_root.check(scope))
            self.assertEqual(calls, [])
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady)
            self.assertFalse(checkpoint_captured(counter))
        namespace = gc.get_referents(type.__dict__["__dict__"].__get__(OtherState))[0]
        del namespace[key]
        OtherState.unrelated = 1
        self.assertTrue(fallback_root.check(scope))
        self.assertEqual(calls, [])
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(counter.frame_count, steady)
        original_state.initialized = True
        state = original_state
        self.assertFalse(root.check(self.frame_locals(fn, x=x, state=state)))
        for size in (8, 11, 15):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady + 1)
            self.assertTrue(checkpoint_captured(counter))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_unknown_retry_predicate_preserves_prefix_guards(self, device):
        class State:
            def __init__(self):
                self.initialized = False
                self.prefix = False

        def body(x, state):
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            y = (x + 1) * 3 if state.prefix else x + 1
            return checkpoint(body, y, state, use_reentrant=False) * 2

        state, calls = State(), []
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        x = self.compare_effects(fn, compiled, device, calls, (state,))
        root = self.native_guards(fn)
        scope = self.frame_locals(fn, x=x, state=state)
        self.assertIn("initialized", str(root))
        self.assertIn("prefix", str(root))

        class Name(str):
            __slots__ = ()
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append("eq")
                return str.__eq__(self, other)

        namespace = gc.get_referents(type.__dict__["__dict__"].__get__(State))[0]
        namespace[Name("unrelated_protocol")] = None
        State.unrelated = 1
        state.initialized = True
        calls.clear()
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(calls, [])
        before = counter.frame_count
        state.prefix = True
        self.assertFalse(root.check(scope))
        self.assertEqual(calls, [])
        for index, size in enumerate((8, 11, 15)):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            if index == 0:
                steady = counter.frame_count
                self.assertGreater(steady, before)
            self.assertEqual(counter.frame_count, steady)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_completed_unrelated_branches_do_not_retry(self, device):
        def body(x, state, unrelated):
            if unrelated:
                x = x + 0
            if state.unrelated:
                x = x + 0
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state, unrelated):
            return checkpoint(body, x + 1, state, unrelated, use_reentrant=False) * 2

        state, unrelated = InitializationState(), {}
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for index in range(7):
            self.compare(fn, compiled, device, (state, unrelated), size=8 + index)
            self.assertFalse(checkpoint_captured(counter))
            frames.append(counter.frame_count)
            state.unrelated = not state.unrelated
            setattr(state, f"extra_{index}", index)
            unrelated[str(index)] = index
        self.assertEqual(frames, [2] * 7)

    @parametrize("kind", ("local_full", "local_pre", "global_full", "global_pre"))
    @parametrize("checkpointed", (False, True))
    def test_backward_hook_removal_recovers(self, device, kind, checkpointed):
        module, calls = HookModule().to(device), []

        def full_hook(module, grad_input, grad_output):
            calls.append("full")
            return tuple(
                value * 3 if value is not None else None for value in grad_input
            )

        def pre_hook(module, grad_output):
            calls.append("pre")
            return tuple(
                value * 3 if value is not None else None for value in grad_output
            )

        nn_module = torch.nn.modules.module
        if kind == "local_full":
            handle = module.register_full_backward_hook(full_hook)
        elif kind == "local_pre":
            handle = module.register_full_backward_pre_hook(pre_hook)
        elif kind == "global_full":
            handle = nn_module.register_module_full_backward_hook(full_hook)
        else:
            handle = nn_module.register_module_full_backward_pre_hook(pre_hook)
        self.addCleanup(handle.remove)

        def body(x, module):
            return module(x)

        def fn(x, module):
            if checkpointed:
                return checkpoint(body, x + 1, module, use_reentrant=False) * 2
            return module(x + 1) * 2

        counter = CompileCounterWithBackend("eager")
        with torch._dynamo.config.patch(
            nested_graph_breaks=checkpointed, compiled_autograd=False
        ):
            self.assertFalse(compiled_autograd.compiled_autograd_enabled)
            compiled = torch.compile(fn, backend=counter, dynamic=True)
            frames = []
            for index, size in enumerate((8, 8, 11, 15, 8)):
                calls.clear()
                self.compare(fn, compiled, device, (module,), size=size)
                self.assertEqual(len(calls), 2 if index == 0 else 0)
                capture = (
                    checkpoint_captured(counter)
                    if checkpointed
                    else whole_region_captured(counter)
                )
                self.assertEqual(capture, index != 0)
                frames.append(counter.frame_count)
                if index == 0:
                    handle.remove()
            self.assertEqual(
                frames, [2, 3, 3, 3, 3] if checkpointed else [3, 4, 4, 4, 4]
            )

    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_same_hook_gate_does_not_retry(self, device):
        module, calls = HookModule().to(device), []

        def full_hook(module, grad_input, grad_output):
            calls.append("full")
            return None

        def pre_hook(module, grad_output):
            calls.append("pre")
            return None

        handles = [module.register_full_backward_hook(full_hook)]
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(hook_outer, backend=counter, dynamic=True)
        for _ in range(3):
            calls.clear()
            self.compare(hook_outer, compiled, device, (module,))
            self.assertEqual(len(calls), 2)
        steady = counter.frame_count
        nn_module = torch.nn.modules.module

        def check(count):
            calls.clear()
            self.compare(hook_outer, compiled, device, (module,))
            self.assertEqual(len(calls), 2 * count)
            self.assertEqual(counter.frame_count, steady)
            self.assertFalse(whole_region_captured(counter))

        handles[0].remove()
        handles.append(module.register_full_backward_hook(full_hook))
        check(1)
        for count in (2, 3, 4):
            handles.append(module.register_full_backward_hook(full_hook))
            check(count)
        module._backward_hooks = collections.OrderedDict(module._backward_hooks)
        check(4)
        global_pre = nn_module.register_module_full_backward_pre_hook(pre_hook)
        handles.append(global_pre)
        module._backward_hooks.clear()
        check(1)
        local_pre = module.register_full_backward_pre_hook(pre_hook)
        handles.append(local_pre)
        global_pre.remove()
        check(1)
        global_full = nn_module.register_module_full_backward_hook(full_hook)
        handles.append(global_full)
        local_pre.remove()
        check(1)
        with mock.patch.object(
            nn_module,
            "_global_backward_hooks",
            collections.OrderedDict(nn_module._global_backward_hooks),
        ):
            check(1)
        local_full = module.register_full_backward_hook(full_hook)
        handles.append(local_full)
        global_full.remove()
        check(1)
        local_full.remove()
        self.compare(hook_outer, compiled, device, (module,))
        self.assertEqual(counter.frame_count, steady + 1)
        self.assertTrue(whole_region_captured(counter))

    @parametrize(
        "protocol",
        (
            "class_full",
            "instance_full",
            "class_pre",
            "instance_pre",
            "ordered_full_values",
            "ordered_pre_values",
        ),
    )
    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_overridden_hook_collection_is_not_retained(self, device, protocol):
        class ClassFull(HookModule):
            _get_backward_hooks = unfiltered_backward_hooks

        class ClassPre(HookModule):
            _get_backward_pre_hooks = empty_backward_pre_hooks

        if protocol == "class_full":
            module = ClassFull().to(device)
        elif protocol == "class_pre":
            module = ClassPre().to(device)
        else:
            module = HookModule().to(device)
        if protocol == "instance_full":
            module._get_backward_hooks = types.MethodType(
                unfiltered_backward_hooks, module
            )
        elif protocol == "instance_pre":
            module._get_backward_pre_hooks = types.MethodType(
                empty_backward_pre_hooks, module
            )
        calls = []

        def full_hook(module, grad_input, grad_output):
            calls.append("full")
            return tuple(
                value * 3 if value is not None else None for value in grad_input
            )

        def pre_hook(module, grad_output):
            calls.append("pre")
            return tuple(
                value * 3 if value is not None else None for value in grad_output
            )

        full_handle = module.register_full_backward_hook(full_hook)
        self.addCleanup(full_handle.remove)
        if protocol not in ("class_full", "instance_full"):
            pre_handle = module.register_full_backward_pre_hook(pre_hook)
            self.addCleanup(pre_handle.remove)
        if protocol == "ordered_full_values":
            module._backward_hooks.values = list
        elif protocol == "ordered_pre_values":
            module._backward_pre_hooks.values = list
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(hook_outer, backend=counter, dynamic=True)
        frames = []
        for index in range(4):
            calls.clear()
            self.compare(hook_outer, compiled, device, (module,))
            self.assertEqual(
                calls,
                ["pre", "pre"]
                if protocol == "ordered_full_values"
                else ["full", "full"],
            )
            frames.append(counter.frame_count)
            if index == 2 and protocol in ("class_full", "instance_full"):
                module._is_full_backward_hook = False
        self.assertEqual(frames, [frames[0]] * 4)
        self.assertNotIn(
            "___nn_module_backward_hooks", str(self.native_guards(hook_outer))
        )

    def hook_alias_fixture(self, device):
        module = HookModule().to(device)
        hook_calls = []

        def hook(module, grad_input, grad_output):
            hook_calls.append("full")
            return tuple(
                value * 3 if value is not None else None for value in grad_input
            )

        handle = module.register_full_backward_hook(hook)
        self.addCleanup(handle.remove)
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(hook_outer, backend=counter, dynamic=True)
        namespace = hook_outer.__globals__
        for index in range(3):
            hook_calls.clear()
            self.compare(hook_outer, compiled, device, (module,))
            self.assertEqual(hook_calls, ["full", "full"])
            name = f"unused_hook_global_{index}"
            namespace[name] = index
            self.addCleanup(namespace.pop, name)
        self.assertEqual(counter.frame_count, 3)
        self.assertEqual(
            [
                {
                    node.target
                    for node in graph.graph.nodes
                    if node.op == "call_function"
                }
                for graph in counter.graphs
            ],
            [{operator.add}, {torch.sin, torch.cos}, {operator.mul}],
        )
        root = self.native_guards(hook_outer)
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        self.assertTrue(root.check({"x": x, "module": module}))
        alias = "__import_torch_dot_nn_dot_modules_dot_module"
        saved = namespace.pop(alias)
        self.assertFalse(root.check({"x": x, "module": module}))
        namespace[alias] = types.ModuleType("foreign_hook_module")
        self.assertFalse(root.check({"x": x, "module": module}))
        namespace.pop(alias)
        calls = []
        alias_hash = hash(alias)

        class ForeignKey:
            def __hash__(self):
                calls.append("hash")
                return alias_hash

            def __eq__(self, other):
                calls.append("eq")
                return False

        key = ForeignKey()
        namespace[key] = 1
        namespace[alias] = saved
        calls.clear()

        def cleanup():
            namespace.pop(key, None)
            namespace[alias] = saved

        self.addCleanup(cleanup)
        return module, x, compiled, counter, root, calls, hook_calls, key, saved

    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_hook_alias_guard_is_pure(self, device):
        module, x, compiled, counter, root, calls, _, _, _ = self.hook_alias_fixture(
            device
        )
        scope = {"x": x, "module": module}
        accepted = root.check(scope)
        self.assertEqual(calls, [])
        self.assertTrue(accepted)

    @parametrize("alias_state", ("present", "absent", "rebound"))
    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_hook_alias_unsafe_globals_keep_fallback(self, device, alias_state):
        module, x, compiled, counter, root, calls, hook_calls, key, saved = (
            self.hook_alias_fixture(device)
        )
        eager_x = x.detach().clone().requires_grad_()
        hook_calls.clear()
        expected = hook_outer(eager_x, module)
        eager_calls = list(calls)
        calls.clear()
        actual = compiled(x, module)
        self.assertEqual(actual, expected)
        self.assertEqual(calls, eager_calls)
        self.assertEqual(
            torch.autograd.grad(actual.sum(), x)[0],
            torch.autograd.grad(expected.sum(), eager_x)[0],
        )
        self.assertEqual(hook_calls, ["full", "full"])
        self.assertEqual(counter.frame_count, 3)
        namespace = hook_outer.__globals__
        alias = "__import_torch_dot_nn_dot_modules_dot_module"
        if alias_state == "absent":
            namespace.pop(alias)
        elif alias_state == "rebound":
            namespace[alias] = types.ModuleType("foreign_hook_module")
        module._backward_hooks.clear()
        steady = counter.frame_count
        for index, size in enumerate((8, 11, 15)):
            calls.clear()
            hook_calls.clear()
            self.assertTrue(root.check({"x": x, "module": module}))
            self.compare(hook_outer, compiled, device, (module,), size=size)
            self.assertEqual(calls, [])
            self.assertEqual(hook_calls, [])
            if index == 0:
                steady = counter.frame_count
            self.assertEqual(counter.frame_count, steady)
            self.assertEqual(len(_debug_get_cache_entry_list(hook_outer.__code__)), 1)
            self.assertFalse(whole_region_captured(counter))
        self.assertEqual(
            [
                {
                    node.target
                    for node in graph.graph.nodes
                    if node.op == "call_function"
                }
                for graph in counter.graphs[3:]
            ],
            [{torch.sin, torch.cos}, {operator.mul}],
        )
        namespace.pop(key)
        calls.clear()
        self.assertFalse(root.check({"x": x, "module": module}))
        if alias_state == "rebound":
            namespace[alias] = saved
        for size in (8, 11, 15):
            self.compare(hook_outer, compiled, device, (module,), size=size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady + 1)
            self.assertEqual(len(_debug_get_cache_entry_list(hook_outer.__code__)), 2)
            self.assertTrue(whole_region_captured(counter))

    @parametrize("dictionary_protocol", ("same", "different", "raising"))
    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_same_globals_protocol_preserves_eager_call(
        self, device, dictionary_protocol
    ):
        module, calls, graphs, executed = HookModule().to(device), [], [], []

        def hook(module, grad_input, grad_output):
            calls.append("hook")
            return tuple(
                value * 3 if value is not None else None for value in grad_input
            )

        handle = module.register_full_backward_hook(hook)
        self.addCleanup(handle.remove)

        def backend(graph, inputs):
            index = len(graphs)
            graphs.append(graph)

            def forward(*args):
                executed.append(index)
                return graph.forward(*args)

            return forward

        def fn(x, module):
            return module(x + 1) * 2

        compiled = torch.compile(fn, backend=backend, dynamic=True)
        x = self.compare_effects(fn, compiled, device, calls, (module,))
        self.assertEqual(calls, ["hook"])
        self.assertEqual(len(graphs), 3)
        root = self.native_guards(fn)
        scope = {"x": x, "module": module}
        self.assertIn("___nn_module_backward_hooks", str(root))
        globals_module = torch.nn.modules.module
        globals_type = type(globals_module)
        getter = types.ModuleType.__dict__["__dict__"].__get__
        namespace = getter(globals_module)

        class GlobalShadow(globals_type):
            @property
            def __dict__(self):
                calls.append("global_dict")
                if dictionary_protocol == "raising":
                    raise RuntimeError("Unexpected globals dictionary lookup")
                if dictionary_protocol == "different":
                    return {}
                return getter(self)

        try:
            object.__setattr__(globals_module, "__class__", GlobalShadow)
            self.assertIs(getter(globals_module), namespace)
            for size in (8, 11, 15):
                calls.clear()
                accepted = root.check(scope)
                self.assertEqual(calls, [])
                executed.clear()
                self.compare_effects(fn, compiled, device, calls, (module,), size)
                self.assertEqual(calls, ["hook"])
                self.assertTrue(accepted)
                self.assertEqual(executed, [0, 1, 2])
                self.assertEqual(len(graphs), 3)

            handle.remove()
            for index, size in enumerate((8, 11, 15)):
                calls.clear()
                self.assertTrue(root.check(scope))
                self.assertEqual(calls, [])
                executed.clear()
                self.compare_effects(fn, compiled, device, calls, (module,), size)
                self.assertEqual(calls, [])
                self.assertEqual(len(executed), 3)
                if index == 0:
                    steady = len(graphs)
                self.assertEqual(len(graphs), steady)
                self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 1)
        finally:
            object.__setattr__(globals_module, "__class__", globals_type)

        self.assertFalse(root.check(scope))
        for size in (8, 11, 15):
            calls.clear()
            executed.clear()
            self.compare_effects(fn, compiled, device, calls, (module,), size)
            self.assertEqual(calls, [])
            self.assertEqual(len(graphs), steady + 1)
            self.assertEqual(executed, [steady])
        self.assertEqual(
            {
                node.target
                for node in graphs[-1].graph.nodes
                if node.op == "call_function"
            },
            {operator.add, torch.sin, torch.cos, operator.mul},
        )

    @parametrize("helper_name", ("_get_backward_hooks", "_get_backward_pre_hooks"))
    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_hook_guard_late_mro_namespace_is_pure(self, device, helper_name):
        class HeapHook(HookModule):
            pass

        module, calls = HeapHook().to(device), []

        def hook(module, grad_input, grad_output):
            calls.append("hook")
            return tuple(
                value * 3 if value is not None else None for value in grad_input
            )

        def pre_hook(module, grad_output):
            calls.append("hook")
            return tuple(
                value * 3 if value is not None else None for value in grad_output
            )

        handle = (
            module.register_full_backward_hook(hook)
            if helper_name == "_get_backward_hooks"
            else module.register_full_backward_pre_hook(pre_hook)
        )
        self.addCleanup(handle.remove)
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(hook_outer, backend=counter, dynamic=True)
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        self.assertEqual(compiled(x, module), hook_outer(x, module))
        root = self.native_guards(hook_outer)
        self.assertIn("___nn_module_backward_hooks", str(root))
        self.assertTrue(root.check({"x": x, "module": module}))

        class Name(str):
            __slots__ = ()
            __hash__ = str.__hash__

            def __eq__(self, other):
                calls.append("eq")
                return str.__eq__(self, other)

        helper = getattr(torch.nn.Module, helper_name)
        base = type("CompatibleBase", (HookModule,), {Name(helper_name): helper})
        self.assertTrue(any(type(key) is Name for key in base.__dict__))
        HeapHook.__bases__ = (base,)
        self.assertIs(type(module), HeapHook)
        self.assertIs(inspect.getattr_static(HeapHook, helper_name), helper)
        calls.clear()
        self.assertTrue(root.check({"x": x, "module": module}))
        self.assertEqual(calls, [])
        steady = counter.frame_count

        def invalidate():
            HeapHook.unrelated = object()

        for size in (8, 11, 15):
            self.compare_effects(
                hook_outer, compiled, device, calls, (module,), size, invalidate
            )
            self.assertEqual(calls, ["eq", "hook"])
            self.assertEqual(counter.frame_count, steady)
        handle.remove()
        for index, size in enumerate((8, 11, 15)):
            calls.clear()
            self.assertTrue(root.check({"x": x, "module": module}))
            self.assertEqual(calls, [])
            self.compare_effects(
                hook_outer, compiled, device, calls, (module,), size, invalidate
            )
            self.assertEqual(calls, [])
            if index == 0:
                steady = counter.frame_count
            self.assertEqual(counter.frame_count, steady)
            self.assertEqual(len(_debug_get_cache_entry_list(hook_outer.__code__)), 1)
            self.assertFalse(whole_region_captured(counter))
        HeapHook.__bases__ = (HookModule,)
        self.assertFalse(root.check({"x": x, "module": module}))
        for size in (8, 11, 15):
            self.compare_effects(hook_outer, compiled, device, calls, (module,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady + 1)
            self.assertTrue(whole_region_captured(counter))

    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_hook_free_fullgraph_with_benign_global_key(self, device):
        module = HookModule().to(device)
        namespace = hook_outer.__globals__
        namespace[37] = None
        self.addCleanup(namespace.pop, 37)
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(
            hook_outer, backend=counter, dynamic=True, fullgraph=True
        )
        for _ in range(3):
            self.compare(hook_outer, compiled, device, (module,))
            self.assertEqual(counter.frame_count, 1)

    @parametrize("producer", ("property", "helper", "receiver_helper"))
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_returned_bool_is_not_a_direct_flag(self, device, producer):
        class PropertyState:
            def __init__(self):
                self.stored = False

            @property
            def initialized(self):
                return self.stored

        state = PropertyState() if producer == "property" else InitializationState()

        def body(x, state):
            if producer == "helper":
                if not read_initialized(state):
                    torch._dynamo.graph_break()
            elif producer == "receiver_helper":
                if not read_state(state).initialized:
                    torch._dynamo.graph_break()
            elif not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for _ in range(4):
            self.compare(fn, compiled, device, (state,))
            frames.append(counter.frame_count)
        self.assertEqual(frames, [2] * 4)
        self.assertFalse(checkpoint_captured(counter))
        self.assertNotIn("___instance_dict_bool", str(self.native_guards(fn)))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_local_only_state_is_not_retained(self, device):
        def body(x):
            state = InitializationState()
            if not state.initialized:
                torch._dynamo.graph_break()
                state.initialized = True
            return torch.cos(torch.sin(x))

        def fn(x):
            return checkpoint(body, x + 1, use_reentrant=False) * 2

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for _ in range(4):
            self.compare(fn, compiled, device)
            frames.append(counter.frame_count)
        self.assertEqual(frames, [2] * 4)
        self.assertFalse(checkpoint_captured(counter))

    @parametrize("mutation", ("receiver", "namespace_alias"))
    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_pending_receiver_write_is_not_retained(self, device, mutation):
        def body(x, state):
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            if mutation == "receiver":
                state.unrelated = True
            else:
                alias = state.__dict__
                alias["unrelated"] = True
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        state = InitializationState()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        for _ in range(4):
            self.compare(fn, compiled, device, (state,))
        self.assertEqual(counter.frame_count, 2)
        self.assertTrue(state.unrelated)
        self.assertFalse(checkpoint_captured(counter))
        self.assertNotIn("___instance_dict_bool", str(self.native_guards(fn)))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_actual_raw_flag_guard_protocol_and_lifetime(self, device):
        class State:
            def __init__(self):
                self.initialized = False

        def body(x, state):
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        state, calls = State(), []
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        result = compiled(x, state)
        self.assertEqual(result, 2 * torch.cos(torch.sin(x + 1)))
        root = self.native_guards(fn)
        self.assertIn("initialized", str(root))
        scope = self.frame_locals(fn, x=x, state=state)
        self.assertTrue(root.check(scope))

        def getter(self, name):
            calls.append(name)
            return (
                True if name == "initialized" else object.__getattribute__(self, name)
            )

        State.__getattribute__ = getter
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        steady = counter.frame_count
        for size in (8, 11, 15):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls.count("initialized"), 2)
            self.assertEqual(counter.frame_count, steady)
        storage = object.__getattribute__(state, "__dict__")
        storage["__dict__"] = {"initialized": True}
        calls.clear()
        self.assertTrue(root.check(scope))
        storage["initialized"] = True
        self.assertTrue(root.check(scope))
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(calls.count("initialized"), 2)
        calls.clear()
        storage.pop("initialized")
        self.assertTrue(root.check(scope))
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(calls.count("initialized"), 2)
        storage["initialized"] = False
        calls.clear()
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        State.__getattribute__ = object.__getattribute__
        storage.pop("initialized")
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        reason = "has no attribute 'initialized'"
        for invoke in (compiled, fn):
            with self.assertRaisesRegex(AttributeError, reason):
                invoke(x, state)
        self.assertEqual(counter.frame_count, steady)
        storage["initialized"] = False

        class ForeignBool:
            def __bool__(self):
                calls.append("bool")
                return False

            def __eq__(self, other):
                calls.append("eq")
                return False

        storage["initialized"] = ForeignBool()
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        self.compare_effects(fn, compiled, device, calls, (state,))
        self.assertEqual(calls, ["bool", "bool"])
        self.assertEqual(counter.frame_count, steady)
        storage["initialized"] = False

        class Shadow:
            @property
            def __dict__(self):
                calls.append("shadow")
                return {"initialized": False}

        object.__setattr__(state, "__class__", Shadow)
        calls.clear()
        self.assertFalse(root.check(scope))
        self.assertEqual(calls, [])
        object.__setattr__(state, "__class__", State)
        storage["initialized"] = True
        self.assertFalse(root.check(scope))
        for size in (8, 11, 15):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady + 1)
            self.assertTrue(checkpoint_captured(counter))
        State.receiver = state
        reference = weakref.ref(state)
        class_reference = weakref.ref(State)
        scope.clear()
        del state, State, result, x
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(class_reference())

    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_actual_hook_guard_namespace_and_bindings(self, device):
        from torch._dynamo.guards import _nn_module_backward_hooks_guard

        class HeapHook(HookModule):
            pass

        module, calls = HeapHook().to(device), []
        handle = module.register_full_backward_hook(lambda *args: None)
        self.addCleanup(handle.remove)
        counter = CompileCounterWithBackend("eager")
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        compiled = torch.compile(hook_outer, backend=counter, dynamic=True)
        result = compiled(x, module)
        self.assertEqual(result, 2 * torch.cos(torch.sin(x + 1)))
        root = self.native_guards(hook_outer)
        self.assertIn("___nn_module_backward_hooks", str(root))
        scope = {"x": x, "module": module}
        self.assertTrue(root.check(scope))

        class Namespace(dict):
            def __iter__(self):
                calls.append("iter")
                return super().__iter__()

            def get(self, *args):
                calls.append("get")
                return super().get(*args)

            def __bool__(self):
                calls.append("bool")
                return True

            def __len__(self):
                calls.append("len")
                return super().__len__()

        namespace = object.__getattribute__(module, "__dict__")
        object.__setattr__(module, "__dict__", Namespace(namespace))
        calls.clear()
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        steady = counter.frame_count
        self.compare_effects(hook_outer, compiled, device, calls, (module,))
        self.assertEqual(counter.frame_count, steady)
        object.__setattr__(module, "__dict__", namespace)
        original = namespace["_backward_hooks"]
        namespace["_backward_hooks"] = Namespace(original)
        calls.clear()
        self.assertTrue(root.check(scope))
        self.assertEqual(calls, [])
        self.compare_effects(hook_outer, compiled, device, calls, (module,))
        self.assertEqual(counter.frame_count, steady)
        namespace["_backward_hooks"] = original
        calls.clear()
        self.assertTrue(root.check(scope))

        globals_module = torch.nn.modules.module
        globals_type = type(globals_module)
        with self.assertRaisesRegex(AttributeError, "readonly attribute"):
            object.__setattr__(
                globals_module, "__dict__", Namespace(globals_module.__dict__)
            )
        self.assertEqual(calls, [])

        class GlobalShadow(globals_type):
            @property
            def __dict__(self):
                calls.append("global_dict")
                return Namespace()

        try:
            object.__setattr__(globals_module, "__class__", GlobalShadow)
            self.assertTrue(root.check(scope))
            self.assertEqual(calls, [])
            self.assertIsNone(_nn_module_backward_hooks_guard()(module, globals_module))
            self.assertEqual(calls, [])
            globals_namespace = hook_outer.__globals__
            alias = "__import_torch_dot_nn_dot_modules_dot_module"
            saved_alias = globals_namespace[alias]
            try:
                globals_namespace[alias] = None
                self.assertFalse(root.check(scope))
                self.assertEqual(calls, [])
                globals_namespace[alias] = GlobalShadow("rebound_retry_globals")
                self.assertFalse(root.check(scope))
                self.assertEqual(calls, [])
            finally:
                globals_namespace[alias] = saved_alias
        finally:
            object.__setattr__(globals_module, "__class__", globals_type)

        temporary = types.ModuleType("retry_globals_lifetime")
        reference = weakref.ref(temporary)
        check = _nn_module_backward_hooks_guard(temporary)
        del temporary
        gc.collect()
        self.assertIsNone(reference())
        self.assertIs(check(module, None), False)
        self.assertIs(check(module, globals_module), False)
        self.assertEqual(calls, [])

        class BindingMeta(type):
            @property
            def __dict__(cls):
                calls.append("binding_dict")
                return {}

        class ForeignBinding(metaclass=BindingMeta):
            pass

        with (
            mock.patch.object(torch.nn, "Module", ForeignBinding),
            mock.patch.object(types, "ModuleType", ForeignBinding),
            mock.patch.object(collections, "OrderedDict", ForeignBinding),
        ):
            self.assertTrue(root.check(scope))
            self.assertEqual(calls, [])
        HeapHook.receiver = module
        reference, class_reference = weakref.ref(module), weakref.ref(HeapHook)
        scope.clear()
        del module, HeapHook, result, x
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(class_reference())

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_global_stored_flag_and_root_namespace(self, device):
        def fn(x):
            return checkpoint(global_initialize, x + 1, use_reentrant=False) * 2

        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        with mock.patch.dict(globals(), global_state=state):
            for index, size in enumerate((8, 8, 11, 15)):
                x = torch.randn(
                    size, device=device, dtype=torch.double, requires_grad=True
                )
                eager_x = x.detach().clone().requires_grad_()
                actual = compiled(x)
                with mock.patch.dict(globals(), global_state=eager_state):
                    expected = fn(eager_x)
                    expected_grad = torch.autograd.grad(expected.sum(), eager_x)[0]
                self.assertEqual(actual, expected)
                self.assertEqual(torch.autograd.grad(actual.sum(), x)[0], expected_grad)
                self.assertEqual(state.calls, eager_state.calls)
                self.assertEqual(state.calls, 1)
                self.assertEqual(checkpoint_captured(counter), index != 0)
                self.assertEqual(counter.frame_count, 2 if index == 0 else 3)

        torch._dynamo.reset()
        state, calls = InitializationState(), []
        with mock.patch.dict(globals(), global_state=state):
            compiled(x)
            root = self.native_guards(fn)
            self.assertIn("___instance_dict_globals_str_keys(G)", str(root))

            class ForeignKey:
                def __hash__(self):
                    return hash("global_state")

                def __eq__(self, other):
                    calls.append("eq")
                    return False

            key = ForeignKey()
            globals()[key] = 1
            try:
                calls.clear()
                self.assertEqual((root.check({"x": x}), calls), (True, []))
                steady = counter.frame_count
                for initialized, size in ((False, 8), (True, 11), (False, 15)):
                    state.initialized = eager_state.initialized = initialized
                    state.calls = eager_state.calls = 0
                    x = torch.randn(
                        size, device=device, dtype=torch.double, requires_grad=True
                    )
                    eager_x = x.detach().clone().requires_grad_()
                    calls.clear()
                    actual = compiled(x)
                    actual_grad = torch.autograd.grad(actual.sum(), x)[0]
                    actual_calls = list(calls)
                    with mock.patch.dict(globals(), global_state=eager_state):
                        calls.clear()
                        expected = fn(eager_x)
                        expected_grad = torch.autograd.grad(expected.sum(), eager_x)[0]
                        expected_calls = list(calls)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual_grad, expected_grad)
                    self.assertEqual(actual_calls, expected_calls)
                    self.assertEqual(state.calls, eager_state.calls)
                    self.assertEqual(state.calls, int(not initialized))
                    self.assertEqual(counter.frame_count, steady)
            finally:
                globals().pop(key)
            calls.clear()
            self.assertFalse(root.check({"x": x}))
            for size in (8, 11, 15):
                x = torch.randn(
                    size, device=device, dtype=torch.double, requires_grad=True
                )
                self.assertEqual(compiled(x), fn(x))
                self.assertEqual(counter.frame_count, steady + 1)
                self.assertTrue(checkpoint_captured(counter))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_direct_global_bool_is_deferred(self, device):
        def fn(x):
            return checkpoint(global_bool_body, x + 1, use_reentrant=False) * 2

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        with mock.patch.dict(globals(), global_initialized=False):
            for initialized in (False, False, True, True):
                globals()["global_initialized"] = initialized
                self.compare(fn, compiled, device)
                self.assertEqual(counter.frame_count, 2)
                self.assertFalse(checkpoint_captured(counter))
        self.assertNotIn("G['global_initialized']", str(self.native_guards(fn)))

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_getter_receiver_is_not_retained(self, device):
        class Owner:
            def __init__(self):
                self._state = InitializationState()
                self.calls = 0

            @property
            def state(self):
                self.calls += 1
                return self._state

        def body(x, owner):
            if not owner.state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, owner):
            return checkpoint(body, x + 1, owner, use_reentrant=False) * 2

        owner, eager_owner = Owner(), Owner()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        for index in range(4):
            self.compare(fn, compiled, device, (owner,), (eager_owner,))
            self.assertEqual(owner.calls, eager_owner.calls)
            self.assertEqual(owner.calls, 2 * (index + 1))
        self.assertEqual(counter.frame_count, 2)
        self.assertFalse(checkpoint_captured(counter))
        self.assertNotIn("___instance_dict_bool", str(self.native_guards(fn)))

    @torch._dynamo.config.patch(nested_graph_breaks=False, compiled_autograd=False)
    def test_pending_hook_registry_alias_is_not_retained(self, device):
        module = HookModule().to(device)

        def hook(module, grad_input, grad_output):
            return None

        handle = module.register_full_backward_hook(hook)
        self.addCleanup(handle.remove)

        def fn(x, module):
            alias = module._backward_hooks
            alias[12345] = hook
            return module(x + 1) * 2

        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for _ in range(4):
            self.compare(fn, compiled, device, (module,))
            self.assertEqual(len(module._backward_hooks), 2)
            frames.append(counter.frame_count)
        self.assertEqual(frames[1:], [frames[1]] * 3)
        self.assertFalse(whole_region_captured(counter))
        self.assertNotIn("___nn_module_backward_hooks", str(self.native_guards(fn)))

    @parametrize("checkpointed", (False, True))
    @parametrize("realized", (False, True))
    def test_direct_backward_hook_constructor_is_constant_fallback(
        self, device, checkpointed, realized
    ):
        module = HookModule().to(device)
        handle = module.register_full_backward_hook(lambda *args: None)
        self.addCleanup(handle.remove)

        def body(x, module):
            if realized and module.training:
                x = x + 0
            torch.utils.hooks.BackwardHook(module, [], [])
            return torch.cos(torch.sin(x))

        def fn(x, module):
            if checkpointed:
                return checkpoint(body, x + 1, module, use_reentrant=False) * 2
            return body(x + 1, module) * 2

        counter = CompileCounterWithBackend("eager")
        with torch._dynamo.config.patch(
            nested_graph_breaks=checkpointed, compiled_autograd=False
        ):
            self.assertFalse(compiled_autograd.compiled_autograd_enabled)
            compiled = torch.compile(fn, backend=counter, dynamic=True)
            frames = []
            for index in range(5):
                self.compare(fn, compiled, device, (module,))
                frames.append(counter.frame_count)
                if index == 1:
                    handle.remove()
            self.assertEqual(frames, [frames[0]] * 5)
            self.assertFalse(checkpoint_captured(counter))
            self.assertFalse(whole_region_captured(counter))
            self.assertNotIn("___nn_module_backward_hooks", str(self.native_guards(fn)))
            torch._dynamo.reset()
            fullgraph = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "Module-level backwards hooks require compiled autograd",
            ):
                fullgraph(x, module)

    @torch._dynamo.config.patch(nested_graph_breaks=True)
    def test_raw_flag_guard_foreign_namespace_key(self, device):
        def body(x, state):
            if not state.initialized:
                torch._dynamo.graph_break()
            return torch.cos(torch.sin(x))

        def fn(x, state):
            return checkpoint(body, x + 1, state, use_reentrant=False) * 2

        state, calls = InitializationState(), []
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        self.assertEqual(compiled(x, state), 2 * torch.cos(torch.sin(x + 1)))
        root = self.native_guards(fn)
        scope = self.frame_locals(fn, x=x, state=state)
        self.assertTrue(root.check(scope))

        class ForeignKey:
            def __hash__(self):
                return hash("initialized")

            def __eq__(self, other):
                calls.append("eq")
                return False

        object.__setattr__(state, "__dict__", {ForeignKey(): 1, "initialized": False})
        calls.clear()
        self.assertEqual((root.check(scope), calls), (True, []))
        steady = counter.frame_count

        def getter(receiver, name):
            if name == "initialized":
                namespace = object.__getattribute__(receiver, "__dict__")
                return dict.__getitem__(namespace, name)
            return object.__getattribute__(receiver, name)

        with mock.patch.object(InitializationState, "__getattribute__", getter):
            for initialized in (False, True, False):
                state.initialized = initialized
                calls.clear()
                self.assertEqual((root.check(scope), calls), (True, []))
                self.compare_effects(fn, compiled, device, calls, (state,))
                self.assertEqual(calls, ["eq", "eq"])
                self.assertEqual(counter.frame_count, steady)

        class Namespace(dict):
            def keys(self):
                calls.append("keys")
                return super().keys()

            def __iter__(self):
                calls.append("iter")
                return super().__iter__()

        object.__setattr__(state, "__dict__", Namespace(initialized=False))
        calls.clear()
        self.assertEqual((root.check(scope), calls), (True, []))
        for size in (8, 11, 15):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady)
        object.__setattr__(state, "__dict__", {"initialized": True})
        self.assertFalse(root.check(scope))
        for size in (8, 11, 15):
            self.compare_effects(fn, compiled, device, calls, (state,), size)
            self.assertEqual(calls, [])
            self.assertEqual(counter.frame_count, steady + 1)
            self.assertTrue(checkpoint_captured(counter))

    @torch._dynamo.config.patch(
        nested_graph_breaks=False,
        compiled_autograd=False,
        inline_single_use_invoke_subgraph=False,
    )
    def test_successful_nested_region_with_failed_call(self, device):
        @torch.compiler.nested_compile_region
        def region(x):
            return x + 0

        def fn(x, module):
            x = region(x)
            return region(module(x + 1)) * 2

        module = HookModule().to(device)
        handle = module.register_full_backward_hook(lambda *args: None)
        self.addCleanup(handle.remove)
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(fn, backend=counter, dynamic=True)
        frames = []
        for _ in range(4):
            self.compare(fn, compiled, device, (module,))
            frames.append(counter.frame_count)
        self.assertEqual(frames[1:], [frames[1]] * 3)
        self.assertFalse(whole_region_captured(counter))
        self.assertTrue(
            any(
                node.target is torch.ops.higher_order.invoke_subgraph
                for graph in counter.graphs
                for node in graph.graph.nodes
            )
        )
        handle.remove()
        self.compare(fn, compiled, device, (module,))
        self.assertTrue(whole_region_captured(counter))

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_fullgraph_cold_and_hot(self, device):
        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        compiled = torch.compile(
            initialize_outer, backend=counter, fullgraph=True, dynamic=True
        )
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"Call to `torch\._dynamo\.graph_break\(\)`",
        ):
            compiled(x, state)
        self.assertFalse(state.initialized)
        self.assertEqual(state.calls, 0)
        state.initialized = eager_state.initialized = True
        self.compare(initialize_outer, compiled, device, (state,), (eager_state,))
        self.assertEqual(counter.frame_count, 1)

    def test_direct_capture_context(self, device):
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import get_metrics_context

        state = InitializationState()
        x = torch.randn(8, device=device, dtype=torch.double, requires_grad=True)
        with (
            get_metrics_context(),
            self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                r"Call to `torch\._dynamo\.graph_break\(\)`",
            ),
        ):
            fullgraph_capture(initialize_outer, (x, state))
        state.initialized = True
        with get_metrics_context():
            captured = fullgraph_capture(initialize_outer, (x, state))
        compiled = captured.forward_callable()
        self.compare(initialize_outer, compiled, device, (state,))

    @parametrize("caching", (False, True))
    @torch._dynamo.config.patch(nested_graph_breaks=False, strict_precompile=True)
    def test_package_context(self, device, caching):
        from torch._dynamo.package import CompilePackage

        state, eager_state = InitializationState(), InitializationState()
        counter = CompileCounterWithBackend("eager")
        package = CompilePackage(initialize_outer)
        with torch._dynamo.config.patch(caching_precompile=caching):
            compiled = torch._dynamo.optimize(counter, package=package)(
                initialize_outer
            )
            for _ in range(3):
                self.compare(
                    initialize_outer, compiled, device, (state,), (eager_state,)
                )
        self.assertEqual(state.calls, 1)
        self.assertFalse(whole_region_captured(counter))
        self.assertNotIn(
            "___instance_dict_bool",
            str(self.native_guards(initialize_outer)),
        )


class GraphBreakRetryColdTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "method",
        (
            "_get_backward_hooks",
            "_get_backward_pre_hooks",
            "_call_impl",
            "module_alias",
        ),
    )
    def test_lazy_compile_after_method_replacement(self, method):
        code = textwrap.dedent(
            """
            import sys
            import types
            from unittest import mock

            import torch

            method = sys.argv.pop(1)
            if "torch._dynamo.guards" in sys.modules or torch.__spec__._initializing:
                raise AssertionError("Expected completed Torch import without Dynamo guards")

            class Forwarder:
                def __init__(self, function):
                    self.function = function

                def __get__(self, receiver, owner=None):
                    return self if receiver is None else types.MethodType(self, receiver)

                def __call__(self, receiver, *args, **kwargs):
                    return self.function(receiver, *args, **kwargs)

            class AddOne(torch.nn.Module):
                def forward(self, x):
                    return x + 1

            graphs = []

            def backend(graph, inputs):
                graphs.append(graph)
                return graph.forward

            def fn(x):
                return x.sin() * 2

            if method == "module_alias":
                patch = mock.patch.object(torch.nn, "Module", AddOne)
            else:
                replacement = Forwarder(getattr(torch.nn.Module, method))
                patch = mock.patch.object(torch.nn.Module, method, replacement)
            with patch:
                receiver = AddOne()
                if method == "module_alias" and "__dict__" in vars(torch.nn.Module):
                    raise AssertionError("Expected an inherited instance dictionary descriptor")
                if method == "_get_backward_hooks":
                    receiver.register_full_backward_hook(lambda *args: None)
                    if [len(value) for value in receiver._get_backward_hooks()] != [1, 0]:
                        raise AssertionError("Expected faithful full backward hook collection")
                elif method == "_get_backward_pre_hooks":
                    receiver.register_full_backward_pre_hook(lambda *args: None)
                    if len(receiver._get_backward_pre_hooks()) != 1:
                        raise AssertionError("Expected faithful pre-hook collection")
                else:
                    producer_x = torch.arange(8, dtype=torch.double)
                    if not torch.equal(receiver(producer_x), producer_x + 1):
                        raise AssertionError("Expected faithful eager Module call")
                if "torch._dynamo.guards" in sys.modules:
                    raise AssertionError("Eager producer unexpectedly loaded Dynamo guards")
                compiled = torch.compile(fn, backend=backend, fullgraph=True)

                from torch.testing._internal.common_device_type import instantiate_device_type_tests
                from torch.testing._internal.common_utils import HardwareClassification, run_tests, TestCase

                class ColdCompileTests(TestCase):
                    hw_classification = HardwareClassification.ACCELERATOR

                    def test_unrelated_compilation(self, device):
                        x = torch.arange(8, device=device, dtype=torch.double).requires_grad_()
                        eager_x = x.detach().clone().requires_grad_()
                        before = len(graphs)
                        actual, expected = compiled(x), fn(eager_x)
                        self.assertEqual(actual, expected)
                        self.assertEqual(
                            torch.autograd.grad(actual.sum(), x)[0],
                            torch.autograd.grad(expected.sum(), eager_x)[0],
                        )
                        self.assertEqual(len(graphs), before + 1)
                        self.assertEqual(compiled(x), expected)
                        self.assertEqual(len(graphs), before + 1)

                instantiate_device_type_tests(ColdCompileTests, globals(), except_for=("mps",))
                run_tests()
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code, method], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


instantiate_device_type_tests(GraphBreakRetryTests, globals(), except_for=("mps",))
instantiate_parametrized_tests(GraphBreakRetryColdTests)

if __name__ == "__main__":
    run_tests()
