# Owner(s): ["module: dynamo"]

import contextlib
import copy
import enum
import functools
import operator
import unittest
from collections import defaultdict, OrderedDict, UserDict
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.utils._pytree as pytree
from functorch.compile import nop
from torch._dynamo import compiled_autograd
from torch._dynamo.comptime import comptime
from torch._dynamo.convert_frame import fullgraph_capture, get_traced_fn
from torch._dynamo.exc import Unsupported
from torch._dynamo.guards import GuardBuilder
from torch._dynamo.source import LocalSource
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._dynamo.variables import higher_order_ops
from torch._dynamo.variables.hashable import HashableTracker
from torch._dynamo.variables.lazy import LazyVariableTracker
from torch._dynamo.variables.object_protocol import generic_getattr
from torch._functorch.aot_autograd import aot_module_simplified
from torch.nn.modules import module as hook_api
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
    preserves_device_selection,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.utils.hooks import RemovableHandle


def compiler_fn(gm):
    return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)


def global_hook_0(grad):
    return grad * 4


def global_hook_1(grad):
    return grad / 2


def global_hook_2(grad):
    return grad * 3


h0 = None
_global_hook_flags_alias = None
global_post_hooks = None


def identity_forward_hook(module, args, output):
    return output


def add_one(module, args, *rest):
    return rest[-1] + 1


def add_two(module, args, output):
    return output + 2


def hook_arity(module, args, *rest):
    return rest[-1] * 10 + len(rest)


@torch.compiler.nested_compile_region
def call_hooked_module(x, module):
    return module(x)


def count_subgraph_traces():
    speculate = higher_order_ops.speculate_subgraph_with_auto_output_flattening
    return patch.object(higher_order_ops, speculate.__name__, wraps=speculate)


class HookKeyFrozenSet(frozenset):
    pass


@torch.compiler.assume_constant_result
def sum_hook_keys(hooks):
    return sum(hooks)


def update_hook_keys(hooks):
    result = {}
    result.update(hooks)
    return 1 in result


class ClassWithVal:
    def __init__(self, val):
        self.val = val


_mixed_hook_module = torch.nn.Identity()


@instantiate_parametrized_tests
class HookDictionaryTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @staticmethod
    def compile_counted(fn, record=None):
        counter = torch._dynamo.testing.CompileCounter()
        opt = torch._dynamo.optimize(counter, nopython=True, guard_export_fn=record)
        return opt(fn), counter

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("subclass", [False, True])
    @parametrize(
        "operation",
        ["forward", "forward_kwargs", "call_values", "unpack", "empty_unpack"]
        + ["keys", "values", "items", "len_keys"],
    )
    def test_hook_dict_iteration_order(self, mapping_type, operation, subclass):
        kind = type("HookDict", (mapping_type,), {}) if subclass else mapping_type

        def fn(x, module):
            if operation in ("forward", "forward_kwargs"):
                return module(x)
            if operation in ("call_values", "unpack", "empty_unpack"):
                hooks = module._forward_hooks
                values = hooks.values()
                if operation != "call_values":
                    values = [*values]
                for hook in values:
                    x = hook(module, (x,), x)
                if operation == "call_values":
                    return x
                return x + 1, [*hooks], [*hooks.items()]
            if operation == "len_keys":
                return x + 1, len(module._forward_hooks.keys())
            view = getattr(module._forward_hooks, operation)()
            return x + len(view), type(view), list(view)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        callbacks = () if operation == "empty_unpack" else (add_one, hook_arity)
        with_kwargs = operation == "forward_kwargs"
        for _ in range(2):
            module = torch.nn.Identity()
            for callback in callbacks:
                module.register_forward_hook(callback, with_kwargs=with_kwargs)
            module._forward_hooks = kind(module._forward_hooks)
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 1)
        if operation == "empty_unpack":
            return
        hooks = module._forward_hooks
        if mapping_type is OrderedDict:
            hooks.move_to_end(next(iter(hooks)))
        else:
            module._forward_hooks = kind(reversed(hooks.items()))
        self.assertEqual(compiled(x, module), fn(x, module))
        # Plain keys retain storage positions, so returned IDs need no order specialization.
        reusable = operation == "len_keys" or (
            operation == "keys" and mapping_type is dict
        )
        self.assertEqual(counter.frame_count, 1 if reusable else 2)
        if operation == "forward" and kind is OrderedDict:
            module.register_forward_pre_hook(lambda m, args: args)
            dict.clear(hooks)
            with self.assertRaisesRegex(
                Unsupported, "storage differs from linked keys"
            ):
                compiled(x, module)
        if operation == "forward" and kind is dict:
            for callback in (add_one, add_two):
                module._forward_hooks = UserDict({1: callback})
                self.assertEqual(compiled(x, module), module(x))
            self.assertEqual(counter.frame_count, 4)

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("populated", [False, True])
    @parametrize("mutation", ["attribute", "content"])
    def test_hook_subclass_attribute_mutation(self, mapping_type, populated, mutation):
        class Hooks(mapping_type):
            marker = 3

        def fn(x, module):
            hooks = module._forward_hooks
            if mutation == "attribute":
                previous = hooks.marker
                hooks.marker = x + 2
                return module(x) + previous, hooks.marker
            hooks[1] = hook_arity
            return module(x) + 1, hooks[1] is hook_arity

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)

        def run(call, marker):
            module = torch.nn.Identity()
            hooks = Hooks({1: identity_forward_hook} if populated else {})
            module._forward_hooks = hooks
            hooks.marker = marker
            return call(x, module), hooks.marker, hooks.get(1) is hook_arity

        for marker in (3, 3, 7):
            self.assertEqual(run(compiled, marker), run(fn, marker))
        self.assertEqual(counter.frame_count, 2 if mutation == "attribute" else 1)

    @parametrize("kind", [dict, OrderedDict])
    @parametrize("spoof", [False, True])
    @parametrize("method", ["__len__", "__bool__"])
    def test_empty_hook_subclass_truth_protocol(self, kind, spoof, method):
        class Meta(type):
            __hash__ = type.__hash__

            def __eq__(cls, rhs):
                return cls is rhs or (spoof and (rhs is dict or rhs is OrderedDict))

        class Hooks(kind, metaclass=Meta):
            def items(self):
                return [(1, add_two)]

        setattr(Hooks, method, lambda self: True)
        module = torch.nn.Identity()
        module._forward_hooks = Hooks()
        compiled = torch.compile(lambda x, m: m(x), backend="eager", fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x, module), x + 2)

    @torch._dynamo.config.patch(skip_nnmodule_hook_guards=False, compiled_autograd=True)
    def test_empty_backward_hook_subclass_guard(self):
        class Hooks(dict):
            def __bool__(self):
                return True

            def values(self):
                return [lambda module, grads: (3 * grads[0],)]

        module = torch.nn.Identity()
        x = torch.ones(1, requires_grad=True)
        fn, counter = self.compile_counted(lambda x, m: 2 * m(x))
        for hooks, multiplier in (({}, 2), (Hooks(), 6)):
            module._backward_pre_hooks = hooks
            (grad,) = torch.autograd.grad(fn(x, module).sum(), x)
            self.assertEqual(grad, x * multiplier)
        self.assertEqual(counter.frame_count, 2)

    _hook_copies = (lambda h: h.copy(), lambda h: h | {}, lambda h: {} | h)

    @parametrize(
        "kind, copy_fn",
        [(kind, fn) for fn in _hook_copies for kind in (dict, OrderedDict)]
        + [(None, _hook_copies[1])],
    )
    def test_hook_subclass_detached_keys(self, kind, copy_fn):
        base = kind or OrderedDict

        class Hooks(base):
            if kind is None:
                __new__ = staticmethod(lambda cls, *args, **kwargs: [])

        def fn(x, module):
            result = copy_fn(module._forward_hooks)
            return x + (1 in result), type(result), tuple(result)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=kind is dict)
        x, module = torch.ones(1), torch.nn.Identity()
        for keys in ((2, 3), (1, 3)):
            hooks = base.__new__(Hooks)
            base.update(hooks, dict.fromkeys(keys, identity_forward_hook))
            module._forward_hooks = hooks
            if base is OrderedDict:
                torch._dynamo.reset()
                with self.assertRaisesRegex(Unsupported, "subclass allocation"):
                    torch.compile(fn, backend="eager", fullgraph=True)(x, module)
            self.assertEqual(compiled(x, module), fn(x, module))
        if kind is dict:
            self.assertEqual(counter.frame_count, 2)

    def test_always_call_hook(self):
        opt, cnt = self.compile_counted(lambda x, module: module(x))
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(add_one, always_call=True)
            self.assertEqual(opt(x, module), module(x))
        # Existing always_call set insertion observes IDs and recompiles per module.
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("key", [True, 1.0, "1"])
    def test_lazy_hook_key_requires_exact_int(self, key):
        key_var = LazyVariableTracker.create(key, LocalSource("key"))
        with self.assertRaisesRegex(AssertionError, "unrealized exact-int key"):
            HashableTracker.lazy_int_key(key_var)

    @parametrize("name", ["_forward_hooks", "_backward_hooks"])
    def test_hook_dictionary_direct_mutation(self, name):
        def fn(x, module):
            hooks = getattr(module, name)
            hooks[1] = identity_forward_hook
            return x + len(hooks)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            self.assertEqual(compiled(x, module), x + 1)
            self.assertEqual(tuple(getattr(module, name)), (1,))
            self.assertIs(getattr(module, name)[1], identity_forward_hook)
        self.assertEqual(counter.frame_count, 1)

    @parametrize("method", ["register_forward_hook", "register_full_backward_hook"])
    def test_hook_registration_remains_unsupported(self, method):
        def fn(x, module):
            return x + 1, getattr(module, method)(identity_forward_hook).id

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "RemovableHandle.*next_id"):
            compiled(torch.ones(1), torch.nn.Identity())

    @parametrize("local_first", [False, True])
    @parametrize("backward", [False, True])
    def test_shared_hook_dict_mixed_global_local_alias(self, local_first, backward):
        def fn(x, module):
            first, second = module, _mixed_hook_module
            if not local_first:
                first, second = second, first
            if backward:
                return x + len(first._backward_hooks) + len(second._backward_hooks)
            return first(x) + second(x)

        global_module = torch.nn.Identity()
        if not backward:
            global_module._forward_hooks[1] = add_one
        module = copy.copy(global_module)
        exported = []
        compiled, counter = self.compile_counted(fn, lambda g: exported.append(list(g)))
        x = torch.ones(1)
        with patch(f"{__name__}._mixed_hook_module", global_module):
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 1)
            changed = torch.nn.Identity()
            if backward:
                changed._backward_hooks[1] = add_one
            self.assertEqual(compiled(x, changed), fn(x, changed))
            self.assertEqual(counter.frame_count, 2)
            globals()["_mixed_hook_module"] = changed
            self.assertEqual(compiled(x, module), fn(x, module))
        name = "_backward_hooks" if backward else "_forward_hooks"
        guarded = {
            guard.name
            for guard in exported[0]
            if guard.inner_create_fn() is GuardBuilder.ID_MATCH
        }
        self.assertIn(f"L['module'].{name}", guarded)
        self.assertIn(f"G['_mixed_hook_module'].{name}", guarded)

    def test_empty_hook_dict_membership_keeps_key_lazy(self):
        def fn(x, key, module):
            return x + (key in module._forward_hooks_with_kwargs)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        module = torch.nn.Identity()
        for key in (3, 4):
            self.assertEqual(compiled(x, key, module), x)
        self.assertEqual(counter.frame_count, 1)
        module._forward_hooks_with_kwargs[4] = True
        self.assertEqual(compiled(x, 4, module), x + 1)
        self.assertEqual(counter.frame_count, 2)

    @parametrize("capture", [False, True])
    @parametrize("global_hook", [False, True])
    @parametrize("local_kwargs", [False, True])
    def test_global_hook_flags_reuse(self, capture, global_hook, local_kwargs):
        def hook(module, args, *rest):
            if isinstance(module, torch.nn.Identity):
                return rest[-1] * 2 + len(rest)

        if global_hook:
            handle = hook_api.register_module_forward_hook(hook, with_kwargs=True)
            self.addCleanup(handle.remove)
        modules = [torch.nn.Identity() for _ in range(5)]
        for module in modules:
            module.register_forward_hook(hook, with_kwargs=local_kwargs)
        x = torch.ones(1)
        if capture:
            with get_metrics_context(), dynamo_timed(""):
                output = fullgraph_capture(modules[0], (x,))
                code = get_traced_fn(modules[0])[0].__code__
                guard = output.graph_capture_output.build_guards(code).guard_manager
            compiled = output.forward_callable()
        else:
            compiled, counter = self.compile_counted(lambda module, x: module(x))
        for module in modules:
            collision = module is modules[-1] and global_hook and not local_kwargs
            if collision:
                # Change only global flag membership for this local callback.
                module._forward_hooks = OrderedDict({handle.id: hook})
            if capture:
                inputs = {"self": module, "args": (x,), "kwargs": {}}
                self.assertEqual(guard.check(inputs), not collision)
            if not capture or not collision:
                self.assertEqual(compiled(module, x), module(x))
            if not capture:
                self.assertEqual(counter.frame_count, 2 if collision else 1)

    def test_global_hook_registry_argument_checks_type(self):
        class AllKeys(dict):
            def __contains__(self, key):
                return True

        def fn(x, registry):
            return x + (5 in registry)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for registry in (hook_api._global_forward_hooks_with_kwargs, AllKeys()):
            self.assertEqual(compiled(x, registry), fn(x, registry))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("source", ["argument", "alias", "vars", "removed"])
    def test_global_hook_registry_reuse(self, source):
        def fn(module, x, registry):
            if source == "removed":
                return module(x)
            if source == "alias":
                registry = global_post_hooks
            elif source == "vars":
                registry = vars(hook_api)["_global_forward_hooks"]
            return len(registry) + module(x)

        registry = hook_api._global_forward_hooks
        self.enterContext(patch(f"{__name__}.global_post_hooks", registry))
        register = hook_api.register_module_forward_hook
        flags = hook_api._global_forward_hooks_with_kwargs
        with register(add_one) as handle:
            compiled, counter = self.compile_counted(fn)
            x = torch.ones(1)
            module = torch.nn.Identity()
            for _ in range(4):
                if source == "removed":
                    removed = register(hook_arity, with_kwargs=True)
                    removed.remove()
                    self.assertNotIn(removed.id, flags)
                else:
                    module = torch.nn.Identity()
                    module.register_forward_hook(hook_arity, with_kwargs=True)
                args = (module, x, registry)
                self.assertEqual(compiled(*args), fn(*args))
            self.assertEqual(counter.frame_count, 1)
            if source == "argument":
                separate = registry.copy()
                registry[handle.id] = add_two
                self.assertEqual(compiled(module, x, separate), fn(module, x, separate))
                self.assertEqual(counter.frame_count, 2)

    @parametrize(
        "operation,container,frames",
        [
            subtest((lambda hooks: hooks.items(), list, 1), name="items"),
            subtest((lambda hooks: hooks.keys(), list, 1), name="keys"),
            subtest((lambda hooks: hooks.keys(), repr, 2), name="keys_repr"),
            subtest((lambda hooks: hooks.items(), repr, 2), name="items_repr"),
            subtest((dict.items, list, 1), name="dict_items"),
            subtest((iter, list, 1), name="iter"),
            subtest((iter, tuple, 1), name="tuple"),
            subtest((OrderedDict.keys, tuple, 1), name="ordered_keys"),
            subtest((OrderedDict.items, tuple, 1), name="ordered_items"),
            subtest((lambda h: [k + 1 for k, _ in h.items()], list, 1), name="add"),
            subtest((lambda h: [hash(k) for k, _ in h.items()], list, 2), name="hash"),
        ],
    )
    def test_hook_dict_items_observe_current_ids(self, operation, container, frames):
        def fn(x, module):
            return x + 1, container(operation(module._forward_hooks))

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(identity_forward_hook)
            self.assertEqual(compiled(x, module), fn(x, module))
        self.assertEqual(counter.frame_count, frames)

    @parametrize(
        "truth_fn",
        [
            subtest(lambda hooks: hooks, name="branch"),
            subtest(bool, name="bool"),
            subtest(operator.not_, name="not"),
            subtest(len, name="len"),
            subtest(lambda hooks: len(hooks.values()), name="view_length"),
        ],
    )
    def test_hook_dict_truth(self, truth_fn):
        def fn(x, module):
            return x + 1 if truth_fn(module._forward_hooks) else x - 1

        module = torch.nn.Linear(2, 2)
        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        self.assertEqual(compiled(x, module), fn(x, module))
        for _ in range(2):
            handle = module.register_forward_hook(lambda *args: None)
            self.assertEqual(compiled(x, module), fn(x, module))
            handle.remove()
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 2)

    @parametrize("with_kwargs", [False, True])
    @parametrize("global_hook", [False, True])
    def test_forward_pre_hook_reuse(self, with_kwargs, global_hook):
        def hook(module, args, *rest):
            return ((args[0] + 3,), rest[0]) if rest else (args[0] + 2,)

        opt, cnt = self.compile_counted(lambda x, module: module(x))
        x = torch.ones(1)

        if global_hook:
            self.addCleanup(hook_api.register_module_forward_pre_hook(hook).remove)

        def check(flag):
            module = torch.nn.Identity()
            module.register_forward_pre_hook(hook, with_kwargs=flag)
            self.assertEqual(opt(x, module), module(x))

        for _ in range(4):
            check(with_kwargs)
        self.assertEqual(cnt.frame_count, 1)
        check(not with_kwargs)
        self.assertEqual(cnt.frame_count, 2)
        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            check(with_kwargs)
            check(not with_kwargs)
        self.assertEqual(cnt.frame_count, 2)

    def test_hook_flag_mutated_before_dispatch(self):
        def fn(x, module):
            module._forward_hooks_with_kwargs.clear()
            return module(x)

        compiled, counter = self.compile_counted(fn)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook_arity, with_kwargs=True)
            self.assertEqual(compiled(torch.ones(1), module), torch.ones(1) * 10 + 1)
            self.assertFalse(module._forward_hooks_with_kwargs)
        self.assertEqual(counter.frame_count, 1)

    def test_cross_module_hook_flags_fallback(self):
        def fn(x, first, second):
            return x + sum(
                key in second._forward_hooks_with_kwargs for key in first._forward_hooks
            )

        first, second = torch.nn.Identity(), torch.nn.Identity()
        first._forward_hooks[1] = identity_forward_hook
        second._forward_hooks_with_kwargs[1] = True
        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        self.assertEqual(compiled(x, first, second), x + 1)
        second._forward_hooks_with_kwargs.clear()
        self.assertEqual(compiled(x, first, second), x)
        self.assertEqual(counter.frame_count, 2)

    @parametrize("suppressed", [False, True])
    def test_hook_membership_guard_objects(self, suppressed):
        def prime(ctx):
            tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
            module = ctx.get_local(
                "module"
            )._i_will_not_complain_if_bc_breaks_VariableTracker()
            hooks = generic_getattr(tx, module, "_forward_hooks")
            flags = generic_getattr(tx, module, "_forward_hooks_with_kwargs")
            with torch._guards.TracingContext.get().guards_context.skip_guard_install():
                flags.sq_contains_impl(tx, next(iter(hooks.items)).vt)

        def fn(x, module):
            if suppressed:
                comptime(prime)
            return module(x)

        exported = []
        compiled, _ = self.compile_counted(fn, lambda g: exported.append(list(g)))
        x = torch.ones(1)
        module, fresh = torch.nn.Identity(), torch.nn.Identity()
        for current in (module, fresh):
            for with_kwargs in (False, True, True, False):
                current.register_forward_hook(hook_arity, with_kwargs=with_kwargs)
            self.assertEqual(compiled(x, current), current(x))
        self.assertEqual(len(exported), 1)
        keys = list(module._forward_hooks)
        module._forward_hooks_with_kwargs.clear()
        module._forward_hooks_with_kwargs.update({keys[0]: True, keys[-1]: True})
        self.assertEqual(compiled(x, module), module(x))
        self.assertEqual(len(exported), 2)
        guards = [
            guard
            for guard in exported[0]
            if guard.inner_create_fn() is GuardBuilder.NN_MODULE_HOOKS_DICT_CONTAINS
        ]
        scope = {"G": globals(), "L": {"module": module}}
        pairs = set()
        for guard in guards:
            self.assertIs(guard.originating_source.get_value(scope, {}, {}), module)
            flags_source = guard.create_fn.keywords["flags_source"]
            flags = flags_source.get_value(scope, {}, {})
            pairs.add((guard.create_fn.keywords["hooks_name"], id(flags)))
        expected_flags = (
            module._forward_hooks_with_kwargs,
            module._forward_hooks_always_called,
            hook_api._global_forward_hooks_with_kwargs,
            hook_api._global_forward_hooks_always_called,
        )
        self.assertEqual(
            pairs, {("_forward_hooks", id(flags)) for flags in expected_flags}
        )
        self.assertEqual(len(guards), len(pairs))

    @parametrize("base_type", [dict, OrderedDict, defaultdict])
    @parametrize("subclass", [False, True])
    @parametrize("operation", ["construct", "update", "ior"])
    def test_mapping_copies_guard_hook_keys(self, base_type, subclass, operation):
        class Mapping(base_type):
            pass

        mapping_type = Mapping if subclass else base_type

        module = torch.nn.Identity()
        key = module.register_forward_hook(identity_forward_hook).id

        def fn(x, module):
            args = (None,) if base_type is defaultdict else ()
            if operation == "construct":
                result = mapping_type(*args, module._forward_hooks)
            else:
                result = mapping_type(*args)
                if operation == "update":
                    result.update(module._forward_hooks)
                else:
                    result |= module._forward_hooks
            return x + (key in result)

        x = torch.zeros(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, module), x + 1)
        module = torch.nn.Identity()
        module.register_forward_hook(identity_forward_hook)
        self.assertEqual(compiled(x, module), x)

    def test_hook_dict_move_to_end_guard_keys(self):
        def hook(module, args, output):
            return output + 1

        first = torch.nn.Identity()
        key = first.register_forward_hook(hook).id

        def fn(x, module):
            try:
                module._forward_hooks.move_to_end(key, last=False)
            except KeyError:
                return x + 2
            return x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(2)
        second = torch.nn.Identity()
        second.register_forward_hook(hook)
        for module in (first, second):
            self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("key_type", [str, bool, enum.IntEnum("HookKey", {"value": 1})])
    def test_mixed_hook_key_types(self, key_type):
        opt, cnt = self.compile_counted(lambda x, module: module(x))
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module._forward_hooks[key_type(1)] = add_one
            module._forward_hooks[100] = hook_arity
            module._forward_hooks_with_kwargs[100] = True
            self.assertEqual(opt(x, module), (x + 1) * 10 + 2)
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=True)
    def test_hook_flag_mutation_with_recursive(self):
        compiled, counter = self.compile_counted(lambda x, module: module(x))
        x = torch.ones(1)
        for first, second in ((1, 2), (3, 4)):
            module = torch.nn.Identity()
            module._forward_hooks.update({first: hook_arity, second: hook_arity})
            module._forward_hooks_with_kwargs[first] = True
            self.assertEqual(compiled(x, module), x * 121)
        self.assertEqual(counter.frame_count, 1)
        module._forward_hooks_with_kwargs.clear()
        module._forward_hooks_with_kwargs[second] = True
        self.assertEqual(compiled(x, module), x * 112)
        self.assertEqual(counter.frame_count, 2)

    @parametrize(
        "global_flags, flags_type, initial_flags",
        [
            (False, dict, ()),
            (True, dict, ()),
            (True, dict, (1,)),
            (True, OrderedDict, ()),
            (True, OrderedDict, (1,)),
        ],
    )
    @parametrize("recursive", [False, True])
    def test_nested_region_hook_membership(
        self, global_flags, flags_type, initial_flags, recursive
    ):
        modules = [torch.nn.Identity(), torch.nn.Identity()]
        for module, flagged in zip(modules, (1, 2)):
            module._forward_hooks.update({1: hook_arity, 2: hook_arity})
            module._forward_hooks_with_kwargs[flagged] = True
        if global_flags:
            modules[1]._forward_hooks = OrderedDict({3: hook_arity, 4: hook_arity})

        @torch.compiler.nested_compile_region
        def region(x, module):
            if global_flags:
                return x + sum(
                    key in _global_hook_flags_alias for key in module._forward_hooks
                )
            return module(x)

        def fn(x, first, second):
            return region(x, first), region(x, second)

        flags = flags_type.fromkeys(initial_flags, True)
        with (
            torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=recursive),
            patch.object(hook_api, "_global_forward_hooks_with_kwargs", flags),
            patch(f"{__name__}._global_hook_flags_alias", flags),
        ):
            x = torch.ones(1)
            expected = (
                (x + len(initial_flags), x) if global_flags else (x * 121, x * 112)
            )
            compiled, counter = self.compile_counted(fn)
            self.assertEqual(fn(x, *modules), expected)
            self.assertEqual(compiled(x, *modules), expected)
            self.assertEqual(counter.frame_count, 1)
            if global_flags:
                flags.clear()
                flags[3] = True
                self.assertEqual(compiled(x, *modules), (x, x + 1))
                self.assertEqual(counter.frame_count, 2)

    @parametrize("initial_flags", [(), (1,)])
    @parametrize("recursive", [False, True])
    @parametrize("mutation", ["flags", "callback", "tensor_shape"])
    def test_nested_region_hook_mutation(self, initial_flags, recursive, mutation):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest) + module.probe.shape[0]

        modules = [torch.nn.Identity(), torch.nn.Identity()]
        for module in modules:
            module.probe = torch.ones(1)
            module._forward_hooks.update({1: hook, 2: hook})
            module._forward_hooks_with_kwargs.update(dict.fromkeys(initial_flags, True))

        def fn(x, y, first, second):
            return call_hooked_module(x, first), call_hooked_module(y, second)

        x, y = torch.ones(1), torch.full((1,), 2.0)
        with (
            torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=recursive),
            count_subgraph_traces() as traces,
        ):
            compiled, counter = self.compile_counted(fn)
            self.assertEqual(compiled(x, y, *modules), fn(x, y, *modules))
            self.assertEqual(traces.call_count, 1)
            if mutation == "flags":
                modules[1]._forward_hooks_with_kwargs.clear()
                modules[1]._forward_hooks_with_kwargs[2] = True
            elif mutation == "callback":
                modules[1]._forward_hooks[1] = hook_arity
            else:
                modules[1].probe = torch.ones(2)
            self.assertEqual(compiled(x, y, *modules), fn(x, y, *modules))
        self.assertEqual(counter.frame_count, 2)

    def test_nested_region_hook_reuse_propagates_guards(self):
        modules = [torch.nn.Identity() for _ in range(4)]
        for module in modules:
            module._forward_hooks.update({1: hook_arity, 2: hook_arity})
            module._forward_hooks_with_kwargs[1] = True

        @torch.compiler.nested_compile_region
        def outer(x, first, second):
            return call_hooked_module(x, first) + call_hooked_module(x, second)

        def fn(x, modules):
            return outer(x, *modules[:2]), outer(x, *modules[2:])

        x = torch.ones(1)
        with count_subgraph_traces() as traces:
            compiled, counter = self.compile_counted(fn)
            self.assertEqual(compiled(x, modules), fn(x, modules))
            self.assertEqual(traces.call_count, 2)
            modules[-1]._forward_hooks_with_kwargs.clear()
            modules[-1]._forward_hooks_with_kwargs[2] = True
            self.assertEqual(compiled(x, modules), fn(x, modules))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("flags_name", ["with_kwargs", "always_called"])
    @parametrize("populated", [False, True])
    def test_global_flags_mapping_type(self, mapping_type, flags_name, populated):
        def fn(x, flags):
            copied = flags.copy()
            if isinstance(copied, defaultdict):
                return x + 1, copied.default_factory is int, repr(flags)
            return x + 1, copied, pytree.tree_map(operator.not_, flags), repr(flags)

        flags = defaultdict(int) if mapping_type is defaultdict else mapping_type()
        if populated:
            flags[1] = True
        with patch.object(hook_api, f"_global_forward_hooks_{flags_name}", flags):
            x = torch.ones(1)
            actual = torch.compile(fn, backend="eager", fullgraph=True)(x, flags)
            self.assertEqual(actual, fn(x, flags))
            if mapping_type is not defaultdict:
                for result in actual[1:3]:
                    self.assertIs(type(result), mapping_type)

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize(
        "operation",
        [
            subtest(lambda d: 1 in set(d), name="set"),
            subtest(lambda d: 1 in {*d}, name="set_unpack"),
            subtest(lambda d: 1 in d.copy(), name="copy"),
            subtest(update_hook_keys, name="update"),
            subtest(sum_hook_keys, name="constant"),
            subtest(
                lambda d: 1 in {key: True for key, _ in d.items()},
                name="dict_comprehension",
            ),
            subtest(lambda d: 1 in {k for k, _ in d.items()}, name="set_comprehension"),
            subtest(lambda d: 1 in frozenset(d), name="frozenset"),
            subtest(lambda d: 1 in HookKeyFrozenSet(d), name="frozenset_subclass"),
            subtest(lambda d: 1 in dict.fromkeys(set(d)), name="fromkeys"),
            subtest(lambda d: 1 in dict.fromkeys(d), name="fromkeys_mapping"),
            subtest(lambda d: 1 in {0}.union(d), name="union"),
            subtest(lambda d: 1 in {1}.intersection(d), name="intersection"),
            subtest(lambda d: 1 in {1}.difference(d), name="difference"),
            subtest(lambda d: {1}.isdisjoint(d), name="isdisjoint"),
            subtest(lambda d: 1 in ({0: True}.keys() | d), name="view_union"),
            subtest(lambda d: 1 in {1: True}.keys() & d, name="view_intersection"),
            subtest(lambda d: 1 in d | {0: True}.keys(), name="reverse_view_union"),
            subtest(
                lambda d: 1 in pytree.tree_map(lambda value: True, d),
                name="tree_map",
            ),
            subtest(
                lambda d: 1 in pytree.tree_map_with_path(lambda path, value: True, d),
                name="tree_map_with_path",
            ),
        ],
    )
    @torch._dynamo.config.patch(allow_unspec_int_on_nn_module=True)
    def test_materialized_hook_keys(self, operation, mapping_type):
        self._check_materialized_hook_keys(operation, mapping_type)

    def _check_materialized_hook_keys(self, operation, mapping_type):
        def check(ctx):
            hooks = ctx.get_local("hooks")
            self.assertFalse(
                hooks._i_will_not_complain_if_bc_breaks_VariableTracker().has_new_items()
            )

        def fn(x, module):
            hooks = module._forward_hooks
            result = operation(hooks)
            comptime(check)
            return x + result

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for keys in ((2, 3), (4, 3), (1, 3), (4, 3)):
            module = torch.nn.Identity()
            module._forward_hooks = mapping_type()
            for key in keys:
                module._forward_hooks[key] = identity_forward_hook
            self.assertEqual(compiled(x, module), fn(x, module))
        self.assertEqual(counter.frame_count, 3)

    @parametrize("recursive", [False, True])
    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("flags_name", ["with_kwargs", "always_called"])
    def test_global_flags_alias_survives_registry_replacement(
        self, mapping_type, flags_name, recursive
    ):
        def fn(x, module):
            return x + sum(
                key in _global_hook_flags_alias for key in module._forward_hooks
            )

        flags = mapping_type({1: True})
        registry_name = f"_global_forward_hooks_{flags_name}"
        with (
            torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=recursive),
            patch.object(hook_api, registry_name, flags),
            patch(f"{__name__}._global_hook_flags_alias", flags),
        ):
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.ones(1)
            for key in (2, 1):
                module = torch.nn.Identity()
                module._forward_hooks[key] = identity_forward_hook
                if key == 1:
                    setattr(hook_api, registry_name, mapping_type())
                self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("populated", [False, True])
    @parametrize(
        "operation",
        ["delete", "setdefault", "insert", "update"]
        + ["insert_contains", "update_contains"],
    )
    def test_hook_dict_key_sensitive_mutation(self, mapping_type, populated, operation):
        def fn(x, module):
            result = module(x)
            hooks = module._forward_hooks
            if operation == "delete":
                try:
                    del hooks[7]
                except KeyError:
                    return result + 1
                return result + 2
            if operation == "setdefault":
                hook = hooks.setdefault(5, add_two)
                return hook(module, (result,), result) + len(hooks)
            if operation.startswith("update"):
                hooks.update({100: add_two})
            else:
                hooks[100] = add_two
            if operation.endswith("contains"):
                return result + (5 in hooks)
            return result + len(hooks)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        second_key = {"delete": 7, "insert": 100, "update": 100}.get(operation, 5)
        for key in (3, second_key, second_key):
            eager, actual = torch.nn.Identity(), torch.nn.Identity()
            for module in (eager, actual):
                module._forward_hooks = mapping_type(
                    {key: identity_forward_hook} if populated else {}
                )
            self.assertEqual(compiled(x, actual), fn(x, eager))
            self.assertEqual(list(actual._forward_hooks), list(eager._forward_hooks))
            for key, hook in eager._forward_hooks.items():
                self.assertIs(actual._forward_hooks[key], hook)
        self.assertEqual(counter.frame_count, 2 if populated else 1)

    @parametrize("registry", ["forward", "backward", "global"])
    @parametrize(
        "operation, populated",
        [
            subtest((type, False), name="type"),
            subtest((type, True), name="populated_type"),
            subtest((lambda d: d.copy(), True), name="populated_copy"),
            subtest((repr, False), name="repr"),
            subtest((ascii, False), name="ascii"),
            subtest((lambda d: ascii((d,)), False), name="nested_ascii"),
            subtest((lambda d: f"{d}", False), name="fstring"),
            subtest(("{}".format, False), name="format"),
            subtest((str, False), name="str"),
            subtest((lambda d: d.copy(), False), name="copy"),
            subtest((lambda d: {} | d, False), name="reflected_union"),
            subtest((lambda d: isinstance(d, OrderedDict), False), name="instance"),
            subtest((lambda d: d.__dict__ is not None, False), name="instance_dict"),
            subtest((lambda d: hasattr(d, "move_to_end"), False), name="hasattr"),
            subtest((lambda d: d.popitem(last=False), False), name="popitem"),
            subtest((lambda d: d.move_to_end(1), False), name="move_to_end"),
            subtest((lambda d: d.move_to_end is not None, False), name="read_method"),
        ],
    )
    def test_hook_dict_observable_type(self, operation, populated, registry):
        def fn(x, module):
            if registry == "global":
                hooks = hook_api._global_forward_hooks_with_kwargs
            else:
                hooks = getattr(module, f"_{registry}_hooks")
            try:
                return x + 1, operation(hooks)
            except (TypeError, KeyError, AttributeError) as error:
                return x + 1, type(error)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for kind in (dict, OrderedDict, dict, OrderedDict):
            module = torch.nn.Identity()
            hooks = kind({1: identity_forward_hook} if populated else {})
            if registry != "global":
                setattr(module, f"_{registry}_hooks", hooks)
            flags = kind(hooks)
            with patch.object(hook_api, "_global_forward_hooks_with_kwargs", flags):
                expected = fn(x, module)
                actual = compiled(x, module)
                self.assertEqual(actual, expected)
                if isinstance(expected[1], dict):
                    self.assertIs(type(actual[1]), type(expected[1]))
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(skip_nnmodule_hook_guards=True)
    @parametrize("hook_mode", ["empty", "local", "global"])
    def test_builtin_hook_dispatch_skips_type_guards(self, hook_mode):
        def fn(x, modules):
            for module in modules:
                x = module(x)
            return x + 1

        seen = []
        opt, cc = self.compile_counted(fn, seen.extend)
        if hook_mode == "global":
            self.enterContext(hook_api.register_module_forward_hook(add_two))
        for _ in range(2):
            modules = [torch.nn.Identity() for _ in range(4)]
            if hook_mode == "local":
                for module in modules:
                    module.register_forward_hook(add_two)
            x = torch.ones(1)
            self.assertEqual(opt(x, modules), fn(x, modules))
        self.assertEqual(cc.frame_count, 1)
        hook_guards = [guard for guard in seen if "_hooks" in guard.name]
        self.assertTrue(hook_guards)
        for guard in hook_guards:
            self.assertNotIn("TYPE_MATCH", guard.guard_types or (), guard.name)

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("populated", [False, True])
    def test_custom_hook_registry_protocol(self, mapping_type, populated):
        class Hooks(mapping_type):
            def items(self):
                return [(7, add_two)]

        def fn(x, module):
            result = module(x)
            hooks = module._forward_hooks
            keys = []
            for key, callback in hooks.items():
                keys.append(key)
                result = callback(module, (result,), result)
            return result + 1, keys, bool(hooks), type(hooks)

        compiled, counter = self.compile_counted(fn)
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module._forward_hooks = Hooks.fromkeys(
                (1,) if populated else (), identity_forward_hook
            )
            self.assertEqual(
                fn(x, module), (x + (5 if populated else 3), [7], populated, Hooks)
            )
            self.assertEqual(compiled(x, module), fn(x, module))
        self.assertEqual(counter.frame_count, 1)
        with patch.object(Hooks, "items", mapping_type.items):
            self.assertEqual(compiled(x, module), fn(x, module))
        self.assertEqual(counter.frame_count, 2)


class HooksTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_tensor_only_register_hook_in_graph_lambda(self):
        def fn(x):
            x.register_hook(lambda grad: grad * 2)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_register_hook_in_graph_lambda(self):
        def fn(x, y, z):
            x.register_hook(lambda grad: grad * 2)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_lambda(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            x.register_hook(lambda grad: grad * 3)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_multi_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle
            z = z * z
            return x, y * y, z, handle, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertNotEqual(h, None)
        self.assertNotEqual(h2, None)
        self.assertEqual(h2, h)

    def test_tensor_register_hook_repeated_handle_return(self):
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            h2 = handle  # noqa: F841
            z = z * z
            return x, y * y, z, handle, handle

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_removed_handle_return(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y, z):
            handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            handle.remove()
            handle.remove()
            return x, y * y, z, handle, handle

        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, y, z, h, h2 = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(cnt.frame_count, 1)
        self.assertIsInstance(h, RemovableHandle)
        self.assertIs(h2, h)

    def test_tensor_register_hook_repeated_handle_not_local(self):
        def fn(x, y, z, mod):
            mod.handle = x.register_hook(lambda grad: grad * 2)
            z = z * z
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts, fullgraph=True)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

        mod = torch.nn.Module()
        mod.handle = None

        v, y, z = fn(v, torch.randn([2, 2]), torch.randn([2, 2]), mod)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

        self.assertNotEqual(mod.handle, None)

    def test_tensor_only_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x):
            x.register_hook(local_hook)
            return x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 0)

    def test_tensor_only_register_hook_in_graph_local_inner(self):
        def fn(x):
            def local_hook(grad):
                return grad * 2

            z = x * x
            x.register_hook(local_hook)
            z.register_hook(local_hook)
            return x, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)
        v[0].backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v[0].grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_local(self):
        def local_hook(grad):
            return grad * 2

        def fn(x, y, z):
            x.register_hook(local_hook)
            return x, y * y, z * z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([2.0, 4.0, 6.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_hook_in_graph_break_handle_local(self):
        def local_hook(grad):
            return grad * 2

        def local_hook2(grad):
            return grad * 3

        def fn(x, y, z):
            handle = x.register_hook(local_hook)
            z = z * z
            handle.remove()
            x.register_hook(local_hook2)
            return x, y * y, z

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v, torch.randn([2, 2]), torch.randn([2, 2]))[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(v.grad, torch.tensor([3.0, 6.0, 9.0]))

    def test_tensor_register_global_hook(self):
        def fn(x):
            x.register_hook(global_hook_0)
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks(self):
        def fn(x):
            x.register_hook(global_hook_0)  # * 4
            x.register_hook(global_hook_1)  # / 2
            x.register_hook(global_hook_2)  # * 3
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v = fn(v)[0]
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_multiple_hooks_handles_in_list(self):
        def fn(x):
            h0 = x.register_hook(global_hook_0)  # * 4
            h1 = x.register_hook(global_hook_1)  # / 2
            h2 = x.register_hook(global_hook_2)  # * 3
            return x, x * x, h0, h1, h2

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r, handle_0, handle_1, handle_2 = fn(v)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([6.0, 12.0, 18.0]))
        handle_0.remove()
        handle_1.remove()
        handle_2.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([7.0, 14.0, 21.0]))

        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_register_global_hooks_handles_in_list(self):
        def fn(x):
            global h0
            h0 = x.register_hook(global_hook_0)  # * 4
            return x, x * x

        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(fn, backend=cnts)
        v = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        v, r = fn(v)

        self.assertIsNotNone(h0)
        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        self.assertEqual(v.grad, torch.tensor([4.0, 8.0, 12.0]))
        h0.remove()

        v.backward(torch.tensor([1.0, 2.0, 3.0]))
        # Handles gone, grad is just applied as is
        self.assertEqual(v.grad, torch.tensor([5.0, 10.0, 15.0]))

        # NYI!
        self.assertEqual(cnts.frame_count, 0)

    def test_hook_on_intermediate(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda grad: grad + 1)
            return y.sum()

        x_compiled = torch.randn(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        x_eager = x_compiled.detach().clone().requires_grad_(True)
        result_eager = fn(x_eager)
        result_eager.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)

    def test_register_hook_identity_alias_fullgraph(self):
        def fn(x):
            y = x * x
            y.register_hook(lambda grad: grad)
            return y.sum()

        x = torch.randn([2, 2], requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        expected = fn(x_ref)
        expected.backward()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        actual = opt_fn(x)
        actual.backward()

        self.assertEqual(actual, expected)
        self.assertEqual(x.grad, x_ref.grad)

    def test_register_hook_input_mutation_rejected(self):
        def fn(x):
            y = x * x
            y.register_hook(lambda grad: grad.mul_(2))
            return y

        with self.assertRaisesRegex(
            RuntimeError,
            "(?s)Encountered input mutation during higher order op tracing.*"
            "Found in register_hook",
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.randn(3, requires_grad=True)
            )

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        x = torch.randn(3, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        grad = torch.randn_like(x)

        expected = fn(x_ref)
        expected.backward(grad.clone())

        actual = torch.compile(fn, backend="eager")(x)
        actual.backward(grad.clone())

        self.assertEqual(actual, expected)
        self.assertEqual(x.grad, x_ref.grad)
        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertEqual(sum(graph_breaks.values()), 1)
        self.assertTrue(
            any(
                "Encountered input mutation during higher order op tracing" in reason
                for reason in graph_breaks
            )
        )

    def test_register_post_accumulate_grad_hook_intermediate_unsupported(self):
        def fn(x):
            y = x.detach().requires_grad_()
            y.register_post_accumulate_grad_hook(lambda tensor: None)
            return (y * 2).sum()

        error = "register_post_accumulate_grad_hook on an intermediate tensor"
        with (
            torch._dynamo.compiled_autograd._disable(),
            self.assertRaisesRegex(RuntimeError, error),
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3))

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        with torch._dynamo.compiled_autograd._disable():
            out = torch.compile(fn, backend="eager")(torch.randn(3))
            out.backward()

        graph_breaks = torch._dynamo.utils.counters["graph_break"]
        self.assertTrue(any(error in reason for reason in graph_breaks))

    def test_hook_on_intermediate_with_container(self):
        glb_list = []
        glb_dict = {}

        def fn(x):
            y = x * 2
            glb_list.append(y)
            glb_dict["tensor"] = y
            a = glb_list[0] * 3  # Should use output of register_hook
            b = glb_dict["tensor"]
            y.register_hook(lambda grad: grad + 1)
            return (a + b).sum()

        glb_list.clear()
        glb_dict.clear()
        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        glb_list.clear()
        glb_dict.clear()
        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_compiled.grad, x_eager.grad)
        # Without hook: dloss/dy = 4, dloss/dx = 8
        # With hook (+1): hooked = 5, dloss/dx = 10
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 10.0))

        glb_list.clear()
        glb_dict.clear()
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(4, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 2;  l_x_ = None
    mul_1 = mul * 3
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(mul, hook_body_0);  mul = hook_body_0 = None
    add = mul_1 + register_hook;  mul_1 = None
    sum_1 = add.sum();  add = None
    return (sum_1, register_hook)""",
        )

    def test_hook_on_intermediate_used_before_and_after(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_with_higher_order_op(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: g * 2)

            def true_fn(t):
                return t + 1

            def false_fn(t):
                return t - 1

            z = torch.cond(x.sum() > 0, true_fn, false_fn, (y,))
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_on_intermediate_returns_none(self):
        def fn(x):
            y = x * 2
            y.register_hook(lambda g: None)
            return y.sum()

        x_eager = torch.ones(4, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(4, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)
        self.assertEqual(x_compiled.grad, torch.full_like(x_compiled, 2.0))

    def test_hook_has_side_effect(self):
        def fn(x):
            y = x * 2
            z = y + 1  # Use y BEFORE hook
            y.register_hook(lambda g: g * 2)
            w = y * 3  # Use y AFTER hook
            return (z + w).sum()

        x_eager = torch.ones(2, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(2, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

    def test_hook_bwd_inside_side_effects(self):
        global_list = []

        def fn(x):
            y = x * 2

            def _hook(grad):
                global_list.append(grad)
                return grad * 2

            y.register_hook(_hook)
            z = y + x
            return z.sum()

        x_eager = torch.ones(3, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        global_list.clear()

        x_compiled = torch.ones(3, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Unsafe side effect"
        ):
            _ = compiled_fn(x_compiled)

    def test_hook_on_intermediate_from_split(self):
        def fn(x):
            splits = x.split(2)
            result = torch.cat(splits)  # use splits before register_hook
            y = splits[0]
            y.register_hook(lambda g: g + 1)
            return result.sum() + y.sum()

        x_eager = torch.ones(6, requires_grad=True)
        result_eager = fn(x_eager)
        result_eager.backward()

        x_compiled = torch.ones(6, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        result_compiled = compiled_fn(x_compiled)
        result_compiled.backward()

        self.assertEqual(x_eager.grad, x_compiled.grad)

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.ones(6, requires_grad=True)
        )
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].code.strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    split = l_x_.split(2);  l_x_ = None
    getitem = split[0]
    getitem_1 = split[1]
    getitem_2 = split[2];  split = None
    cat = torch.cat((getitem, getitem_1, getitem_2));  getitem_1 = getitem_2 = None
    hook_body_0 = self.hook_body_0
    register_hook = torch.ops.higher_order.register_hook(getitem, hook_body_0);  getitem = hook_body_0 = None
    sum_1 = cat.sum();  cat = None
    sum_2 = register_hook.sum();  register_hook = None
    add = sum_1 + sum_2;  sum_1 = sum_2 = None
    return (add,)""",
        )

    def test_intermediary_hooks(self):
        def simple_hook(g):
            return g * 2

        def f(x):
            y = x + 1
            y.register_hook(simple_hook)
            z = y + 1
            return z

        out = torch.randn(1, requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(f, backend=cnts, fullgraph=True)
        res = fn(out)
        res.backward()
        self.assertEqual(res, f(out))
        self.assertEqual(out.grad, torch.Tensor([2.0]))

    def test_intermediary_hooks_same_on_aot_eager(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="aot_eager", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_input_hooks_same(self):
        backends = ["eager", "aot_eager", "inductor"]
        for backend in backends:

            def my_hook(grad, *, k=0):
                return grad + k

            hook = functools.partial(my_hook, k=3)

            class MyMod(torch.nn.Module):
                def forward(self, x):
                    x.register_hook(hook)
                    y = x.mul(2)
                    z = y.mul(3)
                    return (z,)

            mod = MyMod()
            x0 = torch.ones(4, requires_grad=True)
            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            x1 = torch.ones(4, requires_grad=True)
            mod_compiled = aot_module_simplified(mod, (x1,), nop)
            aot_out = mod_compiled(x1)
            aot_out[0].backward(torch.ones(4))

            x2 = torch.ones(4, requires_grad=True)
            dynamo_out = torch.compile(mod, backend=backend, fullgraph=True)(x2)
            with compiled_autograd._enable(compiler_fn):
                dynamo_out[0].backward(torch.ones(4))

            self.assertEqual(dynamo_out, aot_out)
            self.assertEqual(dynamo_out, eager_out)

            self.assertEqual(x0.grad, x1.grad)
            self.assertEqual(x0.grad, x2.grad)

    def test_intermediary_hooks_same_on_inductor(self):
        def my_hook(grad, *, k=0):
            return grad + k

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                hook1 = functools.partial(my_hook, k=3)
                hook2 = functools.partial(my_hook, k=4)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        mod_compiled = aot_module_simplified(mod, (x1,), nop)
        aot_out = mod_compiled(x1)
        aot_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, aot_out)
        self.assertEqual(dynamo_out, eager_out)

        self.assertEqual(x0.grad, x1.grad)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor(self):
        class SomePyClass:
            count = 0

            def do_stuff(self, grad):
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.do_stuff(grad)

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        # Eager 2
        self.assertEqual(obj.count, 2)
        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(dynamo_out, eager_out)

        # Eager 2 + compiled 2
        self.assertEqual(obj.count, 4)
        self.assertEqual(x0.grad, x2.grad)

    def test_complex_state_mutation_in_intermediary_hooks_same_on_inductor_with_graph_break(
        self,
    ):
        class SomePyClass:
            grad_as_str = "None"
            count = 0

            def write_grad_as_str_and_do_stuff(self, grad):
                self.grad_as_str = str(grad)
                if self.count % 2 == 0:
                    r = grad * grad
                else:
                    r = grad + grad
                print("Break!")
                self.count += 1
                return r

        def complex_state_touching_hook(grad, *, obj):
            return obj.write_grad_as_str_and_do_stuff(grad)

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(complex_state_touching_hook, obj=obj)
                hook2 = functools.partial(complex_state_touching_hook, obj=obj)
                y.register_hook(hook1)
                y.register_hook(hook2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj = SomePyClass()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0, obj)
        eager_out[0].backward(torch.ones(4))

        x2 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            dynamo_out = torch.compile(mod, backend="inductor", fullgraph=True)(x2, obj)
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, r"repr\(\) on tensor"
            ):
                dynamo_out[0].backward(torch.ones(4))

        self.assertEqual(obj.count, 2)

    def test_register_hook_partial_guarding(
        self,
    ):
        def some_hook(grad, *, obj):
            return grad + obj.val

        class MyMod(torch.nn.Module):
            def forward(self, x, obj):
                y = x.mul(2)
                hook1 = functools.partial(some_hook, obj=obj)
                y.register_hook(hook1)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        obj3 = ClassWithVal(11)
        cnt = torch._dynamo.testing.CompileCounter()

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)

        with compiled_autograd._enable(compiler_fn):
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x1, obj1)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj2)
            torch.compile(mod, backend=cnt, fullgraph=True)(x0, obj3)
            self.assertEqual(cnt.frame_count, 1)

    def test_hook_with_closure(self):
        def fn(x, obj):
            y = x.sin()
            x.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_hook_with_nested_closure(self):
        def fn(x):
            def run():
                y = x.sin()
                x.register_hook(lambda grad: grad + y)
                z = y.sin()
                return z

            return run()

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        fn(x0).sum().backward()
        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x1).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x1.grad)

    def test_intermediate_hook_with_closure_eager(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_fw = torch._dynamo.testing.CompileCounter()
        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend=cnt_fw, fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_fw.frame_count, 1)
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_intermediate_hook_with_closure_aot(self):
        def fn(x, obj):
            y = x.sin()
            y.register_hook(lambda grad: grad + obj.val)
            z = y.sin()
            return z

        cnt_bw = torch._dynamo.testing.CompileCounter()
        opt = torch.compile(fn, backend="aot_eager", fullgraph=True)

        obj1 = ClassWithVal(torch.tensor(88))
        obj2 = ClassWithVal(torch.tensor(99))
        x0 = torch.ones(4, requires_grad=True)
        x1 = torch.ones(4, requires_grad=True)
        x2 = torch.ones(4, requires_grad=True)
        x3 = torch.ones(4, requires_grad=True)
        fn(x0, obj1).sum().backward()
        fn(x1, obj2).sum().backward()

        with compiled_autograd._enable(
            functools.partial(torch.compile, backend=cnt_bw, fullgraph=True)
        ):
            opt(x2, obj1).sum().backward()
            opt(x3, obj2).sum().backward()
            self.assertEqual(cnt_bw.frame_count, 1)

        self.assertEqual(x0.grad, x2.grad)
        self.assertEqual(x1.grad, x3.grad)

    def test_no_recompile_on_hook_identity_change(self):
        def my_hook(grad, k=0):
            return grad + k

        def my_hook2(grad):
            return grad * 2

        class MyMod(torch.nn.Module):
            def forward(self, x):
                y = x.mul(2)
                y.register_hook(my_hook)
                y.register_hook(my_hook)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x0 = torch.ones(4, requires_grad=True)
        eager_out = mod(x0)
        eager_out[0].backward(torch.ones(4))

        x1 = torch.ones(4, requires_grad=True)
        with compiled_autograd._enable(compiler_fn):
            cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            comp_mod = torch.compile(mod, backend=cnts, fullgraph=True)
            comp_out = comp_mod(x1)
            comp_out[0].backward(torch.ones(4))

            self.assertEqual(cnts.frame_count, 1)
            my_hook = my_hook2
            self.assertEqual(x0.grad, x1.grad)

            eager_out = mod(x0)
            eager_out[0].backward(torch.ones(4))

            comp_out = comp_mod(x1)

            self.assertEqual(cnts.frame_count, 1)
            comp_out[0].backward(torch.ones(4))
            self.assertEqual(x0.grad, x1.grad)

    def test_functools_arg_vary(self):
        def pre_hook(grad, *, k):
            return grad * k

        hook = functools.partial(pre_hook, k=1)

        @torch.compile(backend="eager", fullgraph=True)
        def h(x):
            y = x.mul(2)
            y.register_hook(hook)
            return y.mul(3)

        with compiled_autograd._enable(torch.compile(backend="eager", fullgraph=True)):
            x = torch.randn(2, requires_grad=True)
            h(x).sum().backward()
            orig_grad = x.grad
            x.grad = None

            hook = functools.partial(pre_hook, k=2)
            h(x).sum().backward()
            self.assertEqual(orig_grad * 2, x.grad)

    def test_post_acc_grad_hook(self):
        def hook(input_t):
            input_t.mul_(input_t.grad)
            input_t.grad.mul_(5)

        def reg_and_mul(x, y):
            x.register_post_accumulate_grad_hook(hook)
            return x * y

        cnts = None

        def test_fn(fn):
            fn(x, y)
            b = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
            x.backward(b)
            if cnts:
                self.assertEqual(cnts.frame_count, 1)
            # These same exact assertions run on both eager and compiled
            # X goes to x*2 because of mul_
            self.assertEqual(x, torch.tensor([0.5, 0.5, 0.5]) * 2)
            # This test proves grad aliasing works -
            self.assertEqual(x.grad, b * 5)

        # Eager values
        x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        test_fn(reg_and_mul)

        # Compiled
        for backend in ["eager", "aot_eager", "inductor"]:
            for compiled_bwd in [False, True]:
                torch._dynamo.reset()
                x = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
                y = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

                cnts = torch._dynamo.testing.CompileCounterWithBackend(backend)
                compiled_fn = torch.compile(reg_and_mul, backend=cnts, fullgraph=True)

                compiled_bwd_ctx = (
                    compiled_autograd._enable(
                        torch.compile(backend=backend, fullgraph=True)
                    )
                    if compiled_bwd
                    else contextlib.nullcontext()
                )
                with compiled_bwd_ctx:
                    test_fn(compiled_fn)

    def test_recompile(self):
        def hook(param):
            param.grad *= 2

        x = torch.ones(10)
        x.requires_grad = True

        def run(input):
            return x * input

        x.register_post_accumulate_grad_hook(hook)
        with compiled_autograd._enable(compiler_fn):
            for i in range(5):
                with unittest.mock.patch(
                    "torch._dynamo.config.error_on_recompile", True
                ):
                    # Mimic optimizer.zero_grad() to clear the gradient
                    x.grad = None
                    run(i).sum().backward()

    def test_no_recompile_on_same_hook(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fw_hook(inp):
            return (inp[0] + 1,)

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for _ in range(10):
                    layer = torch.nn.Linear(16, 16)
                    layer.register_forward_pre_hook(lambda _, inp: fw_hook(inp))
                    layer = torch.compile(layer, backend=cnts)
                    self.layers.append(layer)

            def forward(self, x):
                for l in self.layers:
                    x = l(x)
                return x

        mod = Mod()
        x = torch.ones(16, 16, requires_grad=True)
        mod(x)

        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(skip_nnmodule_hook_guards=False)
    def test_nnmodule_hook_guards(self):
        # Compile a model and then apply a hook

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return self.linear(x)

        cnts = torch._dynamo.testing.CompileCounter()

        mod = Mod()

        def fn(x):
            return mod(x)

        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.ones(16, 16)
        opt_fn(x)

        # Register a hook
        def forward_hook(self, inputs, out):
            return out * 2

        mod.register_forward_hook(forward_hook)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(wrap_top_frame=True)
    def test_wrap_top_frame_with_hooks(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = torch.nn.Linear(18, 18, bias=False)

            def forward(self, x):
                return self.net1(x)

        mod = ToyModel()
        mod.register_forward_pre_hook(lambda mod, input: input[0] + 1)

        # Case 1: torch.compile(mod)
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)

        x = torch.rand(18, 18)
        ref = mod(x)
        res = compiled_mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        # Case 2: mod.compile()
        cnts = torch._dynamo.testing.CompileCounter()
        mod.compile(backend=cnts)
        res = mod(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

    def test_global_module_forward_pre_hook(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                return x - 1

        counter = 0

        def hook(mod, args):
            nonlocal counter
            counter += 1
            return args

        x = torch.rand(18, 18)
        mod = Mod()
        compiled_mod = torch.compile(mod, backend="eager")

        try:
            hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
            ref = mod(x)
            self.assertEqual(counter, 1)
            with self.assertWarnsRegex(
                UserWarning,
                r"Using `torch.compile\(module\)` when there are global hooks.*",
            ):
                res = compiled_mod(x)
            self.assertEqual(counter, 3)
            self.assertEqual(ref, res)
        finally:
            hook_handle.remove()

    def test_register_hook_on_intermediate_stride_dependent(self):
        def hook(grad):
            if grad.is_contiguous():
                return grad.sin()
            else:
                return grad.cos()

        def fn(x):
            y = x * 2
            y.register_hook(hook)
            return y.sum()

        # Hooks that branch on grad metadata (e.g. is_contiguous)
        # graph break because grad properties are unknown at trace time.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)


class HooksTestsDevice(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.ACCELERATOR
    setUp = preserves_device_selection(torch._dynamo.test_case.TestCase.setUp)
    setUpClass = classmethod(
        preserves_device_selection(torch._dynamo.test_case.TestCase.setUpClass.__func__)
    )

    @onlyAccelerator
    def test_register_hook_on_intermediate_autograd_cache(self, device):
        from torch._dynamo.utils import counters

        def fn(x):
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            # First compile
            torch._dynamo.reset()
            counters.clear()
            x = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn, fullgraph=True)(x).backward()  # noqa: UNSPECIFIED_BACKEND

            # Second compile (force recompile to test cache)
            torch._dynamo.reset()
            x2 = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn, fullgraph=True)(x2).backward()  # noqa: UNSPECIFIED_BACKEND

            aot_counters = counters["aot_autograd"]
            self.assertEqual(aot_counters.get("autograd_cache_bypass", 0), 0)
        finally:
            torch._functorch.config.enable_autograd_cache = False

    @onlyAccelerator
    def test_register_hook_on_intermediate_autograd_cache_different_hooks(self, device):
        from torch._dynamo.utils import counters

        def fn_a(x):
            y = x * 2
            y.register_hook(lambda g: g * 0.5)
            return y.sum()

        def fn_b(x):
            y = x * 2
            y.register_hook(lambda g: g * 3.0)
            return y.sum()

        torch._functorch.config.enable_autograd_cache = True
        try:
            torch._dynamo.reset()
            counters.clear()

            # Compile fn_a
            x = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn_a, fullgraph=True)(x).backward()  # noqa: UNSPECIFIED_BACKEND

            # Compile fn_b (different hook — must NOT cache hit from fn_a)
            x2 = torch.randn(4, device=device, requires_grad=True)
            torch.compile(fn_b, fullgraph=True)(x2).backward()  # noqa: UNSPECIFIED_BACKEND

            # fn_b should give grad = 2 * 3.0 = 6.0, not 2 * 0.5 = 1.0
            self.assertEqual(x2.grad, torch.tensor([6.0] * 4, device=device))
            self.assertEqual(
                counters["aot_autograd"].get("autograd_cache_bypass", 0), 0
            )
        finally:
            torch._functorch.config.enable_autograd_cache = False


instantiate_device_type_tests(HooksTestsDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
