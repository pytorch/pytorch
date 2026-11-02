# Owner(s): ["module: dynamo"]

import contextlib
import enum
import functools
import operator
import unittest
from collections import defaultdict, OrderedDict
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from functorch.compile import nop
from torch._dynamo import compiled_autograd
from torch._dynamo.convert_frame import fullgraph_capture, get_traced_fn
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._dynamo.variables import higher_order_ops
from torch._functorch.aot_autograd import aot_module_simplified
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.utils import _pytree
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


def identity_forward_hook(module, args, output):
    return output


class HookKeyFrozenSet(frozenset):
    pass


def update_hook_keys(hooks):
    result = {}
    result.update(hooks)
    return 1 in result


class ClassWithVal:
    def __init__(self, val):
        self.val = val


@instantiate_parametrized_tests
class HookDictionaryTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_always_call_hook(self):
        def hook(module, args, output):
            return output + 1

        compiled = torch.compile(
            lambda x, module: module(x), backend="eager", fullgraph=True
        )
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook, always_call=True)
            self.assertEqual(compiled(x, module), module(x))

    def test_empty_hook_dict_membership_keeps_key_lazy(self):
        def fn(x, key, module):
            return x + (key in module._forward_hooks_with_kwargs)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        module = torch.nn.Identity()
        for key in (3, 4):
            self.assertEqual(compiled(x, key, module), x)
        self.assertEqual(counter.frame_count, 1)
        module._forward_hooks_with_kwargs[4] = True
        self.assertEqual(compiled(x, 4, module), x + 1)

    def test_empty_hook_dict_type(self):
        def fn(x, module):
            hooks = module._forward_hooks
            return x + (type(hooks) is dict), type(hooks)

        module = torch.nn.Identity()
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for mapping_type in (dict, OrderedDict, dict):
            module._forward_hooks = mapping_type()
            self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("capture", [False, True])
    @parametrize("global_hook", [False, True])
    def test_global_hook_flags_reuse(self, capture, global_hook):
        def hook(module, args, *rest):
            if isinstance(module, torch.nn.Identity):
                return rest[-1] * 2 + len(rest)

        def fn(module, x):
            return module(x)

        x = torch.ones(1)
        handle = None
        if global_hook:
            handle = torch.nn.modules.module.register_module_forward_hook(
                hook, with_kwargs=True
            )
            self.addCleanup(handle.remove)

        def make_module():
            module = torch.nn.Identity()
            module.register_forward_hook(hook)
            return module

        module = make_module()
        counter = torch._dynamo.testing.CompileCounter()
        if capture:
            # AOT capture uses module.py's globals for the flags.
            with get_metrics_context(), dynamo_timed(""):
                output = fullgraph_capture(module, (x,))
                guard = output.graph_capture_output.build_guards(
                    get_traced_fn(module)[0].__code__
                ).guard_manager
            compiled = output.forward_callable()
        else:
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for _ in range(2):
            module = make_module()
            if capture:
                self.assertTrue(
                    guard.check({"self": module, "args": (x,), "kwargs": {}})
                )
            self.assertEqual(compiled(module, x), module(x))
        if not capture:
            self.assertEqual(counter.frame_count, 1)
        if handle is not None:
            # Manually reused global IDs must still guard flag membership.
            module._forward_hooks = OrderedDict({handle.id: hook})
            if capture:
                self.assertFalse(
                    guard.check({"self": module, "args": (x,), "kwargs": {}})
                )
            else:
                self.assertEqual(compiled(module, x), module(x))
                self.assertEqual(counter.frame_count, 2)

    @parametrize(
        "operation,container,frames",
        [
            subtest((lambda hooks: hooks.items(), list, 1), name="items"),
            subtest((lambda hooks: hooks.keys(), list, 1), name="keys"),
            subtest((dict.items, list, 1), name="dict_items"),
            subtest((iter, list, 1), name="iter"),
            subtest((iter, tuple, 1), name="tuple"),
            subtest((OrderedDict.keys, tuple, 1), name="ordered_keys"),
            subtest((OrderedDict.items, tuple, 1), name="ordered_items"),
            subtest(
                (lambda hooks: [key + 1 for key, _ in hooks.items()], list, 1),
                name="arithmetic",
            ),
            subtest(
                (lambda hooks: [hash(key) for key, _ in hooks.items()], list, 2),
                name="hash",
            ),
        ],
    )
    def test_hook_dict_items_observe_current_ids(self, operation, container, frames):
        def fn(x, module):
            return x + 1, container(operation(module._forward_hooks))

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
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
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x, module), fn(x, module))
        for _ in range(2):
            handle = module.register_forward_hook(lambda *args: None)
            self.assertEqual(compiled(x, module), fn(x, module))
            handle.remove()
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 2)

    @parametrize("flag_name", ["with_kwargs", "always_call"])
    def test_hook_membership_reuse(self, flag_name):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest)

        def fn(x, module):
            result = module.forward(x) if flag_name == "always_call" else module(x)
            if flag_name == "always_call":
                result = result + sum(
                    index + 1
                    for index, (key, _) in enumerate(module._forward_hooks.items())
                    if key in module._forward_hooks_always_called
                )
            return result

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for flags in ((False, True), (False, True), (True, False)):
            module = torch.nn.Identity()
            for flag in flags:
                module.register_forward_hook(hook, **{flag_name: flag})
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 1 if flags == (False, True) else 2)

    @parametrize("key_type", [str, bool, enum.IntEnum("HookKey", {"value": 1})])
    def test_mixed_hook_key_types(self, key_type):
        def hook(module, args, output):
            return output + 1

        def kwargs_hook(module, args, kwargs, output):
            return output + 2

        module = torch.nn.Identity()
        module._forward_hooks[key_type(1)] = hook
        module._forward_hooks[100] = kwargs_hook
        module._forward_hooks_with_kwargs[100] = True
        x = torch.ones(1)
        self.assertEqual(
            torch.compile(module, backend="eager", fullgraph=True)(x), x + 3
        )

    @torch._dynamo.config.patch(use_recursive_dict_tags_for_guards=True)
    def test_hook_flag_mutation_with_recursive_tags(self):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest)

        def fn(x, module):
            return module(x)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for first, second in ((1, 2), (3, 4)):
            module = torch.nn.Identity()
            module._forward_hooks.update({first: hook, second: hook})
            module._forward_hooks_with_kwargs[first] = True
            self.assertEqual(compiled(x, module), x * 121)
        self.assertEqual(counter.frame_count, 1)
        module._forward_hooks_with_kwargs.clear()
        module._forward_hooks_with_kwargs[second] = True
        self.assertEqual(compiled(x, module), x * 112)
        self.assertEqual(counter.frame_count, 2)

    @parametrize("global_flags", [False, True])
    @parametrize("flags_type", [dict, OrderedDict])
    @parametrize("initial_flags", [(), (1,)])
    @parametrize("recursive_tags", [False, True])
    def test_nested_region_hook_membership(
        self, global_flags, flags_type, initial_flags, recursive_tags
    ):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest)

        modules = [torch.nn.Identity(), torch.nn.Identity()]
        for module, flagged in zip(modules, (1, 2)):
            module._forward_hooks.update({1: hook, 2: hook})
            module._forward_hooks_with_kwargs[flagged] = True
        if global_flags:
            modules[1]._forward_hooks = OrderedDict({3: hook, 4: hook})

        @torch.compiler.nested_compile_region
        def region(x, module):
            if global_flags:
                return x + sum(
                    key in _global_hook_flags_alias for key in module._forward_hooks
                )
            return module(x)

        def fn(x, first, second):
            return region(x, first), region(x, second)

        flags = flags_type.fromkeys(initial_flags if global_flags else (), True)
        with (
            torch._dynamo.config.patch(
                use_recursive_dict_tags_for_guards=recursive_tags
            ),
            patch.object(
                torch.nn.modules.module, "_global_forward_hooks_with_kwargs", flags
            ),
            patch(f"{__name__}._global_hook_flags_alias", flags),
        ):
            x = torch.ones(1)
            expected = (
                (x + len(initial_flags), x) if global_flags else (x * 121, x * 112)
            )
            counter = torch._dynamo.testing.CompileCounter()
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            self.assertEqual(fn(x, *modules), expected)
            self.assertEqual(compiled(x, *modules), expected)
            if global_flags:
                flags.clear()
                flags[3] = True
                self.assertEqual(compiled(x, *modules), (x, x + 1))
                self.assertEqual(counter.frame_count, 2)

    @parametrize("initial_flags", [(), (1,)])
    @parametrize("recursive_tags", [False, True])
    @parametrize("mutation", ["flags", "callback", "tensor_shape"])
    def test_nested_region_hook_mutation(self, initial_flags, recursive_tags, mutation):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest) + module.probe.shape[0]

        def replacement_hook(module, args, *rest):
            return rest[-1] * 20 + len(rest)

        modules = [torch.nn.Identity(), torch.nn.Identity()]
        for module in modules:
            module.probe = torch.ones(1)
            module._forward_hooks.update({1: hook, 2: hook})
            module._forward_hooks_with_kwargs.update(dict.fromkeys(initial_flags, True))

        @torch.compiler.nested_compile_region
        def region(x, module):
            return module(x)

        def fn(x, y, first, second):
            return region(x, first), region(y, second)

        counter = torch._dynamo.testing.CompileCounter()
        x, y = torch.ones(1), torch.full((1,), 2.0)
        with (
            torch._dynamo.config.patch(
                use_recursive_dict_tags_for_guards=recursive_tags
            ),
            patch.object(
                higher_order_ops,
                "speculate_subgraph_with_auto_output_flattening",
                wraps=higher_order_ops.speculate_subgraph_with_auto_output_flattening,
            ) as traces,
        ):
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            self.assertEqual(compiled(x, y, *modules), fn(x, y, *modules))
            self.assertEqual(traces.call_count, 1)
            if mutation == "flags":
                modules[1]._forward_hooks_with_kwargs.clear()
                modules[1]._forward_hooks_with_kwargs[2] = True
            elif mutation == "callback":
                modules[1]._forward_hooks[1] = replacement_hook
            else:
                modules[1].probe = torch.ones(2)
            self.assertEqual(compiled(x, y, *modules), fn(x, y, *modules))
        self.assertEqual(counter.frame_count, 2)
        y = torch.ones(2)
        self.assertEqual(compiled(x, y, *modules), fn(x, y, *modules))
        self.assertEqual(counter.frame_count, 3)

    def test_nested_region_hook_reuse_propagates_guards(self):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest)

        modules = [torch.nn.Identity() for _ in range(4)]
        for module in modules:
            module._forward_hooks.update({1: hook, 2: hook})
            module._forward_hooks_with_kwargs[1] = True

        @torch.compiler.nested_compile_region
        def inner(x, module):
            return module(x)

        @torch.compiler.nested_compile_region
        def outer(x, first, second):
            return inner(x, first) + inner(x, second)

        def fn(x, modules):
            return outer(x, *modules[:2]), outer(x, *modules[2:])

        counter = torch._dynamo.testing.CompileCounter()
        x = torch.ones(1)
        with patch.object(
            higher_order_ops,
            "speculate_subgraph_with_auto_output_flattening",
            wraps=higher_order_ops.speculate_subgraph_with_auto_output_flattening,
        ) as traces:
            compiled = torch.compile(fn, backend=counter, fullgraph=True)
            self.assertEqual(compiled(x, modules), fn(x, modules))
            self.assertEqual(traces.call_count, 2)
            modules[-1]._forward_hooks_with_kwargs.clear()
            modules[-1]._forward_hooks_with_kwargs[2] = True
            self.assertEqual(compiled(x, modules), fn(x, modules))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("flags_name", ["with_kwargs", "always_called"])
    @parametrize("populated", [False, True])
    def test_global_flags_preserve_mapping_type(
        self, mapping_type, flags_name, populated
    ):
        def fn(x, flags):
            copied = flags.copy()
            if isinstance(copied, defaultdict):
                return x + 1, copied.default_factory is int, repr(flags)
            return (
                x + 1,
                copied,
                _pytree.tree_map(lambda value: not value, flags),
                repr(flags),
            )

        flags = mapping_type()
        if mapping_type is defaultdict:
            flags.default_factory = int
        if populated:
            flags[1] = True
        with patch.object(
            torch.nn.modules.module, f"_global_forward_hooks_{flags_name}", flags
        ):
            x = torch.ones(1)
            actual = torch.compile(fn, backend="eager", fullgraph=True)(x, flags)
            self.assertEqual(actual, fn(x, flags))
            if mapping_type is not defaultdict:
                for result in actual[1:3]:
                    self.assertIs(type(result), mapping_type)

    @parametrize(
        "operation",
        [
            subtest(lambda hooks: 1 in set(hooks), name="set"),
            subtest(lambda hooks: 1 in {*hooks}, name="set_unpack"),
            subtest(lambda hooks: 1 in hooks.copy(), name="copy"),
            subtest(update_hook_keys, name="update"),
            subtest(
                lambda hooks: 1 in {key: True for key, _ in hooks.items()},
                name="dict_comprehension",
            ),
            subtest(
                lambda hooks: 1 in {key for key, _ in hooks.items()},
                name="set_comprehension",
            ),
            subtest(lambda hooks: 1 in frozenset(hooks), name="frozenset"),
            subtest(
                lambda hooks: 1 in HookKeyFrozenSet(hooks), name="frozenset_subclass"
            ),
            subtest(lambda hooks: 1 in dict.fromkeys(set(hooks)), name="fromkeys"),
            subtest(lambda hooks: 1 in {0}.union(hooks), name="union"),
            subtest(lambda hooks: 1 in {1}.intersection(hooks), name="intersection"),
            subtest(lambda hooks: 1 in {1}.difference(hooks), name="difference"),
            subtest(lambda hooks: {1}.isdisjoint(hooks), name="isdisjoint"),
            subtest(lambda hooks: 1 in ({0: True}.keys() | hooks), name="view_union"),
            subtest(
                lambda hooks: 1 in ({1: True}.keys() & hooks), name="view_intersection"
            ),
            subtest(
                lambda hooks: 1 in (hooks | {0: True}.keys()), name="reverse_view_union"
            ),
            subtest(
                lambda hooks: 1 in _pytree.tree_map(lambda value: True, hooks),
                name="tree_map",
            ),
            subtest(
                lambda hooks: 1
                in _pytree.tree_map_with_path(lambda path, value: True, hooks),
                name="tree_map_with_path",
            ),
        ],
    )
    def test_materialized_hook_keys(self, operation):
        def fn(x, module):
            return x + operation(module._forward_hooks)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for keys in ((2, 3), (1, 3)):
            module = torch.nn.Identity()
            for key in keys:
                module._forward_hooks[key] = identity_forward_hook
            self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("recursive_tags", [False, True])
    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("flags_name", ["with_kwargs", "always_called"])
    def test_global_flags_alias_survives_registry_replacement(
        self, mapping_type, flags_name, recursive_tags
    ):
        def fn(x, module):
            return x + sum(
                key in _global_hook_flags_alias for key in module._forward_hooks
            )

        flags = mapping_type({1: True})
        registry_name = f"_global_forward_hooks_{flags_name}"
        with (
            torch._dynamo.config.patch(
                use_recursive_dict_tags_for_guards=recursive_tags
            ),
            patch.object(torch.nn.modules.module, registry_name, flags),
            patch(f"{__name__}._global_hook_flags_alias", flags),
        ):
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.ones(1)
            for key in (2, 1):
                module = torch.nn.Identity()
                module._forward_hooks[key] = identity_forward_hook
                if key == 1:
                    setattr(torch.nn.modules.module, registry_name, mapping_type())
                self.assertEqual(compiled(x, module), fn(x, module))


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
