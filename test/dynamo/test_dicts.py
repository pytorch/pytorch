# Owner(s): ["module: dynamo"]


import enum
import itertools
import operator
import sys
import types
import unittest
import weakref
from collections import defaultdict, deque, namedtuple, OrderedDict, UserDict
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.fx as fx
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.exc import UncapturedHigherOrderOpError, Unsupported
from torch._dynamo.testing import same
from torch._dynamo.utils import dict_items
from torch.fx.experimental.proxy_tensor import _ModuleStackTracer
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    make_dynamo_test,
    munge_exc,
    parametrize,
    subtest,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class _BadCmpExc(Exception):
    pass


class _BadCmp:
    # Hashable (so it can be a dict value compared via PyObject_RichCompareBool)
    # but its __eq__ raises.
    def __eq__(self, other: object) -> bool:
        raise _BadCmpExc

    def __hash__(self) -> int:
        return 1


class SimpleDict(dict):
    pass


class DummyUserDict(UserDict):
    pass


class FakeMapping:
    def __init__(self, value: Any) -> None:
        self._value = value
        self.keys = lambda: ["a", "b", "c"]  # not required to be a method

    def __getitem__(self, key: str) -> Any:
        return self._value


class DictTests(torch._dynamo.test_case.TestCase):
    @parametrize("operation", ["del", "pop"])
    def test_ordered_dict_delete_reordered_input(self, operation):
        def fn(x, mapping):
            if operation == "del":
                del mapping["a"]
            else:
                mapping.pop("a")
            return x + 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for keys in ("abc", "acb"):
            mapping = OrderedDict((key, ord(key)) for key in keys)
            expected = mapping.copy()
            self.assertEqual(compiled(x, mapping), fn(x, expected))
            self.assertEqual(list(mapping.items()), list(expected.items()))

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("operation", ["not", "bool", "len"])
    def test_input_keys_view_truth_guards(self, mapping_type, operation):
        def fn(x, view):
            if operation == "not":
                return x + (not view)
            if operation == "bool":
                return x + bool(view)
            return x + len(view)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for keys in ((), ("a",), ("a", "b"), ()):
            view = mapping_type.fromkeys(keys).keys()
            self.assertEqual(compiled(x, view), fn(x, view))

    @parametrize("mapping_type", [dict, OrderedDict])
    def test_extend_constant_mapping(self, mapping_type):
        @torch._dynamo.assume_constant_result
        def constant_mapping():
            return mapping_type(a=1, b=2)

        def fn(x):
            result = []
            result.extend(constant_mapping())
            return x + 1, result

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("operation", ["reversed", "repr", "popitem"])
    def test_ordered_dict_linked_order_guards(self, operation):
        def fn(x, mapping):
            if operation == "reversed":
                result = list(reversed(mapping))
            elif operation == "repr":
                result = repr(mapping)
            else:
                result = mapping.popitem()
            return x + 1, result

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for reordered in (False, True, False):
            mapping = OrderedDict(a=1, b=2, c=3)
            if reordered:
                mapping.move_to_end("a")
            expected = mapping.copy()
            self.assertEqual(compiled(x, mapping), fn(x, expected))
            self.assertEqual(list(mapping.items()), list(expected.items()))

    @parametrize("operation", ["copy", "update", "unpack", "extend"])
    def test_dict_materialization_order_guards(self, operation):
        def fn(x, mapping):
            if operation == "copy":
                result = mapping.copy()
            elif operation == "update":
                result = {}
                result.update(mapping)
            elif operation == "unpack":
                result = [*mapping]
            else:
                result = []
                result.extend(mapping)
            return x + 1, list(result)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for keys in ("ab", "ba", "ab"):
            mapping = dict.fromkeys(keys, 1)
            self.assertEqual(compiled(x, mapping), fn(x, mapping))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("size", [0, 1, 2])
    def test_ordered_dict_trivial_order_guard(self, size):
        def fn(x, mapping):
            return x + sum(mapping.values())

        guards = []
        compiled = torch._dynamo.optimize(
            "eager", nopython=True, guard_export_fn=guards.extend
        )(fn)
        x = torch.ones(1)
        mapping = OrderedDict.fromkeys(range(size), 1)
        self.assertEqual(compiled(x, mapping), x + size)
        self.assertEqual(
            any(
                guard.create_fn
                is torch._dynamo.guards.GuardBuilder.ORDERED_DICT_ORDER_MATCH
                for guard in guards
            ),
            size > 1,
        )
        mapping[size] = 1
        self.assertEqual(compiled(x, mapping), x + size + 1)

    @parametrize("same_order", [False, True])
    def test_ordered_dict_comparison_uses_linked_order(self, same_order):
        def fn(x, left, right):
            return x + (left == right)

        left = OrderedDict(a=1, b=2)
        right = OrderedDict(b=2, a=1) if same_order else left.copy()
        left.move_to_end("a")
        x = torch.ones(1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, left, right),
            fn(x, left, right),
        )

    def test_ordered_dict_equal_literal_keys_reuse_graph(self):
        def fn(x, mapping):
            return x + sum(mapping.values())

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            mapping = OrderedDict.fromkeys([b"layer.0.weight".decode(), "other"], 1)
            self.assertEqual(compiled(x, mapping), fn(x, mapping))
        self.assertEqual(counter.frame_count, 1)

    @parametrize("operation", ["copy", "unpack", "update", "expand"])
    def test_ordered_dict_materialization_order(self, operation):
        def fn(x, mapping):
            if operation == "copy":
                result = mapping.copy()
            elif operation == "unpack":
                result = [*mapping]
            elif operation == "update":
                result = {}
                result.update(mapping)
            else:
                result = {**mapping}
            return x + 1, list(result)

        mapping = OrderedDict(a=1, b=2)
        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, mapping), fn(x, mapping))
        mapping.move_to_end("a")
        self.assertEqual(compiled(x, mapping), fn(x, mapping))

    def test_ordered_dict_copy_key_identity_guard(self):
        def fn(x, mapping):
            return x + 1, [id(key) for key in mapping.copy()]

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            mapping = OrderedDict.fromkeys([b"layer.0.weight".decode(), "other"], 1)
            self.assertEqual(compiled(x, mapping), fn(x, mapping))

    @parametrize("operation", ["items", "dict_items", "arithmetic", "repr", "hash"])
    def test_hook_dict_items_observe_current_ids(self, operation):
        def hook(module, args, output):
            return output + 1

        def fn(x, module):
            hooks = module._forward_hooks
            if operation == "repr":
                result = repr(hooks.items())
            elif operation == "hash":
                result = [hash(key) for key, _ in hooks.items()]
            elif operation == "arithmetic":
                result = [key + 1 for key, value in hooks.items()]
            else:
                result = list(
                    dict.items(hooks) if operation == "dict_items" else hooks.items()
                )
            return x + 1, result

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook)
            self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("operation", ["copy", "update", "dict", "set"])
    def test_hook_dict_materialization_keys(self, operation):
        def hook(module, args, output):
            return output

        def fn(x, module):
            if operation == "copy":
                result = module._forward_hooks.copy()
            elif operation == "dict":
                result = {key: True for key, _ in module._forward_hooks.items()}
            elif operation == "set":
                result = {key for key, _ in module._forward_hooks.items()}
            else:
                result = {}
                result.update(module._forward_hooks)
            return x + (1 in result)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for key in (0, 1):
            module = torch.nn.Identity()
            module._forward_hooks[key] = hook
            self.assertEqual(compiled(x, module), fn(x, module))

    def test_empty_dict_membership_keeps_key_lazy(self):
        def fn(x, key, mapping):
            return x + (key in mapping)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for key in (3, 4):
            self.assertEqual(compiled(x, key, {}), x)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(compiled(x, 4, {4: None}), x + 1)

    @parametrize("flag_name", ["with_kwargs", "always_call"])
    def test_hook_membership_reuse(self, flag_name):
        def hook(module, args, *rest):
            return rest[-1] * 10 + len(rest)

        def fn(x, module):
            result = module(x)
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

    def test_always_call_hook_reuse(self):
        def hook(module, args, output):
            return output + 1

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            lambda x, module: module(x), backend=counter, fullgraph=True
        )
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook, always_call=True)
            self.assertEqual(compiled(x, module), module(x))
        self.assertEqual(counter.frame_count, 1)

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

    def test_hook_dict_truth(self):
        def fn(x, module):
            return x + 1 if module._forward_hooks else x - 1

        module = torch.nn.Linear(2, 2)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x, module), fn(x, module))
        handle = module.register_forward_hook(lambda *args: None)
        self.assertEqual(compiled(x, module), fn(x, module))
        handle.remove()
        self.assertEqual(compiled(x, module), fn(x, module))

    def test_constructed_dict_key_guard(self):
        def fn(x, key):
            mapping = {key: 7}
            return x + mapping.get("a", 0)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(1)
        for key in ("b", "a"):
            self.assertEqual(compiled(x, key), fn(x, key))

    def test_ordered_dict_key_identity_guard(self):
        def fn(x, mapping):
            key, other = mapping
            return x + 1, key, id(key)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(2)
        for _ in range(2):
            mapping = OrderedDict.fromkeys([b"layer.0.weight".decode(), "other"], 1)
            result = compiled(x, mapping)
            self.assertEqual(result, fn(x, mapping))
            self.assertIs(result[1], next(iter(mapping)))

    @parametrize("subclass", [False, True])
    @parametrize("storage_order", [False, True])
    def test_ordered_dict_copy_reordered_input(self, storage_order, subclass):
        class Mapping(OrderedDict):
            pass

        mapping_type = Mapping if subclass else OrderedDict

        def fn(x, mapping):
            return x + 1, mapping.copy()

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.randn(3)
        for order in ("abc", "cba", "cba"):
            mapping = mapping_type(
                (key, x + index)
                for index, key in enumerate(order if storage_order else "abc")
            )
            if not storage_order:
                for key in order:
                    mapping.move_to_end(key)
            actual = compiled(x, mapping)
            self.assertEqual(actual, fn(x, mapping))
            self.assertEqual(list(actual[1]), list(mapping))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("failure", ["exception", "graph_break"])
    def test_ordered_dict_copy_key_comparison_exception(self, failure):
        class Key:
            def __init__(self, value):
                self.value = value
                self.fail = False

            def __hash__(self):
                return 1

            def __eq__(self, other):
                if self.fail:
                    if failure == "exception":
                        raise ValueError("key equality failed during copy")
                    torch._dynamo.graph_break()
                return self.value == other.value

        class Mapping(OrderedDict):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if key.value == 1:
                    self.first.fail = True
                return value

        def make_input():
            first, second = Key(1), Key(2)
            mapping = Mapping([(first, 1), (second, 2)])
            mapping.first = first
            return mapping

        def fn(x, mapping):
            try:
                mapping.copy()
            except ValueError as error:
                return x + 1, str(error)
            return x + 2, "not raised"

        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        if failure == "exception":
            self.assertEqual(compiled(x, make_input()), fn(x, make_input()))
        else:
            with self.assertRaisesRegex(Unsupported, "Call to.*graph_break") as caught:
                compiled(x, make_input())
            self.assertNotIn(
                "Dictionary mutated during iteration", str(caught.exception)
            )

    def test_hook_dict_copy_guard_keys(self):
        def hook(module, args, output):
            return output + 1

        def fn(x, module):
            return x + 1, module._forward_hooks.copy()

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(2)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook)
            self.assertEqual(compiled(x, module), fn(x, module))

    @parametrize("owner", [dict, OrderedDict])
    @parametrize("method", ["keys", "values", "items", "input_keys"])
    @parametrize("reordered", [False, True])
    def test_dict_view_iteration_order(self, owner, method, reordered):
        def fn(x, obj):
            view = obj if method == "input_keys" else getattr(owner, method)(obj)
            return x + 1, list(view), type(view)

        mapping = OrderedDict(a=1, b=2)
        if reordered:
            mapping.move_to_end("a")
        obj = owner.keys(mapping) if method == "input_keys" else mapping
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        self.assertEqual(compiled(x, obj), fn(x, obj))
        mapping.move_to_end("b" if reordered else "a")
        self.assertEqual(compiled(x, obj), fn(x, obj))
        self.assertEqual(counter.frame_count, 2 if owner is OrderedDict else 1)

    @parametrize("with_kwargs", [False, True])
    @parametrize("operation", ["forward", "values"])
    def test_hook_dict_iteration_order(self, with_kwargs, operation):
        def add(module, args, *rest):
            return rest[-1] + 1

        def multiply(module, args, *rest):
            return rest[-1] * 2

        def fn(x, module):
            if operation == "forward":
                return module(x)
            for hook in module._forward_hooks.values():
                x = hook(module, (x,), x)
            return x

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for _ in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(add, with_kwargs=with_kwargs)
            module.register_forward_hook(multiply, with_kwargs=with_kwargs)
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, 1)
        module._forward_hooks.move_to_end(next(iter(module._forward_hooks)))
        self.assertEqual(compiled(x, module), fn(x, module))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("return_keys", [False, True])
    def test_hook_dict_keys_guard(self, return_keys):
        def hook(module, args, output):
            return output + 1

        def fn(x, module):
            keys = module._forward_hooks.keys()
            return x + 1, list(keys) if return_keys else len(keys)

        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for index in range(2):
            module = torch.nn.Identity()
            module.register_forward_hook(hook)
            self.assertEqual(compiled(x, module), fn(x, module))
            self.assertEqual(counter.frame_count, index + 1 if return_keys else 1)

    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "operation",
        ["getitem", "contains", "get", "setdefault", "pop"],
    )
    def test_discarded_dict_lookup_keeps_value_lazy(self, operation):
        def fn(x, mapping):
            if operation == "getitem":
                mapping["unused"]
            elif operation == "contains":
                _ = "unused" in mapping
            else:
                getattr(mapping, operation)("unused")
            return x + 1

        x = torch.ones(1)
        # Realizing a set of tensors graph-breaks in fullgraph mode, so this
        # value detects accidental realization even when the lookup is discarded.
        mapping = {"unused": {torch.ones(1)}}
        expected = fn(x, mapping.copy())
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping), expected
        )

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("operation", ["contains", "items_contains", "get", "pop", "delitem"])
    @parametrize("present", [False, True])
    def test_input_dict_lookup_comparison_count(self, mapping_type, operation, present):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return present

        def fn(x, mapping, stored, probe):
            stored.calls = 0
            if operation == "contains":
                result = probe in mapping
            elif operation == "items_contains":
                result = (probe, 7) in mapping.items()
            elif operation == "delitem":
                try:
                    mapping.__delitem__(probe)
                    result = True
                except KeyError:
                    result = False
            else:
                result = getattr(mapping, operation)(probe, 11)
            return x + stored.calls, result

        x = torch.ones(1)
        stored, probe = Key(), Key()
        eager_mapping, compiled_mapping = mapping_type(), mapping_type()
        eager_mapping[stored] = compiled_mapping[stored] = 7
        expected = fn(x, eager_mapping, stored, probe)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(
            x, compiled_mapping, stored, probe
        )
        self.assertEqual(actual, expected)

    @parametrize("key", ["a", 3, (1, 2)])
    @parametrize("items_view", [False, True])
    def test_input_dict_contains_key_guard(self, key, items_view):
        def fn(x, mapping, key):
            if items_view:
                return x + ((key, 1) in mapping.items())
            return x + (key in mapping)

        x = torch.zeros(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for mapping in ({key: 1}, {"other": 1}, {key: 2}):
            self.assertEqual(compiled(x, mapping, key), fn(x, mapping, key))

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("operation", ["contains", "get", "pop", "setitem", "delitem"])
    @parametrize(
        "make_key",
        [
            subtest(lambda: float("nan"), name="float"),
            subtest(lambda: complex(float("nan"), 0), name="complex_real"),
            subtest(lambda: complex(0, float("nan")), name="complex_imag"),
            subtest(lambda: (float("nan"),), name="tuple"),
            subtest(lambda: (0, (float("nan"),)), name="nested_tuple"),
        ],
    )
    def test_input_dict_nan_key_guard(self, mapping_type, operation, make_key):
        def fn(x, mapping, key):
            if operation == "contains":
                return x + (key in mapping)
            if operation == "setitem":
                mapping[key] = 2
                return x + 1
            if operation == "delitem":
                try:
                    del mapping[key]
                except KeyError:
                    return x - 1
                return x + 1
            return x + getattr(mapping, operation)(key, 0)

        absent, present = make_key(), make_key()
        mapping = mapping_type()
        mapping[present] = 1
        x = torch.zeros(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        for key in (present, absent, present, absent):
            expected_mapping, actual_mapping = mapping.copy(), mapping.copy()
            self.assertEqual(
                compiled(x, actual_mapping, key), fn(x, expected_mapping, key)
            )
            self.assertEqual(
                list(actual_mapping.items()), list(expected_mapping.items())
            )

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("with_default", [False, True])
    def test_missing_dict_pop_in_hop(self, mapping_type, with_default):
        def fn(x, mapping):
            def body(x):
                try:
                    value = (
                        mapping.pop("missing", 2)
                        if with_default
                        else mapping.pop("missing")
                    )
                except KeyError:
                    value = 2
                return x + value

            return torch.cond(x.sum() > 0, body, body, (x,))

        x = torch.ones(2)
        mapping = mapping_type()
        mapping["present"] = 1
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping), x + 2
        )
        self.assertEqual(list(mapping.items()), [("present", 1)])

    def test_rejected_dict_pop_preserves_tracked_items(self):
        from torch._dynamo.side_effects import SideEffects
        from torch._dynamo.variables.dicts import ConstDictVariable

        original_check = SideEffects.check_allowed_side_effect
        rejected_sizes = []

        def check(side_effects, variable):
            try:
                return original_check(side_effects, variable)
            except Unsupported:
                if isinstance(variable, ConstDictVariable):
                    rejected_sizes.append(len(variable.items))
                raise

        def fn(x, mapping):
            def body(x):
                return x + mapping.pop("a")

            return torch.cond(x.sum() > 0, body, body, (x,))

        with patch.object(SideEffects, "check_allowed_side_effect", check):
            with self.assertRaises(UncapturedHigherOrderOpError):
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(2), {"a": 1}
                )
        self.assertEqual(rejected_sizes, [1])

    def test_dunder_dict_pop_deleted_key_in_hop(self):
        class Obj:
            def __init__(self):
                self.a = 1
                self.b = 2

        def fn(x, obj):
            del obj.a

            def body(x):
                return x + obj.__dict__.pop("a", 2)

            return torch.cond(x.sum() > 0, body, body, (x,))

        x = torch.ones(2)
        obj = Obj()
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, obj), x + 2
        )
        self.assertNotIn("a", obj.__dict__)

    @parametrize("method", ["keys", "values", "items", "update"])
    def test_ordered_dict_unbound_method_missing_receiver(self, method):
        def fn(x):
            try:
                getattr(OrderedDict, method)()
            except TypeError:
                return x + 1
            return x

        x = torch.ones(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("present", [False, True])
    def test_ordered_dict_subclass_setdefault_dispatch(self, present):
        class Mapping(OrderedDict):
            def __contains__(self, key):
                self.calls.append("contains")
                return present

            def __getitem__(self, key):
                self.calls.append("getitem")
                return 17

            def __setitem__(self, key, value):
                self.calls.append("setitem")
                super().__setitem__(key, value + 1)

        def fn(x):
            mapping = Mapping()
            mapping.calls = []
            result = mapping.setdefault("key", 3)
            return x + result, mapping.calls, list(mapping.items())

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_ordered_dict_subclass_setdefault_lookup_error(self):
        class Mapping(OrderedDict):
            pass

        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 7

            def __eq__(self, other):
                self.calls += 1
                if self.calls == 2:
                    raise ValueError("second lookup")
                return True

        def fn(x):
            stored, probe = Key(), Key()
            mapping = Mapping()
            mapping[stored] = 2
            try:
                value = mapping.setdefault(probe, 3)
            except ValueError:
                return x + 10
            return x + value

        x = torch.ones(1)
        self.assertEqual(fn(x), x + 10)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "make_iter",
        [
            zip,
            lambda keys, values: map(tuple, zip(keys, values)),
            lambda keys, values: filter(None, zip(keys, values)),
            lambda keys, values: itertools.chain(zip(keys, values)),
        ],
    )
    @parametrize("update", [False, True])
    def test_ordered_dict_from_iterator(self, make_iter, update):
        def fn(x):
            pairs = make_iter(("a", "b"), (x, x + 1))
            if update:
                result = OrderedDict()
                result.update(pairs)
            else:
                result = OrderedDict(pairs)
            return result

        x = torch.randn(3)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(actual), OrderedDict)
        self.assertEqual(list(actual.items()), list(expected.items()))

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    def test_subscription_compares_key_once(self, mapping_type):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return True

        def fn(x):
            stored, probe = Key(), Key()
            mapping = mapping_type()
            mapping[stored] = x
            return mapping[probe] + 1, stored.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 1, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("method", ["keys", "values", "items", "update"])
    def test_unbound_dict_method_input(self, mapping_type, method):
        def fn(x, mapping):
            if method == "update":
                mapping_type.update(mapping, b=x + 1)
                return x + 2
            return x + 1, list(getattr(mapping_type, method)(mapping))

        x = torch.ones(1)
        expected_mapping = mapping_type(a=x)
        actual_mapping = mapping_type(a=x)
        expected = fn(x, expected_mapping)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, actual_mapping)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_mapping, expected_mapping)

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("method", ["keys", "values", "items", "update"])
    @parametrize("receiver", [UserDict, types.MappingProxyType])
    def test_unbound_dict_method_invalid_receiver(self, mapping_type, method, receiver):
        def fn(x, mapping):
            try:
                getattr(mapping_type, method)(mapping)
            except TypeError:
                return x + 1
            return x - 1

        x = torch.ones(1)
        mapping = receiver({"a": 1})
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    @parametrize("method", ["keys", "values", "items", "update"])
    def test_aliased_ordered_dict_method_invalid_receiver(self, method):
        mapping_type = type("Mapping", (dict,), {method: getattr(OrderedDict, method)})

        def fn(x, mapping):
            try:
                getattr(mapping, method)()
            except TypeError as exc:
                return x + 1, str(exc)
            return x - 1, "no error"

        x = torch.ones(1)
        mapping = mapping_type(a=1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    def test_unbound_dict_update_immutable_receiver(self):
        def identity(value):
            return value

        def fn(x, func):
            dict.update(func.keywords, value=x + 1)
            return x + 2

        x = torch.ones(1)
        func = partial(identity, value=None)
        with self.assertRaises(Unsupported) as exc:
            torch.compile(fn, backend="eager", fullgraph=True)(x, func)
        self.assertEqual(exc.exception.gb_type, "Unsupported method call")
        self.assertEqual(torch.compile(fn, backend="eager")(x, func), x + 2)
        self.assertEqual(func(), x + 1)

    @parametrize(
        "source_type",
        [
            "list",
            "tuple",
            "generator",
            "iterator",
            "dict_keys",
            "dict_values",
            "dict_items",
            "deque",
            "set",
        ],
    )
    @parametrize("update", [False, True])
    def test_ordered_dict_update_iterables(self, source_type, update):
        def fn(x):
            pairs = [("a", x), ("b", x + 1)]
            if source_type == "tuple":
                pairs = tuple(pairs)
            elif source_type == "generator":
                pairs = (pair for pair in pairs)
            elif source_type == "iterator":
                pairs = iter(pairs)
            elif source_type == "dict_keys":
                pairs = {pairs[0]: None, pairs[1]: None}.keys()
            elif source_type == "dict_values":
                pairs = {0: pairs[0], 1: pairs[1]}.values()
            elif source_type == "dict_items":
                pairs = dict(pairs).items()
            elif source_type == "deque":
                pairs = deque(pairs)
            elif source_type == "set":
                pairs = set(pairs)
            if update:
                result = OrderedDict()
                result.update(pairs)
            else:
                result = OrderedDict(pairs)
            return result

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("method", ["__getattr__", "__getattribute__"])
    def test_ordered_dict_update_slotted_getattr(self, method):
        class Mapping:
            __slots__ = ()

            def get_attribute(self, name):
                if name == "keys":
                    return lambda: ["a"]
                return object.__getattribute__(self, name)

            if method == "__getattr__":
                __getattr__ = get_attribute
            else:
                __getattribute__ = get_attribute

            def __getitem__(self, key):
                return 2

        def fn(x, source):
            return x + OrderedDict(source)["a"]

        x = torch.ones(1)
        source = Mapping()
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, source), fn(x, source)
        )

    @parametrize("name", ["__name__", "__qualname__", "__mro__", "mro", "missing"])
    def test_iterator_missing_instance_attribute(self, name):
        def fn(x):
            iterator = zip([1], [2])
            return x + 1, hasattr(iterator, name)

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 1, False))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "factory, args",
        [
            (bytearray, ()),
            (itertools.chain, ()),
            (itertools.count, ()),
            (itertools.repeat, (1,)),
            (deque, ()),
            (torch.Size, ([],)),
        ],
    )
    @parametrize("call_method", [False, True])
    def test_builtin_iterable_missing_attribute_message(
        self, factory, args, call_method
    ):
        def fn(x):
            obj = factory(*args)
            try:
                if call_method:
                    obj.sort()
                else:
                    obj.missing
            except AttributeError as exc:
                return x + 1, str(exc)

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("case", [(dict, False), (dict, True), (OrderedDict, True)])
    def test_dict_update_preserves_hash_calls(self, mapping_type, case):
        source_type, fail = case

        class Key:
            def __init__(self):
                self.calls = 0
                self.fail = False

            def __hash__(self):
                self.calls += 1
                if self.fail:
                    raise ValueError("hash failed")
                return 42

        def fn(x):
            key = Key()
            source = source_type([(key, x)])
            key.calls = 0
            key.fail = fail
            result = mapping_type()
            try:
                result.update(source)
            except ValueError:
                return x + 2, key.calls
            return x + 1, key.calls

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("keys", [("a", "b"), (1, 2)])
    def test_dict_update_ordered_input_order(self, mapping_type, keys):
        def fn(x, source):
            result = mapping_type(c=x)
            result.update(source, c=x + 2)
            return list(result.items())

        x = torch.ones(1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        source = OrderedDict(zip(keys, (x, x + 1)))
        self.assertEqual(compiled(x, source), fn(x, source))
        source.move_to_end(keys[0])
        self.assertEqual(compiled(x, source), fn(x, source))

    @parametrize("view_name", ["keys", "values", "items"])
    @parametrize("view_owner", [dict, OrderedDict])
    @parametrize("wrapped", [False, True])
    def test_ordered_dict_subclass_view_reconstruction(
        self, view_name, view_owner, wrapped
    ):
        class Mapping(OrderedDict):
            if wrapped:

                def keys(self):
                    return view_owner.keys(self)

                def values(self):
                    return view_owner.values(self)

                def items(self):
                    return view_owner.items(self)
            else:
                keys = view_owner.keys
                values = view_owner.values
                items = view_owner.items

        view_type = type(getattr(view_owner(), view_name)())

        def fn(x, mapping):
            view = getattr(mapping, view_name)()
            return x + 1, view, type(view), isinstance(view, view_type)

        mapping = Mapping(a=1, b=2)
        mapping.move_to_end("a")
        x = torch.ones(1)
        expected_tensor, expected_view, _, _ = fn(x, mapping)
        actual_tensor, actual_view, actual_type, matches_type = torch.compile(
            fn, backend="eager", fullgraph=True
        )(x, mapping)
        self.assertEqual(actual_tensor, expected_tensor)
        self.assertIs(type(actual_view), type(expected_view))
        self.assertIs(actual_type, view_type)
        self.assertTrue(matches_type)
        self.assertEqual(list(actual_view), list(expected_view))

    @parametrize("view_name", ["keys", "values", "items"])
    def test_ordered_dict_view_hasattr(self, view_name):
        def fn(x):
            view = getattr(OrderedDict(a=1), view_name)()
            return x + 1, tuple(
                hasattr(view, name)
                for name in (
                    "mapping",
                    "__len__",
                    "isdisjoint",
                    "__contains__",
                    "__or__",
                    "missing",
                )
            )

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("ordered_first", [False, True])
    def test_ordered_dict_input_keys_view_type(self, ordered_first):
        mapping = OrderedDict(a=1, b=2)
        ordered_type = type(mapping.keys())

        def fn(x, view):
            return (
                x + 1,
                type(view),
                isinstance(view, ordered_type),
                repr(view),
                list(view),
            )

        x = torch.ones(1)
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        owners = (OrderedDict, dict) if ordered_first else (dict, OrderedDict)
        for owner in owners:
            view = owner.keys(mapping)
            self.assertEqual(compiled(x, view), fn(x, view))
        self.assertEqual(counter.frame_count, 2)

    @parametrize("bound", [False, True])
    def test_ordered_dict_subclass_copy(self, bound):
        class Mapping(OrderedDict):
            def __init__(self):
                self.events = ["init"]
                super().__init__()

            def __getitem__(self, key):
                self.events.append(("get", key))
                return super().__getitem__(key) + 1

            def __setitem__(self, key, value):
                self.events.append(("set", key))
                super().__setitem__(key, value)

            def __iter__(self):
                raise AssertionError("copy must use OrderedDict's stored keys")

        def fn(x):
            mapping = Mapping()
            OrderedDict.__setitem__(mapping, "a", x)
            result = mapping.copy() if bound else OrderedDict.copy(mapping)
            return result, result.events, mapping.events

        x = torch.ones(1)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(actual[0]), Mapping)
        self.assertEqual(actual[1:], (["init", ("set", "a")], ["init", ("get", "a")]))
        self.assertEqual(actual, expected)

    @parametrize("method", ["__getitem__", "__setitem__"])
    def test_ordered_dict_subclass_copy_mutates_source(self, method):
        state = types.SimpleNamespace(source=None)

        class Mapping(OrderedDict):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if method == "__getitem__":
                    state.source.pop("b")
                return value

            def __setitem__(self, key, value):
                if method == "__setitem__":
                    state.source.pop("b")
                super().__setitem__(key, value)

        def fn(x):
            source = Mapping()
            OrderedDict.__setitem__(source, "a", x)
            OrderedDict.__setitem__(source, "b", x + 1)
            state.source = source
            return source.copy()

        with self.assertRaisesRegex(Unsupported, "Dictionary mutated during iteration"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    @parametrize("method", ["keys", "items"])
    @parametrize("update", [False, True])
    def test_ordered_dict_update_reads_descriptor_once(self, method, update):
        class Mapping(dict):
            @property
            def keys(self):
                if method == "items":
                    raise AttributeError("keys")
                self.calls += 1
                return lambda: ["a"] if self.calls == 1 else ["missing"]

            @property
            def items(self):
                self.calls += 1
                return lambda: [("a", 2 if self.calls == 1 else 20)]

        def fn(x):
            source = Mapping(a=2)
            source.calls = 0
            if update:
                result = OrderedDict()
                result.update(source)
            else:
                result = OrderedDict(source)
            return x + result["a"], source.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 2, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("bound", [False, True])
    def test_ordered_dict_subclass_using_dict_update(self, bound):
        class Mapping(dict):
            def keys(self):
                raise AssertionError("dict.update must bypass keys")

        class Target(OrderedDict):
            update = dict.update

        def fn(x):
            source = Mapping(a=x)
            target = Target()
            if bound:
                target.update(source)
            else:
                dict.update(target, source)
            return target["a"] + 1

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("bound", [False, True])
    @parametrize("external", [False, True])
    def test_dict_copy_descriptor(self, mapping_type, bound, external):
        def fn(x, mapping):
            if not external:
                mapping = mapping_type(a=x, b=x + 1)
                if mapping_type is OrderedDict:
                    mapping.move_to_end("a")
            if bound:
                result = dict.copy.__get__(mapping)()
            else:
                result = dict.copy(mapping)
            return result, result is mapping, result["a"] is mapping["a"]

        x = torch.ones(1)
        mapping = mapping_type(a=x, b=x + 1)
        if mapping_type is OrderedDict:
            mapping.move_to_end("a")
        expected = fn(x, mapping)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, mapping)
        self.assertEqual(actual, expected)
        self.assertIs(type(actual[0]), dict)
        self.assertEqual(
            list(actual[0]), ["b", "a"] if mapping_type is OrderedDict else ["a", "b"]
        )
        self.assertEqual(actual[1:], (False, True))

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("method", ["copy", "get"])
    def test_aliased_dict_method_descriptor(self, mapping_type, method):
        Mapping = type("Mapping", (mapping_type,), {method: getattr(dict, method)})

        def fn(x):
            mapping = Mapping(a=x)
            if method == "copy":
                return mapping.copy(), x + 1
            return mapping.get("a"), x + 1

        x = torch.ones(1)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual, fn(x))
        if method == "copy":
            self.assertIs(type(actual[0]), dict)

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("bound", [False, True])
    def test_dict_wrapper_descriptor_invalid_receiver(self, mapping_type, bound):
        def fn(x):
            try:
                if bound:
                    mapping_type.__len__.__get__([])()
                else:
                    mapping_type.__len__([])
            except TypeError as exc:
                return x + 1, str(exc)
            return x - 1, "no error"

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("override_iter", [False, True])
    @parametrize("copy", [False, True])
    def test_dict_update_probes_subclass_keys(self, override_iter, copy):
        class Mapping(dict):
            @property
            def keys(self):
                self.calls += 1
                return lambda: ["a"]

            if override_iter:

                def __iter__(self):
                    return super().__iter__()

        def fn(x):
            source = Mapping(a=x)
            source.calls = 0
            if copy:
                result = dict.copy(source)
            else:
                result = {}
                result.update(source)
            return result, source.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), ({"a": x}, int(override_iter) + int(not copy)))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "mapping_type, union",
        [(dict, False), (OrderedDict, False), (defaultdict, False), (dict, True)],
    )
    @parametrize("empty", [False, True])
    def test_dict_copy_descriptor_mapping_override(self, mapping_type, empty, union):
        class Mapping(mapping_type):
            def __iter__(self):
                return super().__iter__()

            def keys(self):
                return ["other"]

            def __getitem__(self, key):
                return dict.__getitem__(self, "a") + 1

            def copy(self):
                raise AssertionError("dict.copy must bypass copy")

        def fn(x):
            mapping = Mapping() if empty else Mapping(a=x)
            result = mapping | {} if union else dict.copy(mapping)
            return result, x + 2

        x = torch.ones(1)
        expected = ({}, x + 2) if empty else ({"other": x + 1}, x + 2)
        self.assertEqual(fn(x), expected)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertIs(type(actual[0]), dict)

    @parametrize("bound", [False, True])
    def test_dict_descriptor_arity_on_ordered_dict(self, bound):
        def fn(x):
            mapping = OrderedDict(a=x)
            try:
                if bound:
                    dict.popitem.__get__(mapping)(False)
                else:
                    dict.popitem(mapping, False)
            except TypeError as exc:
                return x + 1, str(exc)
            return x - 1, "no error"

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_update_input_dict_with_custom_new(self):
        class Mapping(dict):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)

            def keys(self):
                raise AssertionError("dict.update must bypass keys")

            def __getitem__(self, key):
                raise AssertionError("dict.update must bypass __getitem__")

        def fn(x, mapping):
            result = {}
            result.update(mapping)
            return x + result["a"]

        x = torch.randn(3)
        mapping = Mapping(a=2)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    def test_mutate_input_dict_with_custom_new(self, mapping_type):
        class Mapping(mapping_type):
            def __new__(cls, *args, **kwargs):
                return mapping_type.__new__(cls)

        def fn(x, mapping):
            mapping["b"] = x + 1
            mapping.pop("a")
            mapping.update(c=x + 2)
            return list(mapping.items())

        x = torch.ones(1)
        expected = Mapping(a=x)
        actual = Mapping(a=x)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, actual),
            fn(x, expected),
        )
        self.assertEqual(list(actual.items()), list(expected.items()))

    def test_ordered_dict_inplace_union_setitem(self):
        class Mapping(OrderedDict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1)

            def update(self, *args, **kwargs):
                raise AssertionError("update override called")

        def fn(x):
            mapping = Mapping()
            alias = mapping
            mapping |= {"a": x}
            return mapping["a"], mapping is alias

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 1, True))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("update", [False, True])
    @parametrize("args", [({}, {}), ([()],), ([("a",)],), ([(1, 2, 3)],)])
    def test_ordered_dict_update_errors(self, update, args):
        def fn(x):
            try:
                if update:
                    OrderedDict().update(*args)
                else:
                    OrderedDict(*args)
            except (TypeError, ValueError) as exc:
                return x + 1, str(exc)
            return x - 1, "no error"

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("update", [False, True])
    def test_ordered_dict_update_pair_iterator_consumption(self, update):
        def fn(x):
            pair = iter((1, 2, 3, 4))
            try:
                if update:
                    OrderedDict().update([pair])
                else:
                    OrderedDict([pair])
            except ValueError as exc:
                return x + 1, str(exc), list(pair)
            return x - 1, "no error", list(pair)

        x = torch.ones(1)
        self.assertEqual(fn(x)[2], [4])
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("reported_type", [OrderedDict, defaultdict])
    def test_input_dict_subclass_reporting_other_mapping_class(self, reported_type):
        class Mapping(dict):
            value = 2

            @property
            def __class__(self):
                return reported_type

        def fn(x, mapping):
            return x + mapping.value

        x = torch.ones(1)
        mapping = Mapping(a=2)
        self.assertIsInstance(mapping, reported_type)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    @parametrize("reported_type", [dict, OrderedDict])
    def test_input_object_reporting_dict_class(self, reported_type):
        class Proxy:
            value = 2

            @property
            def __class__(self):
                return reported_type

        def fn(x, obj):
            return x + obj.value

        x = torch.randn(3)
        obj = Proxy()
        self.assertIsInstance(obj, reported_type)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, obj), x + 2
        )

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("operation", ["pop", "setdefault", "setitem", "delitem"])
    @parametrize("state", ["empty", "missing", "present"])
    def test_dict_mutation_key_hashes(self, mapping_type, operation, state):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 7

        def fn(x):
            key = Key()
            mapping = mapping_type()
            if state == "present":
                mapping[key] = 1
            if state != "empty":
                mapping["other"] = 2
            calls = key.calls
            try:
                if operation == "pop":
                    result = mapping.pop(key, 3)
                elif operation == "setdefault":
                    result = mapping.setdefault(key, 3)
                elif operation == "setitem":
                    result = mapping.__setitem__(key, 3)
                else:
                    result = mapping.__delitem__(key)
            except KeyError:
                result = "missing"
            return x + len(mapping), key.calls - calls, result

        x = torch.randn(3)
        expected = fn(x)
        hashes = 1
        if operation == "pop" and state == "empty" and mapping_type is not OrderedDict:
            hashes = 0
        elif (
            operation == "setdefault"
            and state != "present"
            and mapping_type is OrderedDict
        ):
            hashes = 2
        self.assertEqual(expected[1], hashes)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("operation", ["pop", "setdefault", "setitem", "delitem"])
    @parametrize("custom_hash", [False, True])
    def test_dict_mutation_key_errors(self, mapping_type, operation, custom_hash):
        class Key:
            def __hash__(self):
                raise TypeError("custom hash failed")

        def fn(x):
            mapping = mapping_type(a=1)
            key = Key() if custom_hash else []
            try:
                if operation == "pop":
                    mapping.pop(key)
                elif operation == "setdefault":
                    mapping.setdefault(key, 3)
                elif operation == "setitem":
                    mapping[key] = 3
                else:
                    del mapping[key]
            except TypeError as error:
                return x + 1, str(error)
            return x - 1, "missing error"

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "mapping_type",
        [dict, OrderedDict, defaultdict],
    )
    @parametrize("present", [False, True])
    def test_get_hashes_key_once(self, mapping_type, present):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 7

        def fn(x):
            key = Key()
            data = {key: 2} if present else {}
            mapping = (
                defaultdict(None, data)
                if mapping_type is defaultdict
                else mapping_type(data)
            )
            calls = key.calls
            value = mapping.get(key, 0)
            return x + value, key.calls - calls

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected[1], 1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    def test_dict_subclass_instantiation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            return sd["x"] * x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_mutation(self):
        def fn(x):
            sd = SimpleDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return z * sd["x"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_contains_enum(self):
        class TensorDim(str, enum.Enum):
            DDP = "ddp"
            FSDP = "fsdp"
            CP = "cp"
            TP = "tp"

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                val = x.sin()
                if TensorDim.DDP == "ddp":
                    val += x.cos()
                if "ddp" == TensorDim.DDP:
                    val += x.cos()
                return val

        inp = torch.randn(4, 4)
        mod = Foo()
        opt_f = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(inp), opt_f(inp))

    def test_dict_subclass_local_with_non_dict_method(self):
        # Checks that add_1 method is inlined
        class MethodDict(dict):
            def add_1(self, x):
                return x + 1

        def fn(x):
            sd = MethodDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return sd.add_1(z * sd["x"])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_contains(self):
        sd = dict()
        sd[2] = 5
        sd[4] = 10

        def fn(x):
            if 1 in sd:
                x = x * 2
            else:
                x = x * 3
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[1] = 15
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure not recompilation because the traced program remains same here.
        sd[2] = 10
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_torch_size_dynamic_key(self):
        class DynamicShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_configs = {
                    torch.Size([32, 64]): torch.nn.Linear(32, 64),
                    torch.Size([64, 128]): torch.nn.Linear(64, 128),
                    torch.Size([128, 256]): torch.nn.Linear(128, 256),
                }
                self.activation_functions = {
                    torch.Size([32]): torch.nn.ReLU(),
                    torch.Size([64]): torch.nn.Tanh(),
                    torch.Size([128]): torch.nn.Sigmoid(),
                }

            def forward(self, x):
                current_shape = torch.tensor(x.shape[1:])
                shape_key = torch.Size([current_shape[0], 64])
                if shape_key in self.layer_configs:
                    x = self.layer_configs[shape_key](x)

                activation_shape = torch.Size([x.shape[1]])
                if activation_shape in self.activation_functions:
                    x = self.activation_functions[activation_shape](x)
                return x

        model = DynamicShapeModel().eval()

        for features in (32, 64, 128, 16):
            x = torch.randn(4, features)
            with torch.no_grad():
                eager_out = model(x)

            torch._dynamo.reset()
            compiled_model = torch.compile(model, backend="eager")
            with torch.no_grad():
                compiled_out = compiled_model(x)

            self.assertTrue(same(eager_out, compiled_out))

    def test_dict_subclass_methods_fallback_readonly(self):
        sd = SimpleDict()
        sd[2] = 5
        sd[4] = 10
        # check that regular attr accesses work well
        sd.attr = 4

        def fn(x):
            for value in sd.values():
                x = x * value
            for key in sd:
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            x = x * sd.attr
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[6] = 15
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_instantiation_return(self):
        def fn(x):
            sd = SimpleDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    @parametrize(
        "variant",
        [
            subtest("plain", name="plain"),
            subtest("bound", name="bound"),
            subtest("setitem", name="setitem"),
            subtest("repeated", name="repeated"),
            subtest("init", name="init"),
            subtest("new", name="new"),
            subtest("super", name="super"),
            subtest("defaultdict", name="defaultdict"),
        ],
    )
    def test_dict_subclass_fromkeys(self, variant):
        class PrefixDict(dict):
            def __setitem__(self, key, value):
                dict.__setitem__(self, f"key_{key}", value)

        class RecordingDict(dict):
            def __init__(self):
                self.calls = []

            def __setitem__(self, key, value):
                self.calls.append(key)
                dict.__setitem__(self, key, value)

        class InitDict(dict):
            def __init__(self):
                dict.__init__(self, initial=1)

        class NewDict(dict):
            def __new__(cls):
                result = dict.__new__(cls)
                result.created = True
                return result

        class SuperDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.inited = True

            def __setitem__(self, key, value):
                super().__setitem__(key.upper(), value)

        class MyDefaultDict(defaultdict):
            pass

        cases = {
            "plain": (SimpleDict, ("a", "b"), False),
            "bound": (SimpleDict, ("a", "b"), True),
            "setitem": (PrefixDict, ("a", "b"), False),
            "repeated": (RecordingDict, ("a", "a", "b"), False),
            "init": (InitDict, ("a", "b"), False),
            "new": (NewDict, ("a",), False),
            "super": (SuperDict, "ab", False),
            "defaultdict": (MyDefaultDict, "ab", False),
        }
        dict_cls, keys, bound = cases[variant]

        def fn(x):
            target = dict_cls() if bound else dict_cls
            return target.fromkeys(keys, x + 1)

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(actual), dict_cls)
        self.assertEqual(actual, expected)
        self.assertEqual(vars(actual), vars(expected))
        if isinstance(expected, defaultdict):
            self.assertIs(actual.default_factory, expected.default_factory)

    def test_dict_subclass_fromkeys_setitem_raises(self):
        class ErrorDict(dict):
            def __setitem__(self, key, value):
                raise ValueError("rejected")

        def fn(x):
            try:
                ErrorDict.fromkeys(("a",), x + 1)
            except ValueError:
                return x + 2
            return x + 3

        x = torch.randn(2)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_dict_subclass_fromkeys_input(self):
        class OtherDict(dict):
            pass

        def fn(cls, d, x):
            y = x + 1
            return cls.fromkeys("a", y), d.fromkeys("b", y), type(d).fromkeys("c", y)

        x = torch.randn(2)
        cnts = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(fn, backend=cnts, fullgraph=True)
        for cls in (SimpleDict, OtherDict):
            expected = fn(cls, cls(), x)
            actual = compiled(cls, cls(), x)
            self.assertEqual([type(d) for d in actual], [cls] * 3)
            self.assertEqual(actual, expected)
        self.assertEqual(cnts.frame_count, 2)

    def test_ordered_dict_subclass_fromkeys(self):
        class MyOrderedDict(OrderedDict):
            pass

        def fn(x):
            return (
                MyOrderedDict.fromkeys("ab", x + 1),
                MyOrderedDict.fromkeys("cd", value=x + 2),
                MyOrderedDict.fromkeys(iterable="ef", value=x + 3),
                OrderedDict.fromkeys(iterable="gh", value=x + 4),
            )

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual([type(d) for d in actual], [MyOrderedDict] * 3 + [OrderedDict])
        self.assertEqual(actual, expected)

    @parametrize("cls", [dict, SimpleDict, defaultdict])
    def test_dict_fromkeys_bad_args(self, cls):
        def fn(x):
            msgs = []
            for args, kwargs in (((), {}), (([1], x, x), {}), (([1],), {"value": x})):
                try:
                    cls.fromkeys(*args, **kwargs)
                except TypeError as e:
                    msgs.append(str(e))
            return msgs

        x = torch.randn(2)
        expected = fn(x)
        self.assertEqual(len(expected), 3)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    def test_dict_fromkeys_iterable_subclass_iter(self):
        class ReversedSet(set):
            def __iter__(self):
                return iter(sorted(super().__iter__(), reverse=True))

        class KeysDict(dict):
            def __iter__(self):
                return iter(["z"])

        def fn(x):
            return (
                dict.fromkeys(ReversedSet({1, 2, 3}), x),
                dict.fromkeys(KeysDict(a=1, b=2), x),
            )

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual([list(d) for d in actual], [list(d) for d in expected])
        self.assertEqual(actual, expected)

    def test_defaultdict_subclass_construct(self):
        class MyDefaultDict(defaultdict):
            pass

        def fn(x):
            d = MyDefaultDict(list)
            d["a"].append(x + 1)
            d["b"]
            d.tag = "t"
            return d

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(actual), MyDefaultDict)
        self.assertIs(actual.default_factory, list)
        self.assertEqual(actual, expected)
        self.assertEqual(vars(actual), vars(expected))

    def test_defaultdict_subclass_construct_custom_init_missing(self):
        class CountingDefaultDict(defaultdict):
            def __init__(self, scale):
                super().__init__(int)
                self.scale = scale
                self.misses = 0

            def __missing__(self, key):
                self.misses += 1
                return super().__missing__(key) + self.scale

        def fn(x):
            d = CountingDefaultDict(3)
            d["a"] += 1
            d["b"]
            return d, x + d["a"]

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(actual[0]), CountingDefaultDict)
        self.assertIs(actual[0].default_factory, int)
        self.assertEqual(actual, expected)
        self.assertEqual(vars(actual[0]), vars(expected[0]))

    def test_dict_subclass_methods_fallback_mutation(self):
        def fn(sd, x):
            for value in sd.values():
                x = x * value
            sd[6] = 14
            for key in sd:
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        sd1 = SimpleDict()
        sd1[2] = 5
        sd1[4] = 10

        sd2 = SimpleDict()
        sd2[2] = 5
        sd2[4] = 10
        self.assertTrue(sd1 == sd2)

        self.assertEqual(fn(sd1, x), opt_fn(sd2, x))
        self.assertTrue(sd1 == sd2)

    def test_dict_subclass_input_key_removal(self):
        # Removals cannot be replayed by dict.update alone, so the replay has
        # to clear the original dict first.
        def fn(sd, x):
            sd.pop(2)
            del sd[4]
            sd[9] = 3
            return x * len(sd)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        sd1 = SimpleDict({2: 5, 4: 10, 6: 15})
        sd2 = SimpleDict({2: 5, 4: 10, 6: 15})

        self.assertEqual(fn(sd1, x), opt_fn(sd2, x))
        self.assertEqual(sd1, sd2)
        self.assertEqual(dict(sd2), {6: 15, 9: 3})

    def test_defaultdict_input_key_mutation(self):
        # A defaultdict input is a DefaultDictVariable, built on a different
        # path than the dict-subclass one, so it needs the same replay.
        def fn(d, x):
            d["a"] += 1
            return x * len(d)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        d1, d2 = defaultdict(int), defaultdict(int)

        self.assertEqual(fn(d1, x), opt_fn(d2, x))
        self.assertEqual(d1, d2)
        self.assertEqual(dict(d2), {"a": 1})

    def test_dict_subclass_setitem(self):
        class SetItemDict(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value + 1)

        def fn(x):
            sd = SetItemDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_custom_iter_dict(self):
        class ReversedDict(dict):
            def __iter__(self):
                return reversed(list(self.keys()))

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            # Forces side effects attribute reapplication logic
            d.sample = 1
            d["baz"] = 4
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        # This is intentional because the dict is mutated, so we will have a recompilation.
        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_custom_keys_iter_dict(self):
        class ReversedDict(dict):
            def keys(self):
                return ["bar", "foo"]

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_dict_guard_on_keys_order(self):
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            for key, value in d.items():
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_dict_guard_on_keys_order2(self):
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            for key in d:
                value = d[key]
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_ordered_dict_reordered_keys(self):
        d = OrderedDict()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            y = 0
            for idx, value in enumerate(d.values()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_ordered_dict_subclass_reordered_keys(self):
        class ODSubclass(OrderedDict):
            def keys(self):
                return super().keys()

        d = ODSubclass()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            y = 0
            for idx, value in enumerate(d.values()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_lazy_key_guarding(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            return x * d["a"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop("c")
        d["d"] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_lazy_key_non_const_guarding(self):
        d = {
            list: 2,
            dict: 3,
            OrderedDict: 5,
            namedtuple: 7,
        }

        def fn(x):
            return x * d[list]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop(dict)
        d[defaultdict] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_dict_mutation_side_effect(self):
        def fn(d):
            d["c"] = d["a"] + d.pop("b")
            return d

        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        if fn(args1) is not args1:
            raise AssertionError("Expected fn(args1) to be args1")
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(args2), args2)
        self.assertTrue(same(args1, args2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_dict_copy_alias(self):
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, d0):
            d1 = d0.copy()
            d1[0] = 1
            return x + 1, d1

        d0 = {}
        res, d1 = run(torch.zeros(1), d0)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(d0, {})
        self.assertEqual(d1, {0: 1})

    def test_dict_subclass_get_method(self):
        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        config = dotdict({"a": 1, "b": 2})

        def fn(x):
            x2 = x * 2  # noqa: F841
            x3 = x * config.get("a", 3)
            return x3

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_order_keys(self):
        def fn(d):
            c = 0
            for v in d.values():
                c += v
            return c

        args1 = {}
        args1["a"] = torch.rand(10)
        args1["b"] = torch.rand(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # A different order of keys recompiles
        args2 = {}
        args2["b"] = args1["b"]
        args2["a"] = args1["a"]
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)
        # Extra calls don't recompile
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_namedtuple(self):
        def fn(d):
            if namedtuple in d:
                return d[3] * 2
            else:
                return d[3] * 3

        args1 = {namedtuple: None, 3: torch.randn(3)}
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        # Test a failing namedtuple guard
        args2 = {2: None, 3: torch.randn(3)}
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_tensors(self):
        def fn(d, x):
            return d[x] + 3

        args1 = {}
        x = torch.randn(10)
        y = torch.randn(10)
        z = torch.randn(10)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_modules(self):
        def fn(d, x):
            return d[x](torch.ones(2, 2))

        args1 = {}
        x = torch.nn.Linear(2, 2)
        y = torch.nn.Linear(2, 2)
        z = torch.nn.Linear(2, 2)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_contains_dunder_dict(self):
        class UserDefined:
            def __init__(self) -> None:
                self.a = 3
                self.b = 5

            def run(self, x):
                if "a" in self.__dict__:
                    x = x * self.a
                if "b" in self.__dict__:
                    x = x * self.b
                self.c = 7
                if "c" in self.__dict__:
                    x = x * self.c
                return x * self.__dict__.get("a") * self.__dict__.get("z", 2)

        obj = UserDefined()

        def fn(x):
            return obj.run(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_len_dunder_dict_after_delete(self):
        # After deleting an attribute during tracing, len(obj.__dict__) must
        # reflect the deletion. The bug: SideEffectsProxyDict.__len__ returned
        # len(item_dict) + len(side_effects_table), double-counting deleted
        # entries and keys mutated from item_dict.
        class UserDefined:
            def __init__(self) -> None:
                self.a = 3
                self.b = 5

            def run(self, x):
                del self.a
                # len should be 1 (only b remains), not 3 (item_dict=2 + mutations=1)
                return x * len(self.__dict__)

        obj1 = UserDefined()
        obj2 = UserDefined()

        def fn(x, obj):
            return obj.run(x)

        x = torch.ones(2)
        ref = fn(x, obj1)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj2)
        self.assertEqual(ref, res)

    def test_contains_module_dunder_dict(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = 1
                self.bar = 2
                self.baz = 3

            def forward(self, x):
                if "foo" in self.__dict__:
                    return x * self.bar
                return x * self.baz

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_update_dunder_dict(self):
        class UserDefined:
            def run(self, x):
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        obj1 = UserDefined()
        obj2 = UserDefined()

        def fn(x, obj):
            return obj.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, obj1)
        res = opt_fn(x, obj2)
        self.assertEqual(ref, res)
        # Make sure only `a` is updated.
        self.assertEqual(obj1.__dict__, obj2.__dict__)

    def test_update_module_dunder_dict(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_dict_reconstruct_keeps_original_order(self):
        def fn():
            modules = OrderedDict([("act", torch.nn.ReLU())])
            module_dict = torch.nn.ModuleDict(modules)

            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            modules.update(next_modules.items())
            module_dict.update(next_modules)
            return modules, module_dict

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        modules, module_dict = opt_fn()

        self.assertEqual(len(module_dict), len(modules))
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    # FIXME: see comment in torch/_dynamo/polyfills/__init__.py:mutable_mapping_update
    @unittest.expectedFailure
    def test_dict_construct_from_mapping_like(self):
        def fn(x):
            fm = FakeMapping(x)
            d = dict(fm, x=x)
            return d

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_no_args(self):
        def fn(x):
            d = {"a": x}
            result = d.update()
            return d["a"], result is None, len(d)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_from_mapping_like(self):
        class MappingLike:
            def __init__(self, x):
                self.d = {"a": x, "b": x + 1}

            def keys(self):
                return self.d.keys()

            def __getitem__(self, key):
                return self.d[key]

        def fn(x):
            d = {"a": x - 1}
            result = d.update(MappingLike(x))
            return d["a"], d["b"], result is None

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    def test_dict_update_from_dict_subclass_iter_override(self, mapping_type):
        class Mapping(dict):
            def __init__(self):
                super().__init__(a=1, b=3)
                self.calls = []

            def __iter__(self):
                return super().__iter__()

            def keys(self):
                self.calls.append("keys")
                return ["a"]

            def __getitem__(self, key):
                self.calls.append(("getitem", key))
                return 2

        def fn(x):
            source = Mapping()
            result = mapping_type()
            result.update(source)
            return result, source.calls, x + result["a"]

        x = torch.randn(2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(expected, ({"a": 2}, ["keys", ("getitem", "a")], x + 2))
        self.assertEqual(actual, expected)

    def test_dict_update_from_mapping_proxy(self):
        def fn(x):
            source = {"a": x, "b": x + 1}
            d = {"a": x - 1}
            d.update(types.MappingProxyType(source))
            return d["a"], d["b"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_rejects_bad_sequence_element_length(self):
        def fn():
            try:
                {}.update([(1, 2, 3)])
            except ValueError as exc:
                return "length 3; 2 is required" in str(exc)
            return False

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), opt_fn())

    def test_dict_update_rejects_too_many_args(self):
        def fn():
            try:
                {}.update({}, {})
            except TypeError:
                return True
            return False

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), opt_fn())

    def test_dict_subclass_initialization_in_graph(self):
        for super_class in (
            OrderedDict,
            dict,
        ):

            class CustomDict(super_class):
                def __new__(cls, *args, **kwargs):
                    return super().__new__(cls, *args, **kwargs)

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

            def fn(x):
                c = CustomDict()
                c["key"] = x
                assert "key" in c  # noqa: S101
                return c["key"] + 1

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

            x = torch.rand(4)
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_list_values(self):
        def inner_fn(args):
            return [x[1].shape for x in args]

        @torch.compile(backend="eager")
        def fn(tensors):
            return inner_fn(zip(itertools.count(), tensors["args"]))

        fn({"args": [torch.ones(5, 5), torch.ones(5, 6), torch.ones(5, 7)]})
        fn({"args": [torch.ones(5, 5)]})

    def test_dict_iter(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = {"my": 1, "const": 2, "dict": 3, "variable": 4}
                tot = 0
                for key in z:
                    tot += z[key]

                return tot

        x = torch.tensor([0])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)

        self.assertEqual(y, 10)

    def test_dict_subclass_contains(self):
        # pattern from huggingface
        class ClassInstantier(OrderedDict):
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, d):
            if "key1" in d:
                x = x + 2
            if "key2" in d:
                x = x + 4
            x = x + 8
            return x

        result = f(torch.ones(8), ClassInstantier({"key1": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 11.0)))

        result = f(torch.ones(8), ClassInstantier({"key2": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 13.0)))

    def test_dict_tag_guard(self):
        class Foo:
            def __init__(self) -> None:
                self.scalar = 10

        def fn(d, x):
            return d["a"] * d["b"] * d["c"].scalar * x

        foo = Foo()

        d = {"a": 2, "b": 3, "c": foo}

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        d["a"] = 4
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        # Check that recompilation happens
        foo.scalar = 12
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

    def test_empty_dict_recompilation(self):
        def fn(d, x):
            if d:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn({}, x), opt_fn({}, x))
        self.assertEqual(fn({"a": 1}, x), opt_fn({"a": 1}, x))

    def test_udf_dict_reconstruction(self):
        class MyDict(dict):
            pass

        def fn(x, klass):
            x = x * 2
            sc_dict = dict.__new__(klass)
            sc_dict["x"] = x
            if isinstance(sc_dict, MyDict):
                sc_dict.attr = 3
            return sc_dict

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x, MyDict)
        res = opt_fn(x, MyDict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, MyDict))
        self.assertEqual(ref.attr, res.attr)

        ref = fn(x, dict)
        res = opt_fn(x, dict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, dict))

    def test_weakref_dict(self):
        states = weakref.WeakKeyDictionary()

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()

        states[mod1] = 2
        states[mod2] = 3

        def fn(x):
            if mod1 in states:
                x = torch.sin(x)
            if mod2 in states:
                x = torch.cos(x)
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_weakkeydict_attr_proxy_key(self):
        class Child(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

        root = Root()
        tracer = _ModuleStackTracer(root)
        proxy = tracer.proxy_type(root.child, "child")
        states = weakref.WeakKeyDictionary({proxy: 1})

        def fn(d, key, x):
            offset = d[key]
            return key(x) + offset

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(states, proxy, x), opt_fn(states, proxy, x))

        sin_node = next(
            node for node in backend.graphs[0].graph.nodes if node.name == "sin"
        )
        module_paths = [path for path, _ in sin_node.meta["nn_module_stack"].values()]
        key_source = (
            "L['args'][1]"
            if torch._dynamo.config.debug_force_nested_calls
            else "L['key']"
        )
        self.assertEqual(module_paths, [f"{key_source}.get_base()"])

    def test_attr_proxy_cross_scope_reuse_guard(self):
        class Cell(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.tensor(-1.0)

        first = Cell()
        second = Cell()
        tracer = _ModuleStackTracer(torch.nn.Module())
        local_proxy = tracer.proxy_type(first, "local")
        global_proxy = tracer.proxy_type(first, "global")
        other_global_proxy = tracer.proxy_type(second, "other")

        def local_proxy_first(proxy, x):
            proxy.value = x
            return _attr_proxy_global.value + 1  # noqa: F821

        counter = torch._dynamo.testing.CompileCounter()
        with patch.dict(
            local_proxy_first.__globals__, {"_attr_proxy_global": global_proxy}
        ):
            opt_fn = torch.compile(local_proxy_first, backend=counter, fullgraph=True)

            self.assertEqual(opt_fn(local_proxy, torch.tensor(2.0)), torch.tensor(3.0))
            self.assertEqual(counter.frame_count, 1)

            local_proxy_first.__globals__["_attr_proxy_global"] = other_global_proxy
            first.value = torch.tensor(-1.0)
            second.value = torch.tensor(-1.0)

            self.assertEqual(opt_fn(local_proxy, torch.tensor(5.0)), torch.tensor(0.0))
            self.assertEqual(counter.frame_count, 2)

        def global_proxy_first(module, x):
            _attr_proxy_global.value = x  # noqa: F821
            return module.value + 1

        counter = torch._dynamo.testing.CompileCounter()
        with patch.dict(
            global_proxy_first.__globals__, {"_attr_proxy_global": global_proxy}
        ):
            opt_fn = torch.compile(global_proxy_first, backend=counter, fullgraph=True)

            self.assertEqual(opt_fn(first, torch.tensor(2.0)), torch.tensor(3.0))
            self.assertEqual(counter.frame_count, 1)

            first.value = torch.tensor(-1.0)
            second.value = torch.tensor(-1.0)

            self.assertEqual(opt_fn(second, torch.tensor(5.0)), torch.tensor(0.0))
            self.assertEqual(counter.frame_count, 2)

    def test_construct_user_dict_and_return(self):
        def fn(x):
            return DummyUserDict({"a": x + 1})

        x = torch.randn(4)
        res = fn(x)
        self.assertEqual(res["a"], x + 1)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(res["a"], opt_fn(x)["a"])

    def test_mapping_proxy_for_local(self):
        def fn(x):
            d = {"a": 2, "b": 3, "c": 5 * x}
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

    def test_mapping_proxy_for_nonlocal(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            d["d"] = 4
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

        # check update to d is reflected in res
        d["e"] = 5
        self.assertEqual(d["e"], res["e"])

    def test_mapping_proxy_existing(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x, mp):
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            if isinstance(mp, types.MappingProxyType):
                y *= 2
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        mp = types.MappingProxyType(d)
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

        d["a"] = 3
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

        d.pop("b")
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

    def test_mapping_proxy_len_does_not_call_getitem(self):
        class Data(dict):
            reads = 0

            def __getitem__(self, key):
                self.reads += 1
                return super().__getitem__(key)

        def fn(x, proxy):
            return x + len(proxy)

        data = Data(a=1, b=2)
        proxy = types.MappingProxyType(data)
        x = torch.ones(2)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, proxy), fn(x, proxy)
        )
        self.assertEqual(data.reads, 0)

    @parametrize("override", ["iter", "keys"])
    def test_mapping_proxy_items_order(self, override):
        class Inputs(OrderedDict):
            def __iter__(self):
                if override == "iter":
                    return reversed(super().keys())
                return super().__iter__()

            def keys(self):
                if override == "keys":
                    return reversed(super().keys())
                return super().keys()

        def fn(x, mapping):
            return x + 1, tuple(mapping.items())

        x = torch.randn(3)
        mapping = types.MappingProxyType(Inputs(a=1, b=2))
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    def test_dict_construction_from_mapping_proxy(self):
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x, mp):
            d = dict(mp)
            y = torch.sin(x * d["a"])
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        mp = types.MappingProxyType(d)
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

    def test_mapping_proxy_existing_mutation(self):
        d = {"a": 2, "b": 3, "c": 5}

        mp = types.MappingProxyType(d)

        def fn(x):
            d["d"] = 4
            y = torch.sin(x * mp["d"])
            return y

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        ref = torch.sin(x * 4)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(d.keys(), mp.keys())

    def test_mapping_proxy_existing_local_mutation(self):
        d = {"a": 2, "b": 3, "c": 5}

        mp = types.MappingProxyType(d)

        def fn(x):
            # Dynamo should not cause a graph break here because it knows that
            # the existing proxy can't point to this new dict
            other_dict = {}
            other_dict["d"] = 4
            y = torch.sin(x * mp["c"])
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = torch.sin(x * mp["c"])
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(d.keys(), mp.keys())

    @parametrize("mapping_type", [dict, OrderedDict])
    @parametrize("name", ["keys", "values", "items"])
    def test_dict_base_view_reconstruction(self, mapping_type, name):
        class Mapping(mapping_type):
            def keys(self):
                return ["override"]

            values = items = keys

        def fn(x):
            mapping = Mapping(a=x + 1)
            return mapping, getattr(mapping_type, name)(mapping)

        x = torch.randn(3)
        mapping, view = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = getattr(mapping_type, name)(mapping)
        self.assertIs(type(view), type(expected))
        self.assertEqual(list(view), list(expected))
        self.assertIs(view.mapping["a"], mapping["a"])

    @parametrize(
        "mapping_type",
        [dict, OrderedDict, defaultdict],
    )
    @parametrize("candidate", [[], ["a", 1], ("a",), ("a", 1, 2)])
    def test_items_contains_non_pair(self, mapping_type, candidate):
        def fn(x):
            mapping = mapping_type(a=1)
            try:
                return x + (candidate in mapping.items())
            except TypeError:
                return x - 1

        x = torch.randn(3)
        self.assertEqual(fn(x), x)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), x)

    def test_dict_view_mapping(self):
        # dict_keys/values/items expose a read-only mappingproxy via .mapping
        mappingproxy = type(type.__dict__)

        def fn(x):
            d = {"a": 1, "b": 2}
            m_keys = d.keys().mapping
            m_values = d.values().mapping
            m_items = d.items().mapping
            y = torch.sin(x)
            for m in (m_keys, m_values, m_items):
                if isinstance(m, mappingproxy) and m == d:
                    y = y + 1
            return y, m_keys, m_values, m_items

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        for m in res[1:]:
            self.assertTrue(type(m) is mappingproxy)
            self.assertEqual(m, {"a": 1, "b": 2})

    def test_dict_view_mapping_reflects_mutation(self):
        # The mappingproxy returned by .mapping proxies the live dict, so a
        # later mutation must be visible through it.
        mappingproxy = type(type.__dict__)

        def fn(x):
            d = {}
            m = d.keys().mapping
            d["foo"] = "bar"
            return torch.sin(x), isinstance(m, mappingproxy), dict(m)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertTrue(res[1])
        self.assertEqual(res[2], {"foo": "bar"})

    def test_move_to_end(self):
        def fn(x):
            d = OrderedDict({"a": torch.cos(x), "b": 3, "c": 5})
            d.move_to_end("a")
            return d

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(["b", "c", "a"], list(opt_fn(x).keys()))
        self.assertEqual(fn(x), opt_fn(x))

    def test_mapping_proxy_ban_muation_on_dict_realization(self):
        def fn(x):
            class Foo:
                b = 4

            d = dict(Foo.__dict__)
            y = torch.sin(x) * d["b"]
            # This should cause a graph break, because otherwise the
            # Foo.__dict__ will not be updated.
            Foo.bar = 3
            return Foo, y * Foo.__dict__["bar"]

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        foo1, ref = fn(x)
        foo2, res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(foo1.bar, foo2.bar)

    def test_overridden_get_item(self):
        class MyDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def __getitem__(self, key):
                self.calls += 1
                return super().__getitem__(key) + 1

        def fn(x, d):
            d["d"] = 4
            return x * d["a"] + d["b"] + d["c"] + d["d"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        d1 = MyDict({"a": 2, "b": 3, "c": 5})
        ref = fn(x, d1)

        d2 = MyDict({"a": 2, "b": 3, "c": 5})
        res = opt_fn(x, d2)
        self.assertEqual(ref, res)
        self.assertEqual(d1.calls, d2.calls)

    def test_items_type(self):
        def fn():
            d = dict({"a": 1, "b": "2", "c": torch.tensor(3)})  # noqa: C418
            return d.items()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_fn()
        self.assertEqual(ref, res)
        self.assertEqual(type(res), dict_items)

    def test_builtin_or_with_invalid_types(self):
        args = (
            1,  # int
            1.0,  # float
            "a",  # str
            (1, 2),  # tuple
            [1, 2],  # list
        )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(b: Any):
            a = {"one": torch.ones(1)}
            return a | b

        for arg in args:
            with self.assertRaises(Unsupported):
                _ = fn(arg)

    def test_builtin_or_with_diff_keys(self):
        def f():
            a = {"one": torch.ones(1)}
            b = {"two": torch.ones(2)}
            return a, b, a | b, b | a, a.__or__(b), b.__or__(a)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_builtin_or_with_same_keys(self):
        def f():
            a = {"one": torch.ones(1), "two": torch.ones(2)}
            b = {"one": torch.ones(1), "three": torch.ones(3)}
            return a, b, a | b, b | a, a.__or__(b), b.__or__(a)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_builtin_ior_(self):
        def f():
            a = {"one": torch.ones(1)}
            b = {"two": torch.ones(2)}
            a |= b
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("construct", [False, True])
    @parametrize("reverse", [False, True])
    def test_defaultdict_union_partial_factory(self, construct, reverse):
        def fn(x, mapping):
            if construct:
                mapping = defaultdict(partial(torch.zeros_like, x), existing=x)
            other = {"present": x + 1}
            result = other | mapping if reverse else mapping | other
            missing = result["missing"]
            return result, missing, result.default_factory is mapping.default_factory

        x = torch.randn(3)
        mapping = defaultdict(partial(torch.zeros_like, x), existing=x)
        expected = fn(x, mapping)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, mapping)
        self.assertIs(type(actual[0]), defaultdict)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], torch.zeros_like(x))
        self.assertTrue(actual[2])

    @parametrize("override_iter", [False, True])
    def test_defaultdict_union_probes_keys(self, override_iter):
        class Mapping(dict):
            @property
            def keys(self):
                self.reads += 1
                return lambda: dict.keys(self)

            if override_iter:

                def __iter__(self):
                    return dict.__iter__(self)

        def fn(x):
            source = Mapping(b=x + 1)
            source.reads = 0
            result = defaultdict(int, a=x) | source
            return result, source.reads

        x = torch.ones(1)
        expected = fn(x)
        self.assertEqual(expected[1], int(override_iter))
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("factory_kind", ["callable", "noncallable", "list"])
    @parametrize("reverse", [False, True])
    def test_defaultdict_union_constructed_factory(self, factory_kind, reverse):
        class Factory:
            if factory_kind == "callable":

                def __call__(self):
                    return 7

        def fn(x):
            factory = [x] if factory_kind == "list" else Factory()
            mapping = defaultdict(None, a=x)
            mapping.default_factory = factory
            try:
                result = {"b": x + 1} | mapping if reverse else mapping | {"b": x + 1}
            except TypeError as error:
                return x + 1, str(error)
            return x + result["missing"], result.default_factory is factory

        x = torch.ones(1)
        expected = (
            (x + 7, True)
            if factory_kind == "callable"
            else (x + 1, "first argument must be callable or None")
        )
        self.assertEqual(fn(x), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("reverse", [False, True])
    def test_defaultdict_union_noncallable_factory(self, reverse):
        def fn(x, mapping):
            try:
                if reverse:
                    {"a": x} | mapping
                else:
                    mapping | {"a": x}
            except TypeError as error:
                return x + 1, str(error)
            return x, "no error"

        x = torch.randn(3)
        mapping = defaultdict(int)
        mapping.default_factory = 42
        expected = (x + 1, "first argument must be callable or None")
        self.assertEqual(fn(x, mapping), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping), expected
        )

    def test_defaultdict_inplace_union_preserves_factory(self):
        def f():
            d = defaultdict(int, {1: 1, 2: 2})
            d |= [(0, "zero"), (1, "one")]
            result = d.__ior__({3: "three"})
            return (
                result is d,
                d.default_factory is int,
                type(d) is defaultdict,
                dict(d),
                list(d),
                d[4],
            )

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_defaultdict_shallow_copy_preserves_factory(self):
        # defaultdict.copy(), dd.__copy__(), and copy.copy(dd) all produce a
        # new defaultdict with the same default_factory and a shallow copy of
        # the contents. The copies are consumed in-graph because reconstructing
        # an escaping sourceless defaultdict is a separate unsupported case.
        import copy

        def f():
            d = defaultdict(list, {1: 1, 2: 2})
            c1 = d.copy()
            c2 = d.__copy__()
            c3 = copy.copy(d)
            return (
                c1.default_factory is list,
                c2.default_factory is list,
                c3.default_factory is list,
                dict(c1),
                dict(c2),
                dict(c3),
                c1 == d,
                c3 == d,
                c1[5],
            )

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_newly_constructed_default_dict(self):
        def f(x):
            d = defaultdict(list)
            d[0] = [
                42,
            ]
            return x + 1, d

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_newly_constructed_default_dict_no_default_factory(self):
        def f1(x):
            d = defaultdict()
            try:
                d[1] += 42
            except KeyError:
                d[1] = 1
            return x + 1, d

        x = torch.ones(2)
        ref = f1(x)
        res = torch.compile(f1, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

        def f2(x):
            d = defaultdict(None)
            try:
                d[1] += 42
            except KeyError:
                d[1] = 1
            return x + 1, d

        ref = f2(x)
        res = torch.compile(f2, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

        def f3(x):
            d = defaultdict(None, {1: 10})
            d[1] += 42
            try:
                d[2] += 24
            except KeyError:
                d[2] = 1
            return x + 1, d

        ref = f3(x)
        res = torch.compile(f3, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_newly_constructed_default_dict_with_dict(self):
        def f(x):
            d = dict([("a", 1), ("b", 2)], c=3)
            dd = defaultdict(list, d, d=4, e=5)
            dd["x"].append(42)
            return x + 1, d, dd

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_iter_default_dict(self):
        def f(x):
            d = defaultdict(list)
            d[0] = 42
            for k in d:
                d[k] += 1
            return x + 1, d

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_dict_new_ignores_extra_args(self):
        """dict.__new__ ignores extra args (CPython behavior).

        This matters for instantiate_user_defined_class_object which calls
        cls.__new__(cls, *args, **kwargs) — dict.__new__ should not fail
        on the extra args.
        """

        def f(x):
            od = OrderedDict(a=1, b=2)
            return x + od["a"]

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_ordered_dict_repr(self):
        """repr() on OrderedDictVariable should not graph break."""

        def f(x):
            od = OrderedDict(a=1, b=2)
            r = repr(od)
            # Just verify repr doesn't graph break and returns a string
            return x + (1 if isinstance(r, str) else 0)

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_c_new_init_args_ignored_for_dict(self):
        """C-level __new__ for dict/set passes init_args=[] since they
        ignore extra args. This ensures generators passed to OrderedDict()
        don't end up in reconstruction."""

        def whoo(t):
            yield 1, t + 1
            yield 2, t + 2

        def f(t):
            return OrderedDict(whoo(t))

        t = torch.randn(2)
        ref = f(t)
        res = torch.compile(f, backend="eager", fullgraph=True)(t)
        self.assertEqual(ref, res)

    def test_ordered_dict_as_python_constant_preserves_type(self):
        """as_python_constant should return OrderedDict, not plain dict."""

        def f(x):
            od = OrderedDict(a=1, b=2)
            # Enum functional API calls as_python_constant on the OrderedDict
            import enum

            E = enum.Enum("E", od)
            return x + E.a.value

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_default_dict_as_python_constant_preserves_type(self):
        """as_python_constant should return defaultdict, not plain dict."""

        def f(x):
            dd = defaultdict(int, a=1, b=2)
            # isinstance triggers as_python_constant internally for
            # constant folding the type check
            assert isinstance(dd, defaultdict)  # noqa: S101
            return x + dd["a"]

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    @parametrize("op", ["or_", "and_", "xor", "sub"])
    def test_dict_keys_binop(self, op):
        op = getattr(operator, op)

        def f():
            a = {"one": torch.ones(1), "two": torch.ones(2)}
            b = {"one": torch.ones(1), "three": torch.ones(3)}
            return op(a.keys(), b.keys()), op(b.keys(), a.keys())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["ior", "iand", "ixor", "isub"])
    def test_dict_keys_inplace_binop(self, op):
        op = getattr(operator, op)

        def f():
            a = {"one": torch.ones(1), "two": torch.ones(2)}.keys()
            b = {"one": torch.ones(1), "three": torch.ones(3)}.keys()
            c = {"one": torch.ones(1), "two": torch.ones(2)}.keys()
            a = op(a, b)
            b = op(b, c)
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub"])
    def test_dict_items_binop(self, op):
        op = getattr(operator, op)

        def f():
            a = {"one": 1, "two": 2}
            b = {"one": 1, "three": 3}
            return op(a.items(), b.items()), op(b.items(), a.items())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["ior", "iand", "ixor", "isub"])
    def test_dict_items_inplace_binop(self, op):
        op = getattr(operator, op)

        def f():
            a = {"one": 1, "two": 2}.items()
            b = {"one": 1, "three": 3}.items()
            c = {"one": 1, "two": 2}.items()
            a = op(a, b)
            b = op(b, c)
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_set(self, op):
        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2, "three": 3}.keys()
            s = {"one", "four"}
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_items_vs_set(self, op):
        op = getattr(operator, op)

        def f():
            items = {"one": 1, "two": 2}.items()
            s = {("one", 1), ("three", 3)}
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_dict_items(self, op):
        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2}.keys()
            items = {"three": 3, "four": 4}.items()
            return op(keys, items), op(items, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2, "three": 3}.keys()
            s = MySet({"one", "four"})
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_items_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            items = {"one": 1, "two": 2, "three": 3}.items()
            s = MySet({"one", "four"})
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_set_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            s = {"one", "two", "three"}
            u = MySet({"one", "four"})
            return op(s, u), op(u, s)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_set(self, op):
        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2}.keys()
            s = {"one", "two", "three"}
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_items_vs_set(self, op):
        op = getattr(operator, op)

        def f():
            items = {"one": 1, "two": 2}.items()
            s = {("one", 1), ("two", 2), ("three", 3)}
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_dict_items(self, op):
        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2}.keys()
            items = {"one": 1, "two": 2}.items()
            return op(keys, items), op(items, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_set_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            s = {"one", "two"}
            u = MySet({"one", "two", "three"})
            return op(s, u), op(u, s)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            keys = {"one": 1, "two": 2}.keys()
            u = MySet({"one", "two", "three"})
            return op(keys, u), op(u, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_items_vs_user_defined_set(self, op):
        class MySet(set):
            pass

        op = getattr(operator, op)

        def f():
            items = {"one": 1, "two": 2}.items()
            u = MySet({("one", 1), ("two", 2), ("three", 3)})
            return op(items, u), op(u, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_dict_view_iand_rebinds_variable(self):
        def f_keys():
            d = {"one": 1, "two": 2, "three": 3}
            tmp = d.keys()
            tmp &= {"one", "four"}
            return tmp, d

        opt_f_keys = torch.compile(f_keys, backend="eager", fullgraph=True)
        eager_tmp, eager_d = f_keys()
        compiled_tmp, compiled_d = opt_f_keys()
        self.assertIsInstance(eager_tmp, set)
        self.assertEqual(eager_tmp, compiled_tmp)
        self.assertEqual(eager_d, compiled_d)

        def f_items():
            d = {"one": 1, "two": 2}
            tmp = d.items()
            tmp &= {("one", 1), ("three", 3)}
            return tmp, d

        opt_f_items = torch.compile(f_items, backend="eager", fullgraph=True)
        eager_tmp, eager_d = f_items()
        compiled_tmp, compiled_d = opt_f_items()
        self.assertIsInstance(eager_tmp, set)
        self.assertEqual(eager_tmp, compiled_tmp)
        self.assertEqual(eager_d, compiled_d)

    def test_dict_union_result_type(self):
        def_dict = defaultdict(int, {1: 1, 2: 2})
        ord_dict = OrderedDict({3: 3, 4: 4})
        base_dict = {5: 5, 6: 6}

        exp_def = def_dict | ord_dict
        self.assertIsInstance(exp_def, defaultdict)

        exp_def_2 = def_dict | base_dict
        self.assertIsInstance(exp_def_2, defaultdict)

        exp_def_3 = base_dict | def_dict
        self.assertIsInstance(exp_def_3, defaultdict)

        exp_ord = ord_dict | def_dict
        self.assertIsInstance(exp_ord, OrderedDict)

        exp_ord_2 = base_dict | ord_dict
        self.assertIsInstance(exp_ord_2, OrderedDict)

        exp_base = base_dict | base_dict
        self.assertIsInstance(exp_base, dict)

    @parametrize(
        "mapping_type,reverse",
        [
            subtest((OrderedDict, False), name="ordered_forward"),
            subtest((OrderedDict, True), name="ordered_reverse"),
            subtest((defaultdict, False), name="default_forward"),
            subtest((defaultdict, True), name="default_reverse"),
        ],
    )
    def test_dict_subclass_union_constructor(self, mapping_type, reverse):
        class Mapping(mapping_type):
            def __init__(self, *args):
                super().__init__(*args)
                self.initial_items = list(self.items())

            def copy(self):
                raise AssertionError("copy override called")

            def update(self, *args, **kwargs):
                raise AssertionError("update override called")

            def __setitem__(self, key, value):
                super().__setitem__(key, value + 10)

        def fn(x, source):
            other = {"b": 3, "c": 4}
            result = other | source if reverse else source | other
            return result, result.initial_items, x + result["b"]

        source = (
            Mapping(int, {"a": 1, "b": 2})
            if mapping_type is defaultdict
            else Mapping({"a": 1, "b": 2})
        )
        x = torch.randn(2)
        expected = fn(x, source)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, source)
        self.assertIs(type(actual[0]), Mapping)
        self.assertEqual(actual, expected)
        offset = 10 if mapping_type is OrderedDict else 0
        self.assertEqual(
            list(actual[0].items()),
            [("b", 2 + 2 * offset), ("c", 4 + offset), ("a", 1 + 2 * offset)]
            if reverse
            else [("a", 1 + 2 * offset), ("b", 3 + offset), ("c", 4 + offset)],
        )
        self.assertEqual(
            actual[1],
            [("b", 3 + offset), ("c", 4 + offset)]
            if reverse
            else [("a", 1 + 2 * offset), ("b", 2 + 2 * offset)],
        )
        if mapping_type is defaultdict:
            self.assertIs(actual[0].default_factory, int)

    @parametrize("reverse", [False, True])
    def test_ordered_dict_union_mapping_overrides(self, reverse):
        class Mapping(dict):
            def keys(self):
                return ["b"]

            def __getitem__(self, key):
                return super().__getitem__(key) + 10

            def items(self):
                raise AssertionError("items override called")

        def fn(x, source):
            mapping = OrderedDict()
            # Call __ror__ explicitly: dict subclasses inherit dict.__or__,
            # which handles the left operand without delegating to OrderedDict.
            result = mapping.__ror__(source) if reverse else mapping | source
            return tuple(result), x + result["b"]

        x = torch.ones(1)
        source = Mapping(a=1, b=2)
        expected = (("b",), x + 12)
        self.assertEqual(fn(x, source), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, source), expected
        )

    def test_range_as_dict_key(self):
        def fn(x):
            d = {range(5): x * 2, range(10, 15): x * 3}
            return d[range(0, 5, 1)] + d[range(10, 15)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_tuple_as_dict_key(self):
        def fn(x):
            d = {(1, 2): x * 2, (3, 4, 5): x * 3}
            return d[(1, 2)] + d[(3, 4, 5)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_enum_as_dict_key(self):
        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            d = {Color.RED: x * 2, Color.GREEN: x * 3, Color.BLUE: x * 4}
            return d[Color.RED] + d[Color.GREEN] + d[Color.BLUE]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_intenum_as_dict_key(self):
        class Priority(enum.IntEnum):
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x):
            d = {Priority.LOW: x * 2, Priority.MEDIUM: x * 3, Priority.HIGH: x * 4}
            return d[Priority.LOW] + d[Priority.MEDIUM] + d[Priority.HIGH]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_frozenset_as_dict_key(self):
        def fn(x):
            d = {frozenset([1, 2]): x * 2, frozenset([3, 4, 5]): x * 3}
            return d[frozenset([1, 2])] + d[frozenset([3, 4, 5])]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_typing_union_as_dict_key(self):
        from typing import Union

        def fn(x):
            d = {Union[int, str]: x * 2, Union[float, bool]: x * 3}  # noqa: UP007
            return d[Union[int, str]] + d[Union[float, bool]]  # noqa: UP007

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_numpy_dtype_as_dict_key(self):
        import numpy as np

        def fn(x):
            d = {np.float32: x * 2, np.int64: x * 3, np.bool_: x * 4}
            return d[np.float32] + d[np.int64] + d[np.bool_]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_method_wrapper_as_dict_key(self):
        add_method = list.__add__
        mul_method = list.__mul__

        def fn(x):
            # Method wrappers are the type of bound methods on built-in types
            d = {add_method: x * 2, mul_method: x * 3}
            return d[add_method] + d[mul_method]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_torch_builtin_function_as_dict_key(self):
        def fn(x, y):
            # Using torch built-in functions as dictionary keys
            d = {torch.add: x * 2, torch.mul: y * 3, torch.sub: x + y}
            return d[torch.add] + d[torch.mul] + d[torch.sub]

        x = torch.randn(4)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, y), opt_fn(x, y))

    def test_frozen_dataclass_as_dict_key(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            x: int
            y: int

        def fn(tensor):
            p1 = Point(1, 2)
            p2 = Point(3, 4)
            d = {p1: tensor * 2, p2: tensor * 3}
            return d[Point(1, 2)] + d[Point(3, 4)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_list_as_dict_key_raises_typeerror(self):
        def fn(x):
            d = {[1, 2, 3]: x * 2}
            return d[[1, 2, 3]]

        x = torch.randn(4)

        # First check that eager execution raises TypeError
        with self.assertRaises(TypeError):
            fn(x)

        # Also check that compiled version raises TypeError
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Observed exception"):
            opt_fn(x)

    def test_property_backed_by_dunder_dict(self):
        class Obj:
            def __init__(self, x):
                self.__dict__["x"] = x

            @property
            def x(self):
                return self.__dict__["x"]

        def fn(obj):
            return obj.x

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_property_backed_by_dunder_dict_mutated(self):
        class Obj:
            def __init__(self, x):
                self.__dict__["x"] = x

            @property
            def x(self):
                return self.__dict__["x"]

            @x.setter
            def x(self, value):
                self.__dict__["x"] = value

        def fn(obj):
            a = obj.x
            obj.x = a + 1
            return obj.x

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(Obj(3)), opt_fn(obj))

    def test_property_backed_by_dunder_dict_pending_side_effects(self):
        class Obj:
            def __init__(self, x):
                self.__dict__["x"] = x

            @property
            def x(self):
                return self.__dict__["x"]

        def fn(obj):
            a = obj.__dict__["x"]
            obj.__dict__["x"] = a + 1
            return obj.__dict__["x"]

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(Obj(3)), opt_fn(obj))

    def test_dunder_dict_getitem_guard_recompiles(self):
        class Obj:
            def __init__(self):
                self.__dict__["x"] = 3

            @property
            def x(self):
                return 999

        def fn(obj, t):
            return t + obj.__dict__["x"]

        obj = Obj()
        t = torch.tensor(0.0)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        self.assertEqual(opt_fn(obj, t), torch.tensor(3.0))
        self.assertEqual(cnt.frame_count, 1)

        obj.__dict__["x"] = 5
        self.assertEqual(opt_fn(obj, t), torch.tensor(5.0))
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("use_vars", [False, True])
    @parametrize("delete_attribute", [False, True])
    def test_dunder_dict_getitem_deleted(self, use_vars, delete_attribute):
        class Obj:
            pass

        def fn(x):
            obj = Obj()
            obj.a = 1
            if delete_attribute:
                del obj.a
            else:
                del obj.__dict__["a"]
            mapping = vars(obj) if use_vars else obj.__dict__
            try:
                mapping["a"]
            except KeyError:
                return x + 1
            return x - 1

        x = torch.ones(1)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_dunder_dict_get_with_property(self):
        class Obj:
            def __init__(self, x):
                self.__dict__["x"] = x

            @property
            def x(self):
                return self.__dict__["x"]

        def fn(obj):
            return obj.__dict__.get("x")

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_dunder_dict_getitem_tensor(self):
        class Obj:
            def __init__(self, x):
                self.__dict__["x"] = x

            @property
            def x(self):
                return self.__dict__["x"]

        def fn(obj):
            return obj.x + 1

        obj = Obj(torch.tensor(3.0))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_get_default_nowrap_functions_as_dict_key(self):
        def fn(x):
            # Get the set of default nowrap functions
            nowrap_funcs = torch.overrides.get_default_nowrap_functions()
            # Use the set as a dict key and search for Tensor.grad.__get__ in it
            d = {frozenset(nowrap_funcs): x * 2}
            # Check if Tensor.grad.__get__ is in the set
            if torch.Tensor.grad.__get__ in nowrap_funcs:
                return d[frozenset(nowrap_funcs)] + x
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_custom_bool(self):
        class CustomBoolDict:
            def __init__(self, bool_result: bool):
                super().__init__()
                self._bool_result = bool_result

            def __bool__(self):
                return self._bool_result

        def fn(t: torch.Tensor, d, apply_not: bool):
            if apply_not:
                return t.sin(), bool(not d)
            else:
                return t.sin(), bool(d)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        t = CustomBoolDict(True)
        self.assertTrue(
            opt_fn(x, t, False)[1],
            "CustomBoolDict(True) should evaluate to True in boolean context",
        )
        self.assertFalse(
            opt_fn(x, t, True)[1],
            "not CustomBoolDict(True) should evaluate to False in boolean context",
        )

        f = CustomBoolDict(False)
        self.assertFalse(
            opt_fn(x, f, False)[1],
            "CustomBoolDict(False) should evaluate to False in boolean context",
        )
        self.assertTrue(
            opt_fn(x, f, True)[1],
            "not CustomBoolDict(False) should evaluate to True in boolean context",
        )

    def _get_fw_graphs_for_dict_orders(self):
        """Compile a model with two different dict key orders, return both
        forward graphs from AOTAutograd (the level where has_same_nodes operates).

        Uses asymmetric paths (deep vs shallow) so that different dict iteration
        orders produce different op orderings in the graph.
        """

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deep = torch.nn.Sequential(
                    torch.nn.Linear(8, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                )
                self.shallow = torch.nn.Linear(8, 16)

            def forward(self, d):
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.deep(val))
                    else:
                        results.append(self.shallow(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        backend1 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        torch.compile(model, backend=backend1)(d1).backward()
        torch._dynamo.reset()
        model.zero_grad()
        backend2 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        torch.compile(model, backend=backend2)(d2).backward()

        return backend1.fw_graphs[0].graph, backend2.fw_graphs[0].graph

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=False)
    def test_name_based_hash_diverges_on_dict_order(self):
        # Demonstrates the original bug: the name-based hash used in
        # has_same_nodes diverges when different ranks trace with different
        # dict iteration orders, producing different node orderings.
        import hashlib

        graph1, graph2 = self._get_fw_graphs_for_dict_orders()

        def name_based_hash(graph):
            node_str = "/".join(x.name for x in graph.nodes)
            return hashlib.sha256(node_str.encode("utf-8")).hexdigest()

        self.assertNotEqual(name_based_hash(graph1), name_based_hash(graph2))

    def test_canonical_names_invariant_to_dict_order(self):
        # The canonical naming produces identical mappings for structurally
        # equivalent graphs traced with different dict iteration orders.
        from torch._functorch.partitioners import _canonical_node_names

        graph1, graph2 = self._get_fw_graphs_for_dict_orders()

        canonical1 = _canonical_node_names(graph1)
        canonical2 = _canonical_node_names(graph2)

        # Build {canonical_name: (op, target)} for each graph.
        # For placeholders, exclude target (it's the rank-local name like
        # primals_N which differs across ranks).
        def structure(graph, canonical):
            return {
                canonical[n]: (n.op, str(n.target) if n.op != "placeholder" else "")
                for n in graph.nodes
            }

        self.assertEqual(structure(graph1, canonical1), structure(graph2, canonical2))

    def test_canonical_names_different_models(self):
        from torch._functorch.partitioners import _canonical_node_names

        class ModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, x):
                return self.linear(x)

        class ModelB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, x):
                return self.linear(x).relu()

        inp = torch.randn(4, 8)
        backend1 = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(ModelA(), backend=backend1)(inp)
        torch._dynamo.reset()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(ModelB(), backend=backend2)(inp)

        c1 = _canonical_node_names(backend1.graphs[0].graph)
        c2 = _canonical_node_names(backend2.graphs[0].graph)
        structure1 = {
            c1[n]: (n.op, str(n.target)) for n in backend1.graphs[0].graph.nodes
        }
        structure2 = {
            c2[n]: (n.op, str(n.target)) for n in backend2.graphs[0].graph.nodes
        }
        self.assertNotEqual(structure1, structure2)

    def test_stable_target_str(self):
        # A call_function whose target is a plain Python function (e.g.
        # torch.sym_not) must stringify to its qualified name, not its repr,
        # which bakes in the per-process memory address and so differs across
        # ranks (poisoning the cross-rank node_str hash).
        from torch._functorch.partitioners import _stable_target_str

        # Plain Python function: qualified name, no memory address.
        self.assertEqual(_stable_target_str(torch.sym_not), "torch.sym_not")
        self.assertNotIn("0x", _stable_target_str(torch.sym_not))
        self.assertIn("0x", str(torch.sym_not))  # the behavior being fixed

        # OpOverload targets: stable qualified name (matches the FX printer),
        # never an address.
        from torch.fx.node import _get_qualified_name

        add = torch.ops.aten.add.Tensor
        self.assertEqual(_stable_target_str(add), _get_qualified_name(add))
        self.assertNotIn("0x", _stable_target_str(add))

        # Non-callable targets (e.g. get_attr names) fall through to str().
        self.assertEqual(_stable_target_str("_param_constant0"), "_param_constant0")

        # Callables _get_qualified_name cannot resolve fall back to str().
        class NoName:
            def __call__(self):
                return None

        obj = NoName()
        self.assertEqual(_stable_target_str(obj), str(obj))

    def test_canonical_node_str_invariant_to_function_target_address(self):
        # Regression test: two ranks tracing the same graph reference the same
        # torch.sym_not function, but the function object lives at a different
        # memory address in each process. The old code stringified the target
        # with str(), baking that address into the cross-rank hash, so
        # structurally identical graphs looked different and the sync was
        # skipped. Simulate two ranks with two distinct function objects that
        # share a qualified name but differ in repr (address).
        import hashlib

        from torch._functorch.partitioners import (
            _canonical_node_names,
            _stable_target_str,
        )

        def _make_sym_not():
            def sym_not(x):
                return not x

            return sym_not

        fn_rank0 = _make_sym_not()
        fn_rank1 = _make_sym_not()
        # Same qualified name, different repr (mimics differing addresses).
        self.assertNotEqual(str(fn_rank0), str(fn_rank1))
        self.assertEqual(_stable_target_str(fn_rank0), _stable_target_str(fn_rank1))

        def build_graph(fn):
            g = torch.fx.Graph()
            p = g.placeholder("x")
            p.meta["val"] = torch.tensor(True)
            n = g.create_node("call_function", fn, (p,))
            n.meta["val"] = torch.tensor(False)
            g.output((n,))
            return g

        graph0 = build_graph(fn_rank0)
        graph1 = build_graph(fn_rank1)

        # Reconstruct the node_str hash the way _sync_decision_cross_ranks does.
        def node_str(graph, stringify):
            canonical = _canonical_node_names(graph)

            def hash_str(n):
                if n.op == "placeholder":
                    return f"{canonical[n]}:{n.op}"
                return f"{canonical[n]}:{n.op}:{stringify(n.target)}"

            joined = "/".join(
                hash_str(n) for n in sorted(graph.nodes, key=lambda n: canonical[n])
            )
            return hashlib.sha256(joined.encode("utf-8")).hexdigest()

        # Old behavior: str(target) bakes in the address -> hashes diverge.
        self.assertNotEqual(node_str(graph0, str), node_str(graph1, str))
        # Fixed behavior: _stable_target_str -> identical hashes.
        self.assertEqual(
            node_str(graph0, _stable_target_str),
            node_str(graph1, _stable_target_str),
        )

    def _get_graph_node_names(self, model, inp):
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(model, backend=backend)(inp)
        return [n.name for n in backend.graphs[0].graph.nodes]

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deep = torch.nn.Sequential(
                    torch.nn.Linear(8, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                )
                self.shallow = torch.nn.Linear(8, 16)

            def forward(self, d):
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.deep(val))
                    else:
                        results.append(self.shallow(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": torch.randn(4, 8), "a": torch.randn(4, 8)}

        names1 = self._get_graph_node_names(model, d1)
        torch._dynamo.reset()
        names2 = self._get_graph_node_names(model, d2)
        self.assertEqual(names1, names2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_correctness(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = torch.nn.Linear(8, 16)
                self.linear_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                return self.linear_a(d["a"]) + self.linear_b(d["b"])

        model = Model()

        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        eager_result = model(d1)
        compiled_result1 = torch.compile(model, backend="eager")(d1)
        torch._dynamo.reset()
        compiled_result2 = torch.compile(model, backend="eager")(d2)

        self.assertEqual(eager_result, compiled_result1)
        self.assertEqual(eager_result, compiled_result2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_aot_eager(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_a = torch.nn.Linear(8, 16)
                self.proj_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.proj_a(val))
                    else:
                        results.append(self.proj_b(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        backend1 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        loss1 = torch.compile(model, backend=backend1)(d1)
        loss1.backward()

        torch._dynamo.reset()
        model.zero_grad()

        backend2 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        loss2 = torch.compile(model, backend=backend2)(d2)
        loss2.backward()

        self.assertEqual(loss1, loss2)

        fw_names1 = [n.name for n in backend1.fw_graphs[0].graph.nodes]
        fw_names2 = [n.name for n in backend2.fw_graphs[0].graph.nodes]
        self.assertEqual(fw_names1, fw_names2)
        bw_names1 = [n.name for n in backend1.bw_graphs[0].graph.nodes]
        bw_names2 = [n.name for n in backend2.bw_graphs[0].graph.nodes]
        self.assertEqual(bw_names1, bw_names2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_idempotent(self):
        from torch._dynamo.output_graph import _canonicalize_graph

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, d):
                return self.linear(d["a"]) + self.linear(d["b"])

        model = Model()
        d = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(model, backend=backend)(d)
        graph = backend.graphs[0].graph

        names_once = [n.name for n in graph.nodes]
        _canonicalize_graph(graph)
        names_twice = [n.name for n in graph.nodes]
        self.assertEqual(names_once, names_twice)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_overlapping_unsqueeze_with_mutation(self):
        def f(x, y):
            x.add_(1)
            y.add_(1)
            return x

        base = torch.ones(10)
        inputs = [base.unsqueeze(0), base.unsqueeze(0)]
        out_eager = f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)
        base = torch.ones(10)
        inputs = [base.unsqueeze(0), base.unsqueeze(0)]
        out_compiled = optf(*inputs)

        self.assertEqual(out_eager, out_compiled)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_barrier_preserves_order(self):
        from torch._dynamo.output_graph import _canonicalize_graph

        graph = fx.Graph()
        x = graph.placeholder("x")
        # Pure ops that could be reordered
        a = graph.call_function(torch.relu, (x,))
        b = graph.call_function(torch.sigmoid, (x,))
        # In-place call_method acts as barrier
        barrier = graph.call_method("add_", (x, a))
        # Pure op after barrier
        c = graph.call_function(torch.neg, (x,))
        graph.output((a, b, barrier, c))

        _canonicalize_graph(graph)
        ops = [n.name for n in graph.nodes if n.op in ("call_function", "call_method")]
        barrier_idx = next(i for i, name in enumerate(ops) if "add_" in name)
        neg_idx = next(i for i, name in enumerate(ops) if "neg" in name)
        # neg must come after the barrier even though it only depends on x
        self.assertGreater(neg_idx, barrier_idx)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_in_place_ops_are_barriers(self):
        def f(x, y):
            a = y * 2
            x.add_(1)
            b = y * 3
            return a + b + x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        out = torch.compile(f, backend=backend)(torch.randn(4), torch.randn(4))
        out_eager = f(torch.randn(4), torch.randn(4))
        self.assertEqual(out.shape, out_eager.shape)

        graph = backend.graphs[0].graph
        ops = [
            (n.name, n.op)
            for n in graph.nodes
            if n.op in ("call_function", "call_method")
        ]
        barrier_idx = next(
            i
            for i, (name, op) in enumerate(ops)
            if op == "call_method" and "add_" in name
        )
        mul_before = [
            i
            for i, (name, _) in enumerate(ops)
            if name.startswith("mul") and i < barrier_idx
        ]
        mul_after = [
            i
            for i, (name, _) in enumerate(ops)
            if name.startswith("mul") and i > barrier_idx
        ]
        self.assertTrue(len(mul_before) >= 1)
        self.assertTrue(len(mul_after) >= 1)

    def test_canonical_graph_config_gating(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = torch.nn.Linear(8, 16)
                self.linear_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.linear_a(val))
                    else:
                        results.append(self.linear_b(val))
                return torch.cat(results, dim=-1)

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": torch.randn(4, 8), "a": torch.randn(4, 8)}

        with torch._dynamo.config.patch(canonicalize_output_graph_node_order=False):
            names1 = self._get_graph_node_names(model, d1)
            torch._dynamo.reset()
            names2 = self._get_graph_node_names(model, d2)
        self.assertNotEqual(names1, names2)

        torch._dynamo.reset()
        with torch._dynamo.config.patch(canonicalize_output_graph_node_order=True):
            names3 = self._get_graph_node_names(model, d1)
            torch._dynamo.reset()
            names4 = self._get_graph_node_names(model, d2)
        self.assertEqual(names3, names4)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_is_safe_to_reorder(self):
        import operator

        from torch.fx.passes.canonicalize import _is_safe_to_reorder

        graph = fx.Graph()
        x = graph.placeholder("x")

        # Reorderable call_function nodes must carry a tensor or symbolic
        # example_value/val, as Dynamo attaches to every data-producing node.
        def pure(node):
            node.meta["example_value"] = torch.empty(0)
            return node

        pure_call = pure(graph.call_function(torch.relu, (x,)))
        self.assertTrue(_is_safe_to_reorder(pure_call))

        inplace_method = graph.call_method("add_", (x, x))
        self.assertFalse(_is_safe_to_reorder(inplace_method))

        safe_method = graph.call_method("add", (x, x))
        self.assertTrue(_is_safe_to_reorder(safe_method))

        iadd_node = graph.call_function(operator.iadd, (x, x))
        self.assertFalse(_is_safe_to_reorder(iadd_node))

        # operator.invert is pure despite starting with "i"
        invert_node = pure(graph.call_function(operator.invert, (x,)))
        self.assertTrue(_is_safe_to_reorder(invert_node))

        index_node = pure(graph.call_function(operator.index, (x,)))
        self.assertTrue(_is_safe_to_reorder(index_node))

        # out= kwarg makes a node unsafe
        out_node = graph.call_function(torch.add, (x, x), {"out": x})
        self.assertFalse(_is_safe_to_reorder(out_node))

        # Functions with no FX Node arguments are treated as barriers
        def _fake_state_fn():
            pass

        no_input_node = graph.call_function(_fake_state_fn, ())
        self.assertFalse(_is_safe_to_reorder(no_input_node))

        # _add_batch_dim / _remove_batch_dim are barriers
        add_batch = pure(graph.call_function(torch._add_batch_dim, (x, x, x)))
        self.assertFalse(_is_safe_to_reorder(add_batch))

        remove_batch = pure(graph.call_function(torch._remove_batch_dim, (x, x, x, x)))
        self.assertFalse(_is_safe_to_reorder(remove_batch))

        # State functions that consume the token produced by their enter node
        # (so the no-node-arguments heuristic misses them) are barriers:
        # inference_mode via _side_effectful_functions, arbitrary ones (e.g.
        # _sdpa_kernel, _maybe_exchange_device) via the lack of a tensor or
        # symbolic example_value/val.
        enter_node = graph.call_function(
            torch.autograd.grad_mode._enter_inference_mode, (True,)
        )
        self.assertFalse(_is_safe_to_reorder(enter_node))
        exit_node = graph.call_function(
            torch.autograd.grad_mode._exit_inference_mode, (enter_node,)
        )
        self.assertFalse(_is_safe_to_reorder(exit_node))

        def _fake_exit_fn(token):
            pass

        token_consumer = graph.call_function(_fake_exit_fn, (no_input_node,))
        self.assertFalse(_is_safe_to_reorder(token_consumer))
        self.assertTrue(_is_safe_to_reorder(pure(token_consumer)))

        # HOPs are exempt from the value heuristic: graph passes (e.g. graph
        # deduplication) create invoke_subgraph nodes without example_value/val.
        hop = torch.ops.higher_order.invoke_subgraph
        hop_node = graph.call_function(hop, (x, "subgraph_0", x))
        self.assertTrue(_is_safe_to_reorder(hop_node))

    @parametrize(
        "operation",
        [
            operator.getitem,
            operator.contains,
            lambda d, k: d.get(k),
            lambda d, k: operator.contains(d.keys(), k),
            lambda d, k: (k, 1) in d.items(),
        ],
    )
    @parametrize("custom_hash", [False, True])
    @parametrize(
        "mapping_type",
        [dict, OrderedDict, defaultdict],
    )
    def test_lookup_type_errors(self, operation, custom_hash, mapping_type):
        class Key:
            def __hash__(self):
                raise TypeError("custom hash failed")

        def fn(x):
            mapping = mapping_type(a=x)
            key = Key() if custom_hash else []
            try:
                operation(mapping, key)
            except TypeError as error:
                return x + 1, str(error)
            return x - 1, "missing error"

        x = torch.randn(3)
        expected = fn(x)
        if sys.version_info >= (3, 14):
            self.assertIn("cannot use", expected[1])
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )


instantiate_parametrized_tests(DictTests)


class DictGuardTests(LoggingTestCase):
    hw_classification = HardwareClassification.GENERIC

    thetype = dict

    @make_logging_test(recompiles=True)
    def test_popitem(self, records):
        d = self.thetype()
        d[1] = 2
        d[3] = 4

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            k, v = d.popitem()
            if k == 3 and v == 4:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        d[3] = 5
        y = fn(x)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d")
        self.assertIn(
            """d[3] == 4""",
            munge_exc(record),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_eq(self, records):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            if d1 == d2:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.cos())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.sin())
        record = self.getRecord(records, "d2")
        self.assertIn(
            "(HINT: Dictionary d2 must not contain key 3",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_ne(self, records):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            if d1 == d2:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.cos())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.sin())
        record = self.getRecord(records, "d2")
        self.assertIn(
            "(HINT: Dictionary d2 must not contain key 3",
            munge_exc(record.getMessage()),
        )

    @parametrize("present_during_tracing", [True, False])
    def test_contains_recompilation_message(self, present_during_tracing):
        failures = []

        def fn(x, d):
            if "scale" in d:
                return x.sin()
            return x.cos()

        compiled_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=lambda failure: failures.append(failure.reason)
        )(fn)
        x = torch.tensor(1.0)
        initial = {"scale": 1} if present_during_tracing else {}
        changed = {} if present_during_tracing else {"scale": 1}
        compiled_fn(x, initial)
        compiled_fn(x, changed)

        self.assertEqual(len(failures), 1)
        expectation = "present" if present_during_tracing else "absent"
        requirement = "contain" if present_during_tracing else "not contain"
        negation = "" if present_during_tracing else "not "
        self.assertIn(
            f"{negation}___dict_contains('scale', d)",
            failures[0],
        )
        self.assertIn(
            f"(HINT: Dictionary d must {requirement} key 'scale'; "
            f"Dynamo specialized the compiled code on this key being {expectation}.",
            failures[0],
        )

    @make_logging_test(recompiles=True)
    def test_cmp_or(self, records):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            d = d1 | d2
            if d.get(5, False):
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d2")
        self.assertIn(
            """list(dict.keys(d2))[1] == 5""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_ior(self, records):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            d2 |= d1
            if d2.get(3, False):
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        d3, d4 = d2.copy(), d2.copy()
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        y = fn(x, d3, d4)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d1")
        self.assertIn(
            """list(dict.keys(d1))[1] == 3""",
            munge_exc(record.getMessage()),
        )


class DictMethodsTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    thetype = dict

    # Methods:
    # + bool
    # + clear
    # + copy
    # + fromkeys
    # + get
    # + items
    # + keys
    # + pop
    # + popitem
    # + setdefault
    # + update
    # + values
    # BinOps:
    # ==, !=, |

    def setUp(self):
        self._prev_trace_unittest = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev_trace_unittest
        return super().tearDown()

    def assertEqual(self, x, y):
        self.assertTrue(x == y, lambda msg: f"{msg}\nExpected {x} to be equal to {y}")

    def assertNotEqual(self, x, y):
        self.assertFalse(
            x == y, lambda msg: f"{msg}\nExpected {x} to not be equal to {y}"
        )

    @make_dynamo_test
    def test_dict_items_cmp_value_eq_raises(self):
        # dictitems_contains calls PyObject_RichCompareBool(found, value, Py_EQ)
        # on the stored value, so a value whose __eq__ raises must propagate that
        # exception. Equal-length views push the comparison past
        # dictview_richcompare's length short-circuit for eq/ne/le/ge.
        d1 = self.thetype({1: _BadCmp()})
        d2 = self.thetype({1: _BadCmp()})
        for op in (operator.eq, operator.ne, operator.le, operator.ge):
            with self.assertRaises(_BadCmpExc):
                op(d1.items(), d2.items())
        # lt/gt reach the value comparison only on a proper-subset length match.
        d3 = self.thetype({1: _BadCmp(), 2: _BadCmp()})
        with self.assertRaises(_BadCmpExc):
            d1.items() < d3.items()  # noqa: B015
        with self.assertRaises(_BadCmpExc):
            d3.items() > d1.items()  # noqa: B015

    @make_dynamo_test
    def test_dict_items_cmp_value_present_absent(self):
        # Normal value comparison: matching/non-matching stored values resolve
        # membership without raising, matching eager dict_items semantics.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2, "c": 3})
        d3 = self.thetype({"a": 1, "b": 99})
        self.assertEqual(d1.items() == d2.items(), False)
        self.assertEqual(d1.items() <= d2.items(), True)
        self.assertEqual(d1.items() < d2.items(), True)
        self.assertEqual(d1.items() == d3.items(), False)
        self.assertEqual(d1.items() <= d3.items(), False)
        self.assertEqual(d2.items() >= d1.items(), True)

    @make_dynamo_test
    def test_cmp_eq_key_raises(self):
        # A key whose __eq__ raises must propagate that exception during dict
        # comparison, mirroring CPython's PyObject_RichCompareBool lookup.
        d1 = self.thetype({_BadCmp(): 1})
        d2 = self.thetype({1: 1})
        with self.assertRaises(_BadCmpExc):
            d1 == d2  # noqa: B015

    @make_dynamo_test
    def test_cmp_eq(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2})
        d3 = self.thetype({"a": 1, "b": 3})
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)

        # Test the == operator
        self.assertEqual(d1 == d2, True)
        self.assertEqual(d1 == d3, False)

        # Test the __eq__ method
        self.assertEqual(d1.__eq__(d2), True)
        self.assertEqual(d1.__eq__(d3), False)

        # Test Dict.__eq__
        self.assertEqual(dict.__eq__(d1, d2), True)
        self.assertEqual(self.thetype.__eq__(d1, d3), False)

    @make_dynamo_test
    def test_cmp_ne(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2})
        d3 = self.thetype({"a": 1, "b": 3})
        self.assertNotEqual(d1, d3)
        self.assertEqual(d1, d2)

        # Test the != operator
        self.assertEqual(d1 != d3, True)
        self.assertEqual(d1 != d2, False)

        # Test the __ne__ method
        self.assertEqual(d1.__ne__(d3), True)
        self.assertEqual(d1.__ne__(d2), False)

        # Test Dict.__ne__
        self.assertEqual(dict.__ne__(d1, d3), True)
        self.assertEqual(self.thetype.__ne__(d1, d2), False)

    @make_dynamo_test
    def test_binop_or(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test the | operator
        self.assertEqual(d1 | d2, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d2 | d1, {"a": 1, "b": 2, "c": 4})

        # Test the __or__ method
        self.assertEqual(d1.__or__(d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d2.__or__(d1), {"a": 1, "b": 2, "c": 4})

        # Test Dict.__or__
        self.assertEqual(dict.__or__(d1, d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(self.thetype.__or__(d2, d1), {"a": 1, "b": 2, "c": 4})

        # Test with non-dict types
        self.assertRaises(TypeError, lambda: d1 | 1)

    @make_dynamo_test
    def test_binop_ior(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test the |= operator
        d3, d4 = d1.copy(), d2.copy()
        d3 |= d2
        d4 |= d1
        self.assertEqual(d3, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4, {"a": 1, "b": 2, "c": 4})

        # Test the __ior__ method
        d3, d4 = d1.copy(), d2.copy()
        d3.__ior__(d2)
        d4.__ior__(d1)
        self.assertEqual(d3, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4, {"a": 1, "b": 2, "c": 4})

        # Test Dict.__or__
        d3, d4 = d1.copy(), d2.copy()
        self.assertEqual(dict.__ior__(d3, d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(self.thetype.__ior__(d4, d1), {"a": 1, "b": 2, "c": 4})

        # Test return value
        d3, d4 = d1.copy(), d2.copy()
        self.assertEqual(d3.__ior__(d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(dict.__ior__(d4, d1), {"a": 1, "b": 2, "c": 4})

        # Test with non-dict types
        self.assertRaises(TypeError, lambda: dict.__ior__(d1, 1))

    @make_dynamo_test
    def test_binop_ior_iterable(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})
        d3, d4 = d1.copy(), d2.copy()

        def fn(d):
            yield from d.items()

        self.assertEqual(d3.__ior__(d2.items()), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4.__ior__(fn(d1)), {"a": 1, "b": 2, "c": 4})

    @make_dynamo_test
    def test_clear(self):
        d = self.thetype({"a": 1, "b": 2})
        d.clear()
        self.assertEqual(d, {})

        # Test that clear returns None
        d = self.thetype({"a": 1, "b": 2})
        self.assertIsNone(d.clear())

        # Test Dict.clear
        d = self.thetype({"a": 1, "b": 2})
        dict.clear(d)
        self.assertEqual(d, {})

        d = self.thetype({"a": 1, "b": 2})
        self.thetype.clear(d)
        self.assertEqual(d, {})

        # Test invalid usage
        self.assertRaises(TypeError, d.clear, 1)

    @make_dynamo_test
    def test_copy(self):
        d = self.thetype({"a": 1, "b": 2})
        d2 = d.copy()
        self.assertEqual(d, d2)

        # Test that copy returns a new instance
        self.assertIsNot(d, d2)

        # Test Dict.copy
        self.assertEqual(dict.copy(d), d2)
        self.assertEqual(self.thetype.copy(d), d2)

        # Test invalid usage
        self.assertRaises(TypeError, d.copy, 1)

    @make_dynamo_test
    def test_fromkeys(self):
        d = self.thetype.fromkeys(["a", "b"], 1)
        self.assertEqual(d, {"a": 1, "b": 1})
        p = self.thetype.fromkeys(["a", "b"], None)
        self.assertEqual(p, {"a": None, "b": None})

        # Test Dict.fromkeys
        d2 = self.thetype.fromkeys(["c", "d"], 2)
        self.assertEqual(d2, {"c": 2, "d": 2})

        # Test invalid usage
        self.assertRaises(TypeError, self.thetype.fromkeys)
        self.assertRaises(TypeError, self.thetype.fromkeys, 1, 2)

    @make_dynamo_test
    def test_get(self):
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.get("a"), 1)
        self.assertEqual(d.get("c", 3), 3)
        self.assertIsNone(d.get("c"))

        # Test Dict.get
        self.assertEqual(dict.get(d, "b"), 2)
        self.assertEqual(self.thetype.get(d, "b"), 2)

        # Test invalid usage
        self.assertRaises(TypeError, d.get)

    @make_dynamo_test
    def test_items(self):
        d = self.thetype({"a": 1, "b": 2})
        items = d.items()
        self.assertEqual(set(items), {("a", 1), ("b", 2)})

        # Test Dict.items
        self.assertEqual(set(dict.items(d)), {("a", 1), ("b", 2)})
        self.assertEqual(set(self.thetype.items(d)), {("a", 1), ("b", 2)})

        # Test invalid usage
        self.assertRaises(TypeError, d.items, 1)

    @make_dynamo_test
    def test_keys(self):
        d = self.thetype({"a": 1, "b": 2})
        keys = d.keys()
        self.assertEqual(set(keys), {"a", "b"})

        # Test Dict.keys
        self.assertEqual(set(dict.keys(d)), {"a", "b"})
        self.assertEqual(set(self.thetype.keys(d)), {"a", "b"})

        # Test invalid usage
        self.assertRaises(TypeError, d.keys, 1)

    @make_dynamo_test
    def test_pop(self):
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.pop("a"), 1)
        self.assertEqual(d, {"b": 2})
        self.assertIsNone(d.pop("c", None))

        # Test Dict.pop
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(dict.pop(d, "b"), 2)
        self.assertEqual(self.thetype.pop(d, "a"), 1)

        # Test invalid usage
        self.assertRaises(KeyError, d.pop, "c")
        self.assertRaises(TypeError, d.pop)

    @make_dynamo_test
    def test_popitem(self):
        d = self.thetype({"a": 1})
        key, value = d.popitem()
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)
        self.assertEqual(len(d), 0)
        # check LIFO
        d = self.thetype()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d.popitem(), ("b", 2))

        # Test Dict.popitem
        d = self.thetype({"a": 1})
        key, value = dict.popitem(d)
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)

        d = self.thetype({"a": 1})
        key, value = self.thetype.popitem(d)
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)

        # Test invalid usage
        if self.thetype is not OrderedDict:
            # OrderedDict accepts a keyword arg
            self.assertRaises(TypeError, d.popitem, 1)

    @make_dynamo_test
    def test_setdefault(self):
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.setdefault("a", 3), 1)
        self.assertEqual(d.setdefault("c", 3), 3)
        self.assertIsNone(d.setdefault("d"), None)
        self.assertEqual(d, {"a": 1, "b": 2, "c": 3, "d": None})

        # Test Dict.setdefault
        self.assertEqual(dict.setdefault(d, "f", 5), 5)
        self.assertEqual(self.thetype.setdefault(d, "e", 5), 5)

        # Test invalid usage
        self.assertRaises(TypeError, d.setdefault)
        self.assertRaises(TypeError, d.setdefault, [[]])

    @make_dynamo_test
    def test_update(self):
        d = self.thetype({"a": 1, "b": 2})
        d.update({"b": 3, "c": 4})
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4})

        # Test with another dict
        d2 = self.thetype({"d": 5})
        d.update(d2)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5})

        # Test Dict.update
        d3 = self.thetype({"e": 6})
        dict.update(d, d3)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6})
        d4 = self.thetype({"f": 7})
        self.thetype.update(d, d4)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7})

        # Test with keyword arguments
        d.update(f=7, g=8)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8})

        # Test Dict.update with keyword arguments
        self.thetype.update(d, h=9, i=10)
        self.assertEqual(
            d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8, "h": 9, "i": 10}
        )

        # Test invalid usage
        self.assertRaises(TypeError, d.update, 1)

    @make_dynamo_test
    def test_values(self):
        d = self.thetype({"a": 1, "b": 2})
        values = d.values()
        self.assertEqual(set(values), {1, 2})

        # Test Dict.values
        self.assertEqual(set(dict.values(d)), {1, 2})
        self.assertEqual(set(self.thetype.values(d)), {1, 2})

        # Test invalid usage
        self.assertRaises(TypeError, d.values, 1)

    @make_dynamo_test
    def test_keys_contains(self):
        d = self.thetype({"a": 1, "b": 2})
        keys = d.keys()
        self.assertTrue("a" in keys)
        self.assertTrue("b" in keys)
        self.assertFalse("c" in keys)

        # Test Dict.keys contains
        self.assertTrue("a" in dict.keys(d))
        self.assertFalse("c" in dict.keys(d))
        self.assertTrue("b" in self.thetype.keys(d))

    @make_dynamo_test
    def test_items_contains(self):
        d = self.thetype({"a": 1, "b": 2})
        items = d.items()
        self.assertTrue(("a", 1) in items)
        self.assertTrue(("b", 2) in items)
        self.assertFalse(("a", 2) in items)
        self.assertFalse(("c", 1) in items)

        # Test Dict.items contains
        self.assertTrue(("a", 1) in dict.items(d))
        self.assertFalse(("c", 1) in dict.items(d))
        self.assertTrue(("b", 2) in self.thetype.items(d))

    @make_dynamo_test
    def test_values_contains(self):
        d = self.thetype({"a": 1, "b": 2})
        values = d.values()
        self.assertTrue(1 in values)
        self.assertTrue(2 in values)
        self.assertFalse(3 in values)

        # Test Dict.values contains
        self.assertTrue(1 in dict.values(d))
        self.assertFalse(3 in dict.values(d))
        self.assertTrue(2 in self.thetype.values(d))

    @make_dynamo_test
    def test_type(self):
        d = self.thetype({"a": 1, "b": 2})
        self.assertIsInstance(d, self.thetype)
        self.assertIs(type(d), self.thetype)

    @make_dynamo_test
    def test_dict_type_comparison(self):
        types = (dict, OrderedDict, defaultdict)
        self.assertEqual(self.thetype, self.thetype)
        self.assertTrue(self.thetype is self.thetype)
        for other in types:
            if self.thetype == other:
                continue
            self.assertNotEqual(self.thetype, other)
            self.assertTrue(
                self.thetype is not other,
                lambda msg: f"{msg}\n{self.thetype=}, {other=}",
            )

    @make_dynamo_test
    def test_dict___iter__(self):
        d = self.thetype({1: 2})
        it = d.__iter__()
        self.assertEqual(next(it), 1)

    def test_functools_partial_key(self):
        def gn(x, y):
            return x + y

        def fn(x):
            new_dict = {}
            new_gn1 = partial(gn, x=1)
            new_dict[new_gn1] = 5
            return x * new_dict[new_gn1]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_namedtuple_functools(self):
        class Container(NamedTuple):
            partial_fn: Callable
            const: int

        def gn(x, y):
            return x + y

        def fn(x):
            new_dict = {}

            new_gn = partial(gn, x=1)
            key = Container(new_gn, 4)
            new_dict[key] = 5
            # Make another key that should hash to the same value
            key1 = Container(new_gn, 4)
            return x * new_dict[key1]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_custom_object_as_dict_key(self):
        """Test that custom objects with __hash__ as dict keys are properly handled.

        This test verifies that when using custom objects with overridden __hash__
        and __eq__ as dictionary keys, two instances with the same hash and equality
        should be recognized as the same key.
        """

        class CustomKey:
            def __init__(self, value, name):
                self.value = value
                self.name = name

        def fn(x):
            d = {}
            # Create first instance
            key1 = CustomKey(42, "test")
            d[key1] = x * 2

            # Create second instance with same values - should hash to same value
            key2 = CustomKey(42, "test")
            d[key2] = x * 3  # This should overwrite the first value

            return d[key1] * d[key2]

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertTrue(same(opt_fn(x), fn(x)))

    def test_user_defined_object(self):
        class A:
            def __init__(self):
                self.x = {}
                REF[self] = {}

        REF = {}

        def f(a, x):
            REF[a]["foo"] = x
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        x = torch.randn(4)
        self.assertTrue(same(f(A(), x), opt_f(A(), x)))

    @make_dynamo_test
    def test_bool(self):
        p = self.thetype()
        q = self.thetype({"a": 1, "b": 2})
        if p:
            self.fail("empty mapping must compare to False")
        if not q:
            self.fail("non-empty mapping must compare to True")
        if bool(p):
            self.fail("empty mapping must compare to False")
        if not bool(q):
            self.fail("non-empty mapping must compare to True")


class DictSubclassMethodsTests(DictMethodsTests):
    hw_classification = HardwareClassification.GENERIC

    thetype = SimpleDict


class OrderedDictMethodsTests(DictMethodsTests):
    hw_classification = HardwareClassification.GENERIC

    thetype = OrderedDict

    # Methods:
    # - popitem - Inherited from DictMethodsTest
    # + move_to_end

    @make_dynamo_test
    def test_move_to_end(self):
        d = self.thetype.fromkeys("abcde")
        self.assertEqual("".join(d), "abcde")
        d.move_to_end("b")
        self.assertEqual("".join(d), "acdeb")

        # Test OrderedDict.move_to_end
        self.thetype.move_to_end(d, "a")
        self.assertEqual("".join(d), "cdeba")

        # Test last=False
        self.thetype.move_to_end(d, "a", last=False)
        self.assertEqual("".join(d), "acdeb")

        # Test KeyError
        self.assertRaises(KeyError, d.move_to_end, "f")

    def test_cmp_eq_order(self):
        a = self.thetype.fromkeys("abc")
        b = self.thetype.fromkeys("bca")
        self.assertFalse(a == b)

    @make_dynamo_test
    def test_binop_or_return_type(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test return type
        self.assertIs(type(d1 | d2), OrderedDict)
        self.assertIs(type(dict(d1) | d2), OrderedDict)
        self.assertIs(type(d1 | dict(d2)), OrderedDict)

    @make_dynamo_test
    def test_binop_ior_return_type(self):
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test return type
        d3, d4 = d1.copy(), d2.copy()
        self.assertIs(type(d3.__ior__(d2)), OrderedDict)
        self.assertIs(type(dict.__ior__(d4, d2)), OrderedDict)
        self.assertIs(type(self.thetype.__ior__(d4, d2)), OrderedDict)

        d3, d4 = d1.copy(), d2.copy()
        self.assertIs(type(dict.__ior__(d3, dict(d2))), OrderedDict)
        self.assertIs(type(dict.__ior__(dict(d3), d2)), dict)
        self.assertIs(type(dict(d4).__ior__(d2)), dict)

    @make_dynamo_test
    def test_popitem_kwarg(self):
        d = self.thetype.fromkeys("abcdf")
        self.assertEqual(d.popitem(last=True), ("f", None))
        self.assertEqual(list(d), list("abcd"))
        self.assertEqual(d.popitem(last=False), ("a", None))
        self.assertEqual(list(d), list("bcd"))
        self.assertEqual(d.popitem(False), ("b", None))
        self.assertEqual(list(d), list("cd"))
        self.assertEqual(d.popitem(True), ("d", None))
        self.assertEqual(list(d), list("c"))


class OrderedDictSubclassOverload(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        self._prev_trace_unittest = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self._prev_trace_unittest
        return super().tearDown()

    def assertEqual(self, x, y):
        self.assertTrue(x == y, lambda msg: f"{msg}\nExpected {x} to be equal to {y}")

    def assertNotEqual(self, x, y):
        self.assertFalse(
            x == y, lambda msg: f"{msg}\nExpected {x} to not be equal to {y}"
        )

    class OrderedDictSubclass(OrderedDict):
        def get(self, key, default=None, /):
            return default

        def move_to_end(self, key, last=True, /):
            # change the behavior to something else
            self.pop(key)

    thetype = OrderedDictSubclass

    @make_dynamo_test
    def test_move_to_end(self):
        p = self.thetype({"a": 1, "b": 2, "c": 3})
        p.move_to_end("a")
        self.assertEqual(list(p.keys()), list("bc"))


class DunderDictVariableTests(torch._dynamo.test_case.TestCase):
    """Tests for DunderDictVariable (object.__dict__ handling in Dynamo)"""

    hw_classification = HardwareClassification.GENERIC

    def test_dunder_dict_items_includes_mutations(self):
        """Test that __dict__.items() includes both original and mutated keys"""

        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.c = 3
            items = dict(obj.__dict__.items())
            return items

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_dunder_dict_keys_includes_mutations(self):
        """Test that __dict__.keys() includes both original and mutated keys"""

        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.c = 3
            obj.d = 4
            keys = sorted(obj.__dict__.keys())
            return keys

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, ["a", "b", "c", "d"])

    def test_dunder_dict_values_includes_mutations(self):
        """Test that __dict__.values() includes both original and mutated values"""

        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.c = 3
            values = sorted(obj.__dict__.values())
            return values

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, [1, 2, 3])

    def test_dunder_dict_copy_does_not_alias_instance_dict(self):
        class MyClass:
            def __init__(self):
                self.y = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.__dict__.copy()["x"] = 4
            return "x" in obj.__dict__, dict(obj.__dict__.items())

        obj = MyClass()
        has_x, live_dict = fn(obj)
        self.assertFalse(has_x)
        self.assertEqual(live_dict, {"y": 1})
        self.assertNotIn("x", obj.__dict__)

    def test_dunder_dict_comprehension_with_mutations(self):
        """Test dict comprehension over __dict__ with mutations (scheduler use case)"""

        class SimpleScheduler:
            def __init__(self):
                self.base_lrs = [0.1]
                self.last_epoch = 0

            def state_dict(self):
                state = dict(self.__dict__.items())
                return state

        @torch.compile(backend="eager", fullgraph=True)
        def fn(scheduler):
            scheduler.last_epoch = 5
            scheduler.new_attr = "test"
            return scheduler.state_dict()

        scheduler = SimpleScheduler()
        result = fn(scheduler)
        self.assertEqual(result["base_lrs"], [0.1])
        self.assertEqual(result["last_epoch"], 5)
        self.assertEqual(result["new_attr"], "test")

    def test_dunder_dict_multiple_mutations(self):
        """Test __dict__ operations with multiple mutations"""

        class MyClass:
            def __init__(self):
                self.x = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.y = 2
            obj.z = 3
            obj.y = 20
            items = dict(obj.__dict__.items())
            keys = set(obj.__dict__.keys())
            return items, keys

        obj = MyClass()
        items, keys = fn(obj)
        self.assertEqual(items, {"x": 1, "y": 20, "z": 3})
        self.assertEqual(keys, {"x", "y", "z"})

    def test_dunder_dict_pop_reinsert_order(self):
        # CPython appends a re-inserted key at the end of the dict rather than
        # reusing its old slot. obj.__dict__ ordering is observable, so popping
        # then re-adding a key must move it to the end (mirrors
        # test_dict.DictTest.test_splittable_pop).
        class MyClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            obj = MyClass()
            obj.x, obj.y, obj.z = 1, 2, 3
            d = obj.__dict__
            d.pop("y")
            d["y"] = 42
            return list(d), d["y"]

        keys, value = fn()
        self.assertEqual(keys, ["x", "z", "y"])
        self.assertEqual(value, 42)

    def test_dunder_dict_pop_reinsert_order_with_other_mutations(self):
        # Mutating other keys after a pop must not disturb their relative order;
        # only the re-inserted popped key moves to the end. Updating an existing
        # key keeps its slot, a brand-new key appends, and the reinserted key
        # lands last.
        class MyClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            obj = MyClass()
            obj.a, obj.b, obj.c = 1, 2, 3
            d = obj.__dict__
            d.pop("b")  # a, c
            d["c"] = 30  # update existing -> stays in place: a, c
            d["e"] = 5  # new key -> appended: a, c, e
            d["b"] = 20  # reinsert popped -> end: a, c, e, b
            return list(d), dict(d)

        keys, value = fn()
        self.assertEqual(keys, ["a", "c", "e", "b"])
        self.assertEqual(value, {"a": 1, "c": 30, "e": 5, "b": 20})

    def test_dunder_dict_pop_missing_raises_keyerror(self):
        class MyClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            obj = MyClass()
            obj.x = 1
            d = obj.__dict__
            d.pop("x")
            try:
                d.pop("x")
                raised = False
            except KeyError:
                raised = True
            return list(d), raised

        self.assertEqual(fn(), ([], True))

    def test_dunder_dict_pop_default(self):
        class MyClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            obj = MyClass()
            obj.x = 1
            d = obj.__dict__
            return d.pop("missing", "fallback"), d.pop("x"), list(d)

        self.assertEqual(fn(), ("fallback", 1, []))

    def test_dunder_dict_items_iteration(self):
        """Test iterating over __dict__.items() with mutations"""

        class MyClass:
            def __init__(self):
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            obj.c = 3
            result = []
            for k, v in obj.__dict__.items():
                result.append((k, v))
            return sorted(result)

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, [("a", 1), ("b", 2), ("c", 3)])

    def test_method_dict_direct_fullgraph(self):
        """
        Regression test: Accessing __dict__ directly on UserMethodVariable.
        This should fail with: unsupported variable type for __dict__ access
        """

        class Foo:
            def bar(self):
                return 42

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            obj = Foo()
            # Access __dict__ on bound method - creates UserMethodVariable
            method = obj.bar
            d = method.__dict__
            return d

        # This should not raise Unsupported
        fn()

    def test_dunder_dict_non_str_key_setitem(self):
        # CPython's instance __dict__ accepts arbitrary hashable keys when set
        # via the mapping API; only attribute access via setattr requires str.
        # Mirrors test_dict.DictTest.test_object_set_item_single_instance_non_str_key.
        class Foo:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            f = Foo()
            f.__dict__[1] = 1
            f.a = "a"
            return dict(f.__dict__), list(f.__dict__)

        d, keys = fn()
        self.assertEqual(d, {1: 1, "a": "a"})
        self.assertEqual(keys, [1, "a"])

    def test_dunder_dict_non_str_key_roundtrip(self):
        # Non-str instance-dict keys must read back and delete correctly.
        class Foo:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            f = Foo()
            d = f.__dict__
            d[1] = 10
            d[2.5] = 20
            d[(3, 4)] = 30
            got = (d[1], d[2.5], d[(3, 4)])
            del d[2.5]
            return got, dict(d)

        got, d = fn()
        self.assertEqual(got, (10, 20, 30))
        self.assertEqual(d, {1: 10, (3, 4): 30})

    def test_dunder_dict_iteration_no_recompile_on_mutation(self):
        # DunderDictVariable.install_dict_keys_match_guard() is a deliberate
        # no-op: __dict__ mutations are already tracked via side effects, so
        # guarding on its keys would cause needless recompiles. tp_iter_impl
        # must route through that overridable hook (not install the guard
        # directly) so plain `for k in obj.__dict__` iteration respects it too.
        class Foo:
            pass

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def fn(obj, x):
            total = 0
            for _ in obj.__dict__:
                total += 1
            return x + total

        f = Foo()
        f.a, f.b = 1, 2
        x = torch.zeros(1)
        self.assertEqual(fn(f, x), x + 2)
        self.assertEqual(cnts.frame_count, 1)

        # Mutating __dict__'s shape between calls must not trigger a
        # recompile: DunderDictVariable deliberately suppresses
        # DICT_KEYS_MATCH, so the guard tree must stay unaffected by this.
        f.c = 3
        fn(f, x)
        self.assertEqual(cnts.frame_count, 1)

        f.d, f.e = 4, 5
        fn(f, x)
        self.assertEqual(cnts.frame_count, 1)


instantiate_parametrized_tests(DictGuardTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
