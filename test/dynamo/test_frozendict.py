# Owner(s): ["module: dynamo"]

import builtins
import operator
import unittest
from collections import defaultdict, OrderedDict
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch.utils._pytree as python_pytree
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.testing import CompileCounter
from torch._dynamo.variables.builder import SourcelessBuilder
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    runWithoutCompiledAutograd,
    subtest,
)


pytree_modules = {
    "python": python_pytree,
}
if python_pytree._cxx_pytree_dynamo_traceable:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
    pytree_modules["native_optree"] = cxx_pytree.optree

parametrize_pytree_module = parametrize(
    "pytree",
    [subtest(module, name=name) for name, module in pytree_modules.items()],
)


@unittest.skipIf(not torch._has_frozendict, "requires builtins.frozendict")
@instantiate_parametrized_tests
class FrozenDictTests(torch._dynamo.test_case.TestCase):
    @parametrize("mapping_input", [False, True])
    @parametrize("key_error", ["unhashable", "custom", "subclass"])
    def test_constructor_key_type_errors(self, mapping_input, key_error):
        class KeyTypeError(TypeError):
            pass

        class Key:
            def __hash__(self):
                if key_error == "subclass":
                    raise KeyTypeError("custom hash failed", 17)
                raise TypeError("custom hash failed")

        class Mapping:
            def __init__(self, key, value):
                self.key = key
                self.value = value

            def keys(self):
                return [self.key]

            def __getitem__(self, key):
                return self.value

        def fn(x):
            key = [] if key_error == "unhashable" else Key()
            source = Mapping(key, x) if mapping_input else [(key, x)]
            try:
                builtins.frozendict(source)
            except TypeError as error:
                return x + 1, type(error), error.args
            return x - 1, None, ()

        x = torch.randn(3)
        expected = fn(x)
        if key_error == "subclass":
            self.assertIs(expected[1], KeyTypeError)
            self.assertEqual(expected[2], ("custom hash failed", 17))
        else:
            self.assertIs(expected[1], TypeError)
            self.assertIn("as a frozendict key", expected[2][0])
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("key_error", ["unhashable", "custom", "subclass"])
    def test_fromkeys_key_type_errors(self, key_error):
        class KeyTypeError(TypeError):
            pass

        class Key:
            def __hash__(self):
                if key_error == "subclass":
                    raise KeyTypeError("custom hash failed", 17)
                raise TypeError("custom hash failed")

        def fn(x):
            key = [] if key_error == "unhashable" else Key()
            try:
                builtins.frozendict.fromkeys([key], x)
            except TypeError as error:
                return x + 1, type(error), error.args
            return x - 1, None, ()

        x = torch.randn(3)
        expected = fn(x)
        if key_error == "subclass":
            self.assertIs(expected[1], KeyTypeError)
            self.assertEqual(expected[2], ("custom hash failed", 17))
        else:
            self.assertIs(expected[1], TypeError)
            self.assertIn("as a frozendict key", expected[2][0])
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("explicit_new", [False, True])
    def test_constructor_preserves_key_comparisons(self, explicit_new):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return self.calls > 1

        def fn(x):
            first, second = Key(), Key()
            if explicit_new:
                mapping = builtins.frozendict.__new__(
                    builtins.frozendict, [(first, x), (second, x + 1)]
                )
            else:
                mapping = builtins.frozendict([(first, x), (second, x + 1)])
            return x + len(mapping), first.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 2, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_subscription_compares_key_once(self):
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
            mapping = builtins.frozendict([(stored, x)])
            return mapping[probe] + 1, stored.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 1, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("backend", ["eager", "aot_eager"])
    @parametrize("construct", [False, True])
    def test_allowed_function_argument(self, backend, construct):
        @torch.compiler.allow_in_graph
        def consume(mapping):
            if type(mapping) is not builtins.frozendict:
                raise AssertionError("expected frozendict")
            if type(mapping["nested"]) is not builtins.frozendict:
                raise AssertionError("expected nested frozendict")
            return mapping["x"] + mapping["nested"]["y"]

        def fn(x, mapping):
            if construct:
                mapping = builtins.frozendict(x=x, nested=builtins.frozendict(y=x * 2))
            return consume(mapping)

        try:
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for x in (torch.ones(2), torch.full((2,), 3.0)):
                mapping = builtins.frozendict(x=x, nested=builtins.frozendict(y=x * 2))
                self.assertEqual(compiled(x, mapping), fn(x, mapping))
        finally:
            torch._dynamo.disallow_in_graph(consume)

    def test_sourceless_custom_key_hash(self):
        class Meta(type):
            hash_value = 1
            calls = 0

            def __hash__(cls):
                Meta.calls += 1
                return Meta.hash_value

        class Key(metaclass=Meta):
            pass

        mapping = builtins.frozendict({Key: 5})
        Meta.hash_value = 2
        calls = Meta.calls
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Preexisting frozendict key"
        ):
            SourcelessBuilder.create(Mock(), mapping)
        self.assertEqual(Meta.calls, calls)
        literal = builtins.frozendict(a=5)
        tx = Mock()
        with InstructionTranslator.set_current_tx(tx):
            self.assertEqual(
                SourcelessBuilder.create(tx, literal).as_python_constant(), literal
            )

    def test_class_getitem(self):
        def fn(x):
            return x + 1, builtins.frozendict.__class_getitem__((str, int))

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("case", ["missing", "object", "unrelated", "extra"])
    def test_new_invalid_arguments(self, case):
        class NotAType:
            pass

        receiver = NotAType() if case == "object" else dict

        def fn(x):
            try:
                if case == "missing":
                    builtins.frozendict.__new__()
                elif case == "extra":
                    builtins.frozendict.__new__(builtins.frozendict, {}, {})
                else:
                    builtins.frozendict.__new__(receiver)
            except TypeError:
                return x + 1
            return x - 1

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("construct", [False, True])
    @parametrize(
        "name,args",
        [
            ("get", ("a",)),
            ("keys", ()),
            ("values", ()),
            ("items", ()),
            ("__getitem__", ("a",)),
            ("__len__", ()),
        ],
    )
    def test_dict_descriptor_rejects_frozen_receiver(self, construct, name, args):
        def fn(x, mapping):
            if construct:
                mapping = builtins.frozendict(a=x)
            try:
                getattr(dict, name)(mapping, *args)
            except TypeError as error:
                return x + 1, str(error)
            return x - 1, "missing error"

        x = torch.ones(1)
        mapping = builtins.frozendict(a=x)
        expected = fn(x, mapping)
        self.assertEqual(expected[0], x + 1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping), expected
        )

    @parametrize("construct", [False, True])
    def test_reads_and_reconstruction(self, construct):
        def fn(x, mapping):
            if construct:
                mapping = builtins.frozendict(mapping, c=x + 1)
            result = builtins.frozendict((k, v * 2) for k, v in mapping.items())
            return (
                result,
                list(mapping),
                list(reversed(mapping)),
                mapping.get("missing", x),
                "a" in mapping,
            )

        x = torch.randn(3)
        mapping = builtins.frozendict(a=x, b=x + 2)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, mapping)
        self.assertEqual(result, fn(x, mapping))
        self.assertIs(type(result[0]), builtins.frozendict)

    def test_arbitrary_keys_and_guards(self):
        counter = CompileCounter()

        def fn(mapping, key):
            return mapping[key] + next(iter(mapping.values()))

        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        tensor_key = torch.randn(2)
        for key in (None, (1, "a"), tensor_key, object()):
            mapping = builtins.frozendict(
                [(key, torch.randn(3)), ("last", torch.randn(3))]
            )
            self.assertEqual(compiled(mapping, key), fn(mapping, key))
        self.assertGreater(counter.frame_count, 1)

    @parametrize(
        "wrap",
        [
            lambda key: key,
            lambda key: (key,),
            lambda key: frozenset([key]),
            python_pytree.MappingKey,
            python_pytree.SequenceKey,
            python_pytree.GetAttrKey,
        ],
    )
    def test_preexisting_custom_key_hash(self, wrap):
        class Key:
            def __init__(self):
                self.value = 1
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return self.value

        key = Key()
        wrapped = wrap(key)
        mapping = builtins.frozendict([(wrapped, 1)])
        key.value = 2
        calls = key.calls

        def fn(x, mapping):
            return x + len(mapping)

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "stored key hash"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1), mapping)
        self.assertEqual(key.calls, calls)
        self.assertEqual(torch.compile(fn, backend="eager")(torch.ones(1), mapping), 2)
        self.assertEqual(key.calls, calls)

    def test_constructed_custom_keys(self):
        class Key:
            def __hash__(self):
                return 42

        def fn(x):
            first, second = Key(), Key()
            mapping = builtins.frozendict([(first, x), (second, x + 1)])
            return mapping[first] + mapping[second], list(mapping.values())

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("with_kwargs", [False, True])
    def test_conversion_preserves_stored_key_hashes(self, with_kwargs):
        class Key:
            def __init__(self):
                self.value = 1
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return self.value

        def fn(x):
            key = Key()
            mapping = builtins.frozendict([(key, x)])
            key.value = 2
            copied = dict(mapping, extra=x + 1) if with_kwargs else dict(mapping)
            unpacked = {**mapping}
            rebuilt = builtins.frozendict(mapping, extra=x + 1)
            return (
                next(iter(copied.values())),
                next(iter(unpacked.values())),
                next(iter(rebuilt.values())),
                rebuilt["extra"],
                key.calls,
            )

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_constructor_identity(self):
        def fn(x):
            mapping = builtins.frozendict(a=x)
            rebuilt = builtins.frozendict(mapping)
            allocated = builtins.frozendict.__new__(builtins.frozendict, mapping)
            return x + 1, rebuilt is mapping, allocated is mapping

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertEqual(result[1:], (True, False))

    def test_explicit_subclass_allocation(self):
        class Frozen(builtins.frozendict):
            pass

        def fn(x):
            try:
                mapping = builtins.frozendict.__new__(Frozen, a=x)
            except TypeError:
                return x - 1
            return mapping["a"] + 1

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("name", ["copy", "keys", "__len__"])
    def test_descriptor_receiver_errors(self, name):
        def fn(x):
            try:
                getattr(builtins.frozendict, name)({})
            except TypeError:
                return x + 1
            return x - 1

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_guard_invalidation(self):
        counter = CompileCounter()

        def fn(mapping):
            return next(iter(mapping.values())) + len(mapping), list(mapping)

        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for items in (
            [("a", torch.randn(3))],
            [("a", torch.randn(3))],
            [("b", torch.randn(3))],
            [("a", torch.randn(3)), ("b", torch.randn(3))],
            [("b", torch.randn(3)), ("a", torch.randn(3))],
        ):
            mapping = builtins.frozendict(items)
            self.assertEqual(compiled(mapping), fn(mapping))
        self.assertEqual(counter.frame_count, 4)

    def test_views_and_aliases(self):
        def fn(x):
            values = [x]
            mapping = builtins.frozendict(a=values)
            values.append(x + 1)
            return (
                mapping,
                mapping.keys(),
                mapping.values(),
                mapping.items(),
                mapping.keys().mapping,
            )

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result[0]["a"], [x, x + 1])
        self.assertEqual(list(result[1]), ["a"])
        self.assertIs(next(iter(result[2])), result[0]["a"])
        self.assertIs(next(iter(result[3]))[1], result[0]["a"])
        self.assertIs(result[4]["a"], result[0]["a"])
        with self.assertRaisesRegex(TypeError, "does not support item assignment"):
            result[4]["a"] = x

    @parametrize("name", ["__eq__", "__ne__", "__or__"])
    def test_dict_descriptors(self, name):
        def fn(x):
            op = getattr(dict, name)
            return (
                x + 1,
                op({"a": x}, builtins.frozendict(a=x)),
                op({"a": x}, builtins.frozendict(b=x)),
            )

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("operation", [operator.setitem, operator.delitem])
    def test_immutable(self, operation):
        def fn(x):
            mapping = builtins.frozendict(a=x)
            try:
                if operation is operator.setitem:
                    operation(mapping, "a", x + 1)
                else:
                    operation(mapping, "a")
            except TypeError:
                return mapping["a"] + 2
            return x - 100

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_graph_break_reconstruction(self):
        def fn(x):
            mapping = builtins.frozendict(a=x + 1)
            view = mapping.values()
            torch._dynamo.graph_break()
            return mapping, view, next(iter(view)) * 2

        x = torch.randn(3)
        mapping, view, value = torch.compile(fn, backend="eager")(x)
        self.assertIs(type(mapping), builtins.frozendict)
        self.assertIs(next(iter(view)), mapping["a"])
        self.assertEqual(value, (x + 1) * 2)

    @parametrize_pytree_module
    def test_pytree_roundtrip(self, pytree):
        def fn(x):
            tree = builtins.frozendict(b=[x + 1], a=x * 2)
            kwargs = {"namespace": "torch"} if pytree.__name__ == "optree" else {}
            leaves, spec = pytree.tree_flatten(tree, **kwargs)
            leaves = [v.sin() for v in leaves]
            if pytree.__name__ == "optree":
                return pytree.tree_unflatten(spec, leaves)
            return pytree.tree_unflatten(leaves, spec)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), builtins.frozendict)
        self.assertEqual(result, fn(x))
        self.assertEqual(list(result), ["b", "a"])

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @parametrize("construct", [False, True])
    def test_autograd(self, construct):
        def fn(x, y, inputs):
            if construct:
                inputs = builtins.frozendict(y=y, x=x)
            return torch.autograd.grad((x * x + 3 * y).sum(), inputs)

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        inputs = builtins.frozendict(y=y, x=x)
        expected = fn(x, y, inputs)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, y, inputs)
        self.assertIs(type(result), dict)
        self.assertEqual(list(result), ["y", "x"])
        self.assertEqual(result, expected)
        self.assertEqual(result["y"], torch.full_like(y, 3))
        self.assertEqual(result["x"], 2 * x)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_backward_and_unused(self):
        def fn(x, y):
            inputs = builtins.frozendict(y=y, x=x)
            grad = torch.autograd.grad((x * x).sum(), inputs, allow_unused=True)
            (x * x + 3 * y).sum().backward(inputs=inputs)
            return grad

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(x, y)
        self.assertIsNone(result["y"])
        self.assertEqual(result["x"], 2 * x)
        self.assertEqual(x.grad, 2 * x)
        self.assertEqual(y.grad, torch.full_like(y, 3))

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @runWithoutCompiledAutograd("Test external GradientEdge validation")
    def test_empty_and_invalid_autograd_inputs(self):
        x = torch.randn(3, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            torch.autograd.grad(x.sum(), builtins.frozendict())
        edge = torch.autograd.graph.get_gradient_edge(x)
        self.assertEqual(
            torch.autograd.grad((x * x).sum(), builtins.frozendict(x=edge))["x"],
            2 * x,
        )

        def fn(x, inputs):
            return torch.autograd.grad((x * x).sum(), inputs)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "external GradientEdge"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(
                x, builtins.frozendict(x=edge)
            )

    def test_dunder_dict_union(self):
        class Holder:
            def __init__(self, x):
                self.a = x
                self.b = x + 1

        def fn(obj, x):
            return obj.__dict__ | builtins.frozendict(b=x + 2, c=x)

        x = torch.randn(3)
        obj = Holder(x)
        result = torch.compile(fn, backend="eager", fullgraph=True)(obj, x)
        self.assertEqual(result, fn(obj, x))
        self.assertIs(type(result), dict)
        self.assertEqual(list(result), ["a", "b", "c"])
        self.assertIs(result["a"], obj.a)
        self.assertIs(result["c"], x)

    def test_missing_key_exception(self):
        def fn(x):
            mapping = builtins.frozendict(a=x)
            try:
                return mapping["missing"]
            except KeyError as error:
                return x + 1, error.args

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("other_type", [dict, getattr(builtins, "frozendict", dict)])
    def test_equality_unequal_values(self, other_type):
        def fn(x):
            left = builtins.frozendict(a=1, b=2)
            right = other_type(a=1, b=3)
            return x + 1, left == right, right == left, left != right, right != left

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "iterable_type", [dict, getattr(builtins, "frozendict", dict), set, frozenset]
    )
    def test_fromkeys_container_inputs(self, iterable_type):
        def fn(x, keys):
            result = builtins.frozendict.fromkeys(keys, x + 1)
            return result, list(result)

        x = torch.randn(3)
        keys = iterable_type({"b": 0, "a": 1})
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, keys)
        self.assertEqual(result, fn(x, keys))
        self.assertIs(type(result[0]), builtins.frozendict)
        self.assertIs(result[0]["a"], result[0]["b"])

    def test_fromkeys_does_not_compare_iterable_type(self):
        class Meta(type):
            def __eq__(cls, other):
                if other is builtins.frozendict:
                    raise RuntimeError("fromkeys must not compare types")
                return cls is other

            __hash__ = type.__hash__

        class Keys(list, metaclass=Meta):
            pass

        def fn(x, keys):
            return builtins.frozendict.fromkeys(keys, x + 1)

        x = torch.randn(3)
        keys = Keys(["b", "a"])
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, keys), fn(x, keys)
        )

    def test_copy_and_fromkeys(self):
        def fn(x):
            mapping = builtins.frozendict(a=x)
            fromkeys = mapping.fromkeys(["b", "a", "b"], x + 1)
            return mapping.copy() is mapping, fromkeys

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertTrue(result[0])
        self.assertEqual(list(result[1]), ["b", "a"])
        self.assertIs(result[1]["a"], result[1]["b"])

    @parametrize("mapping_type", [OrderedDict, defaultdict])
    @parametrize("reverse", [False, True])
    @parametrize("direct", [False, True])
    def test_union_preserves_mutable_mapping_type(self, mapping_type, reverse, direct):
        def fn(x):
            mapping = (
                defaultdict(int, a=x)
                if mapping_type is defaultdict
                else mapping_type(a=x)
            )
            frozen = builtins.frozendict(a=x + 1, b=x + 2)
            if direct:
                return mapping.__ror__(frozen) if reverse else mapping.__or__(frozen)
            return frozen | mapping if reverse else mapping | frozen

        x = torch.randn(3)
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), type(expected))
        self.assertEqual(list(result), list(expected))
        self.assertEqual(result, expected)
        if isinstance(expected, defaultdict):
            self.assertIs(result.default_factory, int)

    def test_defaultdict_union_bypasses_update_override(self):
        class Mapping(defaultdict):
            def update(self, *args, **kwargs):
                raise RuntimeError("union must not call update override")

        def fn(x):
            return Mapping(int, a=x) | builtins.frozendict(b=x + 1)

        x = torch.randn(3)
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), Mapping)
        self.assertIs(result.default_factory, int)
        self.assertEqual(result, expected)

    @parametrize("operation", [operator.eq, operator.or_])
    def test_input_dict_with_custom_new(self, operation):
        class Mapping(dict):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)

        def fn(x, mapping):
            frozen = builtins.frozendict(a=x)
            return x + 1, operation(frozen, mapping)

        x = torch.randn(3)
        mapping = Mapping(a=x)
        self.assertEqual(torch.compile(fn, backend="eager")(x, mapping), fn(x, mapping))

    def test_union_and_rebinding(self):
        def fn(x):
            left = builtins.frozendict(b=x, a=x + 1)
            right = {"a": x + 2, "c": x + 3}
            alias = left
            left |= right
            return (
                left,
                right | alias,
                alias,
                alias | {},
                builtins.frozendict() | alias,
            )

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertIs(type(result[0]), builtins.frozendict)
        self.assertIs(type(result[1]), dict)
        self.assertEqual(list(result[0]), ["b", "a", "c"])
        self.assertIs(result[2], result[3])
        self.assertIs(result[2], result[4])
        self.assertIsNot(result[0], result[2])

    @parametrize(
        "operation",
        [
            lambda d: d | builtins.frozendict(),
            lambda d: builtins.frozendict(a=0) | d,
            lambda d: builtins.frozendict.fromkeys(d),
        ],
    )
    def test_dict_key_guard_invalidation(self, operation):
        def fn(x, mapping):
            result = operation(mapping)
            return x + len(result), tuple(result)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.randn(3)
        for mapping in ({}, {"a": 1}, {"a": 2}, {"a": 1, "b": 2}, {"c": 1, "d": 2}):
            self.assertEqual(compiled(x, mapping), fn(x, mapping))
        self.assertEqual(counter.frame_count, 4)

    def test_union_preserves_key_comparisons(self):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return self.calls > 1

        def fn(x):
            first, second = Key(), Key()
            mapping = builtins.frozendict([(first, x), (second, x + 1)])
            mapping = mapping | {"tail": x}
            return x + len(mapping), first.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 3, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_equality_and_reflection(self):
        class Reflected:
            def __eq__(self, other):
                return len(other) == 2

        def fn(x):
            mapping = builtins.frozendict(a=x, b=1)
            same = {"b": 1, "a": x}
            return (
                x + (mapping == same),
                same == mapping,
                mapping != {"a": x},
                mapping == Reflected(),
            )

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("dict_side", [None, "left", "right"])
    @parametrize("change_hash", [False, True])
    def test_equality_preserves_stored_key_hashes(self, dict_side, change_hash):
        class Key:
            def __init__(self):
                self.value = 1
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return self.value

        def fn(x):
            key = Key()
            left = {key: 1} if dict_side == "left" else builtins.frozendict([(key, 1)])
            right = (
                {key: 1} if dict_side == "right" else builtins.frozendict([(key, 1)])
            )
            before = key.calls
            if change_hash:
                key.value = 2
            equal, unequal = left == right, left != right
            return x + 1, equal, unequal, key.calls - before

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertEqual(result[1:], (True, False, 0))

    @parametrize("dict_side", [None, "left", "right"])
    def test_equality_looks_up_each_key_once(self, dict_side):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return True

        def fn(x):
            first, second = Key(), Key()
            left = (
                {first: 1} if dict_side == "left" else builtins.frozendict([(first, 1)])
            )
            right = (
                {second: 1}
                if dict_side == "right"
                else builtins.frozendict([(second, 1)])
            )
            equal, unequal = left == right, left != right
            return x + 1, equal, unequal, first.calls + second.calls

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertEqual(result[1:], (True, False, 2))

    def test_hash_and_container_keys(self):
        def fn(x):
            left = builtins.frozendict(a=1, b=(2, 3))
            right = builtins.frozendict(b=(2, 3), a=1)
            table = {left: x + 1}
            members = {left}
            return (
                hash(left),
                hash(right),
                table[right],
                right in members,
                hash(left.keys().mapping),
            )

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_hash_collisions_and_cache(self):
        class Key:
            def __hash__(self):
                return 42

        class Value:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 7

        def fn(x):
            value = Value()
            mapping = builtins.frozendict([(Key(), value), (Key(), value)])
            first = hash(mapping)
            second = hash(mapping)
            return x + value.calls, first, second

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_preexisting_custom_hash_cache(self):
        class Value:
            def __init__(self):
                self.value = 1

            def __hash__(self):
                return self.value

        value = Value()
        mapping = builtins.frozendict(a=value)
        original_hash = hash(mapping)
        value.value = 2

        def fn(x, mapping):
            return x + hash(mapping)

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "cached a hash"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1), mapping)
        self.assertEqual(hash(mapping), original_hash)

    @parametrize("use_key", [False, True])
    @parametrize("graph_break", [False, True])
    @parametrize("subclass", [False, True])
    def test_custom_hash_reconstruction(self, use_key, graph_break, subclass):
        class FrozenMapping(builtins.frozendict):
            pass

        mapping_type = FrozenMapping if subclass else builtins.frozendict

        class Value:
            def __init__(self):
                self.value = 1

            def __hash__(self):
                return self.value

        def fn(x):
            value = Value()
            mapping = mapping_type([(value, 1)] if use_key else [("a", value)])
            saved_hash = hash(mapping)
            value.value = 2
            if graph_break:
                torch._dynamo.graph_break()
            return x + 1, mapping, saved_hash

        x = torch.randn(3)
        if not graph_break:
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "stored key hashes"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(x)
        result, mapping, saved_hash = torch.compile(fn, backend="eager")(x)
        self.assertIs(type(mapping), mapping_type)
        self.assertEqual(result, x + 1)
        self.assertEqual(hash(mapping), saved_hash)

    @parametrize("operation", [lambda d: d | [], lambda d: d < {}, lambda d: hash(d)])
    def test_operator_errors(self, operation):
        def fn(x):
            mapping = builtins.frozendict(a=[])
            try:
                operation(mapping)
            except TypeError:
                return x + 1
            return x - 1

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))


@unittest.skipIf(not torch._has_frozendict, "requires builtins.frozendict")
@instantiate_parametrized_tests
class FrozenDictSubclassTests(torch._dynamo.test_case.TestCase):
    @parametrize("copy", [False, True])
    def test_constructor_preserves_key_comparisons(self, copy):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return self.calls > 1

        class Frozen(builtins.frozendict):
            pass

        def fn(x):
            first, second = Key(), Key()
            mapping = Frozen([(first, x), (second, x + 1)])
            if copy:
                mapping = mapping.copy()
            return x + len(mapping), first.calls

        x = torch.ones(1)
        self.assertEqual(fn(x), (x + 2, 1))
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_subclass_type_guard(self):
        class FrozenMapping(builtins.frozendict):
            def __getitem__(self, key):
                return super().__getitem__(key) + 1

        def fn(mapping):
            return mapping["a"] * 2

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.randn(3)
        for mapping_type in (builtins.frozendict, FrozenMapping):
            mapping = mapping_type(a=x)
            self.assertEqual(compiled(mapping), fn(mapping))
        self.assertEqual(counter.frame_count, 2)

    def test_subclass_attribute_guard(self):
        class FrozenMapping(builtins.frozendict):
            pass

        def fn(mapping):
            if mapping.tag == "first":
                return mapping["a"] + 1
            return mapping["a"] + 2

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        mapping = FrozenMapping(a=torch.randn(3))
        for tag in ("first", "second"):
            mapping.tag = tag
            self.assertEqual(compiled(mapping), fn(mapping))
        self.assertEqual(counter.frame_count, 2)

    def test_fromkeys_seed_mapping_overrides(self):
        class Seeded(builtins.frozendict):
            def __new__(cls, data=None):
                return super().__new__(cls, {"seed": 5} if data is None else data)

            def __iter__(self):
                return iter(self.keys())

            def keys(self):
                return ["virtual"]

            def __getitem__(self, key):
                return 7

        def fn(x):
            result = Seeded.fromkeys(["b", "a", "b"], x)
            return result, list(builtins.frozendict.items(result))

        x = torch.randn(3)
        result, items = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), Seeded)
        self.assertEqual(items, [("virtual", 7), ("b", x), ("a", x)])
        self.assertEqual(items, fn(x)[1])

    def test_construction_attributes_and_reconstruction(self):
        class FrozenMapping(builtins.frozendict):
            def __new__(cls, value):
                return super().__new__(cls, value=value)

            def __init__(self, value):
                self.tag = "initial"

        def fn(x):
            mapping = FrozenMapping(x + 1)
            mapping.tag = "updated"
            return mapping, mapping.copy(), mapping.keys().mapping

        x = torch.randn(3)
        result, copied, proxy = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), FrozenMapping)
        self.assertEqual(result["value"], x + 1)
        self.assertEqual(result.tag, "updated")
        self.assertIs(type(copied), builtins.frozendict)
        self.assertIs(proxy["value"], result["value"])

    def test_overrides_and_base_descriptors(self):
        class FrozenMapping(builtins.frozendict):
            def __getitem__(self, key):
                return super().__getitem__(key) + 1

            def __len__(self):
                return 10

            def __missing__(self, key):
                return 99

            def __ror__(self, other):
                return "reflected"

        def fn(x, mapping):
            return (
                mapping["a"],
                builtins.frozendict.__getitem__(mapping, "a"),
                mapping["missing"],
                len(mapping),
                {} | mapping,
                x + 1,
            )

        x = torch.randn(3)
        mapping = FrozenMapping(a=x)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    @parametrize("construct", [False, True])
    @parametrize("override_iter", [False, True])
    def test_constructor_mapping_overrides(self, construct, override_iter):
        class FrozenMapping(builtins.frozendict):
            def keys(self):
                return ["virtual"]

            def __getitem__(self, key):
                return builtins.frozendict.__getitem__(self, "a") + 1

        if override_iter:
            FrozenMapping.__iter__ = lambda self: iter(["virtual"])

        def fn(x, source):
            if construct:
                source = FrozenMapping(a=x)
            return (
                builtins.frozendict(source),
                builtins.frozendict(b=x) | source,
                source.copy(),
                source | {},
                dict(source),
            )

        x = torch.randn(3)
        source = FrozenMapping(a=x)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, source)
        self.assertEqual(result, fn(x, source))
        self.assertIs(type(result[4]), dict)
        for index in (0, 2, 3):
            self.assertIs(type(result[index]), builtins.frozendict)
            self.assertEqual(
                list(result[index]), ["virtual"] if override_iter else ["a"]
            )

    def test_empty_subclass_copy(self):
        class FrozenMapping(builtins.frozendict):
            def __iter__(self):
                return iter(["virtual"])

            def keys(self):
                raise RuntimeError("empty copies must not call keys")

        def fn(x):
            mapping = FrozenMapping()
            return x + 1, mapping.copy(), mapping | {}

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("construct", [False, True])
    def test_missing_uses_special_lookup(self, construct):
        class FrozenMapping(builtins.frozendict):
            def __missing__(self, key):
                return 99

            def __getattribute__(self, name):
                if name == "__missing__":
                    raise RuntimeError("instance lookup must not run")
                return super().__getattribute__(name)

        def fn(x, mapping):
            if construct:
                mapping = builtins.frozendict.__new__(FrozenMapping, a=x)
            return x + mapping["missing"]

        x = torch.randn(3)
        mapping = FrozenMapping(a=x)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    @parametrize("name", ["keys", "values", "items"])
    def test_base_view_reconstruction(self, name):
        class FrozenMapping(builtins.frozendict):
            def keys(self):
                return ["override"]

            def values(self):
                return ["override"]

            def items(self):
                return ["override"]

        def fn(x):
            mapping = FrozenMapping(a=x + 1)
            return mapping, getattr(builtins.frozendict, name)(mapping)

        x = torch.randn(3)
        mapping, view = torch.compile(fn, backend="eager", fullgraph=True)(x)
        expected = getattr(builtins.frozendict, name)(mapping)
        self.assertIs(type(view), type(expected))
        self.assertEqual(list(view), list(expected))
        self.assertIs(view.mapping["a"], mapping["a"])

    def test_fromkeys_non_frozen_constructor_result(self):
        class Mapping(builtins.frozendict):
            def __new__(cls):
                return {}

        def fn(x):
            return Mapping.fromkeys(iter(["b", "a", "b"]), x + 1)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), dict)
        self.assertEqual(list(result), ["b", "a"])
        self.assertEqual(result, fn(x))

    def test_fromkeys_preserves_initial_contents(self):
        class Seeded(builtins.frozendict):
            def __new__(cls, data=()):
                return super().__new__(cls, {"seed": 5} | dict(data))

        def fn(x):
            return Seeded.fromkeys(["b", "a", "b"], x)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), Seeded)
        self.assertEqual(list(result), ["seed", "b", "a"])
        self.assertEqual(result, fn(x))

    @parametrize(
        "iterable_type",
        [list, iter, dict, getattr(builtins, "frozendict", dict), set, frozenset],
    )
    def test_fromkeys_seed_comparison_order(self, iterable_type):
        state = {"seed_seen": False}

        class Key:
            def __init__(self, name):
                self.name = name

            def __hash__(self):
                return 42

            def __eq__(self, other):
                if self.name == "seed":
                    state["seed_seen"] = True
                return self.name == "a" and other.name == "b" and not state["seed_seen"]

        class Seeded(builtins.frozendict):
            def __new__(cls, data=None):
                if data is None:
                    data = [(Key("seed"), 0)]
                return super().__new__(cls, data)

        def fn(x, *, use_polyfill=False):
            state["seed_seen"] = True
            keys = iterable_type({Key("a"): 0, Key("b"): 0})
            state["seed_seen"] = False
            if use_polyfill:
                result = torch._dynamo.polyfills.frozendict_fromkeys(Seeded, keys, x)
            else:
                result = Seeded.fromkeys(keys, x)
            return x + len(result), [key.name for key in result]

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected, (x + 3, ["seed", "a", "b"]))
        self.assertEqual(fn(x, use_polyfill=True), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize(
        "iterable_type", [dict, getattr(builtins, "frozendict", dict), set, frozenset]
    )
    def test_fromkeys_seed_preserves_key_hashes(self, iterable_type):
        class Key:
            def __init__(self):
                self.calls = 0

            def __hash__(self):
                self.calls += 1
                return 7

        class Seeded(builtins.frozendict):
            def __new__(cls, data=None):
                return super().__new__(cls, {"seed": 5} if data is None else data)

        def fn(x, *, use_polyfill=False):
            key = Key()
            keys = iterable_type({key: 1})
            calls = key.calls
            if use_polyfill:
                result = torch._dynamo.polyfills.frozendict_fromkeys(Seeded, keys, x)
            else:
                result = Seeded.fromkeys(keys, x)
            return x + len(result), key.calls - calls

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected, (x + 2, 0))
        self.assertEqual(fn(x, use_polyfill=True), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize(
        "iterable_type",
        [list, iter, dict, getattr(builtins, "frozendict", dict), set, frozenset],
    )
    def test_fromkeys_seed_comparison_error(self, iterable_type):
        class Key:
            def __hash__(self):
                return 42

            def __eq__(self, other):
                raise RuntimeError("seed collision")

        class Seeded(builtins.frozendict):
            def __new__(cls, data=None):
                return super().__new__(cls, {Key(): 5} if data is None else data)

        def fn(x, *, use_polyfill=False):
            keys = iterable_type({Key(): 0})
            try:
                if use_polyfill:
                    torch._dynamo.polyfills.frozendict_fromkeys(Seeded, keys, x)
                else:
                    Seeded.fromkeys(keys, x)
            except RuntimeError as error:
                return x + 1, error.args
            return x - 1, ()

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected, (x + 1, ("seed collision",)))
        self.assertEqual(fn(x, use_polyfill=True), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("custom_error", [False, True])
    def test_fromkeys_seed_hash_error_context(self, custom_error):
        class HashError(TypeError):
            pass

        error_type = HashError if custom_error else TypeError

        class Key:
            def __hash__(self):
                raise error_type("bad hash")

        class Seeded(builtins.frozendict):
            def __new__(cls, data=None):
                return super().__new__(cls, {"seed": 5} if data is None else data)

        def fn(x, *, use_polyfill=False):
            try:
                if use_polyfill:
                    torch._dynamo.polyfills.frozendict_fromkeys(
                        Seeded, iter([Key()]), x
                    )
                else:
                    Seeded.fromkeys(iter([Key()]), x)
            except TypeError as error:
                return x + 1, error.args, type(error) is error_type
            return x - 1, (), False

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected[0], x + 1)
        self.assertTrue(expected[2])
        if custom_error:
            self.assertEqual(expected[1], ("bad hash",))
        else:
            self.assertIn("as a frozendict key (bad hash)", expected[1][0])
        self.assertEqual(fn(x, use_polyfill=True), expected)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )

    @parametrize("base_type", [dict, OrderedDict, defaultdict])
    def test_fromkeys_dict_constructor_returns_frozendict(self, base_type):
        class Mapping(base_type):
            def __new__(cls, data=(("seed", 5),)):
                return builtins.frozendict(data)

        def fn(x):
            return Mapping.fromkeys(["b", "a", "b"], x + 1)

        x = torch.randn(3)
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), builtins.frozendict)
        self.assertEqual(list(result), ["seed", "b", "a"])
        self.assertEqual(result, expected)

    def test_fromkeys_inherited_new_init_calls(self):
        class FrozenMapping(builtins.frozendict):
            calls = []

            def __init__(self, *args):
                self.calls.append(len(args))

        def fn(x):
            return FrozenMapping.fromkeys(["b", "a", "b"], x + 1)

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(FrozenMapping.calls, [0, 1])
        FrozenMapping.calls.clear()
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(FrozenMapping.calls, [0, 1])
        self.assertIs(type(result), FrozenMapping)
        self.assertEqual(result, expected)

    def test_fromkeys_inherited_new_zero_arg_init(self):
        class FrozenMapping(builtins.frozendict):
            calls = []

            def __init__(self):
                self.calls.append(0)

        def fn(x):
            try:
                FrozenMapping.fromkeys(["a"], x)
            except TypeError:
                return x + 1
            return x - 1

        x = torch.randn(3)
        expected = fn(x)
        self.assertEqual(expected, x + 1)
        self.assertEqual(FrozenMapping.calls, [0])
        FrozenMapping.calls.clear()
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x), expected
        )
        self.assertEqual(FrozenMapping.calls, [0])

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @parametrize("construct", [False, True])
    def test_autograd_items_override(self, construct):
        class FrozenMapping(builtins.frozendict):
            def items(self):
                return [("right", self["y"]), ("left", self["x"])]

        def fn(x, y, inputs):
            if construct:
                inputs = FrozenMapping(x=x, y=y)
            return torch.autograd.grad((x * x + 3 * y).sum(), inputs)

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        inputs = FrozenMapping(x=x, y=y)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, y, inputs)
        self.assertIs(type(result), dict)
        self.assertEqual(list(result), ["right", "left"])
        self.assertEqual(result, fn(x, y, inputs))
        self.assertEqual(result["right"], torch.full_like(y, 3))
        self.assertEqual(result["left"], 2 * x)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    @parametrize("construct", [False, True])
    @parametrize("override", ["items", "values", "both"])
    def test_backward_mapping_overrides(self, construct, override):
        class FrozenMapping(builtins.frozendict):
            def items(self):
                if override in ("items", "both"):
                    return [("x", self["x"])]
                return super().items()

            def values(self):
                if override in ("values", "both"):
                    return [self["y"]]
                return super().values()

        def fn(x, y, inputs):
            if construct:
                inputs = FrozenMapping(x=x, y=y)
            (x * x + 3 * y).sum().backward(inputs=inputs)
            return x.grad, y.grad

        def run(fn):
            x = torch.randn(3, requires_grad=True)
            y = torch.randn(3, requires_grad=True)
            result = fn(x, y, FrozenMapping(x=x, y=y))
            expected_x = 2 * x if override == "items" else None
            self.assertEqual(result, (expected_x, torch.full_like(y, 3)))

        run(fn)
        run(torch.compile(fn, backend="eager", fullgraph=True))

    @parametrize_pytree_module
    def test_registered_pytree_subclass(self, pytree):
        class FrozenMapping(builtins.frozendict):
            pass

        python_pytree.register_pytree_node(
            FrozenMapping,
            lambda d: (list(d.values()), list(d.keys())),
            lambda values, keys: FrozenMapping(zip(keys, values)),
        )
        self.addCleanup(python_pytree._deregister_pytree_node, FrozenMapping)

        def fn(x):
            mapping = FrozenMapping(b=x + 1, a=x * 2)
            kwargs = {"namespace": "torch"} if pytree.__name__ == "optree" else {}
            leaves, spec = pytree.tree_flatten(mapping, **kwargs)
            leaves = [value.sin() for value in leaves]
            if pytree.__name__ == "optree":
                return pytree.tree_unflatten(spec, leaves)
            return pytree.tree_unflatten(leaves, spec)

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertIs(type(result), FrozenMapping)
        self.assertEqual(result, fn(x))
        self.assertEqual(list(result), ["b", "a"])


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
