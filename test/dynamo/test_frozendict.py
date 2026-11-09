# Owner(s): ["module: dynamo"]

import builtins
import io
import operator
import types
import unittest
from collections import defaultdict, OrderedDict
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch.utils._pytree as python_pytree
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.testing import CompileCounter
from torch._dynamo.variables.builder import SourcelessBuilder
from torch._dynamo.variables.hashable import BuiltinMethodKeyTracker
from torch.testing._internal.common_utils import (
    HardwareClassification,
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


class _ChangingHashKey:
    def __init__(self):
        self.value = 1
        self.calls = 0

    def __hash__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(not torch._has_frozendict, "requires builtins.frozendict")
@instantiate_parametrized_tests
class FrozenDictTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize("operation", ["update", "copy", "frozendict"])
    def test_merge_cross_owner_iteration_descriptor(self, operation):
        class Mapping(dict):
            __iter__ = builtins.frozendict.__iter__

            def keys(self):
                return ["b"]

            def __getitem__(self, key):
                return 2

        def fn(x, mapping):
            if operation == "update":
                result = {}
                result.update(mapping)
            elif operation == "copy":
                result = dict.copy(mapping)
            else:
                result = builtins.frozendict(mapping)
            return x + 1, result

        x, mapping = torch.ones(1), Mapping(a=1)
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping),
            fn(x, mapping),
        )

    def test_input_subclass_keys_view(self):
        class Mapping(builtins.frozendict):
            pass

        def fn(x, view):
            return x + len(view), tuple(view)

        x, view = torch.ones(1), Mapping(a=2).keys()
        self.assertEqual(torch.compile(fn, backend="eager")(x, view), fn(x, view))
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Unsupported dictionary view owner"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x, view)

    @parametrize("key_type", [float, complex, tuple, frozenset])
    def test_input_nan_key_identity_guard(self, key_type):
        def make_key():
            value = float("nan")
            return (
                key_type([value]) if key_type in (tuple, frozenset) else key_type(value)
            )

        key = make_key()

        def fn(x, mapping):
            return x + mapping.get(key, 0)

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.ones(1)
        for stored_key in (key, make_key(), key):
            mapping = builtins.frozendict({stored_key: 10})
            self.assertEqual(compiled(x, mapping), fn(x, mapping))
        self.assertEqual(counter.frame_count, 2)

    def test_input_attribute_write_exception(self):
        def fn(x, mapping):
            try:
                mapping.extra = 1
            except AttributeError:
                return x - 1
            return x + 1

        x = torch.ones(1)
        mapping = builtins.frozendict(a=1)
        self.assertEqual(torch.compile(fn, backend="eager")(x, mapping), fn(x, mapping))

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

    @parametrize(
        "operation, explicit_new, streaming",
        [
            ("construct", new, stream)
            for new in (False, True)
            for stream in (False, True)
        ]
        + [("subscribe", False, False)],
    )
    def test_key_comparisons(self, operation, explicit_new, streaming):
        class Key:
            calls = 0

            if operation == "subscribe":

                def __init__(self):
                    self.calls = 0

            def __hash__(self):
                return 42

            def __eq__(self, other):
                self.calls += 1
                return operation == "subscribe" or self.calls > 1

        def fn(x):
            first, second = Key(), Key()
            if operation == "subscribe":
                mapping = builtins.frozendict([(first, x)])
                return mapping[second] + 1, first.calls

            def pairs():
                yield first, x
                yield second, x + 1
                first.calls = 0

            data = pairs() if streaming else [(first, x), (second, x + 1)]
            if explicit_new:
                mapping = builtins.frozendict.__new__(builtins.frozendict, data)
            else:
                mapping = builtins.frozendict(data)
            return x + len(mapping), first.calls

        x = torch.randn(3)
        expected = (
            (x + 1, 1) if operation == "subscribe" else (x + 2, 0 if streaming else 1)
        )
        self.assertEqual(fn(x), expected)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize("backend", ["eager", "aot_eager", "export"])
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

        class Module(torch.nn.Module):
            def forward(self, x, mapping):
                return fn(x, mapping)

        try:
            if backend == "export":
                x = torch.ones(2)
                mapping = builtins.frozendict(
                    x=x + 1, nested=builtins.frozendict(y=x * 2)
                )
                exported = torch.export.export(Module(), (x, mapping), strict=True)
                buffer = io.BytesIO()
                torch.export.save(exported, buffer)
                buffer.seek(0)
                compiled = torch.export.load(buffer).module()
            else:
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for x in (torch.ones(2), torch.full((2,), 3.0)):
                mapping = builtins.frozendict(
                    x=x + 1, nested=builtins.frozendict(y=x * 2)
                )
                self.assertEqual(compiled(x, mapping), fn(x, mapping))
        finally:
            torch._dynamo.trace_rules._allowed_callable_ids.remove(id(consume))
        self.assertFalse(torch._dynamo.trace_rules.is_callable_disallowed(consume))

    @parametrize("instance", [False, True])
    def test_sourceless_custom_key_hash(self, instance):
        class Meta(type):
            hash_value = 1
            calls = 0

            def __hash__(cls):
                Meta.calls += 1
                return Meta.hash_value

        class Key(metaclass=Meta):
            pass

        mapping = builtins.frozendict({Key() if instance else Key: 5})
        Meta.hash_value = 2
        calls = Meta.calls
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Preexisting frozendict key"
        ):
            SourcelessBuilder.create(Mock(), mapping)
        self.assertEqual(Meta.calls, calls)
        literal = builtins.frozendict(a=5)
        tx = Mock()
        tx.output.builtin_method_keys = BuiltinMethodKeyTracker()
        with InstructionTranslator.set_current_tx(tx):
            self.assertEqual(
                SourcelessBuilder.create(tx, literal).as_python_constant(), literal
            )

    def test_constructed_metaclass_key_reconstruction(self):
        class Meta(type):
            def __hash__(cls):
                return cls.hash_value

        class Key(metaclass=Meta):
            hash_value = 1

        def fn(x):
            Key.hash_value = 1
            mapping = builtins.frozendict([(Key, x)])
            Key.hash_value = 2
            return x + 1, mapping

        x = torch.ones(1)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "stored key hashes"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        result, mapping = torch.compile(fn, backend="eager")(x)
        self.assertEqual(result, x + 1)
        self.assertEqual(list(mapping.items()), [(Key, x)])
        self.assertNotIn(Key, mapping)

    def test_class_getitem(self):
        def fn(x):
            return x + 1, builtins.frozendict.__class_getitem__((str, int))

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    @parametrize(
        "name,args",
        [
            subtest(("__new__", ()), name="new_missing"),
            subtest(("__new__", (object(),)), name="new_object"),
            subtest(("__new__", (dict,)), name="new_unrelated"),
            subtest(
                ("__new__", (getattr(builtins, "frozendict", dict), {}, {})),
                name="new_extra",
            ),
            ("keys", ({},)),
            ("__len__", ({},)),
        ],
    )
    def test_descriptor_type_errors(self, name, args):
        def fn(x):
            try:
                if name == "__new__":
                    builtins.frozendict.__new__(*args)
                else:
                    getattr(builtins.frozendict, name)(*args)
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

    @parametrize("construct", [False, True])
    def test_arbitrary_keys_and_guards(self, construct):
        counter = CompileCounter()

        def fn(mapping, key):
            if construct:
                mapping = builtins.frozendict(mapping)
            return mapping[key] + next(iter(mapping.values()))

        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for key in (None, (1, "a"), torch.randn(2), object()):
            items = [(key, torch.randn(3)), ("last", torch.randn(3))]
            mapping = dict(items) if construct else builtins.frozendict(items)
            if isinstance(key, torch.Tensor) and not construct:
                with self.assertRaisesRegex(
                    torch._dynamo.exc.Unsupported, "stored key hash"
                ):
                    compiled(mapping, key)
            else:
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
        Key = _ChangingHashKey

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

    @parametrize(
        "key_type, changed_field",
        [
            (kind, True)
            for kind in (
                python_pytree.MappingKey,
                python_pytree.SequenceKey,
                python_pytree.GetAttrKey,
            )
        ]
        + [(kind, False) for kind in (object, type, types.ModuleType, torch.Tensor)],
    )
    def test_preexisting_changed_key_hash(self, key_type, changed_field):
        if changed_field:
            key = key_type("a" if key_type is python_pytree.GetAttrKey else 0)
        else:

            class Key(key_type):
                def __hash__(self):
                    return object.__hash__(self) + 1

            key = (
                Key("Key", (), {})
                if key_type is type
                else Key(__name__)
                if key_type is types.ModuleType
                else Key()
            )
        mapping = builtins.frozendict([(key, 7)])
        if changed_field:
            changed = "b" if key_type is python_pytree.GetAttrKey else 1
            object.__setattr__(key, next(iter(key_type.__dataclass_fields__)), changed)
        elif key_type is type or key_type is object:
            Key.__hash__ = object.__hash__
        else:
            key.__class__ = key_type

        def fn(x, mapping):
            return x + (mapping.get(key, 0) if changed_field else key in mapping)

        x = torch.ones(1)
        self.assertEqual(torch.compile(fn, backend="eager")(x, mapping), fn(x, mapping))
        torch._dynamo.reset()
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "stored key hash"):
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping)

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
        Key = _ChangingHashKey

        def fn(x):
            key = Key()
            mapping = builtins.frozendict([(key, x)])
            key.value = 2
            copied = dict(mapping, extra=x + 1) if with_kwargs else dict(mapping)
            fromkeys = dict.fromkeys(mapping, x)
            unpacked = {**mapping}
            rebuilt = builtins.frozendict(mapping, extra=x + 1)
            return (
                next(iter(copied.values())),
                next(iter(unpacked.values())),
                next(iter(rebuilt.values())),
                rebuilt["extra"],
                key.calls,
                next(iter(fromkeys.values())),
                key in fromkeys,
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
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

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

    @parametrize("mapping_type", [dict, OrderedDict, defaultdict])
    @parametrize("name", ["__eq__", "__ne__", "__or__", "__ror__"])
    @parametrize("subclass", [False, True])
    def test_dict_descriptors(self, mapping_type, name, subclass):
        class Mapping(builtins.frozendict):
            pass

        def fn(x, mapping):
            return x + 1, getattr(mapping_type, name)(mapping_type(a=x), mapping)

        x = torch.randn(3)
        mapping = (Mapping if subclass else builtins.frozendict)(a=x)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "frozendict"):
            torch.compile(fn, backend="eager", fullgraph=True)(x, mapping)
        self.assertEqual(torch.compile(fn, backend="eager")(x, mapping), fn(x, mapping))

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
    @parametrize("construct, backward", [(False, False), (True, False), (True, True)])
    def test_autograd(self, construct, backward):
        def fn(x, y, inputs):
            if construct:
                inputs = builtins.frozendict(y=y, x=x)
            output = x * x if backward else x * x + 3 * y
            grad = torch.autograd.grad(output.sum(), inputs, allow_unused=backward)
            if backward:
                (x * x + 3 * y).sum().backward(inputs=inputs)
            return grad

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3, requires_grad=True)
        inputs = builtins.frozendict(y=y, x=x)
        if not backward:
            expected = fn(x, y, inputs)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x, y, inputs)
        self.assertIs(type(result), dict)
        self.assertEqual(list(result), ["y", "x"])
        self.assertEqual(result["x"], 2 * x)
        if backward:
            self.assertIsNone(result["y"])
            self.assertEqual(x.grad, 2 * x)
            self.assertEqual(y.grad, torch.full_like(y, 3))
        else:
            self.assertEqual(result, expected)
            self.assertEqual(result["y"], torch.full_like(y, 3))

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

    def test_missing_key_exception(self):
        def fn(x):
            mapping = builtins.frozendict(a=x)
            try:
                return mapping["missing"]
            except KeyError as error:
                return x + 1, error.args

        x = torch.randn(3)
        self.assertEqual(torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x))

    def test_constructed_hash_requires_operators(self):
        def fn(x):
            return x + 1, builtins.frozendict(a=1) in {builtins.frozendict(a=1)}

        x = torch.ones(1)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "frozendict operators"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @parametrize("base_type", [dict, OrderedDict, defaultdict])
    def test_fromkeys_dict_constructor_returns_frozendict(self, base_type):
        class Mapping(base_type):
            def __new__(cls, data=()):
                return builtins.frozendict(data)

        def fn(x):
            return Mapping.fromkeys("ab", x)

        x = torch.ones(1)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "_frozendict_fromkeys"
        ):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        result = torch.compile(fn, backend="eager")(x)
        self.assertIs(type(result), builtins.frozendict)
        self.assertEqual(result, fn(x))


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
