# Owner(s): ["module: dynamo"]

import builtins
import operator

import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    run_tests,
)


pytree_backends = [pytree]
if not IS_FBCODE:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_backends.append(cxx_pytree)


if torch._has_frozendict:

    @instantiate_parametrized_tests
    class FrozenDictTests(torch._dynamo.test_case.TestCase):
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
                pytree.MappingKey,
                pytree.SequenceKey,
                pytree.GetAttrKey,
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

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "stored key hash"
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(1), mapping
                )
            self.assertEqual(key.calls, calls)
            self.assertEqual(
                torch.compile(fn, backend="eager")(torch.ones(1), mapping), 2
            )
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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

        def test_conversion_preserves_stored_key_hashes(self):
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
                copied = dict(mapping)
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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

        @parametrize("name", ["copy", "keys", "__len__"])
        def test_descriptor_receiver_errors(self, name):
            def fn(x):
                try:
                    getattr(builtins.frozendict, name)({})
                except TypeError:
                    return x + 1
                return x - 1

            x = torch.randn(3)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

        def test_guard_invalidation(self):
            counter = CompileCounter()

            def fn(mapping):
                return next(iter(mapping.values())) + len(mapping)

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

        @parametrize("operation", [dict.__eq__, dict.__ne__, dict.__or__])
        def test_dict_descriptors(self, operation):
            def fn(x):
                return x + 1, operation({"a": x}, builtins.frozendict(a=x))

            x = torch.randn(3)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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

        @parametrize("backend", pytree_backends)
        def test_pytree_roundtrip(self, backend):
            def fn(x):
                tree = builtins.frozendict(b=[x + 1], a=x * 2)
                leaves, spec = backend.tree_flatten(tree)
                return backend.tree_unflatten([v.sin() for v in leaves], spec)

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

        @parametrize(
            "operation", [operator.getitem, operator.contains, lambda d, k: d.get(k)]
        )
        @parametrize("custom_hash", [False, True])
        def test_lookup_type_errors(self, operation, custom_hash):
            class Key:
                def __hash__(self):
                    raise TypeError("custom hash failed")

            def fn(x):
                mapping = builtins.frozendict(a=x)
                key = Key() if custom_hash else []
                try:
                    operation(mapping, key)
                except TypeError as error:
                    return x + 1, str(error)
                return x - 1, "missing error"

            x = torch.randn(3)
            expected = fn(x)
            self.assertIn("cannot use", expected[1])
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), expected
            )

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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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
                left = (
                    {key: 1} if dict_side == "left" else builtins.frozendict([(key, 1)])
                )
                right = (
                    {key: 1}
                    if dict_side == "right"
                    else builtins.frozendict([(key, 1)])
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
                    {first: 1}
                    if dict_side == "left"
                    else builtins.frozendict([(first, 1)])
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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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
                torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(1), mapping
                )
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
            self.assertEqual(result, x + 1)
            self.assertEqual(hash(mapping), saved_hash)

        @parametrize(
            "operation", [lambda d: d | [], lambda d: d < {}, lambda d: hash(d)]
        )
        def test_operator_errors(self, operation):
            def fn(x):
                mapping = builtins.frozendict(a=[])
                try:
                    operation(mapping)
                except TypeError:
                    return x + 1
                return x - 1

            x = torch.randn(3)
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

    @instantiate_parametrized_tests
    class FrozenDictSubclassTests(torch._dynamo.test_case.TestCase):
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
            result, copied, proxy = torch.compile(fn, backend="eager", fullgraph=True)(
                x
            )
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
                )

            x = torch.randn(3)
            source = FrozenMapping(a=x)
            result = torch.compile(fn, backend="eager", fullgraph=True)(x, source)
            self.assertEqual(result, fn(x, source))
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
            self.assertEqual(
                torch.compile(fn, backend="eager", fullgraph=True)(x), fn(x)
            )

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

        @parametrize("backend", pytree_backends)
        def test_registered_pytree_subclass(self, backend):
            class FrozenMapping(builtins.frozendict):
                pass

            pytree.register_pytree_node(
                FrozenMapping,
                lambda d: (list(d.values()), list(d.keys())),
                lambda values, keys: FrozenMapping(zip(keys, values)),
            )
            self.addCleanup(pytree._deregister_pytree_node, FrozenMapping)

            def fn(x):
                mapping = FrozenMapping(b=x + 1, a=x * 2)
                leaves, spec = backend.tree_flatten(mapping)
                return backend.tree_unflatten([value.sin() for value in leaves], spec)

            x = torch.randn(3)
            result = torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertIs(type(result), FrozenMapping)
            self.assertEqual(result, fn(x))
            self.assertEqual(list(result), ["b", "a"])


if __name__ == "__main__":
    run_tests()
