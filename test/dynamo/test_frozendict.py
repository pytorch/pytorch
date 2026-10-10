# Owner(s): ["module: dynamo"]

import builtins
import operator
import unittest
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
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

    @parametrize("name", ["keys", "__len__"])
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

    @parametrize("operation", [dict.__eq__, dict.__ne__, dict.__or__])
    def test_dict_descriptors(self, operation):
        def fn(x):
            return x + 1, operation({"a": x}, builtins.frozendict(a=x))

        x = torch.randn(3)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "frozendict"):
            torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(torch.compile(fn, backend="eager")(x), fn(x))

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


if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
