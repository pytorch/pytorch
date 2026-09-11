# Owner(s): ["module: dynamo", "module: custom-operators"]

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._library.dispatchless import (
    _flat_call,
    dispatchless_custom_op,
    flat_dispatchless_call,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.dynamo_pytree_test_utils import PytreeRegisteringTestCase
from torch.utils._python_dispatch import TorchDispatchMode


@dataclass
class Payload:
    value: torch.Tensor
    scale: int


class DispatchlessTest(PytreeRegisteringTestCase):
    def test_eager_direct_call(self):
        class Unregistered:
            def __init__(self, value):
                self.value = value

        x = torch.randn(4, requires_grad=True)
        payload = Unregistered(x)
        calls = []

        class RecordMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                calls.append(func)
                return func(*args, **(kwargs or {}))

        @dispatchless_custom_op
        def op(arg):
            self.assertIs(arg, payload)
            return arg.value.sin()

        with RecordMode():
            result = op(payload)
        self.assertEqual(calls, [torch.ops.aten.sin.default])
        self.assertEqual(result, x.sin())
        self.assertEqual(torch.autograd.grad(result.sum(), x)[0], x.cos())

    @parametrize("tracing_mode", ["real", "fake", "symbolic"])
    @parametrize("pre_dispatch", [False, True])
    def test_make_fx_preserves_call(self, tracing_mode, pre_dispatch):
        @dispatchless_custom_op
        def op(payload, *, scale=2):
            return {"result": payload["x"][0].sin() * scale, "metadata": (None, scale)}

        def fn(x):
            return op({"x": [x]}, scale=3)

        x = torch.randn(4)
        gm = make_fx(fn, tracing_mode=tracing_mode, pre_dispatch=pre_dispatch)(x)
        gm.graph.lint()
        targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
        self.assertEqual(targets.count(flat_dispatchless_call), 1)
        self.assertNotIn(torch.ops.aten.sin.default, targets)
        self.assertEqual(gm(x + 1), fn(x + 1))

    def test_make_fx_decomposition(self):
        @dispatchless_custom_op
        def op(x):
            return {"value": x.sin() + 1}

        x = torch.randn(4)
        gm = make_fx(op, decomposition_table={flat_dispatchless_call: _flat_call})(x)
        targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
        self.assertNotIn(flat_dispatchless_call, targets)
        self.assertIn(torch.ops.aten.sin.default, targets)
        self.assertEqual(gm(x), op(x))

    def test_dynamo_pytree_inputs_outputs(self):
        self.register_pytree_node(
            Payload,
            lambda p: ((p.value, p.scale), None),
            lambda values, _: Payload(*values),
        )

        @dispatchless_custom_op
        def op(payload, *, bias):
            result = Payload(payload.value.sin() + bias, payload.scale)
            return {"output": result, "none": None}

        def fn(x, bias):
            result = op(Payload(x, 2), bias=bias)
            return result["output"].value * result["output"].scale, result["none"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        for _ in range(2):
            x, bias = torch.randn(4), torch.randn(4)
            self.assertEqual(compiled(x, bias), fn(x, bias))
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        gm.graph.lint()
        targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
        self.assertEqual(targets.count(flat_dispatchless_call), 1)
        self.assertNotIn(torch.sin, targets)

    def test_dynamo_dynamic_shapes(self):
        @dispatchless_custom_op
        def op(x):
            return {"result": x.sin(), "size": x.shape[0]}

        def fn(x):
            result = op(x)
            return result["result"] + result["size"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        for size in (4, 7):
            x = torch.randn(size)
            self.assertEqual(compiled(x), fn(x))
        self.assertEqual(len(backend.graphs), 1)

    def test_dynamo_operator_identity(self):
        def make_op(offset):
            @dispatchless_custom_op
            def op(x):
                return x + offset

            return op

        def fn(op, x):
            return op(x)

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(4)
        first, second = make_op(1), make_op(2)
        self.assertEqual(compiled(first, x), x + 1)
        self.assertEqual(compiled(second, x), x + 2)
        self.assertEqual(compiled(first, x), x + 1)
        self.assertEqual(len(backend.graphs), 2)

    def test_output_specs_are_per_call(self):
        @dispatchless_custom_op
        def op(x, *, as_dict):
            result = x.sin()
            return {"value": result} if as_dict else [result]

        def fn(x):
            return op(x, as_dict=True)["value"] + op(x, as_dict=False)[0]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), fn(x))
        targets = [node.target for node in backend.graphs[0].graph.nodes]
        self.assertEqual(targets.count(flat_dispatchless_call), 2)

    def test_nested_operators_and_repeated_inputs(self):
        @dispatchless_custom_op
        def inner(x, y):
            return x * y

        @dispatchless_custom_op
        def outer(values):
            return inner(values[0], values[1]).sin()

        def fn(x):
            return outer([x, x])

        x = torch.randn(4)
        gm = make_fx(fn)(x)
        self.assertEqual(gm(x), fn(x))
        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(flat_dispatchless_call), 1)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn(x))

    def test_registered_constant(self):
        @dataclass(frozen=True)
        class Config:
            scale: int

        self.register_constant(Config)

        @dispatchless_custom_op
        def op(x, config):
            return x * config.scale

        def fn(x, config):
            return op(x, config)

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(4)
        for scale in (2, 3, 2):
            self.assertEqual(compiled(x, Config(scale)), x * scale)
        self.assertEqual(len(backend.graphs), 2)

    @parametrize("output", [False, True])
    def test_unsupported_pytree_leaf(self, output):
        class Unregistered:
            pass

        @dispatchless_custom_op
        def op(x):
            return Unregistered() if output else x

        x = torch.randn(4)
        with self.assertRaisesRegex(RuntimeError, "non-fx-graphable type"):
            make_fx(op)(x if output else Unregistered())

    def test_output_spec_mismatch(self):
        @dispatchless_custom_op
        def op(x):
            return [x.sin()] if x.shape[0] == 4 else {"value": x.sin()}

        gm = make_fx(op)(torch.randn(4))
        with self.assertRaisesRegex(RuntimeError, "same output pytree structure"):
            gm(torch.randn(7))


instantiate_parametrized_tests(DispatchlessTest)


if __name__ == "__main__":
    run_tests()
