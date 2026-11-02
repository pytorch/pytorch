# Owner(s): ["module: dynamo", "module: custom-operators"]

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier

import torch
from torch._dynamo.testing import EagerAndRecordGraphs
from torch._library.dispatchless import (
    _flat_call,
    dispatchless_custom_op,
    flat_dispatchless_call,
)
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx
from torch.testing._internal.common_utils import (
    HardwareClassification,
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

    hw_classification = HardwareClassification.GENERIC

    def trace_for_tensor_checks(self, fn, x, execution):
        if execution == "dynamo":
            return torch.compile(fn, backend="eager", fullgraph=True)(x)
        if execution == "pre_dispatch":
            return make_fx(fn, pre_dispatch=True)(x)
        return make_fx(fn, tracing_mode=execution)(x)

    @parametrize("execution", ["real", "fake", "symbolic", "pre_dispatch", "dynamo"])
    @parametrize("mutation", ["inplace", "view", "out", "foreach", "metadata"])
    def test_rejects_input_mutation(self, execution, mutation):
        @dispatchless_custom_op
        def op(payload):
            x = payload["x"][0]
            if mutation == "inplace":
                x.add_(1)
            elif mutation == "view":
                x.view(-1).add_(1)
            elif mutation == "out":
                torch.add(x, 1, out=x)
            elif mutation == "foreach":
                torch._foreach_add_([x], 1)
            else:
                x.transpose_(0, 1)
            return x.sin()

        def fn(x):
            return op({"x": [x]})

        x = torch.arange(6.0).reshape(2, 3)
        original = x.clone()
        original_stride = x.stride()
        with self.assertRaisesRegex(RuntimeError, "must not mutate input tensors"):
            self.trace_for_tensor_checks(fn, x, execution)
        self.assertEqual(x, original)
        self.assertEqual(x.stride(), original_stride)

    @parametrize("execution", ["real", "fake", "symbolic", "pre_dispatch", "dynamo"])
    @parametrize("alias", ["input", "input_view", "duplicate_output", "output_view"])
    def test_rejects_output_aliases(self, execution, alias):
        @dispatchless_custom_op
        def op(x):
            if alias == "input":
                return {"value": [x]}
            if alias == "input_view":
                return {"value": [x.view(-1)]}
            y = x.sin()
            return {"value": [y, y if alias == "duplicate_output" else y.view(-1)]}

        x = torch.randn(2, 3)
        with self.assertRaisesRegex(
            RuntimeError, "must not alias inputs or other outputs"
        ):
            self.trace_for_tensor_checks(op, x, execution)

    @parametrize("execution", ["real", "fake", "symbolic", "pre_dispatch", "dynamo"])
    @parametrize("size", [0, 4])
    def test_allows_local_mutation_and_repeated_inputs(self, execution, size):
        @dispatchless_custom_op
        def inner(x, y):
            result = x.clone()
            result.add_(y)
            return result.view(-1)

        @dispatchless_custom_op
        def outer(payload):
            x, y = payload
            return {"result": inner(x, y), "other": x.clone(), "constant": None}

        def fn(x):
            return outer([x, x])

        x = torch.arange(float(size))
        result = self.trace_for_tensor_checks(fn, x, execution)
        if execution != "dynamo":
            result = result(x)
        self.assertEqual(result, {"result": x * 2, "other": x, "constant": None})
        self.assertEqual(x, torch.arange(float(size)))

    def test_eager_does_not_check_tensor_purity(self):
        @dispatchless_custom_op
        def op(x):
            return x.add_(1)

        x = torch.zeros(3)
        self.assertIs(op(x), x)
        self.assertEqual(x, torch.ones(3))

    @parametrize("execution", ["real", "fake", "pre_dispatch", "dynamo"])
    def test_inference_tensor_mutation_diagnostic(self, execution):
        @dispatchless_custom_op
        def op(x):
            x.add_(1)
            return x.sin()

        with torch.inference_mode():
            x = torch.zeros(3)
            with self.assertRaisesRegex(RuntimeError, "must not mutate input tensors"):
                self.trace_for_tensor_checks(op, x, execution)
            self.assertEqual(x, torch.zeros(3))

    @parametrize("execution", ["real", "fake", "pre_dispatch", "dynamo"])
    def test_tensor_checks_allow_nested_higher_order_operator(self, execution):
        from torch._higher_order_ops.cond import cond_op

        @dispatchless_custom_op
        def op(x):
            return cond_op(x.sum() > 0, torch.sin, torch.cos, (x,))

        x = torch.ones(3)
        result = self.trace_for_tensor_checks(op, x, execution)
        if execution != "dynamo":
            self.assertEqual(result(-x), (-x).cos())
            result = result(x)
        self.assertEqual(result, x.sin())

    @parametrize("tracing_mode", ["real", "fake"])
    def test_tensor_checks_allow_sparse_inputs(self, tracing_mode):
        @dispatchless_custom_op
        def op(x):
            return x.clone()

        x = torch.sparse_coo_tensor([[0, 1], [1, 0]], [2.0, 3.0], (2, 2))
        gm = make_fx(op, tracing_mode=tracing_mode)(x)
        self.assertEqual(gm(x), x)

    @parametrize("execution", ["real", "fake", "symbolic", "pre_dispatch", "dynamo"])
    def test_factory_output_alias_diagnostic(self, execution):
        @dispatchless_custom_op
        def op():
            x = torch.zeros(3)
            return {"first": x, "second": [x]}

        def fn(unused):
            return op()

        with self.assertRaisesRegex(
            RuntimeError, "must not alias inputs or other outputs"
        ):
            self.trace_for_tensor_checks(fn, torch.zeros(1), execution)

    @parametrize("pre_dispatch", [False, True])
    def test_nested_and_disabled_proxy_modes(self, pre_dispatch):
        @dispatchless_custom_op
        def op(x):
            return x.sin() + 1

        nested_graphs = []

        def fn(x):
            first = op(x)
            with disable_proxy_modes_tracing():
                op(x)
                nested_graphs.append(make_fx(op, pre_dispatch=pre_dispatch)(x))
            return op(first)

        x = torch.randn(4)
        gm = make_fx(fn, pre_dispatch=pre_dispatch)(x)
        self.assertEqual(gm(x + 1), op(op(x + 1)))
        self.assertEqual(len(nested_graphs), 1)
        self.assertEqual(nested_graphs[0](x + 1), op(x + 1))
        for graph, count in ((gm, 2), (nested_graphs[0], 1)):
            targets = [node.target for node in graph.graph.nodes]
            self.assertEqual(targets.count(flat_dispatchless_call), count)

    @parametrize("pre_dispatch", [False, True])
    def test_eager_and_tracing_mode_transitions(self, pre_dispatch):
        @dispatchless_custom_op
        def op(x):
            return x.sin() + 1

        x = torch.randn(4)
        expected = x.sin() + 1
        for _ in range(2):
            self.assertEqual(op(x), expected)
            gm = make_fx(op, pre_dispatch=pre_dispatch)(x)
            self.assertEqual(gm(x), expected)
            targets = [node.target for node in gm.graph.nodes]
            self.assertEqual(targets.count(flat_dispatchless_call), 1)
        self.assertEqual(op(x), expected)

    @parametrize("pre_dispatch", [False, True])
    def test_proxy_mode_is_thread_local(self, pre_dispatch):
        @dispatchless_custom_op
        def op(x):
            return x.sin() + 1

        barrier = Barrier(2, timeout=30)
        x = torch.randn(4)

        def synchronized_call(x):
            barrier.wait()
            result = op(x)
            barrier.wait()
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            traced = pool.submit(
                make_fx(synchronized_call, pre_dispatch=pre_dispatch), x
            )
            eager = pool.submit(synchronized_call, x)
            gm, result = traced.result(), eager.result()

        expected = x.sin() + 1
        self.assertEqual(result, expected)
        self.assertEqual(gm(x), expected)
        targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(targets.count(flat_dispatchless_call), 1)

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

    def test_registered_frozen_constant_without_dispatchless(self):
        @dataclass(frozen=True)
        class Config:
            scale: int

        self.register_constant(Config)

        def fn(x, config):
            leaves, spec = torch.utils._pytree.tree_flatten((x, config))
            value, restored_config = torch.utils._pytree.tree_unflatten(leaves, spec)
            return value * restored_config.scale

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
