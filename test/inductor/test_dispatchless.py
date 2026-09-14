# Owner(s): ["module: inductor", "module: custom-operators"]

from __future__ import annotations

import torch
from torch._dynamo.testing import InductorAndRecordGraphs
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._library.dispatchless import dispatchless_custom_op, flat_dispatchless_call
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize


class DispatchlessTest(TestCase):
    @parametrize("requires_grad", [False, True])
    def test_values_gradients_and_decomposition(self, device, requires_grad):
        @dispatchless_custom_op
        def op(payload, *, scale):
            result = payload[0].sin() * scale
            return {"result": result, "other": (payload[1].cos(), None)}

        def fn(x, y):
            result = op([x, y], scale=2)
            return result["result"] + result["other"][0]

        backend = InductorAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        x = torch.randn(16, device=device, requires_grad=requires_grad)
        y = torch.randn(16, device=device, requires_grad=requires_grad)
        expected = fn(x, y)
        actual = compiled(x, y)
        self.assertEqual(actual, expected)
        if requires_grad:
            self.assertEqual(
                torch.autograd.grad(actual.sum(), (x, y)),
                torch.autograd.grad(expected.sum(), (x, y)),
            )
        self.assertTrue(backend.inductor_graphs)
        self.assertIn(
            flat_dispatchless_call,
            [node.target for node in backend.graphs[0].graph.nodes],
        )
        for gm in backend.inductor_graphs:
            self.assertNotIn(
                flat_dispatchless_call, [node.target for node in gm.graph.nodes]
            )

    def test_fuses_across_operator(self, device):
        @dispatchless_custom_op
        def op(x):
            return x.sin() * 2

        def fn(x):
            return op(x + 1).cos() + 3

        x = torch.randn(256, device=device)
        metrics.reset()
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual, fn(x))
        self.assertEqual(metrics.generated_kernel_count, 1)
        self.assertTrue(code)
        self.assertNotIn("flat_dispatchless_call(", "\n".join(code))

    def test_dynamic_shapes_and_nested_calls(self, device):
        @dispatchless_custom_op
        def inner(x, y):
            return x * y

        @dispatchless_custom_op
        def outer(payload):
            return {"value": inner(payload[0], payload[1]).sin()}

        def fn(x):
            return outer([x, x])["value"] + x.shape[0]

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        for size in (4, 7):
            x = torch.randn(size, device=device, requires_grad=True)
            actual, expected = compiled(x), fn(x)
            self.assertEqual(actual, expected)
            self.assertEqual(
                torch.autograd.grad(actual.sum(), x),
                torch.autograd.grad(expected.sum(), x),
            )


instantiate_device_type_tests(DispatchlessTest, globals(), only_for=("cpu", "cuda"))


if __name__ == "__main__":
    run_tests()
