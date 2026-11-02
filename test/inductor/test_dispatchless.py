# Owner(s): ["module: inductor", "module: custom-operators"]

from __future__ import annotations

import torch
from torch._dynamo.testing import InductorAndRecordGraphs
from torch._inductor import metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._library.dispatchless import dispatchless_custom_op, flat_dispatchless_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, parametrize


def apply_transform(fn, transform):
    if transform == "vmap":
        return torch.vmap(fn)
    if transform == "jvp":
        return lambda x: torch.func.jvp(fn, (x,), (torch.ones_like(x),))
    if transform == "gradgrad":
        def scalar(x):
            return fn(x).sum()

        def grad_sum(x):
            return torch.func.grad(scalar)(x).sum()

        return torch.func.grad(grad_sum)
    raise AssertionError(f"Unknown transform: {transform}")


class DispatchlessTest(TestCase):

    hw_classification = HardwareClassification.GENERIC

    def test_make_fx_autograd(self, device):
        @dispatchless_custom_op
        def op(x):
            return x.sin() * x

        def fn(x):
            (grad,) = torch.autograd.grad(op(x).sum(), x, create_graph=True)
            (gradgrad,) = torch.autograd.grad(grad.sum(), x)
            return grad, gradgrad

        x = torch.randn(4, device=device, requires_grad=True)
        gm = make_fx(fn)(x)
        value = torch.randn_like(x, requires_grad=True)
        expected_grad = value.sin() + value * value.cos()
        expected_gradgrad = 2 * value.cos() - value * value.sin()
        self.assertEqual(gm(value), (expected_grad, expected_gradgrad))

    @parametrize("execution", ["eager", "make_fx", "aot_eager", "inductor"])
    @parametrize("transform", ["vmap", "jvp", "gradgrad"])
    def test_transforms(self, device, execution, transform):
        def body(x):
            return x.sin() * x

        op = dispatchless_custom_op(body)

        fn = apply_transform(op, transform)
        plain = apply_transform(body, transform)
        x = torch.randn(3, 4, device=device)
        if execution == "make_fx":
            actual = make_fx(fn)(x)
        elif execution == "eager":
            actual = fn
        else:
            actual = torch.compile(fn, backend=execution, fullgraph=True)
        for value in (x, x + 1):
            self.assertEqual(actual(value), plain(value))

    @parametrize("execution", ["eager", "make_fx", "compile_eager"])
    def test_functionalize(self, device, execution):
        @dispatchless_custom_op
        def op(x):
            return x.sin() * x

        def fn(x):
            return op(x)

        if execution == "compile_eager":
            fn = torch.compile(fn, backend="eager", fullgraph=True)
        actual = torch.func.functionalize(fn)
        x = torch.randn(3, 4, device=device)
        if execution == "make_fx":
            actual = make_fx(actual)(x)
        for value in (x, x + 1):
            self.assertEqual(actual(value), value.sin() * value)

    @parametrize("backend", ["aot_eager", "inductor"])
    def test_compiled_functionalize_limitation(self, device, backend):
        def body(x):
            return x.sin() * x

        x = torch.randn(3, 4, device=device)
        for fn in (body, dispatchless_custom_op(body)):
            torch._dynamo.reset()
            compiled = torch.compile(
                torch.func.functionalize(fn), backend=backend, fullgraph=True
            )
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported, "_func_increment_nesting"
            ):
                compiled(x)

    @parametrize("execution", ["eager", "make_fx", "aot_eager", "inductor"])
    def test_autocast(self, device, execution):
        @dispatchless_custom_op
        def op(x, y):
            return x @ y

        device_type = torch.device(device).type

        def fn(x, y):
            with torch.autocast(device_type, dtype=torch.bfloat16):
                return op(x, y).sin()

        x, y = (torch.randn(8, 8, device=device) for _ in range(2))
        if execution == "make_fx":
            actual = make_fx(fn)(x, y)
        elif execution == "eager":
            actual = fn
        else:
            actual = torch.compile(fn, backend=execution, fullgraph=True)
        with torch.autocast(device_type, dtype=torch.bfloat16):
            expected = (x @ y).sin()
        result = actual(x, y)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result, expected)

    def test_aot_eager_gradients(self, device):
        @dispatchless_custom_op
        def op(x):
            return {"value": x.sin() * x}

        def fn(x):
            return op(x)["value"].sum()

        x = torch.randn(4, device=device, requires_grad=True)
        result = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))
        self.assertEqual(torch.autograd.grad(result, x), torch.autograd.grad(fn(x), x))


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
