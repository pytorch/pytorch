# Owner(s): ["module: dynamo"]
from functools import wraps
from typing import Any

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import allow_in_graph


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def fn(a, b):
    return a + b * 0.67


@torch.jit.script
class _ScriptBox:
    def __init__(self, value: int):
        self.value = value


class InteropTests(torch._dynamo.test_case.TestCase):
    def _common(self, fn):
        inputs = [
            torch.randn(10, device=device_type),
            torch.randn(10, device=device_type),
        ]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertEqual(ref, res)

    def test_fx_fn(self):
        fx_fn = torch.fx.symbolic_trace(fn)
        self._common(lambda a, b: fx_fn(a, b) + 1)

    def test_script_fn(self):
        script_fn = torch.jit.script(fn)
        self._common(lambda a, b: script_fn(a, b) + 1)

    def test_script_fn_mutable_container_boundary(self):
        @torch.jit.script
        def script_fn(values: list[torch.Tensor]) -> list[torch.Tensor]:
            values.append(values[0] + 1)
            return values

        def fn(values):
            result = script_fn(values)
            result.append(values[0] + 2)
            return result

        value = torch.ones(1)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "mutable container"):
            torch.compile(fn, backend="eager", fullgraph=True)([value])

        value = torch.ones(1)
        values = [value]
        result = torch.compile(fn, backend="eager")(values)

        self.assertEqual(values, [value])
        self.assertEqual(result, [value, value + 1, value + 2])
        self.assertIsNot(values, result)

    def test_script_fn_any_schema(self):
        @torch.jit.script
        def script_fn(value: Any) -> Any:
            return value

        def fn(value):
            return script_fn([value])

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "mutable container"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(1))

    def test_script_fn_class_boundary(self):
        @torch.jit.script
        def script_fn(box: _ScriptBox) -> _ScriptBox:
            return box

        def fn(box):
            result = script_fn(box)
            result.value += 1
            return box.value, result.value

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "class instances"):
            torch.compile(fn, backend="eager", fullgraph=True)(_ScriptBox(3))

        box = _ScriptBox(3)
        self.assertEqual(torch.compile(fn, backend="eager")(box), (3, 4))
        self.assertEqual(box.value, 3)

    def test_script_fn_future_payload(self):
        @torch.jit.script
        def script_fn(
            future: torch.jit.Future[list[torch.Tensor]],
        ) -> torch.jit.Future[list[torch.Tensor]]:
            return future

        def fn(future):
            return script_fn(future)

        future = torch.jit.fork(lambda value: [value + 1], torch.ones(1))
        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(future).wait(),
            [torch.full((1,), 2.0)],
        )

    def test_script_fn_recursive_mutable_schema(self):
        @torch.jit.script
        def nested_return_fn(
            value: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return value, {"result": value + 1}

        def call_nested_return(value):
            return nested_return_fn(value)[0]

        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, "mutable container"):
            torch.compile(call_nested_return, backend="eager", fullgraph=True)(
                torch.ones(1)
            )

    def test_trace_fn(self):
        trace_fn = torch.jit.trace(
            fn,
            [torch.zeros(10, device=device_type), torch.zeros(10, device=device_type)],
        )
        self._common(lambda a, b: trace_fn(a, b) + 1)

    def test_staticmethod_script_fn(self):
        class Foo:
            @staticmethod
            @torch.jit.script
            def _g(a):
                return a**2

            def g(self, a, b):
                return self._g(a) + b

        foo = Foo()
        self._common(lambda a, b: foo.g(a, b) + 1)

    def test_vmap_in_graph(self):
        def traceable(f):
            f = allow_in_graph(f)

            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(3, 5, 3, device=device_type)

        def fn(x):
            return torch.vmap(torch.Tensor.t)(x)

        fn_opt = torch.compile(fn, backend=cnts, fullgraph=True)
        fn_opt_traceable = torch.compile(traceable(fn), backend=cnts, fullgraph=True)

        self.assertEqual(fn(x), fn_opt(x))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(fn_opt(x), fn_opt_traceable(x))
        self.assertEqual(cnts.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
