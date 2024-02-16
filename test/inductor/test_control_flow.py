# Owner(s): ["module: inductor"]
import itertools

import torch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
from torch.testing._internal.triton_utils import requires_cuda


class CondTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_predicates=1,
    ):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        inputs = [inp.to(device=device) for inp in inputs]
        input_sets = [inputs]
        if dynamic:
            larger_inputs = []
            for inp in inputs:
                # tile every first dim 5x
                tiling = [5] + [1] * (inp.ndim - 1)
                larger_inputs.append(torch.tile(inp, tiling))
            input_sets.append(larger_inputs)
            for inputs in input_sets:
                for inp in inputs:
                    # mark every first dim as dynamic
                    torch._dynamo.mark_dynamic(inp, 0)

        for inputs in input_sets:
            # iterate over the cartesian product of predicate values
            for p_values in itertools.product(*([[False, True]] * num_predicates)):
                predicates = [torch.tensor(v, device=device) for v in p_values]
                result = model(*predicates, *inputs)
                result_compiled = compiled_model(*predicates, *inputs)
                self.assertEqual(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("dynamic", [False, True])
    def test_simple_control_flow(self, device, dynamic):
        # cond control flow without nesting
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    return x + y

                def false_fn(x, y):
                    return x - y

                return torch.cond(p, true_fn, false_fn, [a, b])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("dynamic", [False, True])
    def test_nested_control_flow(self, device, dynamic):
        # cond control flow with nesting
        class Model(torch.nn.Module):
            def forward(self, p0, p1, p2, a, b, c):
                def true_fn(x0, y0, z0):
                    def true_true_fn(x1, y1, z1):
                        return (x1 - y1 * z1) * 3.14

                    def true_false_fn(x1, y1, z1):
                        def true_false_true_fn(x2, y2, z2):
                            return (x2 * y2 * z2) / 2.71

                        def true_false_false_fn(x2, y2, z2):
                            return (x2 + y2 + z2) * 1.23

                        return torch.cond(
                            p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                        )

                    return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

                def false_fn(x0, y0, z0):
                    def false_true_fn(x1, y1, z1):
                        def false_true_true_fn(x2, y2, z2):
                            return (x2 - y2 - z2) + 1.23

                        def false_true_false_fn(x2, y2, z2):
                            return (x2 / y2 / z2) - 3.14

                        return torch.cond(
                            p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                        )

                    def false_false_fn(x1, y1, z1):
                        return (x1 - y1 * z1) / 2.71

                    return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

                return torch.cond(p0, true_fn, false_fn, [a, b, c])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_predicates=3,
        )

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("dynamic", [False, True])
    def test_outer_code_before_after(self, device, dynamic):
        # some code before and after the conditional
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                c = a * b + 3.14
                d = a / b - 2.71

                def true_fn(x, y):
                    return x + y

                def false_fn(x, y):
                    return x - y

                e = torch.cond(p, true_fn, false_fn, [c, d])

                return e * e / 1.41

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    @parametrize("dynamic", [False, True])
    def test_multiple_outputs(self, device, dynamic):
        # multiple outputs with different shapes
        class Model(torch.nn.Module):
            def forward(self, p, a, b, c):
                def true_fn(x, y, z):
                    return x * y, z / 2.71, (y - x).sum(dim=1)

                def false_fn(x, y, z):
                    return y / x, z * 3.14, (x + y).mean(dim=1)

                return torch.cond(p, true_fn, false_fn, [a, b, c])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(30, 40),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_advanced_dynamic_shapes(self, device):
        # subgraphs input shapes include symbolic expressions
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    return torch.cat([x - 3, y * 3], dim=1)

                def false_fn(x, y):
                    return torch.cat([x / 3, y - 3], dim=1)

                c = torch.cat([a, b], dim=0)
                d = c * 2
                e = c / 2

                return torch.cond(p, true_fn, false_fn, [d, e])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(2, 3, 3),
                torch.randn(4, 3, 3),
            ),
            device=device,
            dynamic=True,
        )

    @requires_cuda
    @parametrize("device", ["cpu", "cuda"])
    def test_use_buffers_from_outer_scope(self, device):
        # subgraphs input shapes include symbolic expressions
        class Model(torch.nn.Module):
            def forward(self, p, a, b, c):
                d = a * 2
                e = b / 2

                def true_fn(x):
                    return x + d

                def false_fn(x):
                    return x - e

                return torch.cond(p, true_fn, false_fn, [c])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=False,
        )


instantiate_parametrized_tests(CondTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests(needs="filelock")
