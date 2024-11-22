# Owner(s): ["module: inductor"]
import contextlib
import itertools
import unittest

import torch
import torch._dynamo.testing
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


def _prepend_product_of_values(inputs, possible_values, num_to_prepend=1):
    result = []
    device = inputs[0].device
    # iterate over the cartesian product of predicate values
    for values in itertools.product(*([possible_values] * num_to_prepend)):
        prepended = [torch.tensor(v, device=device) for v in values]
        result.append((*prepended, *inputs))
    return result


def prepend_predicates(inputs, num_predicates=1):
    return _prepend_product_of_values(inputs, [False, True], num_predicates)


def prepend_counters(inputs, num_counters=1, counter_values=(0, 1, 5)):
    return _prepend_product_of_values(inputs, counter_values, num_counters)


class CondModels:
    class Simple(torch.nn.Module):
        def forward(self, p, a, b):
            def true_fn(x, y):
                return x + y

            def false_fn(x, y):
                return x - y

            return torch.cond(p, true_fn, false_fn, [a, b])

    class SimpleWithIntClosure(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num = 3

        def forward(self, p, a, b):
            return torch.cond(
                pred=p,
                true_fn=lambda a, b: [a + b + self.num],
                false_fn=lambda a, b: [a - b - self.num],
                operands=(a, b),
            )

    class Nested(torch.nn.Module):
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

    class Parameters(torch.nn.Module):
        class InnerModel1(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer = torch.nn.Linear(20, 30, device=device)

            def forward(self, x):
                return self.layer(x + 1) * 3.14

        class InnerModel2(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer1 = torch.nn.Linear(20, 10, device=device)
                self.layer2 = torch.nn.Linear(10, 30, device=device)

            def forward(self, x):
                return self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            super().__init__()
            self.true_fn = self.InnerModel1(device)
            self.false_fn = self.InnerModel2(device)

        def forward(self, p, a):
            return torch.cond(p, self.true_fn, self.false_fn, [a])

    class ReinterpretView(torch.nn.Module):
        def forward(self, p, a, b):
            def true_fn(x, y):
                z1 = x + y
                z2 = x - y
                return z1[2:], z2[:, 4:]

            def false_fn(x, y):
                z1 = x - y
                z2 = x + y
                return z1[2:], z2[:, 4:]

            return torch.cond(p, true_fn, false_fn, [a[:-1], b[:-1]])

    class MultipleOutputs(torch.nn.Module):
        def forward(self, p, a, b, c):
            def true_fn(x, y, z):
                return x * y, z / 2.71, (y - x).sum(dim=1)

            def false_fn(x, y, z):
                return y / x, z * 3.14, (x + y).mean(dim=1)

            return torch.cond(p, true_fn, false_fn, [a, b, c])

    class OuterCode(torch.nn.Module):
        def forward(self, p, a, b):
            c = a * b + 3.14
            d = a / b - 2.71

            def true_fn(x, y):
                return x + y

            def false_fn(x, y):
                return x - y

            e = torch.cond(p, true_fn, false_fn, [c, d])

            return e * e / 1.41

    class OuterBuffers(torch.nn.Module):
        def forward(self, p, a, b, c):
            d = a * 2
            e = b / 2

            def true_fn(x):
                return x + d

            def false_fn(x):
                return x - e

            return torch.cond(p, true_fn, false_fn, [c])

    class WithNonTensorPredicate(torch.nn.Module):
        def forward(self, a, b):
            def true_fn(x, y):
                return x.sum(0) / 3.14

            def false_fn(x, y):
                return y.sum(0) * 2.71

            return torch.cond(a.size(0) > b.size(0), true_fn, false_fn, [a, b])


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
            for inputs_with_predicates in prepend_predicates(inputs, num_predicates):
                cloned_inputs = [inp.clone() for inp in inputs_with_predicates]
                result = model(*inputs_with_predicates)
                result_compiled = compiled_model(*inputs_with_predicates)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_predicates)
                torch.testing.assert_close(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_simple_control_flow(self, device, dynamic):
        # cond control flow without nesting
        self._run_test(
            model=CondModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_simple_with_int_closure(self, device):
        self._run_test(
            model=torch.compile(CondModels.SimpleWithIntClosure(), dynamic=True),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
        )

    @requires_gpu
    def test_cond_control_flow_with_precomputed_size(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(
                    512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                )
                self.threshold = 20

            def forward(self, x: torch.Tensor, index) -> torch.Tensor:
                def true_fn(x: torch.Tensor):
                    return self.conv2d(x)

                def false_fn(x: torch.Tensor):
                    return self.conv2d(x)

                return torch.cond(
                    index < self.threshold and index >= 0, true_fn, false_fn, (x,)
                )

        main_model = TestModel().to(GPU_TYPE)
        x1 = torch.rand(2, 512, 128, 72).to(GPU_TYPE)
        x2 = torch.rand(2, 512, 96, 96).to(GPU_TYPE)

        opt_model = torch.compile(main_model)
        out1 = main_model(x1, 1)
        opt_out1 = opt_model(x1, 1)
        self.assertTrue(torch.allclose(out1, opt_out1, atol=1e-5))

        out2 = main_model(x2, 30)
        opt_out2 = opt_model(x2, 30)
        self.assertTrue(torch.allclose(out2, opt_out2, atol=1e-5))

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_nested_control_flow(self, device, dynamic):
        # cond control flow with nesting
        self._run_test(
            model=CondModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_predicates=3,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_outer_code_before_after(self, device, dynamic):
        # some code before and after the conditional
        self._run_test(
            model=CondModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_multiple_outputs(self, device, dynamic):
        # multiple outputs with different shapes
        self._run_test(
            model=CondModels.MultipleOutputs(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(30, 40),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_advanced_dynamic_shapes(self, device):
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

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_outer_to_inner(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.cos(x)

                def false_fn(x):
                    return torch.sin(x)

                nz = torch.nonzero(a)
                b = torch.ones([nz.size(0), 8], device=nz.device)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_inner(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @unittest.skip("unbacked symints from inner to outer graph not supported yet")
    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_inner_to_outer(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                y = torch.cond(p, true_fn, false_fn, [b])
                return torch.sin(y)

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    def test_cond_use_buffers_from_outer_scope(self):
        # subgraphs input shapes include symbolic expressions
        self._run_test(
            model=CondModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=False,
        )

    @requires_gpu
    def test_cond_reintepret_view_inputs_outputs(self):
        # ReinterpretView in inputs and outputs of the subgraphs
        self._run_test(
            model=CondModels.ReinterpretView(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_subgraphs_with_parameters(self, device, dynamic):
        # nested Modules with parameters
        self._run_test(
            model=CondModels.Parameters(device),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, device, dynamic):
        # model with a boolean predicate
        for b_size_0 in [5, 15]:
            torch._dynamo.reset()
            self._run_test(
                model=CondModels.WithNonTensorPredicate(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(b_size_0, 20),
                ),
                device=device,
                dynamic=dynamic,
                num_predicates=0,
            )

    @requires_gpu
    def test_cond_aliasing_outputs(self):
        # output aliasing in subgraphs: not supported
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    z = x + y
                    return z, z[1:]

                def false_fn(x, y):
                    z = x - y
                    return z, z[1:]

                return torch.cond(p, true_fn, false_fn, [a, b])

        # AssertionError: Output aliasing is currently not supported...
        with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed):
            torch.compile(Model())(
                torch.tensor(True),
                torch.randn(10, 20),
                torch.randn(10, 20),
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.zeros_like(x)

                def false_fn(x):
                    return torch.ones_like(x)

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph_recursive(self, device):
        def inner_fn1(x):
            return torch.zeros_like(x)

        def inner_fn2(x):
            return torch.ones_like(x)

        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.cond(p, inner_fn2, inner_fn1, [x])

                def false_fn(x):
                    return torch.cond(p, inner_fn1, inner_fn2, [x])

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    def test_cond_inductor_fx_passes_recursively_applied(self):
        counters = {"pre_grad": 0, "post_grad": 0}

        def pre_grad_pass_counter(gm):
            counters["pre_grad"] += 1

        def post_grad_pass_counter(gm):
            counters["post_grad"] += 1

        with torch._inductor.config.patch(
            {
                "pre_grad_custom_pass": pre_grad_pass_counter,
                "post_grad_custom_pre_pass": post_grad_pass_counter,
                # The above patches don't pickle
                "fx_graph_cache": False,
            }
        ):
            self._run_test(
                model=CondModels.Nested(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=GPU_TYPE,
                dynamic=True,
                num_predicates=3,
            )

        self.assertEqual(counters["pre_grad"], 11)
        self.assertEqual(counters["post_grad"], 11)


class WhileLoopModels:
    class Simple(torch.nn.Module):
        def forward(self, ci, a, b):
            def cond_fn(i, x, y):
                return i > 0

            def body_fn(i, x, y):
                return i - 1, x + y, y - x

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = torch._higher_order_ops.while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )

                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class Parameters(torch.nn.Module):
        class InnerModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer1 = torch.nn.Linear(20, 30, device=device)
                self.layer2 = torch.nn.Linear(30, 20, device=device)

            def forward(self, c, x):
                return c - 1, self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            super().__init__()
            self.body_fn = self.InnerModel(device)
            self.cond_fn = lambda c, x: c > 0

        def forward(self, c, a):
            return torch._higher_order_ops.while_loop(
                self.cond_fn, self.body_fn, [c, a]
            )

    class OuterCode(torch.nn.Module):
        def forward(self, c, a, b):
            d = a * b + 3.14
            e = a / b - 2.71

            def cond_fn(c, x, y):
                return c > 0

            def body_fn(c, x, y):
                return c - 1, y - x, x + y

            _, f, g = torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, d, e])

            return f * g / 1.41

    # TODO(aakhundov): add while_loop test with outer buffers
    # with dynamic=True once dynamo / export allows while_loop
    # closure capture with mark_dynamic:
    # https://github.com/pytorch/pytorch/issues/123596
    class OuterBuffers(torch.nn.Module):
        def forward(self, c, a, b):
            d = a * 2
            e = b / 2

            def cond_fn(c, x, y):
                return c > 0

            def body_fn(c, x, y):
                return c - 1, x + d, y - e

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, a, b])


class WhileLoopTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_counters=1,
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
                    if inp.ndim:
                        torch._dynamo.mark_dynamic(inp, 0)

        for inputs in input_sets:
            for inputs_with_counters in prepend_counters(inputs, num_counters):
                cloned_inputs = [inp.clone() for inp in inputs_with_counters]
                result = model(*inputs_with_counters)
                with torch.no_grad():
                    result_compiled = compiled_model(*inputs_with_counters)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_counters)
                torch.testing.assert_close(
                    result, result_compiled, atol=1e-4, rtol=1e-4
                )

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_while_loop_simple_control_flow(self, device, dynamic):
        # while_loop control flow without nesting
        self._run_test(
            model=WhileLoopModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_while_loop_nested_control_flow(self, device, dynamic):
        # while_loop control flow with nesting
        self._run_test(
            model=WhileLoopModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_counters=2,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_while_loop_with_outer_code(self, device, dynamic):
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_while_loop_with_parameters(self, device, dynamic):
        # while_loop control flow with parameters
        self._run_test(
            model=WhileLoopModels.Parameters(device),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    # dynamic=True doesn't work now due to
    # https://github.com/pytorch/pytorch/issues/123596
    @parametrize("dynamic", [False])
    def test_while_loop_with_outer_buffers(self, device, dynamic):
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )


class AssociativeScanTests(TestCase):
    @requires_gpu
    @parametrize("combine_mode", ["pointwise", "generic"])
    @parametrize("backend", ["inductor"])
    @parametrize("device", [torch.device("cpu"), GPU_TYPE])
    # This test will fail as flip in combination with particular input lenghts
    # produces weird results.
    # This is under investigations in
    # https://github.com/pytorch/pytorch/issues/131805
    @decorateIf(unittest.skip, lambda params: params["device"] == GPU_TYPE)
    def test_associative_scan_CUDA_flip(self, combine_mode, backend, device):
        def fct(x: torch.Tensor, y: torch.Tensor):
            return x + y

        # for n in range(10):
        for n in [9]:
            x = torch.arange(n, device=device)
            torch.compiler.reset()
            associative_scan1 = torch.compile(
                associative_scan, backend=backend, fullgraph=True
            )
            associative_scan2 = associative_scan

            if combine_mode == "pointwise" and device == torch.device("cpu"):
                with self.assertRaisesRegex(Exception, r"."):
                    associative_scan1(
                        fct, x, 0, reverse=False, combine_mode=combine_mode
                    )

                # Skipping test because combine_mode currently only suppors CUDA tensors
                return

            result1 = associative_scan1(
                fct, x, 0, reverse=False, combine_mode=combine_mode
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=False, combine_mode=combine_mode
            )
            result3 = torch.cumsum(x, 0)

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Flip only non-compiled and compare with compiled reverse=True
            result1 = associative_scan1(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result2 = torch.flip(
                associative_scan2(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Flip only compiled and compare with non-compiled reverse=True
            result1 = torch.flip(
                associative_scan1(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Use reverse=False, but flip both results before and after
            result1 = torch.flip(
                associative_scan1(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result2 = torch.flip(
                associative_scan2(
                    fct, torch.flip(x, [0]), 0, reverse=False, combine_mode=combine_mode
                ),
                [0],
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)

            # Reverse=True
            result1 = associative_scan1(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result2 = associative_scan2(
                fct, x, 0, reverse=True, combine_mode=combine_mode
            )
            result3 = torch.flip(torch.cumsum(torch.flip(x, [0]), 0), [0])

            self.assertEqual(result1, result2)
            self.assertEqual(result1, result3)


class ScanModels:
    class SimpleScan(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim

        def forward(self, _input, weight, bias):
            def combine_fn(carry, x):
                from torch.utils import _pytree as pytree

                new_carry = {
                    "param": carry["param"] @ x + carry["bias"],
                    "bias": carry["bias"].sin(),
                }
                return new_carry, (
                    pytree.tree_map(lambda x: x.clone(), new_carry),
                    {"dummy": x.sin()},
                )

            return scan(
                combine_fn,
                {"param": weight, "bias": bias},
                _input,
                reverse=self.reverse,
                dim=self.dim,
            )

    class ScanLinearWithView(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.linear = torch.nn.Linear(9, 9)

        def forward(self, scan_op, init, xs):
            def combine_fn(carry, x):
                prev_sz = x.size()
                x = self.linear(x.view(-1, x.size(-1)))
                x_view = x.view(*prev_sz)
                return x_view, x_view.clone()

            return scan_op(combine_fn, init, xs, dim=self.dim, reverse=self.reverse)

    class ScanConv(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.conv2d = torch.nn.Conv2d(4, 4, (3, 3), stride=(1, 1), padding=(1, 1))

        # init = torch.randn(4, 4, 9, 9)
        # xs = torch.randn(scan_dim, 4, 4, 9, 9)
        def forward(self, scan_op, init, xs):
            def combine_fn(carry, x):
                x = self.conv2d(x)
                return x, x.clone()

            return scan_op(combine_fn, init, xs, dim=self.dim, reverse=self.reverse)

    class ScanConvAndLinear(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.conv2d = torch.nn.Conv2d(4, 4, (3, 3), stride=(1, 1), padding=(1, 1))
            self.linear = torch.nn.Linear(9, 9)

        # init = torch.randn(4, 4, 9, 9)
        # xs = torch.randn(scan_dim, 4, 4, 9, 9)
        def forward(self, scan_op, init, xs):
            def combine_fn(carry, x):
                # size is still 4, 4, 9, 9 after conv2d
                x = self.conv2d(x)
                sizes = x.size()
                x = self.linear(x.view(-1, sizes[-1]))
                x = x.view(*sizes)
                new_carry = carry + x
                y = new_carry.view(-1)
                return new_carry, y

            return scan_op(combine_fn, init, xs, dim=self.dim, reverse=self.reverse)

    class ScanInScan(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.linear = torch.nn.Linear(9, 9)
            self.linear2 = torch.nn.Linear(9, 9)

        # init: (4, 9)
        # xs: (scan_length, 4, 9, 9)
        def forward(self, scan_op, init, xs):
            # carry: (4, 9)
            # x: (9, 9)
            def inner_combine_fn(carry, x):
                prev_sz = x.size()
                x = self.linear(x.view(-1, x.size(-1)))
                x_view = x.view(*prev_sz)
                # x_view: (9, 9)
                return carry + torch.sum(x_view, 0, keepdim=True), x_view.clone()

            # carry: (4, 9)
            # x: (4, 9, 9)
            def outer_combine_fn(carry, x):
                new_carry = self.linear2(carry)
                new_carry2, y2 = scan_op(inner_combine_fn, new_carry, x)
                return new_carry2, y2

            return scan_op(
                outer_combine_fn, init, xs, reverse=self.reverse, dim=self.dim
            )

    class ScanInCond(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.true_scan_linear = ScanModels.ScanConvAndLinear(reverse, dim)
            self.false_scan_linear = ScanModels.ScanConvAndLinear(not reverse, dim)

        def forward(self, scan_op, pred, init, xs):
            def true_fn():
                last_carry, y = self.true_scan_linear(scan_op, init, xs)
                return last_carry.sum(), y.sin()

            def false_fn():
                last_carry, y = self.false_scan_linear(scan_op, init, xs)
                return -last_carry.sum(), y.cos()

            return torch.cond(pred, true_fn, false_fn, tuple())

    class CondInScan(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim
            self.true_linear = torch.nn.Linear(9, 9)
            self.false_linear = torch.nn.Linear(9, 9)

        def forward(self, scan_op, init, xs):
            def combine_fn(carry, x):
                old_sizes = carry.size()
                carry_view = carry.view(-1, carry.size()[-1])
                new_carry_out = torch.cond(
                    torch.all(carry_view > 1),
                    lambda: self.true_linear(carry_view).sin(),
                    lambda: self.false_linear(carry_view).cos(),
                    tuple(),
                )
                return carry + new_carry_out.view(*old_sizes), new_carry_out

            return scan_op(
                combine_fn,
                init,
                xs,
                dim=self.dim,
                reverse=self.reverse,
            )

    class SimpleWithPytreeInOuts(torch.nn.Module):
        def __init__(self, reverse, dim):
            super().__init__()
            self.reverse = reverse
            self.dim = dim

        def forward(self, scan_op, _input, weight, bias):
            def combine_fn(carry, x):
                from torch.utils import _pytree as pytree

                new_carry = {
                    "param": carry["param"] @ x + carry["bias"],
                    "bias": carry["bias"].sin(),
                }
                return new_carry, (
                    pytree.tree_map(lambda x: x.clone(), new_carry),
                    {"dummy": x.sin()},
                )

            return scan_op(
                combine_fn,
                {"param": weight, "bias": bias},
                _input,
                reverse=self.reverse,
                dim=self.dim,
            )

    class ChunkedCE(torch.nn.Module):
        def __init__(self, chunk_size):
            super().__init__()
            self.chunk_size = chunk_size
            self.ce = lambda logits, target: torch.abs(target - logits).sum()

        def forward(self, scan_op, _input, weight, target, bias):
            CHUNK_SIZE = self.chunk_size

            def compute_loss(input_chunk, weight, bias, target):
                logits = torch.addmm(bias, input_chunk, weight.t())
                logits = logits.float()
                loss = self.ce(logits, target)
                return loss

            grad_weight = torch.zeros_like(weight)
            grad_bias = torch.zeros_like(bias)
            loss_acc = torch.zeros((), device=_input.device)

            chunks = _input.shape[0] // CHUNK_SIZE

            _input_chunks = _input.view(CHUNK_SIZE, chunks, *_input.shape[1:])
            target_chunks = target.view(CHUNK_SIZE, chunks, *target.shape[1:])

            def combine_fn(carry, xs):
                grad_weight, grad_bias, loss_acc = carry
                input_chunk, target_chunk = xs
                (
                    chunk_grad_input,
                    chunk_grad_weight,
                    chunk_grad_bias,
                ), chunk_loss = torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 2)
                )(
                    input_chunk, weight, bias, target_chunk
                )
                return (
                    (
                        grad_weight + chunk_grad_weight,
                        grad_bias + chunk_grad_bias,
                        loss_acc + chunk_loss,
                    ),
                    chunk_grad_input,
                )

            (grad_weight, grad_bias, loss_acc), grad_inputs = scan_op(
                combine_fn,
                (grad_weight, grad_bias, loss_acc),
                (_input_chunks, target_chunks),
            )
            return (
                grad_weight / chunks,
                grad_bias / chunks,
                loss_acc / chunks,
                grad_inputs.view(-1, *_input.shape[1:]) / chunks,
            )

    class ChunkedCENoScan(torch.nn.Module):
        def __init__(self, chunk_size):
            super().__init__()
            self.chunk_size = chunk_size
            self.ce = lambda logits, target: torch.abs(target - logits).sum()

        def forward(self, scan_op, _input, weight, target, bias):
            CHUNK_SIZE = self.chunk_size

            def compute_loss(input_chunk, weight, bias, target):
                logits = torch.addmm(bias, input_chunk, weight.t())
                logits = logits.float()
                loss = self.ce(logits, target)
                return loss

            grad_weight = torch.zeros_like(weight)
            grad_inputs = []
            grad_bias = torch.zeros_like(bias)
            loss_acc = torch.zeros((), device=_input.device)

            chunks = _input.shape[0] // CHUNK_SIZE

            def accumulate_chunk(input_chunk, target_chunk):
                (
                    chunk_grad_input,
                    chunk_grad_weight,
                    chunk_grad_bias,
                ), chunk_loss = torch.func.grad_and_value(
                    compute_loss, argnums=(0, 1, 2)
                )(
                    input_chunk, weight, bias, target_chunk
                )
                grad_weight.add_(chunk_grad_weight)
                grad_bias.add_(chunk_grad_bias)
                loss_acc.add_(chunk_loss)
                return chunk_grad_input

            accumulate_chunk = torch.compile(accumulate_chunk)

            input_chunks = torch.chunk(_input, chunks=chunks, dim=0)
            target_chunks = torch.chunk(target, chunks=chunks, dim=0)
            for input_chunk, target_chunk in zip(input_chunks, target_chunks):
                grad_inputs.append(accumulate_chunk(input_chunk, target_chunk))
            return (
                grad_weight / chunks,
                grad_bias / chunks,
                loss_acc / chunks,
                torch.cat(grad_inputs, dim=0) / chunks,
            )


class ScanTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic,
        requires_grad=False,
    ):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True, dynamic=dynamic)(
            model
        )

        inputs = [inp.to(device=device) for inp in inputs]
        model = model.to(device=device)
        cloned_inputs = [inp.clone() for inp in inputs]
        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        with grad_ctx:
            result = model(scan, *cloned_inputs)
            result_exp = model(_fake_scan, *cloned_inputs)

            result_compiled = compiled_model(scan, *cloned_inputs)
            result_compiled_exp = compiled_model(_fake_scan, *cloned_inputs)

        self.assertEqual(result, result_exp)
        self.assertEqual(result_exp, result_compiled)
        self.assertEqual(result_compiled, result_compiled_exp)

    def _compare_result(
        self,
        model1,
        model2,
        inputs,
        device,
    ):
        inp_on_device = [elem.to(device=device) for elem in inputs]
        cloned_inputs = [arg.clone() for arg in inp_on_device]
        model1_out = model1(scan, *cloned_inputs)
        model2_out = model2(scan, *cloned_inputs)
        self.assertEqual(model1_out, model2_out)

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2])
    def test_scan_pytree_in_out(self, device, dynamic, reverse, dim):
        self._run_test(
            model=ScanModels.SimpleWithPytreeInOuts(reverse=reverse, dim=dim),
            inputs=(
                torch.randn(3, 3, 3),
                torch.randn(4, 3),
                torch.randn(3),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("scan_length", [1, 5])
    def test_scan_nn_modules(self, device, dynamic, reverse, dim, scan_length):
        init = torch.randn(20, 16, 9, 9)
        xs = torch.randn(scan_length, 20, 16, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanLinearWithView(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("scan_length", [1, 5])
    def test_scan_conv(self, device, dynamic, reverse, dim, scan_length):
        init = torch.randn(4, 4, 9, 9)
        xs = torch.randn(scan_length, 4, 4, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanConv(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("scan_length", [1, 5])
    def test_scan_conv_linear(self, device, dynamic, reverse, dim, scan_length):
        init = torch.randn(4, 4, 9, 9)
        xs = torch.randn(scan_length, 4, 4, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanConvAndLinear(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("scan_length", [1, 5])
    def test_scan_in_scan(self, device, dynamic, reverse, dim, scan_length):
        init = torch.randn(4, 9)
        xs = torch.randn(scan_length, 4, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanInScan(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    # TODO(yidi): when setting dynamic=True, some symints are lifted to the subgraph of cond, which
    # is not supported by cond yet.
    @parametrize("dynamic", [False])
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("pred", [True, False])
    @parametrize("scan_length", [1, 5])
    def test_scan_in_cond(self, device, dynamic, reverse, dim, pred, scan_length):
        init = torch.randn(4, 4, 9, 9)
        xs = torch.randn(scan_length, 4, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.ScanInCond(reverse=reverse, dim=dim),
            inputs=(
                torch.tensor(pred),
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("reverse", [True, False])
    @parametrize("dim", [0, 1, 2, 3])
    @parametrize("scan_length", [1, 5])
    def test_cond_in_scan(self, device, dynamic, reverse, dim, scan_length):
        init = torch.randn(4, 4, 9, 9)
        xs = torch.randn(scan_length, 4, 9, 9)
        xs = xs.movedim(0, dim)
        self._run_test(
            model=ScanModels.CondInScan(reverse=reverse, dim=dim),
            inputs=(
                init,
                xs,
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_scan_chunked_ce(self, device, dynamic):
        self._run_test(
            model=ScanModels.ChunkedCE(10),
            inputs=(
                torch.randn(100, 20),
                torch.randn(20, 20),
                torch.randn(100, 20),
                torch.randn(20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_scan_compare_chunked_ce_with_no_scan(self, device, dynamic):
        for trunk_size, B, T in zip([10, 20], [10, 100], [20, 40]):
            self._compare_result(
                model1=torch.compile(ScanModels.ChunkedCE(trunk_size), dynamic=dynamic),
                model2=ScanModels.ChunkedCENoScan(trunk_size),
                inputs=(
                    torch.randn(B, T),
                    torch.randn(T, T),
                    torch.randn(B, T),
                    torch.randn(T),
                ),
                device=device,
            )


instantiate_parametrized_tests(CondTests)
instantiate_parametrized_tests(WhileLoopTests)
instantiate_parametrized_tests(AssociativeScanTests)
instantiate_parametrized_tests(ScanTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
