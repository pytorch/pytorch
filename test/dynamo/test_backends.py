# Owner(s): ["module: dynamo"]
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch._dynamo
import torch._dynamo.backends
import torch._dynamo.test_case
from torch._dynamo.backends.debugging import ExplainWithBackend
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.backends.tvm import has_tvm
from torch._dynamo.testing import same
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyHPU,
)
from torch.testing._internal.common_utils import skipIfHpu
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TestOptimizations(torch._dynamo.test_case.TestCase):
    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertEqual(r1.size(), r2.size())
        self.assertEqual(r1.stride(), r2.stride())
        self.assertEqual(r1.dtype, r2.dtype)

        self.assertEqual(r1.size(), r3.size())
        self.assertEqual(r1.stride(), r3.stride())
        self.assertEqual(r1.dtype, r3.dtype)

    def test_example_inputs_runtime_use(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            def fwd(*args):
                nonlocal r1
                r = graph.forward(*args)
                r1 = r[0]
                return r

            return fwd

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)
        self.assertTrue(same(r1, r2))
        self.assertTrue(same(r1, r3))

    def _check_backend_works(self, backend, device, boxed=True, options=None):
        model = Seq().eval()
        model.to(device)

        if not boxed:
            compiled_model = torch.compile(model, backend=backend, options=options)
        else:

            def boxed_assert(gm, *example_args):
                fn = lookup_backend(backend)(gm, *example_args)
                assert fn._boxed_call
                return fn

            compiled_model = torch.compile(model, backend=boxed_assert, options=options)

        input1 = torch.randn(2, 10, device=device, requires_grad=True)
        input2 = input1.detach().clone().requires_grad_(True)

        r1 = model(input1)
        r2 = compiled_model(input2)
        self.assertTrue(same(r1, r2.float(), tol=0.01))

        r1.sum().backward()
        r2.sum().backward()
        self.assertTrue(same(input1.grad, input2.grad.float(), tol=0.01))

        # Clean up compilation state before test returns to avoid false positive
        # memory leak detection (leak check runs before tearDown)
        torch._dynamo.reset()

    def test_eager(self, device):
        self._check_backend_works("eager", device, boxed=False)

    def test_eager_noexcept(self, device):
        self._check_backend_works("eager_noexcept", device, boxed=False)

    @skipIfHpu
    @_force_skip_lazy_graph_module()
    def test_torchscript(self, device):
        self._check_backend_works("ts", device, boxed=False)

    def test_aot_eager(self, device):
        self._check_backend_works("aot_eager", device)

    def test_aot_eager_decomp_partition(self, device):
        self._check_backend_works("aot_eager_decomp_partition", device)

    @skipIfHpu
    @_force_skip_lazy_graph_module()
    def test_aot_ts(self, device):
        self._check_backend_works("aot_ts", device)

    @requires_cuda_and_triton
    def test_aot_cudagraphs(self, device):
        self._check_backend_works("cudagraphs", device)

    @unittest.skipIf(not has_tvm(), "requires tvm")
    def test_tvm(self, device):
        self._check_backend_works("tvm", device)
        self._check_backend_works("tvm", device, options={"scheduler": None})
        self._check_backend_works("tvm", device, options={"opt_level": 0})

    @onlyHPU
    def test_intel_gaudi_backend(self, device):
        self._check_backend_works("hpu_backend", device)

    def test_list_backends(self):
        self.assertIn("inductor", torch._dynamo.list_backends())
        self.assertIn("inductor", torch._dynamo.list_backends(exclude_tags=None))
        self.assertNotIn("eager", torch._dynamo.list_backends())
        self.assertNotIn("eager", torch._dynamo.list_backends(exclude_tags=["debug"]))
        self.assertIn("eager", torch._dynamo.list_backends(exclude_tags=[]))


class NormalizeIRTests(torch._dynamo.test_case.TestCase):
    def test_inplace_normalize(self):
        def fn(a, b):
            x = torch.cos(a)
            x += b
            return torch.sin(x)

        a = torch.randn(10)
        b = torch.randn(10).to(torch.float64)

        ref = fn(a, b)

        optimized_fn = torch.compile(fn, backend="aot_eager")
        res = optimized_fn(a, b)
        self.assertTrue(same(ref, res))


class MPSSupportedTest(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not torch.backends.mps.is_available(), "requires mps")
    def test_mps_supported(self):
        model = Seq().to("mps")
        example_input = torch.randn(1, 10).to("mps")
        rc_eager = model(example_input)
        rc = torch.compile(model, backend="inductor")(example_input)
        self.assertEqual(rc, rc_eager)


class TestExplainWithBackend(torch._dynamo.test_case.TestCase):
    def test_explain_with_backend(self):
        def fn3(x):
            x = torch.sin(x)
            torch._dynamo.graph_break()
            x = torch.sin(x)
            return x

        def fn2(x):
            x = torch.cos(x)
            x = fn3(x)
            x = torch.cos(x)
            return x

        def fn1(x):
            x = torch.tan(x)
            x = fn2(x)
            x = torch.tan(x)
            return x

        def fn(x):
            x = torch.sigmoid(x)
            x = fn1(x)
            x = torch.sigmoid(x)
            return x

        # Wrap TorchInductor with explain backend
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        input_tensor = torch.randn(5)
        result = optimized_fn(input_tensor)

        # Check that fn still produces the same output when wrapped by ExplainWithBackend
        self.assertTrue(torch.allclose(result, fn(input_tensor)))

        # Verify ExplainOutput object contents, output might change but make sure these fields are present
        explain_output = eb.output()
        explain_str = str(explain_output)
        self.assertIn("Graph Count", explain_str)
        self.assertIn("Graph Break Count", explain_str)
        self.assertIn("Op Count", explain_str)
        self.assertIn("Break Reasons", explain_str)

        # Verify that for the given functions above, we report the correct number of graphs, graph breaks, and ops
        self.assertEqual(8, explain_output.graph_count)
        self.assertEqual(7, explain_output.graph_break_count)
        self.assertEqual(8, explain_output.op_count)


class TestCustomBackendAPI(torch._dynamo.test_case.TestCase):
    """Test APIs documented by https://pytorch.org/docs/main/torch.compiler_custom_backends.html"""

    def test_register_backend_api(self):
        from torch._dynamo import register_backend
        from torch._dynamo.backends import registry as backend_registry

        backend_run = False
        backend_name = "my_custom_backend"

        def cleanup_backend():
            backend_registry._COMPILER_FNS.pop(backend_name, None)
            backend_registry._BACKENDS.pop(backend_name, None)

        self.addCleanup(cleanup_backend)

        @register_backend(name=backend_name)
        def my_custom_backend(gm, example_inputs):
            nonlocal backend_run
            backend_run = True
            return gm.forward

        def f(x):
            return torch.relu(x)

        opt_f = torch.compile(f, backend="my_custom_backend")
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_aot_autograd_api(self):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        backend_run = False

        def my_compiler(gm, example_inputs):
            nonlocal backend_run
            backend_run = True
            return make_boxed_func(gm.forward)

        my_backend = aot_autograd(fw_compiler=my_compiler)

        def f(x):
            return torch.relu(x)

        opt_f = torch.compile(f, backend=my_backend)
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_lookup_backend(self):
        from torch._dynamo import lookup_backend

        backend_run = False

        def my_compiler(gm, example_inputs):
            nonlocal backend_run
            backend_run = True
            try:
                trt_compiled = lookup_backend("tensorrt")(gm, example_inputs)
                if trt_compiled is not None:
                    return trt_compiled
            except Exception:
                pass
            # first backend failed, try something else...
            try:
                inductor_compiled = lookup_backend("inductor")(gm, example_inputs)
                if inductor_compiled is not None:
                    return inductor_compiled
            except Exception:
                pass
            return gm.forward

        def f(x):
            return torch.relu(x)

        opt_f = torch.compile(f, backend=my_compiler)
        opt_f(torch.randn(3, 3))
        self.assertTrue(backend_run)

    def test_lookup_custom_backend(self):
        from torch._dynamo import list_backends

        backends_group = "torch_dynamo_backends"
        name = "mycustombackend"

        mock_3_10 = MagicMock()
        mock_3_10.load.return_value = lambda: "mocked 3.10"

        def mock_eps(group=None):
            assert group == backends_group, group
            mock_group = MagicMock()
            mock_group.names = [name]
            mock_group[name] = mock_3_10
            return mock_group

        with patch("importlib.metadata.entry_points", mock_eps):
            from torch._dynamo.backends import registry

            orig_backends = dict(registry._BACKENDS)
            orig_compiler_fns = dict(registry._COMPILER_FNS)

            def restore_registry():
                registry._BACKENDS.clear()
                registry._BACKENDS.update(orig_backends)
                registry._COMPILER_FNS.clear()
                registry._COMPILER_FNS.update(orig_compiler_fns)
                registry._lazy_import.cache_clear()
                registry._discover_entrypoint_backends.cache_clear()

            self.addCleanup(restore_registry)

            registry._lazy_import.cache_clear()
            registry._discover_entrypoint_backends.cache_clear()

            backends = list_backends()
            assert name in backends, (name, backends)

    def test_backend_recompilation(self):
        def fn(x):
            return x + x

        input = torch.tensor(2.0)

        opt_fn = torch.compile(
            fn, backend="inductor", options={"_raise_error_for_testing": False}
        )
        opt_fn(input)
        with self.assertRaises(torch._dynamo.exc.BackendCompilerFailed):
            opt_fn = torch.compile(
                fn, backend="inductor", options={"_raise_error_for_testing": True}
            )
            opt_fn(input)

    def test_backend_graph_freeze(self):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        backend_run = False

        def my_compiler(gm, example_inputs):
            nonlocal backend_run
            if tracing_context := torch._guards.TracingContext.try_get():
                fw_metadata = tracing_context.fw_metadata
                params_flat = tracing_context.params_flat
                self.assertTrue(fw_metadata is not None)
                self.assertTrue(params_flat is not None)
                self.assertTrue(len(params_flat) == 2)
            backend_run = True
            return make_boxed_func(gm.forward)

        my_backend = aot_autograd(fw_compiler=my_compiler)

        class MyClass(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.p1 = torch.nn.Parameter(torch.randn(2, 3))
                self.p2 = torch.nn.Parameter(torch.randn(2, 3))

            @torch._dynamo.config.patch("prepare_freezing", True)
            def forward(self, x):
                t = self.p1 + x
                out = t / self.p2
                return out

        mod = MyClass()

        opt_mod = torch.compile(mod, backend=my_backend)
        opt_mod(torch.randn(2, 3))
        self.assertTrue(backend_run)


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(TestOptimizations, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
