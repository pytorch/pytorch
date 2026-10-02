# Owner(s): ["module: PrivateUse1"]

import os

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounterWithBackend


class TestBackendRegistration(TestCase):
    def test_backend_in_list(self):
        from torch._dynamo.backends.registry import list_backends

        backends = list_backends(exclude_tags=())
        self.assertIn("openreg", backends)


class TestGraphCapture(TestCase):
    def test_simple_compile(self):
        @torch.compile(backend="openreg")
        def fn(x):
            return x + 1

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, x + 1)

    def test_compile_fullgraph_multiple_ops(self):
        @torch.compile(backend="openreg", fullgraph=True)
        def fn(x):
            y = x + 1
            z = y * 2
            return torch.relu(z)

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, torch.relu((x + 1) * 2))


class TestFakeTensor(TestCase):
    def test_device_metadata(self):
        captured_gm = None

        def capture_backend(gm, example_inputs):
            nonlocal captured_gm
            captured_gm = gm
            return gm.forward

        @torch.compile(backend=capture_backend)
        def fn(x):
            return x + 1

        x = torch.randn(4, device="openreg")
        fn(x)

        self.assertIsNotNone(captured_gm)
        for node in captured_gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                fake = node.meta["val"]
                if isinstance(fake, torch.Tensor):
                    self.assertEqual(fake.device.type, "openreg")

    def test_fake_tensor_mode(self):
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="openreg")
            self.assertEqual(x.device.type, "openreg")
            y = x + 1
            self.assertEqual(y.device.type, "openreg")


class TestGuards(TestCase):
    def test_backend_recompilation(self):
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            return x + 1

        x = torch.randn(4, device="openreg")
        fn(x)
        self.assertEqual(counter.frame_count, 1)

        fn(x)
        self.assertEqual(counter.frame_count, 1)

    def test_recompilation_on_shape_change(self):
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            return x + 1

        x1 = torch.randn(4, device="openreg")
        fn(x1)
        self.assertEqual(counter.frame_count, 1)

        x2 = torch.randn(8, device="openreg")
        fn(x2)
        self.assertEqual(counter.frame_count, 2)


class TestExecution(TestCase):
    def test_autograd(self):
        @torch.compile(backend="openreg")
        def fn(x):
            return (x * 2).sum()

        x = torch.randn(4, device="openreg", requires_grad=True)
        loss = fn(x)
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad, torch.full_like(x, 2.0))

    def test_nn_module(self):
        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        m = SimpleModule().to("openreg")
        compiled_m = torch.compile(m, backend="openreg")

        x = torch.randn(2, 4, device="openreg")
        result = compiled_m(x)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.shape, torch.Size([2, 4]))

        eager_result = m(x)
        self.assertEqual(result, eager_result)


class TestDynamicShapes(TestCase):
    def test_dynamic_shapes(self):
        @torch.compile(backend="openreg", dynamic=True)
        def fn(x):
            return x + 1

        x4 = torch.randn(4, device="openreg")
        x8 = torch.randn(8, device="openreg")

        self.assertEqual(fn(x4), x4 + 1)
        self.assertEqual(fn(x8), x8 + 1)


class TestGraphBreaks(TestCase):
    def test_device_transfer_traces_fully(self):
        @torch.compile(backend="openreg", fullgraph=True)
        def fn(x):
            y = x + 1
            z = y.cpu()
            return z * 2

        x = torch.randn(4, device="openreg")
        result = fn(x)
        expected = (x + 1).cpu() * 2
        self.assertEqual(result, expected)
        self.assertEqual(result.device.type, "cpu")

    def test_graph_break_resilience(self):
        counter = CompileCounterWithBackend("openreg")

        @torch.compile(backend=counter)
        def fn(x):
            y = x + 1
            torch._dynamo.graph_break()
            return y * 2

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result, (x + 1) * 2)
        self.assertGreater(counter.frame_count, 1)


class TestAutocast(TestCase):
    def test_compile_with_autocast(self):
        @torch.compile(backend="openreg")
        def fn(x, y):
            with torch.autocast(device_type="openreg", dtype=torch.float16):
                return torch.mm(x, y)

        x = torch.randn(2, 3, device="openreg")
        y = torch.randn(3, 3, device="openreg")
        result = fn(x, y)
        self.assertEqual(result.dtype, torch.float16)


class TestDefaultDevice(TestCase):
    def test_compile_with_default_device(self):
        with torch.device("openreg"):

            @torch.compile(backend="openreg")
            def fn(x):
                y = torch.empty(4)
                return x + y

            x = torch.randn(4, device="openreg")
            result = fn(x)
            self.assertEqual(result.device.type, "openreg")


class TestDeviceInterface(TestCase):
    def test_interface_registered(self):
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertIsNotNone(iface)

    def test_current_device(self):
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        device_idx = iface.current_device()
        self.assertIsInstance(device_idx, int)

    def test_device_count(self):
        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertGreater(iface.device_count(), 0)


class TestInductorRegistration(TestCase):
    def setUp(self):
        super().setUp()
        import torch_openreg.inductor_backend  # noqa: F401

    def test_scheduling_registered(self):
        from torch._inductor.codegen.common import get_scheduling_for_device
        from torch._inductor.codegen.cpp import CppScheduling

        self.assertIs(get_scheduling_for_device("openreg"), CppScheduling)

    def test_wrapper_codegen_registered(self):
        from torch._inductor.codegen.common import get_wrapper_codegen_for_device
        from torch_openreg.inductor_backend import OpenRegWrapperCodegen

        self.assertIs(get_wrapper_codegen_for_device("openreg"), OpenRegWrapperCodegen)

    def test_device_op_overrides_registered(self):
        from torch._inductor.codegen.common import get_device_op_overrides
        from torch_openreg.inductor_backend import OpenRegDeviceOpOverrides

        overrides = get_device_op_overrides("openreg")
        self.assertIsInstance(overrides, OpenRegDeviceOpOverrides)

    def test_device_op_overrides_methods(self):
        from torch._inductor.codegen.common import get_device_op_overrides

        overrides = get_device_op_overrides("openreg")
        self.assertIn(
            "get_raw_stream", overrides.import_get_raw_stream_as("get_raw_stream")
        )
        self.assertIn("_set_device", overrides.set_device(0))
        self.assertEqual(overrides.synchronize(), "pass")


class TestInductorCompile(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["OPENREG_SKIP_MPROTECT"] = "1"
        import torch_openreg.inductor_backend  # noqa: F401

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("OPENREG_SKIP_MPROTECT", None)
        super().tearDownClass()

    def test_inductor_simple_add(self):
        @torch.compile(backend="inductor")
        def fn(x, y):
            return x + y

        x = torch.randn(4, device="openreg")
        y = torch.randn(4, device="openreg")
        self.assertEqual(fn(x, y), x + y)

    def test_inductor_multiple_ops(self):
        def fn(x):
            return torch.relu(x * 2 + 1)

        x = torch.randn(4, device="openreg")
        compiled = torch.compile(fn, backend="inductor")
        self.assertEqual(compiled(x), fn(x))

    def test_inductor_matches_eager(self):
        def fn(x):
            return x * 2 + x

        x = torch.randn(4, device="openreg")
        compiled = torch.compile(fn, backend="inductor")
        self.assertEqual(compiled(x), fn(x))

    def test_inductor_output_device(self):
        @torch.compile(backend="inductor")
        def fn(x):
            return x + 1

        x = torch.randn(4, device="openreg")
        result = fn(x)
        self.assertEqual(result.device.type, "openreg")


if __name__ == "__main__":
    run_tests()
