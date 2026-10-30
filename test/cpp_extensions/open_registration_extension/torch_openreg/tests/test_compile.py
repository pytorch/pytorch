# Owner(s): ["module: PrivateUse1"]

import builtins

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

    def test_backend_metadata_propagation(self):
        captured = []

        def capture_backend(gm, example_inputs):
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    fake = node.meta.get("example_value")
                    if isinstance(fake, torch.Tensor):
                        captured.append(torch._utils.get_tensor_metadata(fake))
            return gm.forward

        @torch.compile(backend=capture_backend, fullgraph=True)
        def fn(x):
            return x + 1

        x = torch.empty(3, 3, device="openreg")
        metadata = {"version_number": True, "format_number": True}
        torch._utils.set_tensor_metadata(x, metadata)
        fn(x)

        # fake_clone is added by clone_for_fake(), confirming the layout metadata
        # reached the compiler backend through the fakeification path.
        self.assertEqual(captured, [{**metadata, "fake_clone": True}])

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


class TestPrivateUse1Routing(TestCase):
    def test_stream_synchronize_exchange_routing(self):
        # The backend calls are the first torch calls traced, so this is the
        # ordering regression: routing must be resolved when the callable is
        # first looked up (get_torch_obj_rule_map, rebuilt on cache clears and
        # backend registration), not later by _get_handlers() which runs only
        # on the first dispatch.
        cnt = CompileCounterWithBackend("eager")
        prev_device = torch.openreg.current_device()

        @torch.compile(backend=cnt, fullgraph=True)
        def compiled_fn():
            torch.openreg.current_stream()
            torch.openreg.synchronize()
            torch.openreg._exchange_device(0)
            torch.openreg._maybe_exchange_device(0)

        compiled_fn()
        compiled_fn()  # second call evaluates the current_device guard
        self.assertEqual(cnt.frame_count, 1)
        # The eager backend executes the emitted graph; the exchange cleanup
        # hook restores the previous device after each call.
        self.assertEqual(torch.openreg.current_device(), prev_device)

        graph = cnt.graphs[0]
        call_targets = {
            node.target for node in graph.graph.nodes if node.op == "call_function"
        }
        # synchronize dispatches to its handler, which emits the
        # device-type-specific synchronize_device op.
        self.assertIn(torch.ops.streams.synchronize_device, call_targets)
        self.assertTrue(
            any(
                node.target is torch.ops.streams.synchronize_device
                and node.args[0] == "openreg"
                for node in graph.graph.nodes
            )
        )
        # _exchange_device/_maybe_exchange_device are traced as call_function
        # nodes; the current_device guard and set_device cleanup hook are
        # exercised by the real backend on the second call.
        self.assertIn(torch.openreg._exchange_device, call_targets)
        self.assertIn(torch.openreg._maybe_exchange_device, call_targets)

        from torch._dynamo import trace_rules
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        fns = (
            torch.openreg.current_stream,
            torch.openreg.synchronize,
            torch.openreg._exchange_device,
            torch.openreg._maybe_exchange_device,
        )
        for fn in fns:
            self.assertIs(trace_rules.lookup(fn), TorchInGraphFunctionVariable)
            self.assertIn(fn, TorchInGraphFunctionVariable._get_handlers())

        # A backend that registers after the rule map was built is routed in
        # place without rebuilding the map, which would drop substitute_in_graph
        # polyfill entries. Exercise the production wiring end-to-end: unbind
        # the module and re-register it through torch._register_device_module,
        # which calls trace_rules._register_privateuse1_backend() for a
        # privateuse1 backend.
        from torch._dynamo.variables import PolyfilledFunctionVariable

        rule_map = trace_rules.get_torch_obj_rule_map()
        for fn in fns:
            del rule_map[fn]
        openreg_module = torch.openreg
        try:
            delattr(torch, "openreg")
            torch._register_device_module("openreg", openreg_module)
            for fn in fns:
                self.assertIn(fn, rule_map)
                self.assertIs(trace_rules.lookup(fn), TorchInGraphFunctionVariable)
            # The in-place re-routing must not rebuild the map: the polyfill
            # entry (and everything else) survives.
            self.assertIs(rule_map[builtins.all], PolyfilledFunctionVariable)
        finally:
            # Restore unconditionally: torch._dynamo.reset() does not clear the
            # rule map, so a failure here must not corrupt it for later tests.
            setattr(torch, "openreg", openreg_module)
            for fn in fns:
                rule_map[fn] = TorchInGraphFunctionVariable

    def test_maybe_exchange_device_invalid_index(self):
        # Invalid indices (negative or out of range) must not switch the device:
        # report no exchange and leave the current device unchanged.
        prev = torch.openreg.current_device()
        for idx in (-1, torch.openreg.device_count() + 10):
            self.assertEqual(torch.openreg._maybe_exchange_device(idx), prev)
            self.assertEqual(torch.openreg.current_device(), prev)

    def test_rename_reroutes_late_backend(self):
        # Renaming a backend after the rule map was built re-routes it in place
        # through the rename hook (torch.utils.rename_privateuse1_backend), the
        # same path a backend author hits when registering under a new name.
        from torch._dynamo import trace_rules
        from torch._dynamo.variables import PolyfilledFunctionVariable
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        fns = (
            torch.openreg.current_stream,
            torch.openreg.synchronize,
            torch.openreg._exchange_device,
            torch.openreg._maybe_exchange_device,
        )
        rule_map = trace_rules.get_torch_obj_rule_map()
        for fn in fns:
            del rule_map[fn]
        try:
            torch.utils.rename_privateuse1_backend("openreg")
            for fn in fns:
                self.assertIn(fn, rule_map)
                self.assertIs(trace_rules.lookup(fn), TorchInGraphFunctionVariable)
            # The re-routing must not rebuild the map: the polyfill entry (and
            # everything else) survives.
            self.assertIs(rule_map[builtins.all], PolyfilledFunctionVariable)
        finally:
            for fn in fns:
                rule_map[fn] = TorchInGraphFunctionVariable

    def test_aliased_exchange_device_functions(self):
        # A backend may define _maybe_exchange_device = _exchange_device (or
        # vice versa); handler registration deduplicates the alias instead of
        # raising "Handler already registered".
        from torch._dynamo import trace_rules
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        exchange = torch.openreg._exchange_device
        maybe = torch.openreg._maybe_exchange_device
        self.assertIsNot(exchange, maybe)
        try:
            torch.openreg._maybe_exchange_device = exchange
            trace_rules._register_privateuse1_backend()
            self.assertIn(exchange, TorchInGraphFunctionVariable._get_handlers())
        finally:
            torch.openreg._maybe_exchange_device = maybe
            # Re-probe the restored (non-aliased) module so the rule map and
            # handler cache are left as they were found.
            trace_rules._register_privateuse1_backend()


if __name__ == "__main__":
    run_tests()
