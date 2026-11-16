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
        # current_stream/synchronize are traced without calling the backend's
        # own implementations (streams.synchronize_device etc.), while
        # _exchange_device/_maybe_exchange_device run the backend's real
        # functions. The exchange leaves the runtime device switched (matching
        # eager semantics), so the current_device guard forces a recompile when
        # the next call starts on another device.
        prev_device = torch.openreg.current_device()
        torch.openreg.set_device(0)
        try:
            cnt = CompileCounterWithBackend("eager")
            # The exchange cleanup hook restores the pre-trace device at the
            # end of tracing, before the backend runs; record the device the
            # backend sees to assert the hook actually restored it.
            devices_at_compile: list[int] = []
            inner_backend = cnt.__call__

            def recording_backend(gm, example_inputs):
                devices_at_compile.append(torch.openreg.current_device())
                return inner_backend(gm, example_inputs)

            @torch.compile(backend=recording_backend, fullgraph=True)
            def compiled_fn():
                stream = torch.openreg.current_stream()
                torch.openreg.synchronize()
                torch.openreg._exchange_device(1)
                torch.openreg._maybe_exchange_device(1)
                return stream

            traced_stream = compiled_fn()
            # The cleanup hook ran at the end of tracing, restoring device 0
            # before the backend was invoked (without it the backend would have
            # seen the switched device 1). The second call starts on device 1
            # (the runtime switch), so its compile records 1.
            self.assertEqual(devices_at_compile, [0, 1])
            # The traced stream was captured for device 0 (the device at trace
            # time); it reconstructs to the eager current stream of device 0.
            self.assertEqual(traced_stream, torch.openreg.current_stream(0))
            # The eager backend executes the emitted graph as-is; the exchange
            # leaves the runtime device switched, matching eager semantics. The
            # cleanup hook restored the device at the end of tracing (see
            # devices_at_compile above), not after each call.
            self.assertEqual(torch.openreg.current_device(), 1)
            # The current_device guard was installed for device 0 at compile
            # time, so starting this call from device 1 recompiles.
            compiled_fn()
            self.assertEqual(cnt.frame_count, 2)

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
            # The exchange helpers are traced as call_function nodes that run
            # the backend's real functions; the current_device guard and the
            # set_device cleanup hook are exercised across the two calls above.
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
                torch.openreg = openreg_module
                for fn in fns:
                    rule_map[fn] = TorchInGraphFunctionVariable
        finally:
            torch.openreg.set_device(prev_device)

    def test_exchange_device_invalid_index(self):
        # Mirror torch.cuda's _exchange_device: a negative index reports no
        # exchange via the -1 convention and must not switch the device; a
        # positive out-of-range index raises rather than leaving the device in
        # an invalid state.
        prev = torch.openreg.current_device()
        self.assertEqual(torch.openreg._exchange_device(-1), -1)
        self.assertEqual(torch.openreg.current_device(), prev)
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.openreg._exchange_device(torch.openreg.device_count() + 10)
        self.assertEqual(torch.openreg.current_device(), prev)

        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertEqual(iface.exchange_device(-1), -1)
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            iface.exchange_device(torch.openreg.device_count() + 10)

    def test_maybe_exchange_device_invalid_index(self):
        # Mirror torch.cuda's _maybe_exchange_device: a negative index reports
        # no exchange via the -1 convention and must not switch the device; a
        # positive out-of-range index is an unusable target, so the current
        # device is returned without switching (never raises).
        prev = torch.openreg.current_device()
        self.assertEqual(torch.openreg._maybe_exchange_device(-1), -1)
        self.assertEqual(torch.openreg.current_device(), prev)
        self.assertEqual(
            torch.openreg._maybe_exchange_device(torch.openreg.device_count() + 10),
            prev,
        )
        self.assertEqual(torch.openreg.current_device(), prev)

        from torch._dynamo.device_interface import get_interface_for_device

        iface = get_interface_for_device("openreg")
        self.assertEqual(iface.maybe_exchange_device(-1), -1)
        self.assertEqual(
            iface.maybe_exchange_device(torch.openreg.device_count() + 10), prev
        )
        self.assertEqual(torch.openreg.current_device(), prev)

    def test_exchange_negative_index_cleanup_hook(self):
        # The exchange cleanup hook restores set_device(prev); prev is -1 when
        # the exchange was called with a negative index, so set_device(-1) must
        # be a no-op (it would otherwise raise and break the compiled call).
        cnt = CompileCounterWithBackend("eager")
        prev_device = torch.openreg.current_device()

        @torch.compile(backend=cnt, fullgraph=True)
        def compiled_fn():
            torch.openreg._exchange_device(-1)

        compiled_fn()
        self.assertEqual(torch.openreg.current_device(), prev_device)
        # The current_device guard is unchanged (no switch happened), so the
        # second call reuses the compiled graph.
        compiled_fn()
        self.assertEqual(cnt.frame_count, 1)

    def test_aliased_exchange_device_functions(self):
        # A backend may define _maybe_exchange_device = _exchange_device (or
        # vice versa); handler registration deduplicates the alias instead of
        # raising "Handler already registered", and compiling through the
        # aliased name emits the shared exchange target.
        from torch._dynamo import trace_rules
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        exchange = torch.openreg._exchange_device
        maybe = torch.openreg._maybe_exchange_device
        self.assertIsNot(exchange, maybe)
        prev_device = torch.openreg.current_device()
        try:
            torch.openreg._maybe_exchange_device = exchange
            trace_rules._register_privateuse1_backend()
            self.assertIn(exchange, TorchInGraphFunctionVariable._get_handlers())

            torch.openreg.set_device(0)
            cnt = CompileCounterWithBackend("eager")

            @torch.compile(backend=cnt, fullgraph=True)
            def compiled_fn():
                torch.openreg._maybe_exchange_device(1)

            compiled_fn()
            graph = cnt.graphs[0]
            exchange_nodes = [
                node
                for node in graph.graph.nodes
                if node.op == "call_function"
                and node.target is torch.openreg._exchange_device
            ]
            # The aliased call routes to the shared exchange handler, which
            # emits the canonical _exchange_device target (self.value).
            self.assertEqual(len(exchange_nodes), 1)
            self.assertEqual(exchange_nodes[0].args, (1,))
            # The exchange leaves the runtime device switched, so the
            # current_device guard (captured for device 0) recompiles the next
            # call, which starts on device 1.
            self.assertEqual(torch.openreg.current_device(), 1)
            compiled_fn()
            self.assertEqual(cnt.frame_count, 2)
        finally:
            torch.openreg._maybe_exchange_device = maybe
            # Re-probe the restored (non-aliased) module so the rule map and
            # handler cache are left as they were found.
            trace_rules._register_privateuse1_backend()
            torch.openreg.set_device(prev_device)

    def test_backend_current_stream_mismatch(self):
        # The backend's own current_stream is routed by name (the reverse
        # lookup cannot resolve it for interfaces that define no
        # current_stream). It must still fail loudly when the symbolic state is
        # tracking a different device instead of silently returning that
        # stream.
        cpu_stream = torch.Stream(device="cpu")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, s):
            with s:
                torch.openreg.current_stream()
            return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported) as ctx:
            fn(torch.tensor(1.0), cpu_stream)
        msg = str(ctx.exception)
        self.assertIn("Dynamo stream mismatch", msg)
        self.assertIn("openreg", msg)
        self.assertIn("cpu", msg)

    def test_exchange_device_missing_contract_methods(self):
        # The exchange handler checks the current_device/set_device contract
        # before tracing: without set_device the cleanup hook would otherwise
        # raise an AttributeError after leaving the process on the switched
        # device. The graph break must fire instead and leave the device
        # untouched.
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        prev_device = torch.openreg.current_device()
        orig_set_device = torch.openreg.set_device
        try:
            del torch.openreg.set_device
            TorchInGraphFunctionVariable._get_handlers.cache_clear()

            @torch.compile(backend="eager", fullgraph=True)
            def fn():
                torch.openreg._exchange_device(1)

            with self.assertRaises(torch._dynamo.exc.Unsupported) as ctx:
                fn()
            msg = str(ctx.exception)
            self.assertIn("privateuse1 backend missing current_device/set_device", msg)
            self.assertIn("torch.openreg", msg)
            # The contract check runs before any exchange, so the failed trace
            # must not have switched the device.
            self.assertEqual(torch.openreg.current_device(), prev_device)
        finally:
            torch.openreg.set_device = orig_set_device
            TorchInGraphFunctionVariable._get_handlers.cache_clear()


if __name__ == "__main__":
    run_tests()
