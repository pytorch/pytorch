# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    device_interfaces,
    register_interface_for_device,
    registered_stream_context_cls,
)
from torch._dynamo.device_trace_rules import registered_device_trace_fns
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.trace_rules import get_torch_obj_rule_map, lookup
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
from torch.testing._internal.common_utils import HardwareClassification


def _fake_stream(stream):
    return stream


def _fake_current_stream(device=None):
    return device


def _fake_set_stream(stream):
    return None


def _fake_default_stream(device=None):
    return device


def _unspecified_stream(stream):
    return stream


def _index_only_current_stream(device=None):
    return device


class _FakeStreamContext:
    def __init__(self, stream: object) -> None:
        self.stream = stream


class _FakeInterface(DeviceInterface):
    StreamContext = _FakeStreamContext
    stream = staticmethod(_fake_stream)
    current_stream = staticmethod(_fake_current_stream)
    set_stream = staticmethod(_fake_set_stream)
    default_stream = staticmethod(_fake_default_stream)


class _IndexOnlyInterface(DeviceInterface):
    current_stream = staticmethod(_index_only_current_stream)


class TestDeviceTraceRules(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        # Warm both caches so registration has to invalidate them.
        get_torch_obj_rule_map()
        TorchInGraphFunctionVariable._get_handlers()

    def tearDown(self) -> None:
        for key in ("fake", "fake:0", "fakectx", "fakemissingctx"):
            device_interfaces.pop(key, None)
        get_torch_obj_rule_map.cache_clear()
        TorchInGraphFunctionVariable._get_handlers.cache_clear()
        super().tearDown()

    def test_registered_stream_fns_are_in_graph(self) -> None:
        self.assertIsNot(lookup(_fake_stream), TorchInGraphFunctionVariable)
        register_interface_for_device("fake", _FakeInterface)
        register_interface_for_device("fake", _FakeInterface)
        register_interface_for_device("fake:0", _FakeInterface)

        for fn in (
            _fake_stream,
            _fake_current_stream,
            _fake_set_stream,
            _fake_default_stream,
        ):
            self.assertIs(lookup(fn), TorchInGraphFunctionVariable)

        handlers = TorchInGraphFunctionVariable._get_handlers()
        self.assertIs(handlers[_fake_stream], handlers[torch.cuda.stream])
        self.assertIs(
            handlers[_fake_current_stream], handlers[torch.cuda.current_stream]
        )
        self.assertNotIn(_fake_set_stream, handlers)
        self.assertNotIn(_fake_default_stream, handlers)
        for base_fn in (
            DeviceInterface.stream,
            DeviceInterface.current_stream,
            DeviceInterface.set_stream,
            DeviceInterface.default_stream,
        ):
            self.assertNotIn(base_fn, registered_device_trace_fns())

        self.assertIs(lookup(torch.cuda.stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.cuda.current_stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.cuda.set_stream), TorchInGraphFunctionVariable)
        self.assertNotIn(torch.cuda.set_stream, handlers)

    def test_missing_stream_context_still_registers(self) -> None:
        from torch._dynamo.variables.builder import _is_device_stream_context

        class BareContext:
            def __init__(self) -> None:
                self.stream = None

        class MissingContextInterface(DeviceInterface):
            stream = staticmethod(_fake_stream)

        register_interface_for_device("fakemissingctx", MissingContextInterface)
        self.assertIs(lookup(_fake_stream), TorchInGraphFunctionVariable)
        self.assertFalse(_is_device_stream_context(BareContext()))

        class LateContext:
            def __init__(self) -> None:
                self.stream = None

        class WithContextInterface(DeviceInterface):
            StreamContext = LateContext
            stream = staticmethod(_fake_stream)

        register_interface_for_device("fakemissingctx", WithContextInterface)
        self.assertTrue(_is_device_stream_context(LateContext()))

    def test_missing_stream_context_compiles_direct_stream(self) -> None:
        class MissingContextInterface(DeviceInterface):
            stream = staticmethod(_unspecified_stream)

        register_interface_for_device("fakemissingctx", MissingContextInterface)
        stream = torch.Stream(device="cpu")

        def fn(x):
            context = _unspecified_stream(stream)
            is_cuda = isinstance(context, torch.cuda.StreamContext)
            with context:
                return x + 1, is_cuda

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
        self.assertEqual(result, (torch.ones(2) + 1, False))

    def test_same_key_stream_context_can_be_replaced(self) -> None:
        class FirstContext:
            pass

        class SecondContext:
            pass

        class FirstInterface(DeviceInterface):
            StreamContext = FirstContext
            stream = staticmethod(_unspecified_stream)

        class SecondInterface(DeviceInterface):
            StreamContext = SecondContext
            stream = staticmethod(_unspecified_stream)

        register_interface_for_device("fakereplace", FirstInterface)
        self.addCleanup(self._drop_registered_key, "fakereplace")
        register_interface_for_device("fakereplace", SecondInterface)
        self.assertIs(registered_stream_context_cls(_unspecified_stream), SecondContext)

    def test_current_stream_device_mismatch_graph_breaks(self) -> None:
        def other_current_stream(device=None):
            return torch.Stream(device="cpu")

        class MissingContextInterface(DeviceInterface):
            stream = staticmethod(_unspecified_stream)

        class OtherInterface(DeviceInterface):
            current_stream = staticmethod(other_current_stream)

        register_interface_for_device("fakemissingctx", MissingContextInterface)
        self.addCleanup(self._drop_registered_key, "fakemissingctx")
        register_interface_for_device("fakeother", OtherInterface)
        self.addCleanup(self._drop_registered_key, "fakeother")
        stream = torch.Stream(device="cpu")

        def fn(x):
            with _unspecified_stream(stream):
                other_current_stream()
                return x + 1

        with self.assertRaises(torch._dynamo.exc.Unsupported) as ctx:
            torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(2))
        self.assertIn("current_stream device mismatch", str(ctx.exception))

    def test_conflicting_stream_context_is_rejected(self) -> None:
        class FirstContext:
            pass

        class SecondContext:
            pass

        class InterfaceA(DeviceInterface):
            StreamContext = FirstContext
            stream = staticmethod(_fake_stream)

        class InterfaceB(DeviceInterface):
            StreamContext = SecondContext
            stream = staticmethod(_fake_stream)

        register_interface_for_device("fakea", InterfaceA)
        self.addCleanup(self._drop_registered_key, "fakea")
        with self.assertRaises(ValueError):
            register_interface_for_device("fakeb", InterfaceB)
        self.assertNotIn("fakeb", device_interfaces)

    def test_index_only_key_reaches_current_stream_handler(self) -> None:
        register_interface_for_device("fake:0", _IndexOnlyInterface)
        self.assertIs(lookup(_index_only_current_stream), TorchInGraphFunctionVariable)
        handlers = TorchInGraphFunctionVariable._get_handlers()
        self.assertIs(
            handlers[_index_only_current_stream], handlers[torch.cuda.current_stream]
        )
        self.assertNotIn(DeviceInterface.stream, registered_device_trace_fns())
        self.assertNotIn(DeviceInterface.default_stream, registered_device_trace_fns())

    def test_in_tree_overridden_streams_stay_in_graph(self) -> None:
        self.assertIs(lookup(torch.mtia.stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.mtia.current_stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.mtia.set_stream), TorchInGraphFunctionVariable)
        handlers = TorchInGraphFunctionVariable._get_handlers()
        self.assertIn(torch.mtia.stream, handlers)
        self.assertIs(handlers[torch.mtia.stream], handlers[torch.cuda.stream])
        self.assertIs(
            handlers[torch.mtia.current_stream], handlers[torch.cuda.current_stream]
        )
        self.assertNotIn(torch.mtia.set_stream, handlers)
        self.assertIs(lookup(torch.cuda.default_stream), TorchInGraphFunctionVariable)

    def test_precreated_stream_context_uses_registered_class(self) -> None:
        from torch._dynamo.variables.builder import _is_device_stream_context

        class RealContext:
            def __init__(self, stream: object) -> None:
                self.stream = stream

        class StreamContext:
            def __init__(self) -> None:
                self.stream = None

        def ctx_stream(s):
            return RealContext(s)

        class CtxInterface(DeviceInterface):
            StreamContext = RealContext
            stream = staticmethod(ctx_stream)

        decoy = StreamContext()
        real = RealContext(None)
        self.assertFalse(_is_device_stream_context(real))
        self.assertFalse(_is_device_stream_context(decoy))
        register_interface_for_device("fakectx", CtxInterface)
        self.addCleanup(self._drop_registered_key, "fakectx")
        self.assertTrue(_is_device_stream_context(real))
        self.assertFalse(_is_device_stream_context(decoy))
        self.assertIn(ctx_stream, registered_device_trace_fns())
        self.assertTrue(_is_device_stream_context(torch.cuda.StreamContext(None)))
        self.assertTrue(_is_device_stream_context(torch.xpu.StreamContext(None)))
        self.assertFalse(_is_device_stream_context(1))

    def test_indexed_only_stream_context_is_recognized(self) -> None:
        from torch._dynamo.variables.builder import _is_device_stream_context

        class IndexedContext:
            def __init__(self) -> None:
                self.stream = None

        class IndexedInterface(DeviceInterface):
            StreamContext = IndexedContext

        ctx = IndexedContext()
        self.assertFalse(_is_device_stream_context(ctx))
        register_interface_for_device("fake:0", IndexedInterface)
        self.assertTrue(_is_device_stream_context(ctx))

    def test_precreated_context_compiles_with_its_own_type(self) -> None:
        class CustomContext:
            def __init__(self, stream: torch.Stream) -> None:
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class CustomInterface(DeviceInterface):
            StreamContext = CustomContext

        ctx = CustomContext(torch.Stream(device="cpu"))
        register_interface_for_device("fakectx", CustomInterface)
        self.addCleanup(self._drop_registered_key, "fakectx")

        def fn(x):
            outside = x + 1
            is_custom = isinstance(ctx, CustomContext)
            is_cuda = isinstance(ctx, torch.cuda.StreamContext)
            with ctx:
                inside = outside + 1
            return inside + 1, is_custom, is_cuda

        x = torch.ones(2)
        eager_out, eager_custom, eager_cuda = fn(x)
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        compiled_out, compiled_custom, compiled_cuda = torch.compile(
            fn, backend=backend, fullgraph=True
        )(x)
        self.assertEqual(compiled_out, eager_out)
        self.assertEqual(eager_custom, True)
        self.assertEqual(eager_cuda, False)
        self.assertEqual(compiled_custom, True)
        self.assertEqual(compiled_cuda, False)
        self.assertEqual(len(graphs), 1)

        stream_ids = []
        for node in graphs[0].graph.nodes:
            name = getattr(node.target, "__name__", "")
            if node.op != "call_function" or name != "add":
                continue
            custom = node.meta.get("custom", {})
            stream_ids.append(custom.get("stream"))
        self.assertEqual(len(stream_ids), 3)
        outside_id, inside_id, after_id = stream_ids
        self.assertIsNotNone(inside_id)
        self.assertNotEqual(inside_id, outside_id)
        self.assertEqual(after_id, outside_id)

    def test_replacing_interface_drops_previous_fns(self) -> None:
        register_interface_for_device("fake", _FakeInterface)
        self.assertIs(lookup(_fake_stream), TorchInGraphFunctionVariable)
        self.assertIn(_fake_default_stream, registered_device_trace_fns())

        register_interface_for_device("fake", _IndexOnlyInterface)
        self.assertIsNot(lookup(_fake_stream), TorchInGraphFunctionVariable)
        self.assertNotIn(_fake_stream, registered_device_trace_fns())
        self.assertNotIn(_fake_default_stream, registered_device_trace_fns())
        self.assertIs(lookup(_index_only_current_stream), TorchInGraphFunctionVariable)

    def _drop_registered_key(self, key: str) -> None:
        device_interfaces.pop(key, None)
        get_torch_obj_rule_map.cache_clear()
        TorchInGraphFunctionVariable._get_handlers.cache_clear()


if __name__ == "__main__":
    run_tests()
