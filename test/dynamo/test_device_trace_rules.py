# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._dynamo.device_trace_rules import _registered_device_trace_fns
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


def _index_only_current_stream(device=None):
    return device


class _FakeInterface(DeviceInterface):
    stream = staticmethod(_fake_stream)
    current_stream = staticmethod(_fake_current_stream)
    set_stream = staticmethod(_fake_set_stream)
    default_stream = staticmethod(_fake_default_stream)


class _IndexOnlyInterface(DeviceInterface):
    current_stream = staticmethod(_index_only_current_stream)


_FAKE_FNS = (
    _fake_stream,
    _fake_current_stream,
    _fake_set_stream,
    _fake_default_stream,
    _index_only_current_stream,
)


class TestDeviceTraceRules(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        # Warm both caches so registration has to invalidate them.
        get_torch_obj_rule_map()
        TorchInGraphFunctionVariable._get_handlers()

    def tearDown(self) -> None:
        for key in ("fake", "fake:0"):
            device_interfaces.pop(key, None)
        for fn in _FAKE_FNS:
            _registered_device_trace_fns.discard(fn)
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
        ):
            self.assertNotIn(base_fn, _registered_device_trace_fns)

        self.assertIs(lookup(torch.cuda.stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.cuda.current_stream), TorchInGraphFunctionVariable)
        self.assertIs(lookup(torch.cuda.set_stream), TorchInGraphFunctionVariable)
        self.assertNotIn(torch.cuda.set_stream, handlers)

    def test_index_only_key_reaches_current_stream_handler(self) -> None:
        register_interface_for_device("fake:0", _IndexOnlyInterface)
        self.assertIs(lookup(_index_only_current_stream), TorchInGraphFunctionVariable)
        handlers = TorchInGraphFunctionVariable._get_handlers()
        self.assertIs(
            handlers[_index_only_current_stream], handlers[torch.cuda.current_stream]
        )
        self.assertNotIn(DeviceInterface.stream, _registered_device_trace_fns)

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

    def test_precreated_stream_context_uses_registered_module(self) -> None:
        from torch._dynamo.variables.builder import _is_device_stream_context

        class StreamContext:
            def __init__(self) -> None:
                self.stream = None

        def ctx_stream(s):
            return StreamContext()

        class CtxInterface(DeviceInterface):
            pass

        CtxInterface.stream = staticmethod(ctx_stream)

        self.assertFalse(_is_device_stream_context(StreamContext()))
        register_interface_for_device("fakectx", CtxInterface)
        self.addCleanup(device_interfaces.pop, "fakectx", None)
        self.assertTrue(_is_device_stream_context(StreamContext()))
        self.assertTrue(_is_device_stream_context(torch.cuda.StreamContext(None)))
        self.assertTrue(_is_device_stream_context(torch.xpu.StreamContext(None)))
        self.assertFalse(_is_device_stream_context(1))


if __name__ == "__main__":
    run_tests()
