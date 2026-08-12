# Owner(s): ["module: dynamo"]

import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import device_interface
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    get_registered_device_interfaces,
    MpsInterface,
    MtiaInterface,
    register_interface_for_device,
    XpuInterface,
)
from torch._dynamo.variables.user_defined import UserDefinedClassVariable


class _StubStream(torch.Stream):
    """A well-behaved backend stream: subclasses torch.Stream."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


class _StubEvent(torch.Event):
    """A well-behaved backend event: subclasses torch.Event."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)


class _StubTensorType:
    """Stand-in for a backend tensor constructor such as torch.foo.FloatTensor."""


class GoodStubInterface(DeviceInterface):
    Stream = _StubStream
    Event = _StubEvent
    tensor_types = frozenset({_StubTensorType})


class BadStubInterface(DeviceInterface):
    """A backend that forgot to subclass torch.Stream/torch.Event.

    It inherits DeviceInterface's placeholders, which only raise
    NotImplementedError, so they must never reach _in_graph_classes().
    """


class InGraphClassesTests(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        for name in ("stub", "badstub"):
            device_interface.device_interfaces.pop(name, None)
        UserDefinedClassVariable._in_graph_classes.cache_clear()
        super().tearDown()

    def _register(self, name, iface):
        register_interface_for_device(name, iface)
        self.assertIs(
            dict(get_registered_device_interfaces())[name],
            iface,
        )

    def test_registered_interface_stream_event_in_graph(self):
        # Prime the cache first: this is the out-of-tree backend's situation,
        # where registration happens lazily, after Dynamo has already run.
        before = UserDefinedClassVariable._in_graph_classes()
        self.assertNotIn(_StubStream, before)
        self.assertNotIn(_StubEvent, before)

        self._register("stub", GoodStubInterface)

        after = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(_StubStream, after)
        self.assertIn(_StubEvent, after)

    def test_registered_interface_tensor_types_in_graph(self):
        self.assertNotIn(_StubTensorType, UserDefinedClassVariable._in_graph_classes())

        self._register("stub", GoodStubInterface)

        self.assertIn(_StubTensorType, UserDefinedClassVariable._in_graph_classes())

    def test_base_class_placeholders_never_in_graph(self):
        self._register("badstub", BadStubInterface)

        in_graph = UserDefinedClassVariable._in_graph_classes()
        # The placeholders raise NotImplementedError with a message telling the
        # backend to subclass torch.Stream/torch.Event.  Putting them in this
        # set would trade that message for an opaque tracing failure.
        self.assertNotIn(DeviceInterface.Stream, in_graph)
        self.assertNotIn(DeviceInterface.Event, in_graph)
        self.assertNotIn(BadStubInterface.Stream, in_graph)
        self.assertNotIn(BadStubInterface.Event, in_graph)

        for cls in in_graph:
            self.assertIsInstance(cls, type)

    def test_tensor_types_defaults_to_empty(self):
        # The base-class default must be inert, so declaring it changes nothing
        # for backends that do not opt in.  Checked against the in-tree
        # interfaces specifically: an out-of-tree backend may well be
        # registered, and is entitled to declare tensor_types.
        self.assertEqual(DeviceInterface.tensor_types, frozenset())
        for iface in (
            CudaInterface,
            XpuInterface,
            MtiaInterface,
            CpuInterface,
            MpsInterface,
        ):
            self.assertEqual(iface.tensor_types, frozenset())
            self.assertNotIn("tensor_types", vars(iface))

    def test_late_registration_invalidates_cache(self):
        # _in_graph_classes() is functools.cache'd over a mutable registry, so
        # register_interface_for_device() has to invalidate it.
        UserDefinedClassVariable._in_graph_classes()
        self._register("stub", GoodStubInterface)
        self.assertIn(_StubStream, UserDefinedClassVariable._in_graph_classes())

    def test_stream_construction_traced_after_registration(self):
        self._register("stub", GoodStubInterface)

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            _StubStream()
            return x + 1

        fn(torch.ones(2))
        self.assertEqual(counter.frame_count, 1)

    @unittest.skipIf(not torch.cuda._is_compiled(), "CUDA not compiled in")
    def test_cuda_stream_event_still_in_graph(self):
        # torch.cuda.Stream/Event used to be listed here by hand.  CudaInterface
        # supplies them now, so the set must be unchanged for a CUDA build.
        # _CudaStreamBase's tp_base is torch.Stream, so the issubclass() check
        # in the loop admits them.
        in_graph = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(torch.cuda.Stream, in_graph)
        self.assertIn(torch.cuda.Event, in_graph)

    @unittest.skipIf(not torch.xpu._is_compiled(), "XPU not compiled in")
    def test_xpu_stream_event_still_in_graph(self):
        in_graph = UserDefinedClassVariable._in_graph_classes()
        self.assertIn(torch.xpu.Stream, in_graph)
        self.assertIn(torch.xpu.Event, in_graph)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
