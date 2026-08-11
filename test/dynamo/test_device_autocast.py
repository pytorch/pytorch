# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import device_interface
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device
from torch._dynamo.variables.torch import _matches_device_autocast_class


class _StubAutocast(torch.amp.autocast_mode.autocast):
    """An out-of-tree backend's autocast, whose device_type is implicit in
    the class itself (omitted from the constructor)."""

    def __init__(self, dtype=torch.float16, enabled=True, cache_enabled=None):
        super().__init__("cpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)


class AutocastInterface(DeviceInterface):
    autocast_classes = {_StubAutocast: "cpu"}


class DeviceAutocastTests(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        device_interface.device_interfaces.pop("stubac", None)
        super().tearDown()

    def test_matches_only_when_registered(self):
        # Positive source-file matching is deliberately not issubclass-based:
        # an arbitrary autocast subclass must not be picked up unless its
        # owning device registered it.
        self.assertFalse(_matches_device_autocast_class(_StubAutocast))

        register_interface_for_device("stubac", AutocastInterface)
        self.assertTrue(_matches_device_autocast_class(_StubAutocast))

    def test_registered_autocast_class_traced(self):
        register_interface_for_device("stubac", AutocastInterface)

        counter = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x):
            with _StubAutocast():
                pass
            return x + 1

        fn(torch.ones(2))
        self.assertEqual(counter.frame_count, 1)

    def test_internal_autocast_subclass_not_matched(self):
        # _UnmanagedAutocast is the class _enter_autocast() constructs during
        # pre-dispatch tracing.  It subclasses autocast but belongs to no
        # device, so a too-broad match would route it to AutocastModeVariable
        # and blow up test__enter__exit_autocast with "setattr() on
        # unsupported type".  This pins the positive source-file matching.
        from torch.amp.autocast_mode import _UnmanagedAutocast

        self.assertFalse(_matches_device_autocast_class(_UnmanagedAutocast))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
