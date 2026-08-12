# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import device_interface
from torch._dynamo.device_interface import (
    DeviceInterface,
    get_device_autocast_classes,
    register_interface_for_device,
)
from torch._dynamo.variables.torch import (
    _matches_device_autocast_class,
    device_type_for_autocast_class,
)


STUB_DEVICE = "privateuseone"


class _StubAutocast(torch.amp.autocast_mode.autocast):
    """An out-of-tree backend's autocast with an implicit device_type."""

    def __init__(self, dtype=torch.float16, enabled=True, cache_enabled=None):
        super().__init__(
            STUB_DEVICE, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )


class AutocastInterface(DeviceInterface):
    autocast_classes = frozenset({_StubAutocast})


class DeviceAutocastTests(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        for device in (STUB_DEVICE, f"{STUB_DEVICE}:3"):
            device_interface.device_interfaces.pop(device, None)
        get_device_autocast_classes.cache_clear()
        super().tearDown()

    def test_slot_defaults_to_empty(self):
        self.assertEqual(DeviceInterface.autocast_classes, frozenset())

    def test_matches_only_when_registered(self):
        self.assertFalse(_matches_device_autocast_class(_StubAutocast))

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertTrue(_matches_device_autocast_class(_StubAutocast))

    def test_device_type_derived_from_registration_key(self):
        self.assertIsNone(device_type_for_autocast_class(_StubAutocast))

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)

    def test_device_type_strips_index_from_key(self):
        register_interface_for_device(f"{STUB_DEVICE}:3", AutocastInterface)

        self.assertEqual(device_type_for_autocast_class(_StubAutocast), STUB_DEVICE)

    def test_late_registration_invalidates_cache(self):
        get_device_autocast_classes()

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertIn(_StubAutocast, get_device_autocast_classes())

    def test_base_autocast_never_matches(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(
            _matches_device_autocast_class(torch.amp.autocast_mode.autocast)
        )

    def test_registered_autocast_class_traced(self):
        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            with _StubAutocast():
                return x + 1

        fn(torch.ones(2))
        self.assertEqual(len(graphs), 1)
        enters = [
            node
            for node in graphs[0].graph.nodes
            if node.op == "call_function" and node.target is torch.amp._enter_autocast
        ]
        self.assertEqual(len(enters), 1)
        self.assertEqual(enters[0].args[0], STUB_DEVICE)

    def test_internal_autocast_subclass_not_matched(self):
        from torch.amp.autocast_mode import _UnmanagedAutocast

        register_interface_for_device(STUB_DEVICE, AutocastInterface)

        self.assertFalse(_matches_device_autocast_class(_UnmanagedAutocast))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
