# mypy: allow-untyped-defs
"""Device-gate tests for torch/_inductor/mkldnn_ir.py.

Pure CPU, deterministic, no hardware required: in-tree mkldnn devices are
resolved from the SUPPORTED_MKLDNN_DEVICES snapshot without touching device
APIs, and out-of-tree opt-in is exercised with a temporarily registered
DeviceInterface (restored via addCleanup).
"""

import torch._dynamo.device_interface as device_interface
from torch._inductor.mkldnn_ir import _is_mkldnn_supported_device
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class FakeMkldnnBackend(device_interface.DeviceInterface):
    @staticmethod
    def is_mkldnn_capable():
        return True


class FakeNonMkldnnBackend(device_interface.DeviceInterface):
    @staticmethod
    def is_mkldnn_capable():
        return False


class FakePlainBackend(device_interface.DeviceInterface):
    pass


class TestMkldnnDeviceGate(TestCase):
    # Label shape follows the in-flight test-tagging PRs (#192987/#193098);
    # adjust to the landed form if it differs at submission time.
    hw_classification = HardwareClassification.GENERIC

    def test_in_tree_mkldnn_devices_allowed(self):
        # Snapshot membership short-circuits before any interface lookup:
        # deterministic on any machine, no hardware needed.
        self.assertTrue(_is_mkldnn_supported_device("cpu"))
        self.assertTrue(_is_mkldnn_supported_device("xpu"))

    def test_cuda_rejected(self):
        # cuda has no mkldnn kernels and declares no capability hook.
        self.assertFalse(_is_mkldnn_supported_device("cuda"))

    def test_unknown_device_rejected(self):
        # No registered interface -> get_interface_for_device raises
        # NotImplementedError -> gate stays closed (same as before this PR).
        self.assertFalse(_is_mkldnn_supported_device("no_such_device"))

    def test_capability_hook_opt_in(self):
        device_interface.register_interface_for_device("fakeoot", FakeMkldnnBackend)
        self.addCleanup(device_interface.device_interfaces.pop, "fakeoot", None)
        self.assertTrue(_is_mkldnn_supported_device("fakeoot"))

    def test_capability_hook_false_rejected(self):
        # An explicit "not capable" declaration keeps the gate closed.
        device_interface.register_interface_for_device(
            "fakecap", FakeNonMkldnnBackend
        )
        self.addCleanup(device_interface.device_interfaces.pop, "fakecap", None)
        self.assertFalse(_is_mkldnn_supported_device("fakecap"))

    def test_interface_without_hook_rejected(self):
        # npu-today semantics: registered interface without the hook falls
        # back to False instead of crashing or opening the gate.
        device_interface.register_interface_for_device(
            "fakeplain", FakePlainBackend
        )
        self.addCleanup(device_interface.device_interfaces.pop, "fakeplain", None)
        self.assertFalse(_is_mkldnn_supported_device("fakeplain"))


if __name__ == "__main__":
    run_tests()
