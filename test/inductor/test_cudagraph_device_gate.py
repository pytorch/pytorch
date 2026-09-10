# Owner(s): ["module: inductor"]

from unittest import mock

import torch
from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor.cudagraph_utils import (
    _graph_capture_compatible_device_type,
    check_multiple_devices_or_any_cpu_nodes,
)
from torch._inductor.test_case import run_tests, TestCase


class _FakeNode:
    name = "n"


class _FakeDevice:
    def __init__(self, device_type: str, index: int = 0) -> None:
        self.type = device_type
        self.index = index

    def __repr__(self) -> str:
        return f"{self.type}:{self.index}"


class _RegisteredInterface(DeviceInterface):
    @staticmethod
    def is_available() -> bool:
        return True


class TestGraphCaptureCompatibleDeviceType(TestCase):
    def test_cuda(self):
        self.assertTrue(_graph_capture_compatible_device_type("cuda"))

    def test_default_privateuseone_slot_unused(self):
        with mock.patch(
            "torch._C._get_privateuse1_backend_name", return_value="privateuseone"
        ):
            self.assertFalse(_graph_capture_compatible_device_type("privateuseone"))

    def test_renamed_without_interface(self):
        with mock.patch("torch._C._get_privateuse1_backend_name", return_value="npu"):
            self.assertFalse(_graph_capture_compatible_device_type("npu"))

    def test_renamed_with_interface(self):
        register_interface_for_device("npu", _RegisteredInterface)
        try:
            with mock.patch(
                "torch._C._get_privateuse1_backend_name", return_value="npu"
            ):
                self.assertTrue(_graph_capture_compatible_device_type("npu"))
        finally:
            device_interfaces.pop("npu", None)


class TestCudagraphDeviceGate(TestCase):
    def test_single_cuda_allowed(self):
        node = _FakeNode()
        mapping = {torch.device("cuda:0"): node}
        self.assertIsNone(check_multiple_devices_or_any_cpu_nodes(mapping))

    def test_single_xpu_skipped(self):
        node = _FakeNode()
        mapping = {torch.device("xpu:0"): node}
        msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)

    def test_renamed_privateuse1_with_interface_allowed(self):
        register_interface_for_device("npu", _RegisteredInterface)
        try:
            node = _FakeNode()
            mapping = {_FakeDevice("npu"): node}
            with mock.patch(
                "torch._C._get_privateuse1_backend_name", return_value="npu"
            ):
                self.assertIsNone(check_multiple_devices_or_any_cpu_nodes(mapping))
        finally:
            device_interfaces.pop("npu", None)

    def test_renamed_privateuse1_without_interface_skipped(self):
        node = _FakeNode()
        mapping = {_FakeDevice("npu"): node}
        with mock.patch("torch._C._get_privateuse1_backend_name", return_value="npu"):
            msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)

    def test_default_privateuseone_skipped(self):
        node = _FakeNode()
        mapping = {torch.device("privateuseone:0"): node}
        with mock.patch(
            "torch._C._get_privateuse1_backend_name", return_value="privateuseone"
        ):
            msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)


if __name__ == "__main__":
    run_tests()
