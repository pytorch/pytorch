# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    get_interface_for_device,
    get_registered_device_interfaces,
    register_interface_for_device,
)
from torch._inductor import config as inductor_config
from torch._inductor.kernel_inputs import architecture_name_from_device, MMKernelInputs
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.test_case import run_tests, TestCase


class _TensorNode:
    def __init__(self, device: torch.device) -> None:
        self._device = device

    def get_device(self) -> torch.device:
        return self._device

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_size(self) -> tuple[int, ...]:
        return (2, 2)

    def get_stride(self) -> tuple[int, ...]:
        return (2, 1)


class _TestMMKernelInputs(MMKernelInputs):
    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        return self.shapes_symbolic()

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        return self.strides_symbolic()


class _NameOnlyInterface(DeviceInterface):
    @staticmethod
    def get_device_properties(device=None):
        return SimpleNamespace(name="TestArch910", gcnArchName=None)

    @staticmethod
    def is_available() -> bool:
        return True


class _LookupArchInterface(DeviceInterface):
    @staticmethod
    def get_lookup_architecture(device=None) -> str | None:
        return "TestArch"

    @staticmethod
    def is_available() -> bool:
        return True


class TestKernelInputsDeviceName(TestCase):
    def setUp(self):
        super().setUp()
        self._prev_pua = device_interfaces.get("privateuseone")

    def tearDown(self):
        if self._prev_pua is None:
            device_interfaces.pop("privateuseone", None)
        else:
            device_interfaces["privateuseone"] = self._prev_pua
        super().tearDown()

    def test_device_name_matches_device_interface(self):
        for name, device_interface in get_registered_device_interfaces():
            if ":" in name or not device_interface.is_available():
                continue
            device = torch.device(name)
            with self.subTest(device=str(device)):
                node = _TensorNode(device)
                expected = get_interface_for_device(device).get_lookup_architecture(
                    device
                )
                self.assertEqual(MMKernelInputs([node, node]).device_name(), expected)

    def test_unregistered_device_returns_none(self):
        self.assertIsNone(architecture_name_from_device(torch.device("meta")))

    def test_runtime_error_from_backend_is_not_swallowed(self):
        calls = {"n": 0}

        class _FlakyLookupArchInterface(DeviceInterface):
            @staticmethod
            def get_lookup_architecture(device=None) -> str | None:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("driver init")
                return "TestArch"

            @staticmethod
            def is_available() -> bool:
                return True

        register_interface_for_device("privateuseone", _FlakyLookupArchInterface)
        device = torch.device("privateuseone:0")
        with self.assertRaisesRegex(RuntimeError, "driver init"):
            architecture_name_from_device(device)
        self.assertEqual(architecture_name_from_device(device), "TestArch")
        self.assertEqual(calls["n"], 2)

    def test_name_only_does_not_enter_lookup(self):
        register_interface_for_device("privateuseone", _NameOnlyInterface)
        device = torch.device("privateuseone:0")
        self.assertIsNone(architecture_name_from_device(device))
        node = _TensorNode(device)
        kernel_inputs = _TestMMKernelInputs([node, node])
        self.assertIsNone(MMKernelInputs([node, node]).device_name())
        choices = LookupTableChoices()
        self.assertIsNone(choices._get_device_key(device))
        specific = choices.make_lookup_key(kernel_inputs, "mm", include_device=True)
        agnostic = choices.make_lookup_key(kernel_inputs, "mm", include_device=False)
        self.assertIsNone(specific)
        self.assertIsNone(agnostic)

    def test_lookup_architecture_hits_table_without_cuda(self):
        register_interface_for_device("privateuseone", _LookupArchInterface)
        device = torch.device("privateuseone:0")
        node = _TensorNode(device)
        kernel_inputs = _TestMMKernelInputs([node, node])
        choices = LookupTableChoices()
        lookup_key = choices.make_lookup_key(kernel_inputs, "mm", include_device=True)
        self.assertIsNotNone(lookup_key)
        self.assertTrue(lookup_key.startswith("TestArch+"))
        self.assertTrue(lookup_key.endswith("+mm"))
        self.assertEqual(kernel_inputs.device_name(), "TestArch")
        table = {
            lookup_key: [{"template_id": "triton", "BLOCK_M": 16}],
        }
        original = inductor_config.lookup_table.table
        inductor_config.lookup_table.table = table
        try:
            with patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(choices._get_lookup_table(), table)
                configs = choices.lookup_template_configs(
                    kernel_inputs, "mm", ["triton"]
                )
            self.assertEqual(configs["triton"][0]["BLOCK_M"], 16)
        finally:
            inductor_config.lookup_table.table = original


if __name__ == "__main__":
    run_tests()
