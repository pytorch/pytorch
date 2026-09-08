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


class TestKernelInputsDeviceName(TestCase):
    def setUp(self):
        super().setUp()
        architecture_name_from_device.cache_clear()
        self._prev_pua = device_interfaces.get("privateuseone")

    def tearDown(self):
        architecture_name_from_device.cache_clear()
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
                try:
                    iface = get_interface_for_device(device)
                    props = iface.get_device_properties(device)
                except NotImplementedError:
                    expected = None
                else:
                    expected = getattr(props, "gcnArchName", None) or getattr(
                        props, "name", None
                    )
                self.assertEqual(MMKernelInputs([node, node]).device_name(), expected)

    def test_unregistered_device_returns_none(self):
        with patch(
            "torch._inductor.kernel_inputs.get_interface_for_device",
            side_effect=NotImplementedError,
        ):
            self.assertIsNone(architecture_name_from_device(torch.device("cpu")))

    def test_missing_gcn_arch_falls_back_to_name(self):
        register_interface_for_device("privateuseone", _NameOnlyInterface)
        device = torch.device("privateuseone:0")
        self.assertEqual(architecture_name_from_device(device), "TestArch910")
        node = _TensorNode(device)
        self.assertEqual(MMKernelInputs([node, node]).device_name(), "TestArch910")

    def test_unimplemented_properties_return_none(self):
        class _Broken(DeviceInterface):
            @staticmethod
            def get_device_properties(device=None):
                raise NotImplementedError

            @staticmethod
            def is_available() -> bool:
                return True

        register_interface_for_device("privateuseone", _Broken)
        self.assertIsNone(
            architecture_name_from_device(torch.device("privateuseone:0"))
        )

    def test_lookup_table_uses_name_without_cuda_gate(self):
        register_interface_for_device("privateuseone", _NameOnlyInterface)
        choices = LookupTableChoices()
        self.assertEqual(
            choices._get_device_key(torch.device("privateuseone:0")), "TestArch910"
        )
        original = inductor_config.lookup_table.table
        inductor_config.lookup_table.table = {"placeholder": []}
        try:
            with patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(choices._get_lookup_table(), {"placeholder": []})
        finally:
            inductor_config.lookup_table.table = original

    def test_lookup_key_hits_table_config(self):
        register_interface_for_device("privateuseone", _NameOnlyInterface)
        device = torch.device("privateuseone:0")
        node = _TensorNode(device)
        kernel_inputs = _TestMMKernelInputs([node, node])
        choices = LookupTableChoices()
        lookup_key = choices.make_lookup_key(kernel_inputs, "mm", include_device=True)
        self.assertIsNotNone(lookup_key)
        self.assertTrue(lookup_key.startswith("TestArch910+"))
        self.assertTrue(lookup_key.endswith("+mm"))
        self.assertEqual(kernel_inputs.device_name(), "TestArch910")
        table = {
            lookup_key: [{"template_id": "triton", "BLOCK_M": 16}],
        }
        original = inductor_config.lookup_table.table
        inductor_config.lookup_table.table = table
        try:
            with patch("torch.cuda.is_available", return_value=False):
                configs = choices.lookup_template_configs(
                    kernel_inputs, "mm", ["triton"]
                )
            self.assertEqual(configs["triton"][0]["BLOCK_M"], 16)
        finally:
            inductor_config.lookup_table.table = original


if __name__ == "__main__":
    run_tests()
