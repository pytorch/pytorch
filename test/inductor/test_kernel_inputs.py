# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.device_interface import (
    get_interface_for_device,
    get_registered_device_interfaces,
)
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.test_case import run_tests, TestCase


class _TensorNode:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def get_device(self) -> torch.device:
        return self.tensor.device


class TestKernelInputsDeviceName(TestCase):
    def test_device_name_matches_device_interface(self):
        for name, device_interface in get_registered_device_interfaces():
            if ":" in name or not device_interface.is_available():
                continue
            device = torch.device(name)
            with self.subTest(device=str(device)):
                node = _TensorNode(torch.empty(2, 2, device=device))
                try:
                    iface = get_interface_for_device(device)
                    props = iface.get_device_properties(device)
                except NotImplementedError:
                    expected = None
                else:
                    expected = getattr(props, "gcnArchName", None)
                self.assertEqual(MMKernelInputs([node, node]).device_name(), expected)


if __name__ == "__main__":
    run_tests()
