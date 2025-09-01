# Owner(s): ["module: PrivateUse1"]

import torch
import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestPinMemory(TestCase):
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        tensor = torch.randn(10)
        self.assertFalse(tensor.is_pinned())
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        slice_tensor = pinned_tensor[2:5]
        self.assertTrue(slice_tensor.is_pinned())

        tensor = torch.randn(10)
        storage = tensor.storage()
        self.assertFalse(storage.is_pinned("openreg"))
        pinned_storage = storage.pin_memory("openreg")
        self.assertTrue(pinned_storage.is_pinned("openreg"))

        tensor = torch.randn(10)
        untyped_storage = tensor.untyped_storage()
        self.assertFalse(untyped_storage.is_pinned("openreg"))
        pinned_untyped_storage = untyped_storage.pin_memory("openreg")
        self.assertTrue(pinned_untyped_storage.is_pinned("openreg"))


class TestCopy(TestCase):
    def test_copy_same_device(self):
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_device_copy(self):
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


if __name__ == "__main__":
    run_tests()
