# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestPinMemory(TestCase):
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        """Test pin memory for tensors, storage, and untyped storage"""
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

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_different_devices(self):
        """Test pin memory on different devices"""
        tensor = torch.randn(10)
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        
        # Test pinning to specific device
        pinned_tensor_openreg = tensor.pin_memory("openreg")
        self.assertTrue(pinned_tensor_openreg.is_pinned("openreg"))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_view(self):
        """Test pin memory with tensor views"""
        tensor = torch.randn(20)
        pinned_tensor = tensor.pin_memory()
        
        # Test various views
        view1 = pinned_tensor[2:5]
        self.assertTrue(view1.is_pinned())
        
        view2 = pinned_tensor[::2]
        self.assertTrue(view2.is_pinned())
        
        view3 = pinned_tensor.view(4, 5)
        self.assertTrue(view3.is_pinned())

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory_storage_sharing(self):
        """Test pin memory with shared storage"""
        tensor = torch.randn(10)
        pinned_tensor = tensor.pin_memory()
        
        # Create another tensor sharing the same storage
        shared_tensor = torch.tensor((), dtype=torch.float32).set_(
            pinned_tensor.storage()
        )
        self.assertTrue(shared_tensor.is_pinned())


if __name__ == "__main__":
    run_tests()
