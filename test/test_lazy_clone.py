# Owner(s): ["module: copy on write"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestLazyCloneDeviceType(TestCase):
    def skip_if_lt_two_accelerators(self):
        if torch.accelerator.device_count() < 2:
            self.skipTest("Only one accelerator device found")

    def get_src_dest_devices(self, case, device):
        device_type = torch.device(device).type

        if device_type == "cpu" and case != "to_same_device":
            self.skipTest("Only case='to_same_device' is run for CPU device")

        if case == "to_cpu":
            src_device = device_type
            dest_device = "cpu"
        elif case == "from_cpu":
            src_device = "cpu"
            dest_device = device_type
        elif case == "to_same_device":
            src_device = device_type
            dest_device = device_type
        elif case == "from_0_to_1":
            self.skip_if_lt_two_accelerators()
            src_device = f"{device_type}:0"
            dest_device = f"{device_type}:1"
        elif case == "from_1_to_0":
            self.skip_if_lt_two_accelerators()
            src_device = f"{device_type}:1"
            dest_device = f"{device_type}:0"
        else:
            raise AssertionError(f"case='{case}' not recognized")

        return src_device, dest_device

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize("materialize_first", ("src", "dest"))
    @parametrize(
        "case",
        [
            "to_cpu",
            "from_cpu",
            "to_same_device",
            "from_0_to_1",
            "from_1_to_0",
        ],
    )
    def test_interdevice_materialize(self, device, materialize_first, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        a = torch.randn(10, device=src_device, pin_memory=pin_memory)
        orig_data_ptr = torch._C._data_address_resolve_unified(a)
        b = a._lazy_clone(device=dest_device)

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        if materialize_first == "src":
            a.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(
                torch._C._data_address_resolve_unified(a), orig_data_ptr
            )
            self.assertTrue(torch._C._is_cow_tensor(b))
            self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

            b.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(
                torch._C._data_address_resolve_unified(a), orig_data_ptr
            )
            self.assertFalse(torch._C._is_cow_tensor(b))
            if src_device_check == dest_device_check:
                self.assertEqual(
                    torch._C._data_address_resolve_unified(b), orig_data_ptr
                )
            else:
                self.assertNotEqual(
                    torch._C._data_address_resolve_unified(b), orig_data_ptr
                )

        elif materialize_first == "dest":
            b.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(
                torch._C._data_address_resolve_unified(b), orig_data_ptr
            )
            self.assertTrue(torch._C._is_cow_tensor(a))
            self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)

            a.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(
                torch._C._data_address_resolve_unified(b), orig_data_ptr
            )
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)

        else:
            raise RuntimeError(f"Not recognized: materialize_first={materialize_first}")

    # Test that COW a tensor with a different target device can be used in read
    # operations.
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize(
        "case",
        [
            "to_cpu",
            "from_cpu",
            "to_same_device",
            "from_0_to_1",
            "from_1_to_0",
        ],
    )
    def test_interdevice_read(self, device, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        orig_tensor = torch.randn(10, device=src_device)
        a = torch.empty(10, device=src_device, pin_memory=pin_memory)
        a.copy_(orig_tensor)

        orig_data_ptr = torch._C._data_address_resolve_unified(a)
        b = a._lazy_clone(device=dest_device)

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        a_clone = a.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        b_clone = b.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        self.assertEqual(a_clone, b_clone)
        self.assertTrue((a == orig_tensor.to(a.device)).all())
        self.assertTrue((b == orig_tensor.to(b.device)).all())
        self.assertEqual(a.cpu(), b.cpu())
        self.assertEqual(a, b)
        self.assertTrue((a.clone() == orig_tensor.to(a.device)).all())
        self.assertTrue((b.clone() == orig_tensor.to(b.device)).all())
        self.assertEqual(a.clone().cpu(), b.clone().cpu())

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

    def test_clone_after_lazy_clone(self, device):
        a = torch.randn(10, device=device)
        orig_data_ptr = torch._C._data_address_resolve_unified(a)
        b = torch._lazy_clone(a)

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        c = b.clone()

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertFalse(torch._C._is_cow_tensor(c))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        self.assertTrue((b == c).all())
        self.assertTrue((a == c).all())
        self.assertTrue((b.clone() == c).all())
        self.assertTrue((a.clone() == c).all())
        self.assertTrue((b.clone() == c.clone()).all())
        self.assertTrue((a.clone() == c.clone()).all())

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertFalse(torch._C._is_cow_tensor(c))
        self.assertEqual(torch._C._data_address_resolve_unified(a), orig_data_ptr)
        self.assertEqual(torch._C._data_address_resolve_unified(b), orig_data_ptr)

        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(b, c)


instantiate_device_type_tests(
    TestLazyCloneDeviceType, globals(), allow_mps=True, only_for=["cpu", "mps"]
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
