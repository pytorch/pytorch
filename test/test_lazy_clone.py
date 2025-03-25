# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
    skipXLA,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestLazyCloneDeviceType(TestCase):
    def skip_if_lt_two_devices(self, device_type):
        if device_type == "cuda":
            if torch.cuda.device_count() < 2:
                self.skipTest("Only one CUDA device found")
        elif device_type == "mps":
            if torch.mps.device_count() < 2:
                self.skipTest("Only one MPS device found")
        else:
            self.skipTest(f"Index not supported for device type {device_type}")

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
            self.skip_if_lt_two_devices(device_type)
            src_device = f"{device_type}:0"
            dest_device = f"{device_type}:1"
        elif case == "from_1_to_0":
            self.skip_if_lt_two_devices(device_type)
            src_device = f"{device_type}:1"
            dest_device = f"{device_type}:0"
        else:
            assert False

        return src_device, dest_device

    # TODO: Add test compatible with dynamo/inductor
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @skipXLA
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
    def test_lazy_clone_to_device(self, device, materialize_first, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        a = torch.randn(10, device=src_device, pin_memory=pin_memory)
        orig_data_ptr = a.data_ptr()
        b = a._lazy_clone(device=dest_device)

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address(b), orig_data_ptr)

        if materialize_first == "src":
            a[0] = 1

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(torch._C._data_address(a), orig_data_ptr)
            self.assertTrue(torch._C._is_cow_tensor(b))
            self.assertEqual(torch._C._data_address(b), orig_data_ptr)
            self.assertEqual(a[0], 1)

            b[0] = 2

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(torch._C._data_address(a), orig_data_ptr)
            self.assertFalse(torch._C._is_cow_tensor(b))
            if src_device_check == dest_device_check:
                self.assertEqual(torch._C._data_address(b), orig_data_ptr)
            else:
                self.assertNotEqual(torch._C._data_address(b), orig_data_ptr)
            self.assertEqual(a[0], 1)
            self.assertEqual(b[0], 2)

        elif materialize_first == "dest":
            b[0] = 2

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(torch._C._data_address(b), orig_data_ptr)
            self.assertTrue(torch._C._is_cow_tensor(a))
            self.assertEqual(torch._C._data_address(a), orig_data_ptr)
            self.assertEqual(b[0], 2)

            a[0] = 1

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(torch._C._data_address(b), orig_data_ptr)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertEqual(torch._C._data_address(a), orig_data_ptr)

            self.assertEqual(a[0], 1)
            self.assertEqual(b[0], 2)

        else:
            raise RuntimeError(f"Not recognized: materialize_first={materialize_first}")

    # Test that COW a tensor with a different target device can be used in read
    # operations.
    @skipCUDAIf(True, "Does not work for CUDA")
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @skipXLA
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
    def test_lazy_clone_to_device_read(self, device, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        a = torch.randn(10, device=src_device, pin_memory=pin_memory)
        orig_data_ptr = a.data_ptr()
        b = a._lazy_clone(device=dest_device)

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address(b), orig_data_ptr)

        a_clone = a.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address(b), orig_data_ptr)

        b_clone = b.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address(b), orig_data_ptr)

        self.assertEqual(a_clone, b_clone)

        self.assertEqual(a, b)

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(torch._C._data_address(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(torch._C._data_address(b), orig_data_ptr)


instantiate_device_type_tests(TestLazyCloneDeviceType, globals(), allow_mps=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
