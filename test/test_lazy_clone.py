# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipXLA,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestLazyCloneDeviceType(TestCase):
    # TODO: Add test compatible with dynamo/inductor
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @skipXLA
    @parametrize(
        "src,dest",
        [
            # NOTE: `None` indicates the device specified by the `device` arg of
            # the test. A number indicates a device index of the same device
            # type means use the device type as specified by the `device` arg.
            ("cpu", None),
            (None, None),
            ("cpu", 0),
            ("cpu", 1),
            (1, 0),
            (0, 1),
            (None, "cpu"),
        ],
    )
    @parametrize("materialize_first", ("src", "dest"))
    def test_lazy_clone_to_device(self, device, src, dest, materialize_first):
        device_type = torch.device(device).type

        def get_device_str(arg):
            if isinstance(arg, str):
                return arg
            elif isinstance(arg, int):
                if device_type == "cuda":
                    if arg >= torch.cuda.device_count():
                        self.skipTest(f"CUDA index {arg} not found")
                elif device_type == "mps":
                    if arg >= torch.mps.device_count():
                        self.skipTest(f"MPS index {arg} not found")
                else:
                    self.skipTest(f"Index not supported for device type {device_type}")

                return f"{device_type}:{arg}"
            elif arg is None:
                return device_type
            else:
                raise AssertionError(f"Test parameter not recognized: {arg}")

        src_device = get_device_str(src)
        dest_device = get_device_str(dest)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device

        a = torch.randn(10, device=src_device)
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


instantiate_device_type_tests(TestLazyCloneDeviceType, globals(), allow_mps=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
