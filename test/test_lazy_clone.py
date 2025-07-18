# Owner(s): ["module: copy on write"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


def data_address_unified(a, device="mps"):
    return torch._C._data_address_unified_mps(a, torch.device(device))


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
    @parametrize(
        "no_device_arg",
        [True, False],
    )
    @parametrize(
        "dest_device",
        ["cpu", "mps"],
    )
    # Ensures correct Copy-on-write semantics specifically for CPU tensors
    # pinned to unified MPS memory.
    def test_lazy_clone_pinned(self, device, dest_device, no_device_arg):
        a_device = torch.device(device)
        b_device = torch.device(dest_device)

        if no_device_arg and a_device != b_device:
            # Need device arg if devices differ.
            self.skipTest("Skipped!")

        # Keep a copy of the original values to make sure they did not change
        orig_values = torch.randn(10, device=a_device)

        a = torch.empty(10, device=a_device, pin_memory=a_device.type == "cpu")
        a.copy_(orig_values)

        orig_values = torch.empty(10, device="cpu").copy_(orig_values)

        a_data_ptr_mps = data_address_unified(a, "mps")
        a_data_ptr_cpu = data_address_unified(a, "cpu")
        self.assertNotEqual(a_data_ptr_cpu, a_data_ptr_mps)

        # Test lazy cloning with and without the device arg
        if no_device_arg:
            b = a._lazy_clone()
        else:
            b = a._lazy_clone(device=dest_device)

        b_data_ptr_mps = data_address_unified(b, "mps")
        b_data_ptr_cpu = data_address_unified(b, "cpu")
        self.assertNotEqual(b_data_ptr_cpu, b_data_ptr_mps)

        def check_tensor(
            x,
            check_data_ptr_cpu,
            check_data_ptr_mps,
            check_device,
            is_materialized,
            reuse_data,
        ):
            self.assertEqual(torch._C._is_cow_tensor(x), not is_materialized)

            if check_device.type == "cpu":
                self.assertEqual(x.device.type, "cpu")
                self.assertTrue(x.is_pinned())
            else:
                self.assertEqual(x.device.type, "mps")
                self.assertFalse(x.is_pinned())

            data_ptr_default = torch._C._data_address(x)
            data_ptr_cpu = data_address_unified(x, "cpu")
            data_ptr_mps = data_address_unified(x, "mps")

            self.assertNotEqual(data_ptr_cpu, data_ptr_mps)

            if is_materialized and not reuse_data:
                self.assertNotEqual(data_ptr_cpu, check_data_ptr_cpu)
                self.assertNotEqual(data_ptr_mps, check_data_ptr_mps)
            else:
                self.assertEqual(data_ptr_mps, check_data_ptr_mps)
                self.assertEqual(data_ptr_cpu, check_data_ptr_cpu)

            if check_device.type == "cpu":
                self.assertEqual(data_ptr_default, data_ptr_cpu)
            else:
                self.assertEqual(data_ptr_default, data_ptr_mps)

            self.assertEqual(orig_values.cpu(), x.clone().cpu())
            self.assertEqual(torch._C._is_cow_tensor(x), not is_materialized)

        check_tensor(a, a_data_ptr_cpu, a_data_ptr_mps, a_device, False, False)
        check_tensor(b, b_data_ptr_cpu, b_data_ptr_mps, b_device, False, False)
        a.data_ptr()
        check_tensor(a, a_data_ptr_cpu, a_data_ptr_mps, a_device, True, False)
        check_tensor(b, b_data_ptr_cpu, b_data_ptr_mps, b_device, False, False)
        b.data_ptr()
        check_tensor(a, a_data_ptr_cpu, a_data_ptr_mps, a_device, True, False)
        check_tensor(
            b,
            b_data_ptr_cpu,
            b_data_ptr_mps,
            b_device,
            True,
            a_device.type == b_device.type,
        )

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize(
        "op",
        [
            "_lazy_clone",
            "to",
        ],
    )
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
    def test_interdevice_materialize(self, device, op, materialize_first, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        a = torch.randn(10, device=src_device, pin_memory=pin_memory)
        orig_data_ptr = data_address_unified(a)

        if op == "_lazy_clone":
            b = a._lazy_clone(device=dest_device)
        elif op == "to":
            if torch.device(device).type != "mps" or case == "to_same_device":
                self.skipTest(
                    "op='to' only runs if device='mps' and if source and dest devices differ"
                )
            b = a.to(device=dest_device)
        else:
            raise AssertionError(f"op='{op}' not recognized")

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        if materialize_first == "src":
            a.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(
                data_address_unified(a),
                orig_data_ptr,
            )
            self.assertTrue(torch._C._is_cow_tensor(b))
            self.assertEqual(
                data_address_unified(b),
                orig_data_ptr,
            )

            b.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertNotEqual(
                data_address_unified(a),
                orig_data_ptr,
            )
            self.assertFalse(torch._C._is_cow_tensor(b))
            if src_device_check == dest_device_check:
                self.assertEqual(
                    data_address_unified(b),
                    orig_data_ptr,
                )
            else:
                self.assertNotEqual(
                    data_address_unified(b),
                    orig_data_ptr,
                )

        elif materialize_first == "dest":
            b.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(
                data_address_unified(b),
                orig_data_ptr,
            )
            self.assertTrue(torch._C._is_cow_tensor(a))
            self.assertEqual(
                data_address_unified(a),
                orig_data_ptr,
            )

            a.data_ptr()

            self.assertEqual(a.device, src_device_check)
            self.assertEqual(b.device, dest_device_check)
            self.assertFalse(torch._C._is_cow_tensor(b))
            self.assertNotEqual(
                data_address_unified(b),
                orig_data_ptr,
            )
            self.assertFalse(torch._C._is_cow_tensor(a))
            self.assertEqual(
                data_address_unified(a),
                orig_data_ptr,
            )

        else:
            raise RuntimeError(f"Not recognized: materialize_first={materialize_first}")

    # Test that COW a tensor with a different target device can be used in read
    # operations.
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @parametrize(
        "op",
        [
            "_lazy_clone",
            "to",
        ],
    )
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
    def test_interdevice_read(self, device, op, case):
        src_device, dest_device = self.get_src_dest_devices(case, device)

        src_device_check = torch.empty(0, device=src_device).device
        dest_device_check = torch.empty(0, device=dest_device).device
        pin_memory = src_device_check.type == "cpu" and dest_device_check.type == "mps"

        orig_tensor = torch.randn(10, device=src_device)
        a = torch.empty(10, device=src_device, pin_memory=pin_memory)
        a.copy_(orig_tensor)

        orig_data_ptr = data_address_unified(a)
        if op == "_lazy_clone":
            b = a._lazy_clone(device=dest_device)
        elif op == "to":
            if torch.device(device).type != "mps" or case == "to_same_device":
                self.skipTest(
                    "op='to' only runs if device='mps' and if source and dest devices differ"
                )
            b = a.to(device=dest_device)
        else:
            raise AssertionError(f"op='{op}' not recognized")

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        a_clone = a.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        b_clone = b.clone()

        self.assertEqual(a.device, src_device_check)
        self.assertEqual(b.device, dest_device_check)
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(b), orig_data_ptr)

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
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(b), orig_data_ptr)

    def test_clone_after_lazy_clone(self, device):
        a = torch.randn(10, device=device)
        orig_data_ptr = data_address_unified(a)
        b = torch._lazy_clone(a)

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        c = b.clone()

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertFalse(torch._C._is_cow_tensor(c))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        self.assertTrue((b == c).all())
        self.assertTrue((a == c).all())
        self.assertTrue((b.clone() == c).all())
        self.assertTrue((a.clone() == c).all())
        self.assertTrue((b.clone() == c.clone()).all())
        self.assertTrue((a.clone() == c.clone()).all())

        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))
        self.assertFalse(torch._C._is_cow_tensor(c))
        self.assertEqual(data_address_unified(a), orig_data_ptr)
        self.assertEqual(data_address_unified(b), orig_data_ptr)

        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(b, c)

    def test_to_compile(self, device):
        pin_memory = torch.device(device).type == "mps"

        @torch.compile
        def fn(x):
            x = x + 1
            x = x + 2
            x = x.to(device=device)
            x = x + 3
            x = x + 4
            x = x.cpu()
            x = x + 5
            x = x + 6
            x = x.to(device=device)
            x = x + 7
            x = x + 8
            x = x.cpu()
            x = x + 9
            x = x + 10
            return x

        fn(torch.randn([2, 2, 10], pin_memory=pin_memory))

    # See Note [CPU pinned to MPS failures]
    # TODO: Once this issue is fixed, remove this test
    def test_isclose_issue(self, device):
        a_device = torch.device(device).type
        b_device = "mps" if a_device == "cpu" else "cpu"

        a_args = [
            torch.randint(
                -9,
                10,
                (5, 10, 5),
                dtype=torch.int32,
                device=a_device,
                pin_memory=a_device == "cpu",
            ),
            torch.randint(
                -9,
                10,
                (5, 10, 5),
                dtype=torch.int32,
                device=a_device,
                pin_memory=a_device == "cpu",
            ),
        ]

        b_args = [arg.to(b_device) for arg in a_args]

        # This op call mutates the first arg's data
        if a_device == "mps":
            torch.isclose(*a_args)
        else:
            torch.isclose(*b_args)

        self.assertEqual(a_args[1], b_args[1])
        # THIS one fails
        self.assertEqual(a_args[0], b_args[0])


instantiate_device_type_tests(
    TestLazyCloneDeviceType, globals(), allow_mps=True, only_for=["cpu", "mps"]
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
