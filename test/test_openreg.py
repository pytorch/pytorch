# Owner(s): ["module: PrivateUse1"]

import os
import types
import unittest

import psutil
import pytorch_openreg  # noqa: F401

import torch
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestPrivateUse1(TestCase):
    """Tests of third-parth device integration mechinasm based PrivateUse1"""

    def test_backend_name(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):  # type: ignore[misc]
            torch.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):  # type: ignore[misc]
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):  # type: ignore[misc]
            torch._register_device_module("openreg", generate_faked_module())

    def test_backend_generate_methods(self):
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):  # type: ignore[misc]
            torch.utils.generate_methods_for_privateuse1_backend()  # type: ignore[misc]

        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_module_function(self):
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):  # type: ignore[misc]
            torch.utils.backend_registration._get_custom_mod_func("func_name_")  # type: ignore[misc]
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2  # type: ignore[misc]
        )

    @skipIfTorchDynamo()
    def test_backend_operator_registration(self):
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::empty.memory_format", torch.DispatchKey.PrivateUse1
            )
        )
        x = torch.empty(3, 3, device="openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertTrue(x.shape, torch.Size([3, 3]))

    def test_backend_dispatchstub(self):
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = torch.abs(x_cpu)
        y_openreg = torch.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())

    def test_backend_fallback(self):
        pass


class TestOpenReg(TestCase):
    """Tests of mimick accelerator named OpenReg based on PrivateUse1"""

    # Stream & Event
    def test_stream_synchronize(self):
        stream = torch.Stream(device="openreg:1")
        stream.synchronize()
        self.assertEqual(True, stream.query())

    def test_stream_wait_stream(self):
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        # Does not crash!
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_record_event(self):
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)
        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)
        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        stream = torch.Stream(device="openreg:1")
        e1 = torch.Event(device="openreg:1", enable_timing=True)
        e1.record(stream)
        e2 = torch.Event(device="openreg:1", enable_timing=True)
        e2.record(stream)

        e2.synchronize()
        self.assertTrue(e2.query())

        ms = e1.elapsed_time(e2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e1 = s1.record_event()
        e1.wait(s2)

    # Copy
    def test_cross_device_copy(self):
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_copy_same_device(self):
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_diff_devices_copy(self):
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))

    # RNG
    def test_generator(self):
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        state = torch.openreg.get_rng_state(0)  # type: ignore[misc]
        torch.openreg.set_rng_state(state, 0)  # type: ignore[misc]

    def test_manual_seed(self):
        torch.openreg.manual_seed_all(2024)  # type: ignore[misc]
        self.assertEqual(torch.openreg.initial_seed(), 2024)  # type: ignore[misc]

    # Autograd
    @unittest.skipIf(not IS_LINUX, "Only works on linux")
    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.ones(2, requires_grad=True, device="openreg").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(torch.accelerator.device_count()):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    # Storage & Pin Memory
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        cpu_a = torch.randn(10)
        self.assertFalse(cpu_a.is_pinned())
        pinned_a = cpu_a.pin_memory()
        self.assertTrue(pinned_a.is_pinned())
        slice_a = pinned_a[2:5]
        self.assertTrue(slice_a.is_pinned())

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_rewrapped_storage(self):
        pinned_a = torch.randn(10).pin_memory()
        rewrapped_a = torch.tensor((), dtype=torch.float32).set_(
            pinned_a.untyped_storage()[2:],
            size=(5,),
            stride=(1,),
            storage_offset=0,
        )
        self.assertTrue(rewrapped_a.is_pinned())
        self.assertNotEqual(pinned_a.data_ptr(), rewrapped_a.data_ptr())

    # Opeartors
    def test_factory(self):
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3]))

        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))

    def test_printing(self):
        a = torch.ones(20, device="openreg")
        # Does not crash!
        str(a)

    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="openreg")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    def test_expand(self):
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())


if __name__ == "__main__":
    run_tests()
