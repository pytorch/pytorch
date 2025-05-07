# Owner(s): ["module: cpp"]

import os
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


class TestOpenReg(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")

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

    def test_factory(self):
        a = torch.empty(50, device="openreg")
        self.assertEqual(a.device.type, "openreg")

        a.fill_(3.5)

        self.assertTrue(a.eq(3.5).all())

    def test_printing(self):
        a = torch.ones(20, device="openreg")
        # Does not crash!
        str(a)

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

    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="openreg")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    def test_generator(self):
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    # TODO(FFFrog): Add more check for rng_state
    def test_rng_state(self):
        state = torch.openreg.get_rng_state(0)
        torch.openreg.set_rng_state(state, 0)

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

    def test_expand(self):
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_empty_tensor(self):
        empty_tensor = torch.tensor((), device="openreg")
        self.assertEqual(empty_tensor.to(device="cpu"), torch.tensor(()))


if __name__ == "__main__":
    run_tests()
