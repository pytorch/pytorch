# Owner(s): ["module: cpp"]

import os
import unittest

import psutil
import pytorch_openreg

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOpenReg(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")

    @unittest.SkipTest
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

        for i in range(pytorch_openreg._device_daemon.NUM_DEVICES):
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

    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="openreg")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))


if __name__ == "__main__":
    run_tests()
