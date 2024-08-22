# Owner(s): ["module: cpp"]

import os
import unittest

import psutil
import pytorch_openreg

import torch
from torch.testing._internal.common_utils import IS_LINUX, run_tests, TestCase


class TestOpenReg(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")

    @unittest.skipIf(not IS_LINUX, "Only works on linux")
    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.rand(2, requires_grad=True, device="openreg").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(pytorch_openreg.NUM_DEVICES):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)


if __name__ == "__main__":
    run_tests()
