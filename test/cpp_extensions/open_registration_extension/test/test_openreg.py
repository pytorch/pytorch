import torch

from torch.testing._internal.common_utils import TestCase, run_tests
import psutil
import os

import pytorch_openreg

class TestOpenReg(TestCase):
    def test_initializes(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")

    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.rand(2, requires_grad=True, device="openreg").sum().backward()

        pid = os.getpid()
        task_path = f'/proc/{pid}/task'
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f'{task_path}/{t.id}/comm', 'r') as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(pytorch_openreg._device_daemon.NUM_DEVICES):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    def test_factory(self):
        a = torch.empty(50, device="openreg")
        self.assertEqual(a.device.type, "openreg")


if __name__ == "__main__":
    run_tests()
