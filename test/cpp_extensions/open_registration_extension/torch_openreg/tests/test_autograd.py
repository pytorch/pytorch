# Owner(s): ["module: PrivateUse1"]

import os

import psutil

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfMPS,
    skipIfTorchDynamo,
    skipIfWindows,
    TestCase,
)


class TestAutograd(TestCase):
    # Support MPS and Windows platform later and fix torchdynamo issue
    @skipIfMPS
    @skipIfWindows()
    @skipIfTorchDynamo()
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


if __name__ == "__main__":
    run_tests()
