# Owner(s): ["module: PrivateUse1"]

import multiprocessing

import torch

import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase


class TestDevice(TestCase):
    def test_device_count(self):
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        torch.accelerator.set_device_index(1)
        self.assertEqual(torch.accelerator.current_device_index(), 1)

        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_device_context(self):
        device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), device)
        self.assertEqual(torch.accelerator.current_device_index(), device)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), device)

    @skipIfWindows(msg="Fork not available on Windows")
    def test_device_poison_fork(self):
        # First, initialize in the parent process
        torch.openreg.init()

        def child(q):
            try:
                # Second, try to initialize in the child process
                torch.openreg.init()
            except Exception as e:
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(not q.empty())

        exc = q.get()
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Cannot re-initialize OpenReg in forked subprocess. "
                "To use OpenReg with multiprocessing, you must use the 'spawn' start method"
            ),
        ):
            raise exc


if __name__ == "__main__":
    run_tests()
