import multiprocessing
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS
import torch


class TestDeviceLazyInit(TestCase):
    @unittest.skipIf(IS_WINDOWS, "Fork not available on Windows")
    def test_cuda_fork_poison_on_lazy_init(self):
        torch.cuda.init()

        def child(q):
            try:
                torch.cuda.init()
            except Exception as e:  # noqa: BLE001
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(not q.empty())
        exc = q.get()
        self.assertIsInstance(exc, RuntimeError)
        self.assertRegex(str(exc), r"forked subprocess.*spawn")

    @unittest.skipIf(IS_WINDOWS, "Fork not available on Windows")
    def test_cuda_registration_idempotent(self):
        def child(q):
            try:
                torch._C._cuda_getDeviceCount()
                torch._C._cuda_getDeviceCount()
            except Exception as e:  # noqa: BLE001
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(q.empty())

if __name__ == "__main__":
    run_tests()
