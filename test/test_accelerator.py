# Owner(s): ["module: tests"]

import unittest

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA,
    TEST_XPU,
    TestCase,
)


TEST_ACCELERATOR = TEST_CUDA or TEST_XPU


class TestAccelerator(TestCase):
    def test_generic_device_behavior(self):
        if not TEST_ACCELERATOR:
            self.assertIsNone(torch.current_accelerator())
            return
        current_device = torch.current_device()
        torch.set_device(current_device)
        self.assertEqual(current_device, torch.current_device())
        if torch.cuda.is_available():
            self.assertTrue(torch.has_accelerator())
            self.assertEqual(torch.current_accelerator(), "cuda")
        if torch.xpu.is_available():
            self.assertTrue(torch.has_accelerator())
            self.assertEqual(torch.current_accelerator(), "xpu")

    @unittest.skipIf(not TEST_ACCELERATOR, "no avaliable accelerators detected")
    def test_generic_multi_device_behavior(self):
        current_device = torch.current_device()
        target_device = (current_device + 1) % torch.device_count()

        with torch.DeviceGuard(target_device):
            self.assertEqual(target_device, torch.current_device())
        self.assertEqual(current_device, torch.current_device())

        s1 = torch.Stream(target_device)
        torch.set_stream(s1)
        self.assertEqual(target_device, torch.current_device())

    @unittest.skipIf(not TEST_ACCELERATOR, "no avaliable accelerators detected")
    def test_generic_stream_behavior(self):
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.set_stream(s1)
        self.assertEqual(torch.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(100)
        b = torch.randn(100)
        c = a + b
        with torch.StreamGuard(s2):
            self.assertEqual(torch.current_stream(), s2)
            a_acc = a.to(torch.current_accelerator(), non_blocking=True)
            b_acc = b.to(torch.current_accelerator(), non_blocking=True)
        self.assertEqual(torch.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)


if __name__ == "__main__":
    run_tests()
