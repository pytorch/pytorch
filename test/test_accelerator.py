# Owner(s): ["module: tests"]

import sys
import unittest

import torch
from torch.testing._internal.common_utils import NoTest, run_tests, TestCase


if not torch.accelerator.is_available():
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIACCELERATOR = torch.accelerator.device_count() > 1


class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        self.assertTrue(torch.accelerator.is_available())
        accelerators = ["cuda", "xpu", "mps"]
        for accelerator in accelerators:
            if torch.get_device_module(accelerator).is_available():
                self.assertEqual(
                    torch.accelerator.current_accelerator().type, accelerator
                )
                self.assertIsNone(torch.accelerator.current_accelerator().index)
                with self.assertRaisesRegex(
                    ValueError, "doesn't match the current accelerator"
                ):
                    torch.accelerator.set_device_index("cpu")

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_generic_multi_device_behavior(self):
        orig_device = torch.accelerator.current_device_index()
        target_device = (orig_device + 1) % torch.accelerator.device_count()

        torch.accelerator.set_device_index(target_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.set_device_index(orig_device)
        self.assertEqual(orig_device, torch.accelerator.current_device_index())

        s1 = torch.Stream(target_device)
        torch.accelerator.set_stream(s1)
        self.assertEqual(target_device, torch.accelerator.current_device_index())
        torch.accelerator.synchronize(orig_device)
        self.assertEqual(target_device, torch.accelerator.current_device_index())

    def test_generic_stream_behavior(self):
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(1000)
        b = torch.randn(1000)
        c = a + b
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream(), s2)
        a_acc = a.to(torch.accelerator.current_accelerator(), non_blocking=True)
        b_acc = b.to(torch.accelerator.current_accelerator(), non_blocking=True)
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event.record(s2)
        event.synchronize()
        c_acc = a_acc + b_acc
        event.record(s2)
        torch.accelerator.synchronize()
        self.assertTrue(event.query())
        self.assertEqual(c_acc.cpu(), c)

    def test_stream_context_manager(self):
        prev_stream = torch.accelerator.current_stream()
        with torch.Stream() as s:
            self.assertEqual(torch.accelerator.current_stream(), s)
        self.assertEqual(torch.accelerator.current_stream(), prev_stream)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_stream_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        src_prev_stream = torch.accelerator.current_stream()
        dst_prev_stream = torch.accelerator.current_stream(dst_device)
        with torch.Stream(dst_device) as dst_stream:
            self.assertEqual(torch.accelerator.current_device_index(), dst_device)
            self.assertEqual(torch.accelerator.current_stream(), dst_stream)
            self.assertEqual(
                torch.accelerator.current_stream(src_device), src_prev_stream
            )
        self.assertEqual(torch.accelerator.current_device_index(), src_device)
        self.assertEqual(torch.accelerator.current_stream(), src_prev_stream)
        self.assertEqual(torch.accelerator.current_stream(dst_device), dst_prev_stream)


if __name__ == "__main__":
    run_tests()
