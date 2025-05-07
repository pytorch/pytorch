# Owner(s): ["module: tests"]

import sys
import unittest

import torch
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_MPS, TestCase


if not torch.accelerator.is_available():
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811
    # Skip because failing when run on cuda build with no GPU, see #150059 for example
    sys.exit()

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

    def test_current_stream_query(self):
        s = torch.accelerator.current_stream()
        self.assertEqual(torch.accelerator.current_stream(s.device), s)
        self.assertEqual(torch.accelerator.current_stream(s.device.index), s)
        self.assertEqual(torch.accelerator.current_stream(str(s.device)), s)
        other_device = torch.device("cpu")
        with self.assertRaisesRegex(
            ValueError, "doesn't match the current accelerator"
        ):
            torch.accelerator.current_stream(other_device)

    def test_device_context_manager(self):
        prev_device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)
        with torch.accelerator.device_index(0):
            self.assertEqual(torch.accelerator.current_device_index(), 0)
        self.assertEqual(torch.accelerator.current_device_index(), prev_device)

    @unittest.skipIf(not TEST_MULTIACCELERATOR, "only one accelerator detected")
    def test_multi_device_context_manager(self):
        src_device = 0
        dst_device = 1
        torch.accelerator.set_device_index(src_device)
        with torch.accelerator.device_index(dst_device):
            self.assertEqual(torch.accelerator.current_device_index(), dst_device)
        self.assertEqual(torch.accelerator.current_device_index(), src_device)

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

    @unittest.skipIf(TEST_MPS, "MPS doesn't support pin memory!")
    def test_pin_memory_on_non_blocking_copy(self):
        t_acc = torch.randn(100).to(torch.accelerator.current_accelerator())
        t_host = t_acc.to("cpu", non_blocking=True)
        torch.accelerator.synchronize()
        self.assertTrue(t_host.is_pinned())
        self.assertEqual(t_acc.cpu(), t_host)

    def test_generic_event_behavior(self):
        event1 = torch.Event(enable_timing=False)
        event2 = torch.Event(enable_timing=False)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)

        event1 = torch.Event(enable_timing=True)
        event2 = torch.Event(enable_timing=True)
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be recorded before calculating elapsed time",
        ):
            event1.elapsed_time(event2)

        # check default value of enable_timing: False
        event1 = torch.Event()
        event2 = torch.Event()
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'",
        ):
            event1.elapsed_time(event2)


if __name__ == "__main__":
    run_tests()
