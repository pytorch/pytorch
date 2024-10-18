# Owner(s): ["module: tests"]

import sys
import unittest

import torch
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    TEST_CUDA,
    TEST_XPU,
    TestCase,
)


if not torch.accelerator.is_available():
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


class TestAccelerator(TestCase):
    def test_current_accelerator(self):
        self.assertTrue(torch.accelerator.is_available())
        accelerators = ["cuda", "xpu"]
        for accelerator in accelerators:
            if torch.get_device_module(accelerator).is_available():
                self.assertEqual(
                    torch.accelerator.current_accelerator().type, accelerator
                )
                with self.assertRaisesRegex(
                    ValueError, "doesn't match the current accelerator"
                ):
                    torch.accelerator.set_device("cpu")

    def test_generic_multi_device_behavior(self):
        orig_device = torch.accelerator.current_device_idx()
        target_device = (orig_device + 1) % torch.accelerator.device_count()

        torch.accelerator.set_device(target_device)
        self.assertEqual(target_device, torch.accelerator.current_device_idx())
        torch.accelerator.set_device(orig_device)
        self.assertEqual(orig_device, torch.accelerator.current_device_idx())

        s1 = torch.Stream(target_device)
        torch.accelerator.set_stream(s1)
        self.assertEqual(target_device, torch.accelerator.current_device_idx())
        torch.accelerator.synchronize(orig_device)
        self.assertEqual(target_device, torch.accelerator.current_device_idx())

    def test_generic_stream_behavior(self):
        s1 = torch.Stream()
        s2 = torch.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream(), s1)
        event = torch.Event()
        a = torch.randn(100)
        b = torch.randn(100)
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

    @unittest.skipIf((not TEST_CUDA) and (not TEST_XPU), "requires CUDA or XPU")
    def test_specific_stream_compatibility(self):
        s1 = torch.cuda.Stream() if torch.cuda.is_available() else torch.xpu.Stream()
        s2 = torch.cuda.Stream() if torch.cuda.is_available() else torch.xpu.Stream()
        torch.accelerator.set_stream(s1)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s1.stream_id)
        torch.accelerator.set_stream(s2)
        self.assertEqual(torch.accelerator.current_stream().stream_id, s2.stream_id)


if __name__ == "__main__":
    run_tests()
