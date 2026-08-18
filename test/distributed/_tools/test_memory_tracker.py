# Owner(s): ["oncall: distributed"]
import io
import os
import pickle
import unittest
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase


class FakeDeviceModule:
    """Stub of a device module so ``MemoryTracker`` can record stats without an accelerator."""

    def __init__(self, allocated_mbs):
        self._allocated_mbs = iter(allocated_mbs)
        self._current = 0

    def memory_allocated(self):
        self._current = next(self._allocated_mbs) * 1024 * 1024
        return self._current

    def memory_reserved(self):
        return self._current * 2

    def memory_stats(self):
        return {"active_bytes.all.current": self._current}


class TestMemoryTracker(TestCase):
    @unittest.skipIf(not torch.accelerator.is_available(), "no accelerator")
    def test_local_model(self):
        """
        Minimal test case to check the memory tracker can collect the expected
        memory stats at operator level, as well as can print the summary result
        without crash.
        """
        device = torch.accelerator.current_accelerator()
        # Create a model with a hierarchy of modules
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ),
            nn.Flatten(start_dim=1),
            nn.Sequential(nn.Linear(64, 2), nn.ReLU(inplace=True)),
        ).to(device)

        # Run one iteration of forward and backward pass
        tracker = MemoryTracker()
        tracker.start_monitor(model)

        x = torch.randn(size=(2, 3, 224, 224), device=device)
        # torch.LongTensor expects cpu device type, not gpu device type in
        # constructor, so calling .to() outside constructor here.
        target = torch.LongTensor([0, 1]).to(device)
        criterion = nn.CrossEntropyLoss()
        criterion(model(x), target).backward()

        self.assertTrue(len(tracker._hooks) > 0)

        tracker.stop()

        self.assertTrue(len(tracker._hooks) == 0)

        path = "memory.trace"
        tracker.save_stats(path)
        tracker.load(path)
        tracker.summary()
        if os.path.exists(path):
            os.remove(path)

        self.assertTrue(tracker._op_index > 0)
        self.assertTrue(len(tracker._operator_names) > 0)
        self.assertEqual(len(tracker.memories_allocated), tracker._op_index)
        self.assertEqual(len(tracker.memories_active), tracker._op_index)
        self.assertEqual(len(tracker.memories_reserved), tracker._op_index)
        self.assertTrue(len(tracker._markers) == 2)
        self.assertTrue(tracker._cur_module_name != "")
        self.assertTrue(hasattr(tracker, "_num_alloc_retries"))

    @staticmethod
    def _capture_summary(tracker):
        buf = io.StringIO()
        with redirect_stdout(buf):
            tracker.summary()
        return buf.getvalue()

    def test_save_load_restores_op_index(self):
        """
        A ``save_stats()``/``load()`` round trip into a fresh tracker should
        restore the operation count so that ``summary()`` reports the same
        per-operator memory deltas as the original tracker.
        """
        original = MemoryTracker()
        original._device_module = FakeDeviceModule([0, 100, 250])
        for op_name in (
            "net.forward.conv2d_0",
            "net.forward.relu_0",
            "net.forward.addmm_0",
        ):
            original._record_memory_stats(op_name)
        self.assertEqual(original._op_index, 3)
        original_summary = self._capture_summary(original)
        self.assertIn("net.forward.addmm_0", original_summary)

        with TemporaryFileName() as path:
            original.save_stats(path)

            loaded = MemoryTracker()
            loaded.load(path)

        self.assertEqual(loaded._op_index, original._op_index)
        self.assertEqual(loaded.memories_allocated, original.memories_allocated)
        self.assertEqual(loaded.memories_active, original.memories_active)
        self.assertEqual(loaded.memories_reserved, original.memories_reserved)
        self.assertEqual(self._capture_summary(loaded), original_summary)

    def test_load_stats_without_op_index(self):
        """
        ``load()`` should restore the operation count from stats files that
        only contain the memory traces (the format ``save_stats()`` has always
        written, with no explicit operation count).
        """
        old_stats = {
            "memories_allocated": {0: ("op_0", 0.0), 1: ("op_1", 128.0)},
            "memories_active": {0: ("op_0", 0.0), 1: ("op_1", 128.0)},
            "memories_reserved": {0: ("op_0", 0.0), 1: ("op_1", 256.0)},
            "markers": {"fw_start": 0},
            "num_alloc_retries": 0,
        }
        with TemporaryFileName() as path:
            with open(path, "wb") as f:
                pickle.dump(old_stats, f, pickle.HIGHEST_PROTOCOL)

            tracker = MemoryTracker()
            tracker.load(path)

        self.assertEqual(tracker._op_index, 2)
        summary = self._capture_summary(tracker)
        self.assertIn("op_1: 128.0MB", summary)


if __name__ == "__main__":
    run_tests()
