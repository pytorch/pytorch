# Owner(s): ["oncall: distributed"]
import io
import os
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_utils import run_tests, TestCase


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

    def _tracker_with_ops(self):
        # Inject traces directly so this does not need a GPU or a real module run.
        tracker = MemoryTracker()
        traces = {
            0: ("net.forward.add_0", 10.0),
            1: ("net.forward.mul_0", 25.0),
            2: ("net.forward.add_1", 28.0),
        }
        tracker.memories_allocated = dict(traces)
        tracker.memories_active = dict(traces)
        tracker.memories_reserved = dict(traces)
        tracker._op_index = len(traces)
        tracker._num_alloc_retries = 2
        tracker._markers = {"fw_start": 0, "fw_bw_boundary": 2}
        return tracker

    def _summary_text(self, tracker):
        buf = io.StringIO()
        with redirect_stdout(buf):
            tracker.summary()
        return buf.getvalue()

    def test_save_load_roundtrip_restores_op_index(self):
        """
        Loading stats into a fresh MemoryTracker should restore the operation
        count and print the same operator summary as the original tracker.
        """
        src = self._tracker_with_ops()
        src_summary = self._summary_text(src)
        self.assertIn("net.forward.mul_0", src_summary)
        self.assertIn("net.forward.add_1", src_summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.trace")
            src.save_stats(path)

            loaded = MemoryTracker()
            loaded.load(path)

            self.assertEqual(loaded._op_index, src._op_index)
            self.assertEqual(loaded.memories_allocated, src.memories_allocated)
            self.assertEqual(loaded._num_alloc_retries, src._num_alloc_retries)
            self.assertEqual(self._summary_text(loaded), src_summary)

            # Older files were written without op_index. Reconstruct it from the traces.
            with open(path, "rb") as f:
                stats = pickle.load(f)
            stats.pop("op_index", None)
            with open(path, "wb") as f:
                pickle.dump(stats, f)

            legacy = MemoryTracker()
            legacy.load(path)
            self.assertEqual(legacy._op_index, src._op_index)
            self.assertEqual(self._summary_text(legacy), src_summary)


if __name__ == "__main__":
    run_tests()
