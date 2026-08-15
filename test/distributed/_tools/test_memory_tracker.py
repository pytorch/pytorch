# Owner(s): ["oncall: distributed"]
import contextlib
import io
import os
import pickle
import tempfile
import unittest

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

    def test_save_load_roundtrip_preserves_op_index(self):
        """
        A save/load round trip must preserve the operation count so that
        summary() reports the same operators as the original tracker.
        Regression test for GH issue 191397.
        """
        tracker = MemoryTracker()
        for i in range(3):
            tracker.memories_allocated[i] = (f"op{i}", float(i))
            tracker.memories_active[i] = (f"op{i}", float(i))
            tracker.memories_reserved[i] = (f"op{i}", float(i))
        tracker._op_index = 3
        tracker._num_alloc_retries = 1

        with tempfile.NamedTemporaryFile() as f:
            tracker.save_stats(f.name)
            loaded = MemoryTracker()
            loaded.load(f.name)

        self.assertEqual(loaded._op_index, tracker._op_index)
        self.assertEqual(loaded._num_alloc_retries, tracker._num_alloc_retries)
        self.assertEqual(loaded.memories_allocated, tracker.memories_allocated)
        self.assertEqual(loaded.memories_active, tracker.memories_active)
        self.assertEqual(loaded.memories_reserved, tracker.memories_reserved)

        orig_output = io.StringIO()
        with contextlib.redirect_stdout(orig_output):
            tracker.summary()
        loaded_output = io.StringIO()
        with contextlib.redirect_stdout(loaded_output):
            loaded.summary()
        self.assertEqual(loaded_output.getvalue(), orig_output.getvalue())

    def test_load_stats_without_op_index_reconstructs_count(self):
        """
        Stats files saved before op_index was persisted must still load,
        reconstructing the operation count from the traces.
        """
        stats = {
            "memories_allocated": {0: ("op0", 0.0), 1: ("op1", 1.0)},
            "memories_active": {0: ("op0", 0.0), 1: ("op1", 1.0)},
            "memories_reserved": {0: ("op0", 0.0), 1: ("op1", 1.0)},
            "markers": {},
            "num_alloc_retries": 0,
        }
        with tempfile.NamedTemporaryFile() as f:
            with open(f.name, "wb") as fh:
                pickle.dump(stats, fh, pickle.HIGHEST_PROTOCOL)
            loaded = MemoryTracker()
            loaded.load(f.name)

        self.assertEqual(loaded._op_index, 2)
        self.assertEqual(loaded.memories_allocated, stats["memories_allocated"])

        loaded_output = io.StringIO()
        with contextlib.redirect_stdout(loaded_output):
            loaded.summary()
        self.assertIn("op1", loaded_output.getvalue())


if __name__ == "__main__":
    run_tests()
