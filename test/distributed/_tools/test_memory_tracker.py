# Owner(s): ["oncall: distributed"]
import os
import pickle
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

    @unittest.skipIf(not torch.accelerator.is_available(), "no accelerator")
    def test_load_restores_op_index_in_fresh_tracker(self):
        """
        Regression test for https://github.com/pytorch/pytorch/issues/191397:
        load() must restore _op_index so summary() works when stats are loaded
        into a fresh tracker (e.g. in a notebook), not only when reusing the
        tracker that recorded them.
        """
        device = torch.accelerator.current_accelerator()
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)

        tracker = MemoryTracker()
        tracker.start_monitor(model)
        model(torch.randn(4, 64, device=device)).sum().backward()
        tracker.stop()

        expected_op_index = tracker._op_index
        self.assertGreater(expected_op_index, 0)

        path = "memory_op_index.trace"
        try:
            tracker.save_stats(path)

            # Loading into a fresh tracker must restore _op_index.
            fresh = MemoryTracker()
            fresh.load(path)
            self.assertEqual(fresh._op_index, expected_op_index)
            self.assertEqual(len(fresh.memories_allocated), fresh._op_index)
            fresh.summary()

            # Backward compatibility: stats saved before this fix lack the
            # "op_index" key, so load() must reconstruct it from the traces.
            with open(path, "rb") as f:
                legacy_stats = pickle.load(f)
            del legacy_stats["op_index"]
            with open(path, "wb") as f:
                pickle.dump(legacy_stats, f, pickle.HIGHEST_PROTOCOL)

            legacy = MemoryTracker()
            legacy.load(path)
            self.assertEqual(legacy._op_index, expected_op_index)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    run_tests()
