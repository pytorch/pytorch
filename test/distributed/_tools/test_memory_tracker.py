# Owner(s): ["oncall: distributed"]
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMemoryTracker(TestCase):
    def test_load_restores_op_index(self):
        with patch.object(torch, "get_device_module"):
            tracker = MemoryTracker()
            loaded_tracker = MemoryTracker()

        tracker.memories_allocated = {
            0: ("initial", 1.0),
            1: ("aten.mm", 3.0),
        }
        tracker.memories_active = {
            0: ("initial", 1.0),
            1: ("aten.mm", 2.0),
        }
        tracker.memories_reserved = {
            0: ("initial", 2.0),
            1: ("aten.mm", 4.0),
        }
        tracker._op_index = 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "memory.trace")
            tracker.save_stats(path)
            loaded_tracker.load(path)

        self.assertEqual(loaded_tracker._op_index, tracker._op_index)
        with redirect_stdout(io.StringIO()) as expected_output:
            tracker.summary()
        with redirect_stdout(io.StringIO()) as actual_output:
            loaded_tracker.summary()
        self.assertEqual(actual_output.getvalue(), expected_output.getvalue())

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
        # The stats must come from a real allocator: with live activations,
        # nonzero allocated memory should be observed at some operator.
        self.assertTrue(
            any(mem > 0 for (_, mem) in tracker.memories_allocated.values())
        )
        self.assertTrue(len(tracker._markers) == 2)
        self.assertTrue(tracker._cur_module_name != "")
        self.assertTrue(hasattr(tracker, "_num_alloc_retries"))


if __name__ == "__main__":
    run_tests()
