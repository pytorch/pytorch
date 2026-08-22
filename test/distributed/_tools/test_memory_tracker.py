# Owner(s): ["oncall: distributed"]
import os
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import hypothesis.strategies as st
from hypothesis import example, given, settings

import torch
import torch.nn as nn
from torch.distributed._tools import MemoryTracker
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMemoryTracker(TestCase):
    @settings(max_examples=10, deadline=200000)
    @given(
        memory_values=st.lists(
            st.floats(
                min_value=0.0,
                max_value=1_000_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @example(memory_values=[1.0, 3.0])
    def test_save_and_load(self, memory_values):
        tracker = MemoryTracker()
        tracker.memories_allocated = {
            index: (f"operation_{index}", value)
            for index, value in enumerate(memory_values)
        }
        tracker.memories_active = {
            index: (f"operation_{index}", value + 1.0)
            for index, value in enumerate(memory_values)
        }
        tracker.memories_reserved = {
            index: (f"operation_{index}", value + 2.0)
            for index, value in enumerate(memory_values)
        }
        tracker._op_index = len(memory_values)

        with redirect_stdout(StringIO()) as expected_output:
            tracker.summary()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "memory.trace")
            tracker.save_stats(path)

            loaded_tracker = MemoryTracker()
            loaded_tracker.load(path)
            with redirect_stdout(StringIO()) as loaded_output:
                loaded_tracker.summary()

            with open(path, "rb") as f:
                legacy_stats = pickle.load(f)
            legacy_stats.pop("op_index")
            with open(path, "wb") as f:
                pickle.dump(legacy_stats, f, pickle.HIGHEST_PROTOCOL)

            legacy_tracker = MemoryTracker()
            legacy_tracker.load(path)
            with redirect_stdout(StringIO()) as legacy_output:
                legacy_tracker.summary()

            self.assertEqual(loaded_tracker._op_index, tracker._op_index)
            self.assertEqual(legacy_tracker._op_index, tracker._op_index)
            self.assertEqual(loaded_output.getvalue(), expected_output.getvalue())
            self.assertEqual(legacy_output.getvalue(), expected_output.getvalue())

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


if __name__ == "__main__":
    run_tests()
