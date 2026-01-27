"""
Test the profiler export_chrome_trace callback mechanism.
"""

import json
import os
import tempfile

import torch
from torch.profiler import register_export_chrome_trace_callback, _export_chrome_trace_callbacks
from torch.testing._internal.common_utils import run_tests, TestCase


class TestExportChromeTraceCallbacks(TestCase):
    """Test the export_chrome_trace callback mechanism."""

    def setUp(self):
        # Clear callbacks before each test
        _export_chrome_trace_callbacks.clear()

    def tearDown(self):
        _export_chrome_trace_callbacks.clear()

    def test_callback_registration(self):
        """Test that callbacks can be registered."""
        def my_callback(data):
            return data

        self.assertEqual(len(_export_chrome_trace_callbacks), 0)
        register_export_chrome_trace_callback(my_callback)
        self.assertEqual(len(_export_chrome_trace_callbacks), 1)

    def test_callback_modifies_trace(self):
        """Test that registered callbacks modify the trace."""
        from torch.profiler import profile, ProfilerActivity

        def add_marker(data):
            data["test_marker"] = "callback_ran"
            return data

        register_export_chrome_trace_callback(add_marker)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                x = torch.randn(10, 10)
                _ = x + x

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                data = json.load(f)

            self.assertEqual(data.get("test_marker"), "callback_ran")

        finally:
            os.unlink(trace_path)

    def test_multiple_callbacks_run_in_order(self):
        """Test that multiple callbacks run in registration order."""
        from torch.profiler import profile, ProfilerActivity

        def add_first(data):
            data["order"] = ["first"]
            return data

        def add_second(data):
            data["order"].append("second")
            return data

        register_export_chrome_trace_callback(add_first)
        register_export_chrome_trace_callback(add_second)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                x = torch.randn(10, 10)
                _ = x + x

            prof.export_chrome_trace(trace_path)

            with open(trace_path) as f:
                data = json.load(f)

            self.assertEqual(data.get("order"), ["first", "second"])

        finally:
            os.unlink(trace_path)

    def test_no_callbacks_no_error(self):
        """Test that export works with no callbacks registered."""
        from torch.profiler import profile, ProfilerActivity

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name

        try:
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                x = torch.randn(10, 10)
                _ = x + x

            # Should not raise
            prof.export_chrome_trace(trace_path)

            # Trace should be valid
            with open(trace_path) as f:
                data = json.load(f)
            self.assertIn("traceEvents", data)

        finally:
            os.unlink(trace_path)


if __name__ == "__main__":
    run_tests()
