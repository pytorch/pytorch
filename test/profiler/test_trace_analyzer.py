"""
Tests for torch.profiler.trace_analyzer module.

These tests verify the trace analysis functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

from torch.profiler.auto_instrumenter import auto_profile_module
from torch.profiler.trace_analyzer import TraceGraph


class SimpleModel(torch.nn.Module):
    """Simple test model for trace analysis"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        z = torch.relu(y)
        return z


class TestTraceAnalyzer(unittest.TestCase):
    """Tests for TraceGraph analyzer"""

    def test_load_and_parse_trace(self):
        """Test that TraceGraph can load and parse a trace file"""
        # Setup: Create and instrument a simple model
        model = SimpleModel()
        model = auto_profile_module(model, "forward")

        # Execute: Run model with profiler and export trace
        x = torch.randn(4, 10)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)

        # Export trace to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            prof.export_chrome_trace(f.name)
            trace_file = f.name

        try:
            # Assert: TraceGraph should successfully load the trace
            trace = TraceGraph(trace_file)
            self.assertIsNotNone(trace)
            self.assertGreater(len(trace.events), 0, "Should have events in trace")
        finally:
            # Cleanup
            Path(trace_file).unlink()

    def test_find_line_annotations(self):
        """Test that TraceGraph finds line annotations from instrumented code"""
        # Setup: Create and instrument model
        model = SimpleModel()
        model = auto_profile_module(model, "forward")

        # Execute: Run with profiler
        x = torch.randn(4, 10)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)

        # Export trace
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            prof.export_chrome_trace(f.name)
            trace_file = f.name

        try:
            # Assert: Should find line annotations
            trace = TraceGraph(trace_file)
            self.assertGreater(
                len(trace.line_annotations), 0, "Should find line annotations"
            )

            # Verify annotation format (unique_key -> event)
            for unique_key, evt in trace.line_annotations.items():
                # unique_key format: occurrence_num:filename:line_number
                parts = unique_key.split(":")
                self.assertGreaterEqual(
                    len(parts), 3, f"Invalid unique_key format: {unique_key}"
                )
                # First part should be occurrence number (digit)
                self.assertTrue(
                    parts[0].isdigit(), f"First part should be occurrence num: {parts[0]}"
                )
                # Second part is filename
                # Third part should be line number (digit)
                self.assertTrue(
                    parts[2].isdigit(), f"Line number should be digit: {parts[2]}"
                )
        finally:
            Path(trace_file).unlink()

    def test_get_all_line_operations(self):
        """Test that TraceGraph can aggregate operations by line"""
        # Setup: Create and instrument model
        model = SimpleModel()
        model = auto_profile_module(model, "forward")

        # Execute: Run with profiler
        x = torch.randn(4, 10)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)

        # Export trace
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            prof.export_chrome_trace(f.name)
            trace_file = f.name

        try:
            # Assert: Should aggregate operations by line
            trace = TraceGraph(trace_file)
            line_ops = trace.get_all_line_operations()

            self.assertGreater(len(line_ops), 0, "Should have operations mapped to lines")

            # Each line should have operations grouped by thread
            for line_id, ops_by_thread in line_ops.items():
                self.assertIsInstance(
                    ops_by_thread, dict, f"Line {line_id} should have dict of ops by thread"
                )
        finally:
            Path(trace_file).unlink()

    def test_multiple_occurrences_same_line(self):
        """Test that TraceGraph handles multiple occurrences of same line correctly"""

        # Setup: Create model with loop (same lines executed multiple times)
        class LoopModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                results = []
                for i in range(2):
                    y = x + i
                    results.append(y)
                return torch.stack(results)

        model = LoopModel()
        model = auto_profile_module(model, "forward")

        # Execute: Run with profiler
        x = torch.randn(4, 4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)

        # Export trace
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            prof.export_chrome_trace(f.name)
            trace_file = f.name

        try:
            # Assert: Should have multiple occurrences with unique keys
            trace = TraceGraph(trace_file)

            # Find annotations for the same base line (y = x + i)
            base_ids = {}
            for unique_key in trace.line_annotations.keys():
                base_id = trace.line_to_base_id[unique_key]
                if base_id not in base_ids:
                    base_ids[base_id] = []
                base_ids[base_id].append(unique_key)

            # Should have at least one line executed multiple times
            multiple_occurrences = [
                (base_id, keys) for base_id, keys in base_ids.items() if len(keys) > 1
            ]
            self.assertGreater(
                len(multiple_occurrences),
                0,
                "Should have lines with multiple occurrences",
            )
        finally:
            Path(trace_file).unlink()
