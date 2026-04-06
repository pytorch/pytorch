#!/usr/bin/env python3
"""Tests for the streams fuzzing template codegen."""

import os
import random
import sys
import unittest


# Add parent directory to path so we can import torchfuzz as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torchfuzz.codegen import convert_graph_to_python_code, StreamFuzzTemplate
from torchfuzz.ops_fuzzer import fuzz_operation_graph, fuzz_spec


class TestStreamsFuzzTemplate(unittest.TestCase):
    def _generate_code(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        target_spec = fuzz_spec("streams")
        graph = fuzz_operation_graph(
            target_spec, max_depth=3, seed=seed, template="streams"
        )
        return convert_graph_to_python_code(graph, seed=seed, template="streams")

    def test_template_inherits_default_ops(self):
        template = StreamFuzzTemplate()
        self.assertGreater(len(template.supported_ops), 0)
        self.assertIn("torch.add", template.supported_ops)
        self.assertIn("torch.matmul", template.supported_ops)

    def test_template_uses_backward_check(self):
        from torchfuzz.checks import EagerVsFullGraphDynamicCompileWithBackwardCheck

        template = StreamFuzzTemplate()
        self.assertIsInstance(
            template.check, EagerVsFullGraphDynamicCompileWithBackwardCheck
        )

    def test_codegen_creates_streams(self):
        code = self._generate_code(seed=999)
        self.assertIn("torch.cuda.Stream()", code)

    def test_codegen_has_stream_context(self):
        code = self._generate_code(seed=999)
        self.assertIn("with torch.cuda.stream(", code)

    def test_codegen_has_final_sync(self):
        code = self._generate_code(seed=999)
        # Should sync all streams before return (wait_stream or wait_event)
        self.assertTrue(
            "torch.cuda.current_stream().wait_stream(" in code
            or "torch.cuda.current_stream().wait_event(" in code
        )

    def test_codegen_has_backward(self):
        code = self._generate_code(seed=999)
        self.assertIn(".sum().backward()", code)

    def test_codegen_has_requires_grad(self):
        """Float tensor args should have requires_grad for backward testing."""
        code = self._generate_code(seed=999)
        self.assertIn("requires_grad_(True)", code)

    def test_codegen_cross_stream_sync(self):
        """Seeds with cross-stream deps should have inter-stream sync."""
        # seed 999 produces a graph with ops on different streams
        code = self._generate_code(seed=999)
        # Should have either wait_stream or event-based sync between streams
        has_wait_stream = ".wait_stream(s" in code
        has_wait_event = ".wait_event(" in code
        self.assertTrue(
            has_wait_stream or has_wait_event,
            "Expected cross-stream synchronization in generated code",
        )

    def test_codegen_event_based_sync(self):
        """Some seeds should use event-based synchronization."""
        found_events = False
        for seed in range(50, 70):
            code = self._generate_code(seed=seed)
            if "torch.cuda.Event()" in code and ".wait_event(" in code:
                found_events = True
                # When events are used, should have record + wait pattern
                self.assertIn(".record(", code)
                break
        self.assertTrue(found_events, "No seed in range produced event-based sync")

    def test_codegen_deterministic(self):
        """Same seed should produce identical code."""
        code1 = self._generate_code(seed=42)
        code2 = self._generate_code(seed=42)
        self.assertEqual(code1, code2)

    def test_codegen_is_valid_python(self):
        """Generated code should be syntactically valid Python."""
        code = self._generate_code(seed=999)
        compile(code, "<test>", "exec")


if __name__ == "__main__":
    unittest.main()
