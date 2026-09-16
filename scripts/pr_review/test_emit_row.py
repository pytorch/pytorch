#!/usr/bin/env python3
"""Tests for the telemetry row builder's one attacker-adjacent string.

`model` is the only value in the row we do not construct ourselves: it is a key
from the usage file's `modelUsage`, produced by a jq pass in the job that reads
untrusted PR code. Influence over it is marginal — but it was the single string
crossing that boundary without validation, and it lands in ClickHouse.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import TestTheSuiteIsWhole  # noqa: E402,F401
from emit_row import safe_model, usage_metrics  # noqa: E402


class TestSafeModel(unittest.TestCase):
    def test_real_bedrock_model_ids_survive(self):
        for good in (
            "global.anthropic.claude-sonnet-4-6",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "claude-opus-4-8",
        ):
            self.assertEqual(safe_model(good), good)

    def test_rendering_and_injection_shapes_are_dropped(self):
        for bad in (
            "claude\nrow_injected: true",  # newline — JSONEachRow record split
            "claude<script>alert(1)</script>",  # markup, if a dashboard renders it
            "claude sonnet",  # space is not in the charset
            "modelo-éé",  # non-ASCII
            "‮model",  # RTL override
            "x" * 129,  # over the length bound
        ):
            self.assertEqual(safe_model(bad), "", f"{bad!r} should not survive")

    def test_non_strings_and_empty_become_empty(self):
        for value in (None, 42, {"a": 1}, ["x"], ""):
            self.assertEqual(safe_model(value), "")


class TestUsageMetricsAppliesIt(unittest.TestCase):
    """The validator has to be wired in, not merely defined."""

    def _write(self, tmp, payload):
        import json

        p = Path(tmp) / "usage.json"
        p.write_text(json.dumps(payload))
        return str(p)

    def test_hostile_model_key_does_not_reach_the_row(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"modelUsage": {"evil\nphase: started": {"in": 1}}})
            self.assertEqual(usage_metrics(path)["model"], "")

    def test_benign_model_key_still_reaches_the_row(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, {"modelUsage": {"global.anthropic.claude-sonnet-4-6": {}}}
            )
            self.assertEqual(
                usage_metrics(path)["model"], "global.anthropic.claude-sonnet-4-6"
            )


if __name__ == "__main__":
    unittest.main()
