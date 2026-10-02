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
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401
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


class TestTokenColumnsCountSubAgents(unittest.TestCase):
    """`usage` is the top-level session only; `modelUsage` includes sub-agents."""

    # Shape and numbers from a headless Claude Code 2.1.280 run with one
    # sub-agent on the same model as the main session.
    RESULT = {
        "total_cost_usd": 0.5766728,
        "usage": {
            "input_tokens": 4,
            "output_tokens": 156,
            "cache_read_input_tokens": 10790,
            "cache_creation_input_tokens": 54627,
        },
        "modelUsage": {
            "claude-opus-4-8": {
                "inputTokens": 12,
                "outputTokens": 2730,
                "cacheReadInputTokens": 59574,
                "cacheCreationInputTokens": 102022,
                "costUSD": 0.5766728,
            }
        },
    }

    def _metrics(self, payload):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "usage.json"
            p.write_text(json.dumps(payload))
            return usage_metrics(str(p))

    def test_tokens_come_from_model_usage(self):
        m = self._metrics(self.RESULT)
        self.assertEqual(
            (
                m["input_tokens"],
                m["output_tokens"],
                m["cache_read_input_tokens"],
                m["cache_creation_input_tokens"],
            ),
            (12, 2730, 59574, 102022),
        )

    def test_several_models_are_summed(self):
        payload = dict(self.RESULT)
        payload["modelUsage"] = {
            "a": {"outputTokens": 100, "inputTokens": 1},
            "b": {"outputTokens": 23, "inputTokens": "x"},
        }
        m = self._metrics(payload)
        self.assertEqual((m["output_tokens"], m["input_tokens"]), (123, 1))

    def test_falls_back_to_usage_without_model_usage_tokens(self):
        for model_usage in ({}, {"m": {}}, {"m": "junk"}, None, [1]):
            payload = dict(self.RESULT, modelUsage=model_usage)
            self.assertEqual(self._metrics(payload)["output_tokens"], 156)


class TestSafeModelIsFullyAnchored(unittest.TestCase):
    """`$` also matches before a final newline, so `.match` let one through."""

    def test_a_trailing_newline_is_rejected(self):
        self.assertEqual(safe_model("claude\n"), "")

    def test_the_bound_cannot_be_exceeded_by_a_newline(self):
        self.assertEqual(safe_model("x" * 128 + "\n"), "")

    def test_an_ordinary_identifier_still_passes(self):
        self.assertEqual(safe_model("claude-sonnet-4"), "claude-sonnet-4")

    def test_the_bound_itself_still_passes(self):
        self.assertEqual(safe_model("x" * 128), "x" * 128)

    def test_over_the_bound_is_still_rejected(self):
        self.assertEqual(safe_model("x" * 129), "")


if __name__ == "__main__":
    run_this_suite()
