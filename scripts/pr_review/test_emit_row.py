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
from emit_row import published_findings, safe_model, usage_metrics  # noqa: E402


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


FINDING = {
    "path": "torch/optim/lr\\_scheduler.py",
    "line": 42,
    "severity": "major",
    "message": "Allocates a new lr tensor every step; see \\[docs\\].",
}


class TestPublishedFindings(unittest.TestCase):
    """Findings reach Dr.CI through `extra`, so the shape is re-checked here."""

    def test_sanitizer_output_survives_unchanged(self):
        self.assertEqual(published_findings({"findings": [FINDING]}), [FINDING])

    def test_malformed_findings_are_dropped(self):
        bad = [
            "not a dict",
            {**FINDING, "extra": "key"},
            {k: v for k, v in FINDING.items() if k != "line"},
            {**FINDING, "line": True},
            {**FINDING, "line": 0},
            {**FINDING, "severity": "critical"},
            {**FINDING, "severity": []},
            {**FINDING, "severity": {"a": 1}},
            {**FINDING, "message": ""},
            {**FINDING, "message": "x" * 601},
            {**FINDING, "message": "a \u2014 b"},
            {**FINDING, "path": "a\nb.py"},
        ]
        self.assertEqual(published_findings({"findings": bad + [FINDING]}), [FINDING])

    def test_capped_at_the_sanitizer_bound(self):
        self.assertEqual(len(published_findings({"findings": [FINDING] * 40})), 25)

    def test_missing_or_wrong_typed_list_is_empty(self):
        for verdict in ({}, {"findings": None}, {"findings": "x"}):
            self.assertEqual(published_findings(verdict), [])


class TestTerminalRowCarriesFindings(unittest.TestCase):
    def _row(self, verdict: dict) -> dict:
        import json
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            verdict_file = Path(tmp) / "verdict.json"
            verdict_file.write_text(json.dumps(verdict))
            out = Path(tmp) / "row.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "emit_row.py"),
                    "--phase",
                    "terminal",
                    "--verdict-file",
                    str(verdict_file),
                    "--out",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            return json.loads(out.read_text())

    def test_succeeded_row_has_findings_json(self):
        import json

        row = self._row(
            {
                "status": "succeeded",
                "verdict": "changes_requested",
                "summary": "s",
                "findings": [FINDING],
            }
        )
        self.assertEqual(json.loads(row["extra"]["findings"]), [FINDING])

    def test_failed_row_has_no_findings(self):
        row = self._row({"status": "model_error", "findings": [FINDING]})
        self.assertEqual(row["extra"], {})

    def test_succeeded_row_without_findings_leaves_extra_empty(self):
        row = self._row(
            {
                "status": "succeeded",
                "verdict": "ready_for_human_review",
                "summary": "s",
                "findings": [],
            }
        )
        self.assertEqual(row["extra"], {})


if __name__ == "__main__":
    run_this_suite()
