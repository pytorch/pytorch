#!/usr/bin/env python3
"""Tests for the in-session findings validator.

The validator's whole justification is that it answers with the SAME verdict the
publishing sanitizer will reach. So the tests here are mostly agreement tests: a
file the validator passes must survive `extract_verdict.py` with nothing dropped,
and a file it rejects must be one that really would lose something. A validator
that is merely "roughly right" is worse than none, because it tells the model its
file is fine and the findings vanish anyway — which is the exact failure that
motivated it.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("validate_findings.py")
sys.path.insert(0, str(SCRIPT.parent))

from _suite_manifest import TestTheSuiteIsWhole  # noqa: F401,E402


# A new 6-line file, so new-side lines 1..6 are anchorable and 7+ are not.
DIFF = """\
diff --git a/pkg/sample.py b/pkg/sample.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/pkg/sample.py
@@ -0,0 +1,6 @@
+import os
+
+
+def run(cmd):
+    os.system(cmd)
+    return 0
"""


def _finding(line: int, path: str = "pkg/sample.py") -> dict:
    return {
        "path": path,
        "line": line,
        "severity": "major",
        "message": "shell injection",
    }


def _verdict(findings: list[dict]) -> dict:
    return {
        "verdict": "changes_requested",
        "summary": "One issue worth fixing before review.",
        "findings": findings,
    }


class ValidatorHarness(unittest.TestCase):
    def run_validator(self, payload, diff: str = DIFF):
        """Run the validator as a subprocess; return (exit_code, stderr)."""
        with tempfile.TemporaryDirectory() as td:
            findings = Path(td) / "findings.json"
            if payload is not None:
                findings.write_text(
                    payload if isinstance(payload, str) else json.dumps(payload)
                )
            diff_file = Path(td) / "diff.txt"
            diff_file.write_text(diff)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--findings-file",
                    str(findings),
                    "--diff-file",
                    str(diff_file),
                ],
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(SCRIPT.parent), "PATH": "/usr/bin:/bin"},
            )
        return proc.returncode, proc.stderr


class TestAgreesWithThePublishingSanitizer(ValidatorHarness):
    """The claim the model is asked to trust: pass here means nothing is lost."""

    def test_a_correctly_anchored_finding_passes(self):
        code, err = self.run_validator(_verdict([_finding(5)]))
        self.assertEqual(code, 0, err)
        self.assertIn("VALID", err)
        self.assertIn("none discarded", err)

    def test_an_empty_findings_list_passes_when_the_verdict_is_positive(self):
        payload = _verdict([])
        payload["verdict"] = "ready_for_human_review"
        code, err = self.run_validator(payload)
        self.assertEqual(code, 0, err)

    def test_changes_requested_with_no_findings_is_refused(self):
        """An objection with no evidence. Caught here rather than at publication,
        where it would discard the whole review instead of one finding."""
        code, err = self.run_validator(_verdict([]))
        self.assertEqual(code, 1)
        self.assertIn("no publishable finding", err)

    def test_a_line_past_the_end_of_the_file_fails(self):
        """The exact shape of the bug this exists to catch: diff-row numbering."""
        code, err = self.run_validator(_verdict([_finding(11)]))
        self.assertEqual(code, 1)
        self.assertIn("line_not_in_diff", err)

    def test_the_failure_names_the_lines_that_would_work(self):
        """A rejection the model cannot act on is a rejection it will repeat."""
        code, err = self.run_validator(_verdict([_finding(11)]))
        self.assertEqual(code, 1)
        self.assertIn("1-6", err)
        self.assertIn("pkg/sample.py", err)

    def test_a_file_the_pr_never_touched_fails_and_lists_the_real_files(self):
        code, err = self.run_validator(_verdict([_finding(1, "pkg/other.py")]))
        self.assertEqual(code, 1)
        self.assertIn("path_not_in_diff", err)
        self.assertIn("pkg/sample.py", err)

    def test_passing_the_validator_means_extract_verdict_drops_nothing(self):
        """Agreement, asserted rather than assumed — same input, both paths."""
        from extract_verdict import build, parse_diff

        payload = _verdict([_finding(4), _finding(5)])
        code, err = self.run_validator(payload)
        self.assertEqual(code, 0, err)
        result = build(payload, parse_diff(DIFF))
        self.assertEqual(result["findings_dropped"], 0)
        self.assertEqual(len(result["findings"]), 2)

    def test_failing_the_validator_means_extract_verdict_really_would_drop(self):
        from extract_verdict import build, parse_diff

        payload = _verdict([_finding(4), _finding(99)])
        code, _ = self.run_validator(payload)
        self.assertEqual(code, 1)
        result = build(payload, parse_diff(DIFF))
        self.assertEqual(result["findings_dropped"], 1)


class TestUnusableInputIsBlamedOnTheRightThing(ValidatorHarness):
    """A model told its own file is broken, when the workflow is, cannot recover."""

    def test_a_missing_file_says_write_it(self):
        code, err = self.run_validator(None)
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)
        self.assertIn("Write tool", err)

    def test_malformed_json_is_reported_as_such(self):
        code, err = self.run_validator("{not json")
        self.assertEqual(code, 1)
        self.assertIn("INVALID", err)

    def test_an_unreadable_diff_is_not_blamed_on_the_findings_file(self):
        """Otherwise every finding reads as unanchorable and the model rewrites
        a file that was already correct."""
        with tempfile.TemporaryDirectory() as td:
            findings = Path(td) / "findings.json"
            findings.write_text(json.dumps(_verdict([_finding(5)])))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--findings-file",
                    str(findings),
                    "--diff-file",
                    str(Path(td) / "absent.txt"),
                ],
                capture_output=True,
                text=True,
                env={"PYTHONPATH": str(SCRIPT.parent), "PATH": "/usr/bin:/bin"},
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("CANNOT VALIDATE", proc.stderr)
        self.assertIn("workflow problem", proc.stderr)

    def test_an_empty_diff_is_never_silently_treated_as_no_files(self):
        """An empty diff anchors nothing, so it must not read as a clean pass."""
        code, err = self.run_validator(_verdict([_finding(5)]), diff="")
        self.assertEqual(code, 1)
        self.assertIn("path_not_in_diff", err)


if __name__ == "__main__":
    unittest.main()
