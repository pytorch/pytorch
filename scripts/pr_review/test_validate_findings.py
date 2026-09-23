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

from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: F401,E402


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
    def run_validator(self, payload, diff: str = DIFF, files: str | None = ""):
        """Run the validator as a subprocess; return (exit_code, stderr).

        `files` is the changed-file list the workflow writes alongside the
        diff. The validator never reads it — it only tells the model where it
        is — so the content is immaterial and its EXISTENCE is not: `None`
        writes no file and exercises the fallback that keeps the model from
        being sent to a path that is not there.
        """
        with tempfile.TemporaryDirectory() as td:
            findings = Path(td) / "findings.json"
            if payload is not None:
                findings.write_text(
                    payload if isinstance(payload, str) else json.dumps(payload)
                )
            diff_file = Path(td) / "diff.txt"
            diff_file.write_text(diff)
            files_file = Path(td) / "pr-files.txt"
            if files is not None:
                files_file.write_text(files)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--findings-file",
                    str(findings),
                    "--diff-file",
                    str(diff_file),
                    "--files-file",
                    str(files_file),
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

    def test_a_file_the_pr_never_touched_fails_and_says_where_the_real_ones_are(
        self,
    ):
        """Names the file holding the list, not the list.

        This used to assert the changed files were printed. They are not any
        more, deliberately — see TestTheAdvisoryChannelIsBounded — so what the
        model needs from this message is a readable path, and that is what is
        asserted. The pointer sentence itself carries no pull-request-authored
        bytes; the drop line above it still names the model's own path, which
        is the residual this class's docstring sets out.
        """
        code, err = self.run_validator(_verdict([_finding(1, "pkg/other.py")]))
        self.assertEqual(code, 1)
        self.assertIn("path_not_in_diff", err)
        self.assertNotIn("pkg/sample.py", err)
        self.assertRegex(err, r"Read \S*pr-files\.txt for the files this PR changed")

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


class TestTheAdvisoryChannelIsBounded(ValidatorHarness):
    """The stderr here reaches the model as a SYSTEM message.

    `validate-post-write.sh` folds it into `hookSpecificOutput.additionalContext`,
    so anything printed arrives wearing trusted framing. Two directions of text
    could reach it, and they are now treated differently.

    THE DIFF'S OWN FILE NAMES no longer reach it. The report used to print the
    changed-file list, capped at 20 names and escaped, and it used to repeat
    the raw diff key in the line-miss hint. Both are gone: the model is pointed
    at the file the workflow already wrote and is already granted `Read` on.
    Capping bounded the channel; removing it is what #196844's finding asked
    for. The tests below pin the removal, not the cap — a re-introduced list
    would be a regression even if it were bounded.

    WHAT REMAINS, and it is not nothing. Each discarded finding's own `path` is
    still printed. That string is the MODEL's, and `sanitize_findings` has
    neutralized it — but the model chose it after reading the pull request, so
    a contributor who names a file after a sentence can still get that sentence
    in front of the model by inducing a finding against it. It is bounded to
    one name per discarded finding and 20 findings, and it is the only way to
    say WHICH finding was discarded, so it stays. Removing it too would mean
    identifying findings by index, which is a change to the message contract
    rather than to this list, and is not made here.
    """

    def _diff_touching(self, names: list[str]) -> str:
        return "".join(
            f"diff --git a/{n} b/{n}\n--- a/{n}\n+++ b/{n}\n@@ -1,0 +1,1 @@\n+x\n"
            for n in names
        )

    def test_no_changed_file_name_reaches_the_model_on_an_anchor_miss(self):
        """The whole finding, stated as a property rather than as a bound.

        An anchor miss is the ONE report that needs the changed-file set, so
        it is the one that used to carry the names. Sixty of them here: not one
        may appear, however short the list or however well escaped.
        """
        names = [f"pkg/f{i:03d}.py" for i in range(60)]
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]),
            diff=self._diff_touching(names),
        )
        self.assertEqual(code, 1)
        leaked = [n for n in names if n in err]
        self.assertEqual(
            leaked,
            [],
            f"{len(leaked)} pull-request-authored file name(s) reached the "
            f"model's system message: {leaked[:5]}",
        )
        # And the old shape specifically, so a partial revert is not silent.
        self.assertNotIn("files total", err)
        self.assertNotIn("Files this PR changed:", err)

    def test_the_model_is_told_where_to_read_the_list_instead(self):
        """Removing the list must not remove the model's way to fix the anchor.

        It cannot run a shell, so if this sentence does not name a readable
        file the finding is simply unfixable and the review loses it.
        """
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]),
            diff=self._diff_touching(["pkg/real.py"]),
            files="pkg/real.py\n",
        )
        self.assertEqual(code, 1)
        self.assertRegex(err, r"Read \S*pr-files\.txt for the files this PR changed")
        # AND THE RULE TRAVELS WITH IT. That file is built by
        # `git diff --name-only`, which lists deletions, renames, binaries and
        # mode-only changes; `parse_diff` admits none of them. A pointer
        # without this sentence tells the model to re-anchor onto an entry
        # that is guaranteed to be dropped, and it has no shell to tell the
        # two kinds of entry apart — so it loops until `--max-turns` ends the
        # session with no valid findings file at all.
        self.assertIn("remove it only if no such line exists", err)
        # And the rule does not over-mandate withdrawal: a defect in a file
        # this PR deleted is often still reportable at the line that now
        # imports it, so re-anchoring must be offered first.
        self.assertIn("Re-anchor the finding to a line that does", err)

    def test_an_absent_list_falls_back_to_the_diff_not_a_dead_pointer(self):
        """A pointer to a file that is not there is worse than no pointer.

        The diff is the fallback because the model is granted `Read` on it too
        — not because it is equivalent. It is strictly weaker: a rename, a
        mode-only change, a binary file or a pure deletion can appear in a diff
        with no usable `+++` path, so the message points at it as a place to
        look rather than promising a complete enumeration.
        """
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]),
            diff=self._diff_touching(["pkg/real.py"]),
            files=None,
        )
        self.assertEqual(code, 1)
        self.assertNotIn("pr-files.txt", err)
        self.assertRegex(err, r"named in the diff at \S*diff\.txt")

    def test_a_diff_with_no_anchorable_line_says_so_instead_of_pointing(self):
        """ "Read the list and anchor to one of those paths" is unfollowable
        when nothing is anchorable — the model would fetch the file and come
        back no better off. The changed-file list said "(the diff touches no
        files)" as a side effect of rendering; a pointer has to say it.

        It also must not blame the workflow. Pure renames and deletions
        legitimately anchor nothing, and the model's unsupported finding is
        still the model's to withdraw.
        """
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]), diff=""
        )
        self.assertEqual(code, 1)
        self.assertIn("path_not_in_diff", err)
        self.assertIn("no anchorable lines at all", err)
        self.assertIn("remove the ones that cannot be", err)
        self.assertNotIn("anchor to one of those paths", err)
        self.assertNotIn("workflow problem", err)

    def test_the_fallback_does_not_offer_a_removal_only_file_as_a_target(self):
        """A MIXED diff, which the two tests either side of this one miss.

        One file has a removal-only hunk (`{"x.py": set()}` — a hunk, and no
        anchorable line) and another has an ordinary edit. So
        `any(touched.values())` is TRUE and the no-anchors branch returns
        early, leaving the fallback to give per-path advice. An earlier draft
        said "anchor to a path that has a hunk there", which names the
        removal-only file as an eligible destination; and the enumeration
        beside it did not cover that case, because the file was not deleted,
        renamed, mode-changed or binary.
        """
        mixed = (
            "diff --git a/pkg/gone.py b/pkg/gone.py\n"
            "--- a/pkg/gone.py\n+++ b/pkg/gone.py\n@@ -1,2 +1,0 @@\n-a\n-b\n"
            "diff --git a/pkg/real.py b/pkg/real.py\n"
            "--- a/pkg/real.py\n+++ b/pkg/real.py\n@@ -0,0 +1,1 @@\n+x\n"
        )
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]), diff=mixed, files=None
        )
        self.assertEqual(code, 1)
        self.assertNotIn("has a hunk there", err)
        self.assertIn("contains an added or unchanged line", err)
        # The rule must name the removal-only case, or the model has no way to
        # know that `pkg/gone.py` — which IS in the diff — cannot take a finding.
        self.assertIn("only remove lines", err)

    def test_a_removal_only_hunk_counts_as_nothing_anchorable(self):
        """`parse_diff` returns `{"x.py": set()}` for a hunk that only removes
        lines — a non-empty mapping with nothing anchorable under it. A
        `if not touched` guard reads that as "fine" and points the model at a
        list it cannot use. Measured against `parse_diff`, not assumed.
        """
        removal_only = (
            "diff --git a/pkg/x.py b/pkg/x.py\n"
            "--- a/pkg/x.py\n+++ b/pkg/x.py\n@@ -1,2 +1,0 @@\n-a\n-b\n"
        )
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/absent.py")]), diff=removal_only
        )
        self.assertEqual(code, 1)
        self.assertIn("no anchorable lines at all", err)

    def test_the_line_hint_names_no_file_because_the_drop_line_already_did(self):
        """ "Lines this PR touches in X: 1-6" printed X as the RAW diff key —
        a second rendering of a contributor-chosen name, differently escaped
        from the one on the line above it, carrying no information the line
        above did not already carry. The ranges stay; the name goes.
        """
        name = "pkg/my_file.py"
        code, err = self.run_validator(
            # The path IS in the diff and the line is not: the line-miss branch.
            _verdict([_finding(99, path=name)]),
            diff=self._diff_touching([name]),
        )
        self.assertEqual(code, 1)
        self.assertIn("line_not_in_diff", err)
        self.assertIn("Lines this PR touches in that file: 1", err)
        # The drop line names the finding once, in the model's own spelling.
        self.assertEqual(err.count("my_file.py"), 1, err)

    def test_the_only_name_printed_is_the_one_the_model_itself_wrote(self):
        """The residual, asserted so it is a decision and not an oversight.

        A contributor can name a file after an instruction. If the model then
        raises a finding against that file and mis-anchors it, the name comes
        back in the drop line. Bounded — one per discarded finding, capped at
        20 — and neutralized, but present. What must NOT happen is a name the
        model never mentioned appearing because it was merely in the diff.
        """
        mentioned = "pkg/model_named_this.py"
        others = [f"pkg/never_mentioned{i:02d}.py" for i in range(30)]
        code, err = self.run_validator(
            _verdict([_finding(99, path=mentioned)]),
            diff=self._diff_touching([mentioned] + others),
        )
        self.assertEqual(code, 1)
        # As the publisher spells it. `neutralize` defuses `@`, `#` and `[`,
        # and deliberately leaves `_` alone — this channel is a system message
        # the model reads, not markdown a human renders.
        self.assertIn(mentioned, err)
        leaked = [n for n in others if n in err]
        self.assertEqual(
            leaked, [], f"unmentioned diff paths reached the model: {leaked}"
        )

    def test_the_drop_report_itself_is_bounded(self):
        """The other direction: what the MODEL wrote comes back through here too.

        `sanitize_findings` neutralizes each dropped `path` but tracks up to 200
        records, so without a cap an injected model could push its own bulk text
        back into its own system message.
        """
        findings = [_finding(1, path=f"pkg/absent{i:03d}.py") for i in range(40)]
        code, err = self.run_validator(
            _verdict(findings), diff=self._diff_touching(["pkg/sample.py"])
        )
        self.assertEqual(code, 1)
        # MAX_FINDINGS caps the input at 25; MAX_REPORTED_DROPS caps the report
        # at 20. The count is still stated in full, so nothing is hidden.
        self.assertIn("25 finding(s) would be DISCARDED", err)
        self.assertIn("showing the first 20", err)
        self.assertIn("absent000.py", err)
        self.assertNotIn("absent024.py", err)

    def test_the_pointer_is_printed_once_per_report_not_once_per_drop(self):
        """The repetition that made the old list expensive is pinned separately.

        Measured before either fix: 100 names near `MAX_PATH` with 20 anchor
        misses produced 161,113 bytes of advisory output, 20 copies of one
        capped list. The list is gone, but the sentence replacing it sits at
        the same call site, so the once-per-report property still needs a test
        — and this one would also catch a re-introduced list, since a revert
        would put it back inside the loop it was moved out of.
        """
        names = [f"pkg/f{i:03d}.py" for i in range(40)]
        findings = [_finding(1, path=f"pkg/absent{i:03d}.py") for i in range(10)]
        code, err = self.run_validator(
            _verdict(findings), diff=self._diff_touching(names)
        )
        self.assertEqual(code, 1)
        self.assertEqual(err.count("for the files this PR changed"), 1, err[:400])
        # Ten drops, so ten lines of model-authored detail and no more.
        self.assertEqual(err.count("path_not_in_diff"), 10, err[:400])

    def test_a_dropped_path_is_reported_as_the_publisher_escaped_it(self):
        """Not re-escaped: it has already been through `sanitize_findings`.

        Running the already-escaped value through `_is_repo_path` again would
        report `pkg/[click].py` — a name the publisher accepts — as unusable,
        and tell the model to fix a file name that was never the problem.
        """
        code, err = self.run_validator(
            _verdict([_finding(1, path="pkg/[click].py")]),
            diff=self._diff_touching(["pkg/sample.py"]),
        )
        self.assertEqual(code, 1)
        self.assertIn("pkg/\\[click\\].py", err)
        self.assertNotIn("not a usable repo path", err)


if __name__ == "__main__":
    run_this_suite()
