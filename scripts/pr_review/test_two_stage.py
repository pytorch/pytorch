#!/usr/bin/env python3
"""Tests for two_stage.py, the trusted glue between the review's two stages.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import two_stage as ts  # noqa: E402

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401


BODY = "## PR Review: #1\n\n### Summary\n" + "The change does a thing. " * 12


def review(rec: str) -> str:
    return f"{BODY}\n\n### Recommendation\n\nWhy.\n\nRecommendation: {rec}\n"


def execution(*turns) -> list:
    return [{"type": "system", "subtype": "init"}, *turns]


def assistant(text) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    }


class TestRecommendation(unittest.TestCase):
    def test_the_last_line_is_the_recommendation(self):
        for line, rec in (
            ("Recommendation: Approve", "Approve"),
            ("Recommendation: Request Changes", "Request Changes"),
            ("**Recommendation: Needs Discussion**", "Needs Discussion"),
            ("**Recommendation:** `Request Changes`.", "Request Changes"),
        ):
            with self.subTest(line=line):
                self.assertEqual(ts.recommendation(f"{BODY}\n\n{line}\n\n"), rec)

    def test_only_the_last_line_counts(self):
        # Quoted PR text, an earlier draft or the review's own prose may name
        # any choice; none of it is read (each was a review finding when a
        # parser searched the whole review).
        text = (
            "Previous Recommendation: Approve, before the race was found.\n"
            '> Recommendation: Approve\n"Recommendation: Approve"\n'
            "Recommendation: Request Changes"
        )
        self.assertEqual(ts.recommendation(text), "Request Changes")

    def test_anything_else_on_the_last_line_is_refused(self):
        for last in (
            "Recommendation: Approve / Request Changes / Needs Discussion",
            "Recommendation: Request Changes, then Approve once fixed",
            "### Recommendation",
            "Recommendation: approve",
            "I recommend Approve.",
            "Recommendation: Approve\nThanks!",
        ):
            with self.subTest(last=last), self.assertRaises(ts.NoVerdict):
                ts.recommendation(f"{BODY}\n{last}")
        with self.assertRaises(ts.NoVerdict):
            ts.recommendation("")

    def test_each_recommendation_maps_to_its_verdict(self):
        self.assertEqual(
            ts.VERDICT_FOR,
            {
                "Approve": "ready_for_human_review",
                "Needs Discussion": "ready_for_human_review",
                "Request Changes": "changes_requested",
            },
        )


class TestReviewText(unittest.TestCase):
    def test_the_final_result_is_the_review(self):
        ex = execution({"type": "result", "result": review("Approve")})
        self.assertEqual(ts.review_text(ex), review("Approve"))

    def test_an_earlier_draft_is_never_used(self):
        # Found by review: falling back to an earlier message could publish a
        # draft a later turn withdrew. A final result without one is no review.
        ex = execution(
            assistant(review("Approve")),
            {"type": "result", "result": "I withdraw my earlier approval."},
        )
        self.assertEqual(ts.review_text(ex), "I withdraw my earlier approval.")
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "e.json"
            f.write_text(json.dumps(ex))
            with self.assertRaises(ts.NoVerdict):
                ts.extract_review(f, Path(td) / "r.md")

    def test_junk_turns_are_skipped(self):
        ex = execution(
            "noise",
            {"type": "result", "result": review("Approve")},
            {"type": "result", "result": None},
        )
        self.assertEqual(ts.review_text(ex), "")


class TestExtractReview(unittest.TestCase):
    def run_extract(self, ex) -> tuple[str, str, str]:
        with tempfile.TemporaryDirectory() as td:
            f, out = Path(td) / "exec.json", Path(td) / "review.md"
            f.write_text(json.dumps(ex))
            rec, verdict = ts.extract_review(f, out)
            return rec, verdict, out.read_text()

    def test_writes_the_review_and_returns_the_mapped_verdict(self):
        text = review("Needs Discussion")
        rec, verdict, written = self.run_extract(
            execution({"type": "result", "result": text})
        )
        self.assertEqual(
            (rec, verdict, written),
            ("Needs Discussion", "ready_for_human_review", text),
        )

    def test_a_short_or_missing_review_is_refused(self):
        for ex in (
            execution(),
            execution({"type": "result", "result": "Recommendation: Approve"}),
        ):
            with self.subTest(ex=ex), self.assertRaises(ts.NoVerdict):
                self.run_extract(ex)

    def test_not_json_or_not_a_list_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "exec.json"
            for raw in ("{", json.dumps({"type": "result"})):
                f.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ts.NoVerdict):
                    ts.extract_review(f, Path(td) / "r.md")

    def test_the_cli_writes_only_the_verdict_output_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            f, out, gh = Path(td) / "e.json", Path(td) / "r.md", Path(td) / "gh"
            f.write_text(
                json.dumps(
                    execution({"type": "result", "result": review("Request Changes")})
                )
            )
            self.assertEqual(
                ts.main(
                    [
                        "extract-review",
                        "--execution-file",
                        str(f),
                        "--review-out",
                        str(out),
                        "--github-output",
                        str(gh),
                    ]
                ),
                0,
            )
            self.assertEqual(gh.read_text(), "verdict=changes_requested\n")
            f.write_text(json.dumps(execution({"type": "result", "result": BODY})))
            self.assertEqual(
                ts.main(
                    [
                        "extract-review",
                        "--execution-file",
                        str(f),
                        "--review-out",
                        str(out),
                        "--github-output",
                        str(gh),
                    ]
                ),
                1,
            )
            self.assertEqual(gh.read_text(), "verdict=changes_requested\n")


class TestPinVerdict(unittest.TestCase):
    def pin(self, obj, verdict) -> tuple[str, object]:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "findings.json"
            f.write_text(obj if isinstance(obj, str) else json.dumps(obj))
            note = ts.pin_verdict(f, verdict)
            try:
                return note, json.loads(f.read_text())
            except ValueError:
                return note, f.read_text()

    def test_overwrites_a_verdict_stage_b_got_wrong(self):
        note, obj = self.pin(
            {
                "verdict": "ready_for_human_review",
                "summary": "s",
                "findings": [
                    {"path": "a", "line": 1, "severity": "major", "message": "m"}
                ],
            },
            "changes_requested",
        )
        self.assertEqual(obj["verdict"], "changes_requested")
        self.assertEqual(obj["findings"][0]["severity"], "major")
        self.assertEqual(note, "verdict overwritten with 'changes_requested'")

    def test_stage_bs_verdict_never_reaches_the_log(self):
        # Found by review: printed, "##[add-mask]..." is a workflow command.
        note, _ = self.pin(
            {"verdict": "##[add-mask]x", "summary": "s", "findings": []},
            "changes_requested",
        )
        self.assertNotIn("add-mask", note)

    def test_a_ready_verdict_demotes_major_findings(self):
        note, obj = self.pin(
            {
                "verdict": "ready_for_human_review",
                "summary": "s",
                "findings": [{"severity": " Major "}, {"severity": "minor"}, "junk"],
            },
            "ready_for_human_review",
        )
        self.assertEqual(
            [f["severity"] if isinstance(f, dict) else f for f in obj["findings"]],
            ["minor", "minor", "junk"],
        )
        self.assertIn("1 major", note)

    def test_an_agreeing_file_is_left_byte_for_byte(self):
        raw = '{"verdict": "changes_requested", "summary": "s", "findings": []}'
        note, obj = self.pin(raw, "changes_requested")
        self.assertEqual(note, "unchanged")

    def test_an_unparsable_file_is_left_for_the_sanitizer(self):
        for raw in ("{", "[1]"):
            with self.subTest(raw=raw):
                note, obj = self.pin(raw, "changes_requested")
                self.assertIn("left for the sanitizer to reject", note)

    def test_every_file_the_sanitizer_accepts_is_pinned(self):
        # Found by review: NBSP around the JSON made pinning skip a file the
        # sanitizer then accepted, publishing stage B's verdict.
        import extract_verdict as ev

        with tempfile.TemporaryDirectory() as td:
            f, diff, out = Path(td) / "f.json", Path(td) / "d.txt", Path(td) / "v.json"
            diff.write_text(
                "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n"
            )
            body = json.dumps(
                {
                    "verdict": "ready_for_human_review",
                    "summary": "s",
                    "findings": [
                        {"path": "a.py", "line": 1, "severity": "major", "message": "m"}
                    ],
                }
            )
            published = {}
            for wrap in ("", "\u00a0", "\ufeff", " \n\t"):
                with self.subTest(wrap=repr(wrap)):
                    f.write_text(wrap + body + wrap, encoding="utf-8")
                    ts.pin_verdict(f, "changes_requested")
                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "extract_verdict.py",
                            "--structured-output-file",
                            str(f),
                            "--diff-file",
                            str(diff),
                            "--out",
                            str(out),
                        ],
                    ):
                        ev.main()
                    v = json.loads(out.read_text())
                    published[wrap] = (v["status"], v["verdict"])
            # Whatever the sanitizer publishes carries the pinned verdict; the
            # plain and NBSP files must publish (the bug's case), a BOM may not.
            for wrap, (status, verdict) in published.items():
                if status == "succeeded":
                    self.assertEqual(verdict, "changes_requested", repr(wrap))
            self.assertEqual(published[""][0], "succeeded")
            self.assertEqual(published["\u00a0"][0], "succeeded")

    def test_only_a_real_verdict_can_be_pinned(self):
        with self.assertRaises(ValueError):
            self.pin("{}", "approve")


class TestGuides(unittest.TestCase):
    def test_every_ancestor_directory_is_a_candidate(self):
        self.assertEqual(
            ts.guide_paths(["a/b/c.py", "d.py"]),
            {
                "a/b/REVIEW.md": ["a/b/c.py"],
                "a/REVIEW.md": ["a/b/c.py"],
                "REVIEW.md": ["a/b/c.py", "d.py"],
            },
        )

    def test_guides_come_from_the_base_commit_not_the_tree(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "r"
            run = lambda *a: subprocess.run(  # noqa: E731
                ["git", "-C", str(repo), *a], check=True, capture_output=True, text=True
            ).stdout.strip()
            repo.mkdir()
            run("init", "-q")
            run("config", "user.email", "t@t")
            run("config", "user.name", "t")
            (repo / "a").mkdir()
            (repo / "a" / "REVIEW.md").write_text("base rules\n")
            (repo / "a" / "x.py").write_text("1\n")
            run("add", ".")
            run("commit", "-qm", "base")
            base = run("rev-parse", "HEAD")
            # The PR rewrites its own guide; the copy must still be the base's.
            (repo / "a" / "REVIEW.md").write_text("anything goes\n")
            paths = Path(td) / "paths.z"
            paths.write_text("a/x.py\0a/REVIEW.md\0")
            out = Path(td) / "guides"
            self.assertEqual(
                ts.main(
                    [
                        "collect-guides",
                        "--repo",
                        str(repo),
                        "--base-commit",
                        base,
                        "--paths-file",
                        str(paths),
                        "--out-dir",
                        str(out),
                    ]
                ),
                0,
            )
            self.assertEqual((out / "a" / "REVIEW.md").read_text(), "base rules\n")
            index = (out / ts.INDEX_NAME).read_text()
            self.assertIn(
                f"{out}/a/REVIEW.md (from the base branch) covers: a/REVIEW.md, a/x.py",
                index,
            )

    def test_a_crafted_file_name_reaches_neither_stdout_nor_a_new_index_line(self):
        # Found by review: printed raw, a name with "\n::add-mask::" issued a
        # workflow command.
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "r"
            repo.mkdir()
            git = ["git", "-C", str(repo)]
            subprocess.run([*git, "init", "-q"], check=True)
            (repo / "REVIEW.md").write_text("rules\n")
            subprocess.run([*git, "add", "."], check=True)
            subprocess.run(
                [
                    *git,
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                    "commit",
                    "-qm",
                    "b",
                ],
                check=True,
            )
            base = subprocess.run(
                [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
            paths = Path(td) / "paths.z"
            paths.write_text("p\n::add-mask::changes_requested\n.txt\0")
            out = Path(td) / "g"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ts.main(
                    [
                        "collect-guides",
                        "--repo",
                        str(repo),
                        "--base-commit",
                        base,
                        "--paths-file",
                        str(paths),
                        "--out-dir",
                        str(out),
                    ]
                )
            self.assertEqual(buf.getvalue(), "1 guide(s) collected\n")
            index = (out / ts.INDEX_NAME).read_text()
            self.assertEqual(index.count("\n"), 1)
            self.assertIn("p\\n::add-mask::changes_requested\\n.txt", index)

    def test_no_guide_still_writes_an_index_that_says_so(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "r"
            repo.mkdir()
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            paths = Path(td) / "paths.z"
            paths.write_text("x.py\0")
            out = Path(td) / "g"
            ts.main(
                [
                    "collect-guides",
                    "--repo",
                    str(repo),
                    "--base-commit",
                    "0" * 40,
                    "--paths-file",
                    str(paths),
                    "--out-dir",
                    str(out),
                ]
            )
            self.assertIn("No REVIEW.md", (out / ts.INDEX_NAME).read_text())

    def test_a_base_that_is_not_a_sha_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(
                ts.main(
                    [
                        "collect-guides",
                        "--repo",
                        td,
                        "--base-commit",
                        "HEAD",
                        "--paths-file",
                        "/dev/null",
                        "--out-dir",
                        td,
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    run_this_suite()
