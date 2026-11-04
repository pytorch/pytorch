"""Tests for interdiff.py: the author's changes since the last review, rebases left out."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import interdiff  # noqa: E402

# Imported so this module cannot run on its own without the guard against a
# shrunk suite. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401


class Repo:
    """A throwaway repository; `commit` writes files on top of a parent and returns the sha."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, parent: str | None, files: dict[str, str]) -> str:
        if parent:
            self.git("checkout", "-q", "--detach", parent)
        for name, text in files.items():
            (self.root / name).write_text(text)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "c")
        return self.git("rev-parse", "HEAD")


class TestInterdiff(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Repo(Path(tmp.name))
        cwd = os.getcwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(tmp.name)
        self.addCleanup(setattr, interdiff, "REPO", interdiff.REPO)
        r = self.repo
        self.b0 = r.commit(None, {"a.txt": "1\n2\n3\n", "b.txt": "x\n"})
        self.v1 = r.commit(
            self.b0, {"a.txt": "1-pr\n2\n3\n"}
        )  # the version the last review saw

    def checkout(self, sha: str) -> None:
        self.repo.git("checkout", "-q", "--detach", sha)

    def test_changes_from_the_base_are_left_out_after_a_rebase(self):
        r = self.repo
        # Upstream edits the PR's file in a place the PR does not touch, and another file.
        b1 = r.commit(self.b0, {"a.txt": "1\n2\n3-up\n", "b.txt": "x-up\n"})
        # The author rebases onto it and changes line 2.
        v2 = r.commit(b1, {"a.txt": "1-pr\n2-pr\n3-up\n"})
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, b1, ["a.txt"])
        self.assertIn("+2-pr", text)
        # Context lines may show the base's text; no added or removed line may.
        changed = [
            ln
            for ln in text.splitlines()
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
        ]
        self.assertEqual(
            changed, ["-2", "+2-pr"], "the base's change was attributed to the author"
        )
        self.assertNotIn(interdiff.CONFLICTED, text)

    def test_a_file_both_sides_changed_gets_the_plain_diff_under_its_heading(self):
        r = self.repo
        b1 = r.commit(
            self.b0, {"a.txt": "1-up\n2\n3\n"}
        )  # conflicts with the PR's line 1
        v2 = r.commit(b1, {"a.txt": "1-pr-up\n2\n3\n"})
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, b1, ["a.txt"])
        head, _, conflicted = text.partition(interdiff.CONFLICTED)
        self.assertNotIn("diff --git", head)
        # Against the earlier commit, not the replayed tree and its conflict markers.
        changed = [
            ln
            for ln in conflicted.splitlines()
            if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
        ]
        self.assertEqual(changed, ["-1-pr", "+1-pr-up"])

    def test_the_prs_gitattributes_cannot_hide_a_conflict(self):
        r = self.repo
        b1 = r.commit(self.b0, {"a.txt": "1-up\n2\n3\n"})
        # With `merge=union` honored, the replay would merge a.txt "cleanly" with both sides' lines.
        v2 = r.commit(
            b1, {"a.txt": "1-pr-up\n2\n3\n", ".gitattributes": "* merge=union\n"}
        )
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, b1, ["a.txt", ".gitattributes"])
        head, _, conflicted = text.partition(interdiff.CONFLICTED)
        self.assertIn("+1-pr-up", conflicted)
        self.assertNotIn("a.txt", head)

    def test_a_reverted_file_shows_up(self):
        # The author answers a finding by dropping the PR's change to a.txt; the PR now changes only b.txt.
        v2 = self.repo.commit(self.v1, {"a.txt": "1\n2\n3\n", "b.txt": "x-pr\n"})
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, self.b0, ["b.txt"])
        self.assertIn("-1-pr", text)
        self.assertIn("+x-pr", text)

    def test_a_rebase_with_no_change_from_the_author_says_so(self):
        r = self.repo
        b1 = r.commit(self.b0, {"b.txt": "x-up\n"})
        v2 = r.commit(b1, {"a.txt": "1-pr\n2\n3\n"})
        self.checkout(v2)
        self.assertEqual(
            interdiff.interdiff(self.v1, self.b0, b1, ["a.txt"]),
            interdiff.NOTE_BASE_ONLY.format(sha=self.v1),
        )

    def test_without_a_rebase_it_is_the_plain_diff(self):
        v2 = self.repo.commit(self.v1, {"a.txt": "1-pr\n2-pr\n3\n"})
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, self.b0, ["a.txt"])
        self.assertIn("+2-pr", text)
        self.assertIn("-2\n", text)

    def test_notes_instead_of_failures(self):
        self.checkout(self.b0)
        self.assertEqual(
            interdiff.interdiff("", "", self.b0, ["a.txt"]), interdiff.NOTE_NO_REVIEW
        )
        self.assertEqual(
            interdiff.interdiff(self.v1, self.b0, self.b0, []), interdiff.NOTE_NO_FILES
        )
        gone = "f" * 40
        self.assertEqual(
            interdiff.interdiff(gone, self.b0, self.b0, ["a.txt"]),
            interdiff.NOTE_UNAVAILABLE.format(sha=gone),
        )
        self.assertEqual(
            interdiff.interdiff(self.v1, "", self.b0, ["a.txt"]),
            interdiff.NOTE_UNAVAILABLE.format(sha=self.v1),
        )

    def test_main_writes_the_file_and_caps_its_size(self):
        v2 = self.repo.commit(self.v1, {"a.txt": "1-pr\n2-pr\n3\n"})
        self.checkout(v2)
        files, out = Path("files.z"), Path("out.txt")
        files.write_bytes(b"a.txt\0")
        args = [
            "--previous",
            self.v1,
            "--previous-base",
            self.b0,
            "--base",
            self.b0,
            "--files",
            str(files),
            "--out",
            str(out),
        ]
        self.assertEqual(interdiff.main([*args, "--max-bytes", "100000"]), 0)
        self.assertIn("+2-pr", out.read_text())
        interdiff.main([*args, "--max-bytes", "10"])
        self.assertEqual(out.read_text(), interdiff.NOTE_TOO_LARGE)

    def test_main_reads_the_repo_it_is_given(self):
        # As in the workflow: run from elsewhere with --repo pointing at the PR's working copy.
        v2 = self.repo.commit(self.v1, {"a.txt": "1-pr\n2-pr\n3\n"})
        self.checkout(v2)
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        os.chdir(elsewhere.name)
        files, out = Path("files.z"), Path("out.txt")
        files.write_bytes(b"a.txt\0")
        interdiff.main(
            [
                "--repo",
                str(self.repo.root),
                "--previous",
                self.v1,
                "--previous-base",
                self.b0,
                "--base",
                self.b0,
                "--files",
                str(files),
                "--out",
                str(out),
                "--max-bytes",
                "100000",
            ]
        )
        self.assertIn("+2-pr", out.read_text())

    def test_file_names_are_literal_not_pathspec_magic(self):
        # As pathspec magic, this name means "every file but a.txt", which would pull in b.txt.
        name = ":(exclude)a.txt"
        v2 = self.repo.commit(self.v1, {name: "new\n", "b.txt": "x-changed\n"})
        self.checkout(v2)
        text = interdiff.interdiff(self.v1, self.b0, self.b0, [name])
        self.assertIn("+new", text)
        self.assertNotIn("b.txt", text)

    @unittest.skipIf(
        sys.platform == "darwin", "APFS rejects file names that are not UTF-8"
    )
    def test_a_file_name_that_is_not_utf8_is_handled(self):
        name = os.fsdecode(b"bad-\xff.txt")
        v2 = self.repo.commit(self.v1, {name: "new\n"})
        self.checkout(v2)
        files, out = Path("files.z"), Path("out.txt")
        files.write_bytes(os.fsencode(name) + b"\0")
        interdiff.main(
            [
                "--previous",
                self.v1,
                "--previous-base",
                self.b0,
                "--base",
                self.b0,
                "--files",
                str(files),
                "--out",
                str(out),
                "--max-bytes",
                "100000",
            ]
        )
        self.assertIn("+new", out.read_text())

    def test_a_rerun_on_the_same_commit_says_nothing_changed(self):
        self.checkout(self.v1)
        self.assertEqual(
            interdiff.interdiff(self.v1, self.b0, self.b0, ["a.txt"]),
            interdiff.NOTE_SAME,
        )
        # Even when the earlier merge base could not be resolved.
        self.assertEqual(
            interdiff.interdiff(self.v1, "", self.b0, ["a.txt"]), interdiff.NOTE_SAME
        )

    def test_an_unexpected_failure_is_a_note_not_a_failed_step(self):
        self.checkout(self.v1)
        files, out = Path("files.z"), Path("out.txt")
        files.write_bytes(b"a.txt\0")
        original = interdiff.interdiff
        try:
            interdiff.interdiff = lambda *a: (_ for _ in ()).throw(ValueError("boom"))
            self.assertEqual(
                interdiff.main(
                    [
                        "--previous",
                        self.v1,
                        "--previous-base",
                        self.b0,
                        "--base",
                        self.b0,
                        "--files",
                        str(files),
                        "--out",
                        str(out),
                        "--max-bytes",
                        "100000",
                    ]
                ),
                0,
            )
        finally:
            interdiff.interdiff = original
        self.assertEqual(out.read_text(), interdiff.NOTE_FAILED)


if __name__ == "__main__":
    run_this_suite()
