"""Tests for check_lint_patch.py, run against a throwaway git repo."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import main, TestCase

import check_lint_patch


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


class TestCheckLintPatch(TestCase):
    def setUp(self) -> None:
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        run("git", "init", "-q", "-b", "main")
        run("git", "config", "user.email", "t@example.com")
        run("git", "config", "user.name", "t")
        for path in ("changed.py", "untouched.py", ".github/changed.yml"):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("x = 1\n")
        os.symlink("changed.py", "link.py")
        run("git", "add", "-A")
        run("git", "commit", "-q", "-m", "base")
        run("git", "checkout", "-q", "-b", "pr")
        for path in ("changed.py", ".github/changed.yml"):
            Path(path).write_text("x  =  2\n")
        os.remove("link.py")
        os.symlink("untouched.py", "link.py")
        run("git", "commit", "-q", "-am", "pr")

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def check(self, change) -> int:
        """Make a change in the work tree, check its diff, then undo it."""
        change()
        run("git", "add", "-A")
        Path("lint.patch").write_text(run("git", "diff", "--cached", "--binary"))
        run("git", "reset", "-q", "--hard")
        with contextlib.redirect_stderr(io.StringIO()):
            return check_lint_patch.main("lint.patch", "main")

    def test_accepts_edit_to_a_file_the_pr_changes(self) -> None:
        self.assertEqual(
            self.check(lambda: Path("changed.py").write_text("x = 2\n")), 0
        )

    def test_rejects_file_the_pr_does_not_change(self) -> None:
        self.assertEqual(
            self.check(lambda: Path("untouched.py").write_text("x = 3\n")), 1
        )

    def test_rejects_github_dir(self) -> None:
        self.assertEqual(
            self.check(lambda: Path(".github/changed.yml").write_text("x = 3\n")), 1
        )

    def test_rejects_new_file(self) -> None:
        self.assertEqual(self.check(lambda: Path("new.py").write_text("x\n")), 1)

    def test_rejects_mode_change(self) -> None:
        self.assertEqual(self.check(lambda: os.chmod("changed.py", 0o755)), 1)

    def test_rejects_rename(self) -> None:
        self.assertEqual(
            self.check(lambda: run("git", "mv", "changed.py", "moved.py")), 1
        )

    def test_rejects_symlink(self) -> None:
        def retarget() -> None:
            os.remove("link.py")
            os.symlink("changed.py", "link.py")

        self.assertEqual(self.check(retarget), 1)


if __name__ == "__main__":
    main()
