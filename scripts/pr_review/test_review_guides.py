#!/usr/bin/env python3
"""Tests for review_guides.py, which combines the trusted REVIEW.md guides.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401
from review_guides import applicable_guides, combine, covering_dirs  # noqa: E402


SCRIPT = Path(__file__).resolve().parent / "review_guides.py"


class TestCoveringDirs(unittest.TestCase):
    def test_every_directory_above_a_path_root_first(self):
        self.assertEqual(covering_dirs("a/b/c.py"), ["", "a", "a/b"])
        self.assertEqual(covering_dirs("setup.py"), [""])

    def test_a_path_git_would_never_emit_is_refused(self):
        for path in ("", "/etc/passwd", "a//b", "a/./b", "../x", "a/../b", "a/"):
            with self.subTest(path=path):
                self.assertIsNone(covering_dirs(path))


class TestApplicableGuides(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name) / "trusted"
        self.root.mkdir()

    def tearDown(self):
        self._td.cleanup()

    def guide(self, rel: str) -> None:
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text("rules\n")

    def test_root_and_nested_guides_cover_the_files_below_them(self):
        for rel in ("REVIEW.md", "a/REVIEW.md", "a/b/REVIEW.md", "c/REVIEW.md"):
            self.guide(rel)
        self.assertEqual(
            applicable_guides(str(self.root), ["a/b/x.py"]),
            ["REVIEW.md", "a/REVIEW.md", "a/b/REVIEW.md"],
        )

    def test_each_guide_is_listed_once_however_many_files_it_covers(self):
        self.guide("a/REVIEW.md")
        self.assertEqual(
            applicable_guides(str(self.root), ["a/x.py", "a/y/z.py"]),
            ["a/REVIEW.md"],
        )

    def test_a_directory_without_a_guide_contributes_nothing(self):
        self.guide("a/REVIEW.md")
        self.assertEqual(applicable_guides(str(self.root), ["b/x.py"]), [])

    def test_only_the_exact_name_is_a_guide(self):
        self.guide("a/review.md")
        self.guide("a/REVIEW.md.txt")
        self.assertEqual(applicable_guides(str(self.root), ["a/x.py"]), [])

    def test_a_symlinked_guide_is_not_read(self):
        outside = Path(self._td.name) / "elsewhere.md"
        outside.write_text("rules from outside\n")
        (self.root / "a").mkdir()
        os.symlink(outside, self.root / "a" / "REVIEW.md")
        self.assertEqual(applicable_guides(str(self.root), ["a/x.py"]), [])

    def test_a_guide_behind_a_link_out_of_the_root_is_not_read(self):
        outside = Path(self._td.name) / "outside"
        outside.mkdir()
        (outside / "REVIEW.md").write_text("rules from outside\n")
        os.symlink(outside, self.root / "a")
        self.assertEqual(applicable_guides(str(self.root), ["a/x.py"]), [])

    def test_a_guide_whose_path_could_break_a_line_fails_the_listing(self):
        """Dropping it would review without a guide that applies."""
        self.guide("a\nb/REVIEW.md")
        self.assertIsNone(applicable_guides(str(self.root), ["a\nb/x.py"]))

    def test_a_control_character_in_a_changed_path_alone_fails_nothing(self):
        self.guide("REVIEW.md")
        self.assertEqual(
            applicable_guides(str(self.root), ["a\nb/x.py"]), ["REVIEW.md"]
        )

    def test_a_malformed_path_fails_the_whole_listing(self):
        self.guide("REVIEW.md")
        self.assertIsNone(applicable_guides(str(self.root), ["ok.py", "../x.py"]))


class TestCombine(unittest.TestCase):
    def test_each_guide_is_a_section_headed_by_its_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "REVIEW.md").write_text("root rules\n")
            # No trailing newline: the end marker still starts its own line.
            (root / "a" / "REVIEW.md").write_text("## a rules")
            text = combine(td, ["REVIEW.md", "a/REVIEW.md"]).decode()
        self.assertIn(
            "## `REVIEW.md`: applies to every changed file\n\n"
            "<!-- begin REVIEW.md -->\nroot rules\n<!-- end REVIEW.md -->\n",
            text,
        )
        self.assertIn(
            "## `a/REVIEW.md`: applies to changed files under `a/`\n\n"
            "<!-- begin a/REVIEW.md -->\n## a rules\n<!-- end a/REVIEW.md -->\n",
            text,
        )
        self.assertLess(text.index("begin REVIEW.md"), text.index("begin a/REVIEW.md"))

    def test_no_guide_says_so(self):
        text = combine("/nonexistent", []).decode()
        self.assertIn("no guide applies", text)
        self.assertNotIn("<!-- begin", text)


class TestTheCommandLine(unittest.TestCase):
    def run_script(self, root: Path, paths: bytes, out: Path):
        changed = root.parent / "paths.z"
        changed.write_bytes(paths)
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(root), str(changed), str(out)],
            capture_output=True,
            text=True,
        )

    def test_it_writes_the_combined_guides(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "trusted"
            (root / "a").mkdir(parents=True)
            (root / "REVIEW.md").write_text("root\n")
            (root / "a" / "REVIEW.md").write_text("a\n")
            out = Path(td) / "guides.md"
            proc = self.run_script(root, b"a/x.py\0b/y.py\0", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(
                out.read_bytes(), combine(str(root), ["REVIEW.md", "a/REVIEW.md"])
            )
            self.assertIn("review guides that apply: 2", proc.stdout)

    def test_no_changed_path_says_no_guide_applies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "trusted"
            root.mkdir()
            (root / "REVIEW.md").write_text("root\n")
            out = Path(td) / "guides.md"
            proc = self.run_script(root, b"", out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("no guide applies", out.read_text())

    def test_a_malformed_path_fails_without_echoing_it(self):
        """The path is PR data, and stderr goes to a log parsed for commands."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "trusted"
            root.mkdir()
            out = Path(td) / "guides.md"
            proc = self.run_script(root, b"../::add-mask::x\0", out)
            self.assertEqual(proc.returncode, 2)
            self.assertNotIn("::", proc.stdout + proc.stderr)
            self.assertFalse(out.exists())

    def test_an_unreadable_path_list_fails(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "guides.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    td,
                    str(Path(td) / "missing.z"),
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    run_this_suite()
