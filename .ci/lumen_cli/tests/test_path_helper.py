# test_path_utils.py
# Run: pytest -q

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cli.lib.common.path_helper import (
    copy,
    ensure_dir_exists,
    force_create_dir,
    get_path,
    is_path_exist,
    remove_dir,
)


class TestPathHelper(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # -------- get_path --------
    def test_get_path_returns_path_for_str(self):
        # Use relative path to avoid absolute-ness
        rel_str = "sub/f.txt"
        os.chdir(self.tmp_path)
        p = get_path(rel_str, resolve=False)
        self.assertIsInstance(p, Path)
        self.assertFalse(p.is_absolute())
        self.assertEqual(str(p), rel_str)

    def test_get_path_resolves(self):
        rel_str = "sub/f.txt"
        p = get_path(str(self.tmp_path / rel_str), resolve=True)
        self.assertTrue(p.is_absolute())
        self.assertTrue(str(p).endswith(rel_str))

    def test_get_path_with_path_input(self):
        p_in = self.tmp_path / "sub/f.txt"
        p_out = get_path(p_in, resolve=False)
        self.assertTrue(str(p_out) == str(p_in))

    def test_get_path_with_none_raises(self):
        with self.assertRaises(ValueError):
            get_path(None)  # type: ignore[arg-type]

    def test_get_path_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            get_path(123)  # type: ignore[arg-type]

    # -------- ensure_dir_exists / force_create_dir / remove_dir --------
    def test_ensure_dir_exists_creates_and_is_idempotent(self):
        d = self.tmp_path / "made"
        ensure_dir_exists(d)
        self.assertTrue(d.exists() and d.is_dir())
        ensure_dir_exists(d)

    def test_force_create_dir_clears_existing(self):
        d = self.tmp_path / "fresh"
        (d / "inner").mkdir(parents=True)
        (d / "inner" / "f.txt").write_text("x")
        force_create_dir(d)
        self.assertTrue(d.exists())
        self.assertEqual(list(d.iterdir()), [])

    def test_remove_dir_none_is_noop(self):
        remove_dir(None)  # type: ignore[arg-type]

    def test_remove_dir_nonexistent_is_noop(self):
        ghost = self.tmp_path / "ghost"
        remove_dir(ghost)

    def test_remove_dir_accepts_str(self):
        d = self.tmp_path / "to_rm"
        d.mkdir()
        remove_dir(str(d))
        self.assertFalse(d.exists())

    # -------- copy --------
    def test_copy_file_to_file(self):
        src = self.tmp_path / "src.txt"
        dst = self.tmp_path / "out" / "dst.txt"
        src.write_text("hello")
        copy(src, dst)
        self.assertEqual(dst.read_text(), "hello")

    def test_copy_dir_to_new_dir(self):
        src = self.tmp_path / "srcdir"
        (src / "a").mkdir(parents=True)
        (src / "a" / "f.txt").write_text("content")
        dst = self.tmp_path / "destdir"
        copy(src, dst)
        self.assertEqual((dst / "a" / "f.txt").read_text(), "content")

    def test_copy_dir_into_existing_dir_overwrite_true_merges(self):
        src = self.tmp_path / "srcdir"
        dst = self.tmp_path / "destdir"
        (src / "x").mkdir(parents=True)
        (src / "x" / "new.txt").write_text("new")
        dst.mkdir()
        (dst / "existing.txt").write_text("old")
        copy(src, dst)
        self.assertEqual((dst / "existing.txt").read_text(), "old")
        self.assertEqual((dst / "x" / "new.txt").read_text(), "new")

    def test_is_str_path_exist(self):
        p = self.tmp_path / "x.txt"
        p.write_text("1")
        self.assertTrue(is_path_exist(str(p)))
        self.assertTrue(is_path_exist(p))
        self.assertFalse(is_path_exist(str(self.tmp_path / "missing")))
        self.assertFalse(is_path_exist(self.tmp_path / "missing"))
        self.assertFalse(is_path_exist(""))


if __name__ == "__main__":
    unittest.main()
