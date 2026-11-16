# Owner(s): ["module: dynamo"]

"""Unit tests for tools/cpython_diff_sync offline sync helpers.

See https://github.com/pytorch/pytorch/issues/189607

Lives at test/test_cpython_diff_sync.py (not under test/cpython/) so run_test.py
does not exclude it via the version-prefix filter, and so it is picked up by
normal pull shards rather than the Dynamo-wrapped cpython suite.

CI enforcement for the full tree also runs via the CPYTHON_DIFF_SYNC lintrunner
adapter (tools/linter/adapters/cpython_diff_sync_linter.py).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.testing._internal.common_utils import run_tests, TestCase


def _load_module(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diff_sync = _load_module("torch_cpython_diff_sync", "tools/cpython_diff_sync.py")
regen = _load_module(
    "torch_regenerate_cpython_diffs", "tools/regenerate_cpython_diffs.py"
)

_FAKE_REL = "test/cpython/v3_13/test_fake.py"
_FAKE_TAG = "v3.13.5"
_FAKE_UPSTREAM = "Lib/test/test_fake.py"
_OUTSIDE_HUNK_LINE = b'@unittest.skip("Dynamo: not supported yet")\n'
_FAKE_HEADER = b"""\
# ======= BEGIN Dynamo patch =======
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_fake.py
# ======= END DYNAMO PATCH =======

"""
_FAKE_BODY = b"""\
def foo():
    return 1

# pad01
# pad02
# pad03
# pad04
# pad05
# pad06
# pad07
# pad08

def bar():
    return 2

# pad09
# pad10
# pad11
# pad12
# pad13
# pad14
# pad15
# pad16

def baz():
    return 3
"""


def _fake_pair_bytes():
    """Pristine / adapted with one middle hunk, plus an edit outside that hunk."""
    pristine = _FAKE_BODY
    adapted = _FAKE_HEADER + _FAKE_BODY.replace(b"    return 2", b"    return 99")
    edited = adapted.replace(b"def baz():", _OUTSIDE_HUNK_LINE + b"def baz():")
    diff_text = diff_sync.make_unified_diff(pristine, adapted, _FAKE_REL)
    return pristine, adapted, edited, diff_text


def _bool_pair():
    for diff_path in diff_sync.iter_diff_paths():
        if diff_path.name == "test_bool.diff":
            py_path = diff_path.with_suffix(".py")
            if not py_path.is_file():
                raise AssertionError(f"missing {py_path}")
            return py_path, diff_path
    raise AssertionError("test_bool.py pair not found")


@unittest.skipIf(shutil.which("git") is None, "git binary required")
class CPythonDiffSyncTests(TestCase):
    def test_verify_all_passes_on_checked_in_tree(self):
        errors = diff_sync.verify_all()
        self.assertEqual(
            errors,
            [],
            "CPython .diff sync check failed. Regenerate with:\n"
            "  python tools/regenerate_cpython_diffs.py\n"
            + "\n".join(f"{rel}: {msg}" for rel, msg in errors),
        )

    def test_parse_header_requires_source_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp) / "test_fake.py"
            py.write_text("# no url here\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                diff_sync.parse_header(py)

    def test_adapted_blob_drift_fails_verify(self):
        py_path, diff_path = _bool_pair()
        text = diff_path.read_text(encoding="utf-8")
        _before, after = diff_sync.parse_index(text)
        # Corrupt the after-hash in a temp copy of the diff (do not touch the tree).
        corrupted = re.sub(
            rf"(index [0-9a-f]+\.\.){re.escape(after)}",
            rf"\g<1>{'0' * len(after)}",
            text,
            count=1,
        )
        self.assertNotEqual(corrupted, text)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_diff = Path(tmp) / "test_bool.diff"
            tmp_diff.write_bytes(corrupted.encode("utf-8"))
            errors = diff_sync.verify_pair(py_path, tmp_diff)
        self.assertTrue(
            any("stale" in msg or "adapted file hash" in msg for _, msg in errors),
            errors,
        )

    def test_missing_index_line_fails_verify(self):
        py_path, diff_path = _bool_pair()
        text = (
            "\n".join(
                line
                for line in diff_path.read_text(encoding="utf-8").split("\n")
                if not line.startswith("index ")
            )
            + "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_diff = Path(tmp) / "test_bool.diff"
            tmp_diff.write_bytes(text.encode("utf-8"))
            errors = diff_sync.verify_pair(py_path, tmp_diff)
        self.assertTrue(
            any("missing git index line" in msg for _, msg in errors), errors
        )

    def test_all_pairs_share_one_upstream_tag(self):
        # Do not hardcode a CPython tag — the suite will move (e.g. to 3.15).
        tags = {
            diff_sync.parse_header(diff.with_suffix(".py"))[0]
            for diff in diff_sync.iter_diff_paths()
            if diff.with_suffix(".py").is_file()
        }
        self.assertEqual(len(tags), 1, tags)

    def test_index_hashes_match_blob_objects(self):
        py_path, diff_path = _bool_pair()
        before, after = diff_sync.parse_index(diff_path.read_text(encoding="utf-8"))
        adapted = diff_sync.normalize_bytes(py_path.read_bytes())
        self.assertEqual(diff_sync.git_hash_object(adapted), after)
        pristine = diff_sync.reverse_apply_to_pristine(py_path, diff_path)
        self.assertEqual(diff_sync.git_hash_object(pristine), before)

    def _reverse_apply_bytes(self, base: bytes, diff_text: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            py = root / _FAKE_REL
            py.parent.mkdir(parents=True)
            py.write_bytes(base)
            diff_path = py.with_suffix(".diff")
            diff_sync.write_utf8(diff_path, diff_text)
            with patch.object(diff_sync, "REPO_ROOT", root):
                return diff_sync.reverse_apply_to_pristine(py, diff_path)

    def test_check_pristine_anchor_refuses_outside_hunk_edit(self):
        _pristine, adapted, edited, diff_text = _fake_pair_bytes()
        old_before, _ = diff_sync.parse_index(diff_text)
        reconstructed = self._reverse_apply_bytes(edited, diff_text)
        self.assertIn(_OUTSIDE_HUNK_LINE, reconstructed)
        self.assertNotEqual(diff_sync.git_hash_object(reconstructed), old_before)
        with self.assertRaisesRegex(
            RuntimeError, "refusing to move the upstream anchor"
        ):
            diff_sync.check_pristine_anchor(
                reconstructed, diff_text, _FAKE_TAG, _FAKE_UPSTREAM
            )
        matching = self._reverse_apply_bytes(adapted, diff_text)
        self.assertEqual(diff_sync.git_hash_object(matching), old_before)
        diff_sync.check_pristine_anchor(matching, diff_text, _FAKE_TAG, _FAKE_UPSTREAM)
        # First full-index write: no index line, nothing to preserve.
        diff_sync.check_pristine_anchor(
            reconstructed, "diff --git a/x b/x\n", _FAKE_TAG, _FAKE_UPSTREAM
        )

    def _run_regen_on_fake_tree(
        self,
        py_bytes: bytes,
        diff_text: str,
        *,
        force: bool,
        pristine_path: Path | None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpython_dir = root / "test" / "cpython" / "v3_13"
            cpython_dir.mkdir(parents=True)
            py_path = cpython_dir / "test_fake.py"
            diff_path = cpython_dir / "test_fake.diff"
            py_path.write_bytes(py_bytes)
            diff_sync.write_utf8(diff_path, diff_text)
            old_diff = diff_path.read_bytes()
            old_before, _ = diff_sync.parse_index(diff_text)
            with patch.object(regen.diff_sync, "CPYTHON_DIR", cpython_dir):
                with patch.object(regen.diff_sync, "REPO_ROOT", root):
                    with patch.object(regen, "REPO_ROOT", root):
                        rc = regen.regenerate(
                            "test_fake.py", force=force, pristine_path=pristine_path
                        )
            return (
                rc,
                old_before,
                old_diff,
                diff_path.read_bytes(),
                py_path.read_bytes(),
            )

    def test_regenerate_refuses_outside_hunk_edit_without_pristine(self):
        _pristine, _adapted, edited, diff_text = _fake_pair_bytes()
        for force in (False, True):
            with self.subTest(force=force):
                rc, old_before, old_diff, new_diff, _py = self._run_regen_on_fake_tree(
                    edited, diff_text, force=force, pristine_path=None
                )
                self.assertEqual(rc, 1)
                self.assertEqual(new_diff, old_diff)
                got_before, _ = diff_sync.parse_index(new_diff.decode("utf-8"))
                self.assertEqual(got_before, old_before)

    def test_regenerate_force_keeps_anchor_when_in_sync(self):
        _pristine, adapted, _edited, diff_text = _fake_pair_bytes()
        rc, old_before, _old_diff, new_diff, _py = self._run_regen_on_fake_tree(
            adapted, diff_text, force=True, pristine_path=None
        )
        self.assertEqual(rc, 0)
        new_before, new_after = diff_sync.parse_index(new_diff.decode("utf-8"))
        self.assertEqual(new_before, old_before)
        self.assertEqual(diff_sync.git_hash_object(adapted), new_after)

    def test_regenerate_with_pristine_records_outside_hunk_edit(self):
        with tempfile.TemporaryDirectory() as ptmp:
            pristine_file = Path(ptmp) / "pristine.py"
            pristine, _adapted, edited, diff_text = _fake_pair_bytes()
            pristine_file.write_bytes(pristine)
            rc, old_before, old_diff, new_diff, written = self._run_regen_on_fake_tree(
                edited, diff_text, force=False, pristine_path=pristine_file
            )
        self.assertEqual(rc, 0)
        self.assertEqual(written, edited)
        self.assertNotEqual(new_diff, old_diff)
        new_before, new_after = diff_sync.parse_index(new_diff.decode("utf-8"))
        self.assertEqual(new_before, old_before)
        self.assertEqual(diff_sync.git_hash_object(edited), new_after)
        self.assertIn(_OUTSIDE_HUNK_LINE.decode("ascii"), new_diff.decode("utf-8"))


if __name__ == "__main__":
    run_tests()
