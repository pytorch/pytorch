# Owner(s): ["module: dynamo"]

"""Unit tests for tools/cpython_diff_sync offline sync helpers.

See https://github.com/pytorch/pytorch/issues/189607

CI enforcement for the full tree also runs via the CPYTHON_DIFF_SYNC lintrunner
adapter (tools/linter/adapters/cpython_diff_sync_linter.py).
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import TestCase, run_tests


def _load_diff_sync():
    path = Path(__file__).resolve().parents[2] / "tools" / "cpython_diff_sync.py"
    spec = importlib.util.spec_from_file_location("torch_cpython_diff_sync", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diff_sync = _load_diff_sync()


def _bool_pair():
    for diff_path in diff_sync.iter_diff_paths():
        if diff_path.name == "test_bool.diff":
            py_path = diff_path.with_suffix(".py")
            if not py_path.is_file():
                raise AssertionError(f"missing {py_path}")
            return py_path, diff_path
    raise AssertionError("test_bool.py pair not found")


class CPythonDiffSyncTests(TestCase):
    def test_verify_all_passes_on_checked_in_tree(self):
        errors = diff_sync.verify_all()
        self.assertEqual(
            errors,
            [],
            "CPython .diff sync check failed. Regenerate with:\n"
            "  python tools/regenerate_cpython_diffs.py\n" + "\n".join(errors),
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
            any("stale" in e or "adapted file hash" in e for e in errors),
            errors,
        )

    def test_missing_index_line_fails_verify(self):
        py_path, diff_path = _bool_pair()
        text = "\n".join(
            line
            for line in diff_path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("index ")
        ) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_diff = Path(tmp) / "test_bool.diff"
            tmp_diff.write_bytes(text.encode("utf-8"))
            errors = diff_sync.verify_pair(py_path, tmp_diff)
        self.assertTrue(any("missing git index line" in e for e in errors), errors)

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
        before, after = diff_sync.parse_index(
            diff_path.read_text(encoding="utf-8")
        )
        adapted = diff_sync.normalize_bytes(py_path.read_bytes())
        self.assertEqual(diff_sync.git_hash_object(adapted), after)
        pristine = diff_sync.reverse_apply_to_pristine(py_path, diff_path)
        self.assertEqual(diff_sync.git_hash_object(pristine), before)


if __name__ == "__main__":
    run_tests()
