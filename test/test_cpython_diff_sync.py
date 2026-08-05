# Owner(s): ["module: dynamo"]

"""Unit tests for test/cpython/diff_sync offline sync helpers.

See https://github.com/pytorch/pytorch/issues/189607

CI enforcement for the full tree also runs via the CPYTHON_DIFF_SYNC lintrunner
adapter (tools/linter/adapters/cpython_diff_sync_linter.py).
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_diff_sync():
    path = Path(__file__).resolve().parent / "cpython" / "diff_sync.py"
    spec = importlib.util.spec_from_file_location("torch_cpython_diff_sync", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diff_sync = _load_diff_sync()


def _bool_pair():
    for py_path, diff_path in diff_sync.iter_diff_pairs():
        if py_path.name == "test_bool.py":
            return py_path, diff_path
    raise AssertionError("test_bool.py pair not found")


class CPythonDiffSyncTests(unittest.TestCase):
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

    def test_drift_in_adapted_content_fails_hash_check(self):
        # Do not mutate the working tree: reverse-apply drifted bytes in-memory.
        py_path, diff_path = _bool_pair()
        rel = py_path.relative_to(diff_sync.CPYTHON_DIR).as_posix()
        entry = diff_sync.load_manifest()["files"][rel]
        repo_rel = py_path.relative_to(diff_sync.REPO_ROOT).as_posix()
        drifted = diff_sync.normalize_bytes(py_path.read_bytes()) + (
            b"\n# intentional sync drift\n"
        )
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
        try:
            pristine = diff_sync._git_apply(
                drifted, diff_text, repo_rel, reverse=True
            )
        except RuntimeError:
            return  # reverse-apply rejecting drifted content is a valid failure mode
        self.assertNotEqual(diff_sync.sha256_bytes(pristine), entry["sha256"])

    def test_wrong_manifest_hash_fails_verify(self):
        py_path, diff_path = _bool_pair()
        rel = py_path.relative_to(diff_sync.CPYTHON_DIR).as_posix()
        entry = dict(diff_sync.load_manifest()["files"][rel])
        entry["sha256"] = "0" * 64
        errors = diff_sync.verify_pair(py_path, diff_path, entry)
        self.assertTrue(any("sha256 mismatch" in e for e in errors), errors)

    def test_all_pairs_share_single_upstream_tag(self):
        tags = {diff_sync.parse_header(py)[0] for py, _ in diff_sync.iter_diff_pairs()}
        self.assertEqual(tags, {"v3.13.5"}, tags)

    def test_reverse_apply_hash_matches_manifest(self):
        self.assertTrue(diff_sync.MANIFEST_PATH.is_file())
        py_path, diff_path = _bool_pair()
        pristine = diff_sync.reverse_apply_to_pristine(py_path, diff_path)
        rel = py_path.relative_to(diff_sync.CPYTHON_DIR).as_posix()
        self.assertEqual(
            diff_sync.sha256_bytes(pristine),
            diff_sync.load_manifest()["files"][rel]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
