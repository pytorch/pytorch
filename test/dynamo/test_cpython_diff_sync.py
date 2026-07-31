# Owner(s): ["module: dynamo"]

"""Ensure test/cpython/v3_13/*.diff stay in sync with adapted CPython tests.

See https://github.com/pytorch/pytorch/issues/189607
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch._dynamo.test_case
from torch.testing._internal.common_utils import run_tests

_DIFF_SYNC_PATH = (
    Path(__file__).resolve().parents[1] / "cpython" / "diff_sync.py"
)
_spec = importlib.util.spec_from_file_location(
    "torch_cpython_diff_sync", _DIFF_SYNC_PATH
)
assert _spec is not None and _spec.loader is not None
_diff_sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diff_sync)
CPYTHON_DIR = _diff_sync.CPYTHON_DIR
iter_diff_pairs = _diff_sync.iter_diff_pairs
parse_header = _diff_sync.parse_header
verify_all = _diff_sync.verify_all


class CPythonDiffSyncTests(torch._dynamo.test_case.TestCase):
    def test_every_adapted_file_has_matching_diff(self):
        pairs = iter_diff_pairs()
        self.assertGreater(len(pairs), 0)
        for py_path, diff_path in pairs:
            self.assertTrue(py_path.is_file(), py_path)
            self.assertTrue(diff_path.is_file(), diff_path)

    def test_headers_point_at_matching_upstream_basename(self):
        for py_path, _diff_path in iter_diff_pairs():
            _tag, upstream = parse_header(py_path)
            self.assertEqual(
                Path(upstream).name,
                py_path.name,
                f"{py_path.relative_to(CPYTHON_DIR)} header upstream mismatch",
            )

    def test_diffs_match_upstream_pristine_hashes(self):
        errors = verify_all()
        if errors:
            self.fail(
                "CPython .diff sync check failed. If you edited an adapted "
                "test under test/cpython/v3_13/, regenerate with:\n"
                "  python tools/regenerate_cpython_diffs.py\n"
                "or for one file:\n"
                "  python tools/regenerate_cpython_diffs.py --only <name>\n\n"
                + "\n".join(errors)
            )


if __name__ == "__main__":
    run_tests()
