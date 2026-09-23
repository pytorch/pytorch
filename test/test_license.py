# Owner(s): ["module: unknown"]

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.license_files_audit import (
    audit_repo_license_files,
    expected_project_license_expression,
    load_project,
)


sys.path.remove(str(REPO_ROOT))

# Audit for https://github.com/pytorch/pytorch/issues/183434:
# explicit included + excluded license paths, unknown discovery fails,
# SPDX expressions from a per-file map

site_packages = os.path.dirname(os.path.dirname(torch.__file__))
distinfo = glob.glob(os.path.join(site_packages, "torch-*dist-info"))


def _audit_fixture(
    project_body: str,
    manifest_body: str,
    files: list[tuple[str, str]],
    git: bool = True,
) -> tuple[list[str], str | None]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text(
            f"[project]\n{project_body}\n", encoding="utf-8"
        )
        manifest = root / "manifest.toml"
        manifest.write_text(manifest_body, encoding="utf-8")
        for relpath, content in files:
            path = root / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if git:
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        with patch("tools.linter.license_files_audit._MANIFEST_PATH", manifest):
            return audit_repo_license_files(root)


class TestLicense(TestCase):
    def test_pyproject_license_metadata(self) -> None:
        """Explicit include/exclude lists, unknown files, SPDX map."""
        errors, skip_reason = audit_repo_license_files(REPO_ROOT)
        if skip_reason:
            self.skipTest(skip_reason)
        if errors:
            self.fail("\n".join(errors))

    def test_audit_overlap_include_excluded(self) -> None:
        errors, skip_reason = _audit_fixture(
            'license = "MIT"\nlicense-files = ["third_party/foo/LICENSE"]',
            'excluded = ["third_party/foo/LICENSE"]\n\n[[spdx]]\nexpression = "MIT"\n'
            'paths = ["third_party/foo/LICENSE"]',
            [("third_party/foo/LICENSE", "MIT\n")],
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(any("manifest excluded list" in e for e in errors), msg=errors)

    def test_audit_unknown_discovery(self) -> None:
        errors, skip_reason = _audit_fixture(
            'license = "MIT"\nlicense-files = ["LICENSE"]',
            'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = ["LICENSE"]',
            [("LICENSE", "MIT\n"), ("third_party/newdep/LICENSE", "MIT\n")],
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(any("New license file(s)" in e for e in errors), msg=errors)

    def test_audit_missing_license_file(self) -> None:
        present = [("third_party/shipped/LICENSE", "MIT\n")]
        present += [(f"third_party/dep{i}/LICENSE", "MIT\n") for i in range(9)]
        listed = [p for p, _ in present] + ["third_party/removed/LICENSE"]
        paths_literal = ", ".join(f'"{p}"' for p in listed)
        errors, skip_reason = _audit_fixture(
            f'license = "MIT"\nlicense-files = [{paths_literal}]',
            f'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = [{paths_literal}]',
            present,
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(
            any("do not exist in the checkout" in e for e in errors), msg=errors
        )

    def test_audit_sparse_checkout_skips(self) -> None:
        listed = ['"LICENSE"', '"third_party/present/LICENSE"'] + [
            f'"third_party/missing_{i}/LICENSE"' for i in range(18)
        ]
        paths_literal = ", ".join(listed)
        errors, skip_reason = _audit_fixture(
            f'license = "MIT"\nlicense-files = [{paths_literal}]',
            f'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = [{paths_literal}]',
            [("LICENSE", "MIT\n"), ("third_party/present/LICENSE", "MIT\n")],
        )
        self.assertIsNotNone(skip_reason)
        self.assertEqual(errors, [])

    def test_project_license_expression_flattens_and(self) -> None:
        spdx = {
            "a": "Apache-2.0 WITH LLVM-exception",
            "b": "BSD-2-Clause AND Zlib",
            "c": "BSD-2-Clause",
            "d": "MIT OR Apache-2.0",
        }
        self.assertEqual(
            expected_project_license_expression(spdx),
            "Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND "
            "(MIT OR Apache-2.0) AND Zlib",
        )

    def test_audit_spdx_table_mismatch(self) -> None:
        present = [(f"third_party/dep{i}/LICENSE", "MIT\n") for i in range(9)]
        listed = [p for p, _ in present] + ["third_party/missing/LICENSE"]
        paths_literal = ", ".join(f'"{p}"' for p in listed)
        spdx_paths_literal = ", ".join(f'"{p}"' for p, _ in present)
        errors, skip_reason = _audit_fixture(
            f'license = "MIT"\nlicense-files = [{paths_literal}]',
            f'excluded = []\n\n[[spdx]]\nexpression = "MIT"\n'
            f"paths = [{spdx_paths_literal}]",
            present,
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(
            any("SPDX table missing paths" in e for e in errors), msg=errors
        )

    def test_audit_license_expression_mismatch(self) -> None:
        errors, skip_reason = _audit_fixture(
            'license = "Apache-2.0"\nlicense-files = ["third_party/shipped/LICENSE"]',
            'excluded = []\n\n[[spdx]]\nexpression = "MIT"\n'
            'paths = ["third_party/shipped/LICENSE"]',
            [("third_party/shipped/LICENSE", "MIT\n")],
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(
            any("does not match SPDX manifest" in e for e in errors), msg=errors
        )

    def test_audit_not_discoverable(self) -> None:
        """inc - discovered: path in license-files not returned by git ls-files."""
        # Need >=90% of inc on disk so _checkout_looks_populated lets the check run.
        present = [(f"third_party/dep{i}/LICENSE", "MIT\n") for i in range(9)]
        present += [("third_party/shipped/LICENSE", "MIT\n")]
        listed = [p for p, _ in present] + ["third_party/removed/LICENSE"]
        paths_literal = ", ".join(f'"{p}"' for p in listed)
        errors, skip_reason = _audit_fixture(
            f'license = "MIT"\nlicense-files = [{paths_literal}]',
            f'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = [{paths_literal}]',
            present,
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(
            any("not returned by git ls-files" in e for e in errors), msg=errors
        )

    def test_audit_stale_excluded(self) -> None:
        """excluded - discovered: excluded entry not returned by git ls-files."""
        errors, skip_reason = _audit_fixture(
            'license = "MIT"\nlicense-files = ["third_party/dep/LICENSE"]',
            'excluded = ["third_party/dep/OLD_LICENSE"]\n\n[[spdx]]\nexpression = "MIT"\n'
            'paths = ["third_party/dep/LICENSE"]',
            [("third_party/dep/LICENSE", "MIT\n")],
        )
        self.assertIsNone(skip_reason)
        self.assertTrue(
            any("excluded list has path(s) not returned" in e for e in errors),
            msg=errors,
        )

    def test_audit_no_git_dir_skips(self) -> None:
        """A populated tree without .git (sdist, release tarball) is a skip."""
        errors, skip_reason = _audit_fixture(
            'license = "MIT"\nlicense-files = ["third_party/dep/LICENSE"]',
            'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = ["third_party/dep/LICENSE"]',
            [("third_party/dep/LICENSE", "MIT\n")],
            git=False,
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(skip_reason)
        self.assertIn("no .git directory", skip_reason)

    def test_audit_git_failure_skips(self) -> None:
        real_run = subprocess.run

        def run(cmd, *args, **kwargs):
            if "ls-files" in cmd:
                raise subprocess.CalledProcessError(128, cmd)
            return real_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=run):
            errors, skip_reason = _audit_fixture(
                'license = "MIT"\nlicense-files = ["third_party/dep/LICENSE"]',
                'excluded = []\n\n[[spdx]]\nexpression = "MIT"\npaths = ["third_party/dep/LICENSE"]',
                [("third_party/dep/LICENSE", "MIT\n")],
            )
        self.assertEqual(errors, [])
        self.assertEqual(skip_reason, "git ls-files failed")

    @unittest.skipIf(len(distinfo) == 0, "no installation in site-package to test")
    def test_distinfo_license(self):
        """Installed wheel ships pyproject.toml license-files."""
        if len(distinfo) > 1:
            raise AssertionError(
                'Found too many "torch-*dist-info" directories '
                f'in "{site_packages}", expected only one'
            )
        licenses_root = os.path.join(distinfo[0], "licenses")
        self.assertTrue(
            os.path.isdir(os.path.join(licenses_root, "third_party")),
            "Installed wheel does not ship licenses/third_party/",
        )
        found = {
            os.path.relpath(path, licenses_root).replace("\\", "/")
            for path in glob.glob(os.path.join(licenses_root, "**"), recursive=True)
            if os.path.isfile(path)
        }
        self.assertTrue(found, "no license files shipped under .dist-info/licenses/")
        self.assertTrue(
            any(path.startswith("third_party/") for path in found),
            "no third-party license files shipped under .dist-info/licenses/",
        )
        try:
            license_files = load_project(REPO_ROOT)["license-files"]
        except (KeyError, OSError) as e:
            self.fail(f"Could not read [project] from pyproject.toml: {e}")
        self.assertLessEqual(found, set(license_files))


if __name__ == "__main__":
    run_tests()
