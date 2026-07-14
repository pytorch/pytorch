# Owner(s): ["module: unknown"]

from __future__ import annotations

import glob
import os
import sys
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.license_files_audit import audit_repo_license_files, load_project

sys.path.remove(str(REPO_ROOT))

# Audit for https://github.com/pytorch/pytorch/issues/183434:
# explicit included + excluded license paths, unknown discovery fails,
# SPDX from a per-file map with aggregate excluding LicenseRef-NvidiaProprietary

site_packages = os.path.dirname(os.path.dirname(torch.__file__))
distinfo = glob.glob(os.path.join(site_packages, "torch-*dist-info"))


class TestLicense(TestCase):
    def test_pyproject_license_metadata(self) -> None:
        """Explicit include/exclude lists, unknown files, SPDX map."""
        errors = audit_repo_license_files(REPO_ROOT)
        self.assertEqual(errors, [], msg="\n".join(errors))

    @unittest.skipIf(len(distinfo) == 0, "no installation in site-package to test")
    def test_distinfo_license(self):
        """Installed wheel ships pyproject.toml license-files."""
        if len(distinfo) > 1:
            raise AssertionError(
                'Found too many "torch-*dist-info" directories '
                f'in "{site_packages}", expected only one'
            )
        licenses_root = os.path.join(distinfo[0], "licenses")
        if not os.path.isdir(os.path.join(licenses_root, "third_party")):
            self.skipTest(
                "Installed wheel uses legacy license layout; rebuild with current "
                "pyproject.toml to populate licenses/third_party/"
            )
        found = {
            os.path.relpath(path, licenses_root).replace("\\", "/")
            for path in glob.glob(os.path.join(licenses_root, "**"), recursive=True)
            if os.path.isfile(path)
        }
        self.assertLessEqual(found, set(load_project(REPO_ROOT)["license-files"]))



if __name__ == "__main__":
    run_tests()
