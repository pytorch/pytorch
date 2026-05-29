# Owner(s): ["module: unknown"]

from __future__ import annotations

import glob
import os
import re
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from packaging.licenses import canonicalize_license_expression

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

# Audit for https://github.com/pytorch/pytorch/issues/183434: the old recursive
# license globs over-collected files (notably dynolog), causing Windows MAX_PATH
# failures. Rules below classify each discovered file; pyproject.toml must match.

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

site_packages = os.path.dirname(os.path.dirname(torch.__file__))
distinfo = glob.glob(os.path.join(site_packages, "torch-*dist-info"))

# Pre-audit recursive patterns; defines the full set of candidate license files.
_LICENSE_GLOBS = (
    "LICENSE",
    "third_party/**/LICENSE",
    "third_party/**/LICENSE.txt",
    "third_party/**/LICENSE.rst",
    "third_party/**/COPYING.BSD",
)

# Path components that indicate non-shipping trees (tests, docs, bindings, ...).
_EXCLUDE_DIRS = frozenset(
    {
        "test", "tests", "testing", "docs", "doc", "examples", "example",
        "googletest", "googlemock", "gtest", "doctest", "python", "dart",
        "swift", "bindings", "hipify_torch", "cpplint", "generator",
        "scripts", "tools",
    }
)


def _project_metadata() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]


def _discover_license_files() -> set[str]:
    found: set[str] = set()
    for pattern in _LICENSE_GLOBS:
        for path in glob.glob(str(REPO_ROOT / pattern), recursive=True):
            if Path(path).is_file():
                found.add(Path(path).relative_to(REPO_ROOT).as_posix())
    return found


def _is_immediate_vendor(parts: tuple[str, ...], segment: str, *, use_last: bool = False) -> bool:
    indices = [i for i, part in enumerate(parts) if part == segment]
    if not indices:
        return False
    index = indices[-1] if use_last else indices[0]
    return index + 2 == len(parts) - 1 and parts[index + 1] not in _EXCLUDE_DIRS


def _is_excluded(path: str) -> bool:
    if "/dynolog/third_party/" in path:
        return True
    parts = Path(path).parts
    if any(part in _EXCLUDE_DIRS for part in parts[1:]):
        return True
    if "3rdparty" in parts or "deps" in parts:
        return not any(_is_immediate_vendor(parts, s) for s in ("3rdparty", "deps"))
    if parts.count("third_party") > 1:
        return not _is_immediate_vendor(parts, "third_party", use_last=True)
    return False


def _shipped_license_files() -> list[str]:
    return sorted(p for p in _discover_license_files() if not _is_excluded(p))


def _classify_license_spdx(path: str) -> str:
    text = (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")[:5000]
    if match := re.search(r"SPDX-License-Identifier:\s*(\S+)", text, re.I):
        return match.group(1)
    if re.search(
        r"Apache License.*Version 2|Licensed under the Apache License, Version 2",
        text,
        re.I | re.S,
    ):
        if re.search(r"University of Illinois|LLVM Exceptions", text, re.I):
            return "Apache-2.0 WITH LLVM-exception"
        return "Apache-2.0"
    if re.search(r"Boost Software License", text, re.I):
        return "BSL-1.0"
    if re.search(r"Permission is hereby granted, free of charge", text, re.I):
        return "MIT"
    if re.search(r"Neither the name|^\s*3\. Neither", text, re.I | re.M):
        return "BSD-3-Clause"
    if re.search(r"Redistribution and use in source and binary", text, re.I):
        return "BSD-2-Clause"
    raise AssertionError(f"Could not classify SPDX license for {path}")


class TestLicense(TestCase):
    def test_pyproject_license_metadata(self) -> None:
        """Audited rules match pyproject.toml license-files and SPDX expression."""
        shipped = _shipped_license_files()
        project = _project_metadata()

        self.assertEqual(project["license-files"], shipped)
        expected_spdx = " AND ".join(sorted({_classify_license_spdx(p) for p in shipped}))
        self.assertEqual(project["license"], expected_spdx)
        canonicalize_license_expression(project["license"])

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
        self.assertEqual(found, set(_project_metadata()["license-files"]))


if __name__ == "__main__":
    run_tests()
