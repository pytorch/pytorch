# Shared by test/test_license.py and tools/linter/adapters/license_files_linter.py
# (https://github.com/pytorch/pytorch/issues/183434, PR #185813 review).
#
# Included paths: [project].license-files in pyproject.toml (only explicit list).
# Excluded paths + SPDX expressions per shipped file: license_audit_manifest.toml next to this file.

from __future__ import annotations

import glob
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from packaging.licenses import canonicalize_license_expression

LICENSE_GLOBS = (
    "LICENSE",
    "third_party/**/LICENSE*",
    "third_party/**/COPYING*",
)

_MANIFEST_PATH = Path(__file__).resolve().parent / "license_audit_manifest.toml"
SPDX_OMIT_FROM_PROJECT_LICENSE = frozenset({"LicenseRef-NvidiaProprietary"})


def _skip_discovery_path(path: str) -> bool:
    # REUSE-style license pools (e.g. ittapi, kleidiai, nlohmann) include GPL
    # texts for non-shipped files; handle separately from glob discovery.
    return "/LICENSES/" in path or path.endswith("/LICENSES")


def _load_license_audit_tables() -> tuple[frozenset[str], dict[str, str]]:
    raw = tomllib.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    ex = raw.get("excluded")
    if not isinstance(ex, list) or not all(isinstance(x, str) for x in ex):
        raise ValueError(f"{_MANIFEST_PATH.name}: 'excluded' must be a list of strings")
    rows = raw.get("spdx")
    if not isinstance(rows, list):
        raise ValueError(f"{_MANIFEST_PATH.name}: 'spdx' must be an array of tables")
    spdx: dict[str, str] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "expression" not in row or "paths" not in row:
            raise ValueError(
                f"{_MANIFEST_PATH.name}: spdx[{i}] must have 'expression' and 'paths'"
            )
        expression = str(row["expression"])
        paths = row["paths"]
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"{_MANIFEST_PATH.name}: spdx[{i}].paths must be a list of strings")
        for path in paths:
            spdx[path] = expression
    return frozenset(ex), spdx


EXCLUDED_LICENSE_FILES, LICENSE_FILE_SPDX = _load_license_audit_tables()


def load_project(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def discover_license_files(repo_root: Path) -> set[str]:
    found: set[str] = set()
    for pattern in LICENSE_GLOBS:
        for path in glob.glob(str(repo_root / pattern), recursive=True):
            p = Path(path)
            if p.is_file():
                rel = p.relative_to(repo_root).as_posix()
                if not _skip_discovery_path(rel):
                    found.add(rel)
    return found


def classify_license_spdx(repo_root: Path, path: str) -> str:
    text = (repo_root / path).read_text(encoding="utf-8", errors="replace")[:5000]
    if m := re.search(r"(?i)SPDX-License-Identifier:\s*(\S+)", text):
        return m.group(1)
    if re.search(
        r"(?is)Apache License.*Version 2|Licensed under the Apache License, Version 2",
        text,
    ):
        if re.search(r"(?i)University of Illinois|LLVM Exceptions", text):
            return "Apache-2.0 WITH LLVM-exception"
        return "Apache-2.0"
    if re.search(r"(?i)Boost Software License", text):
        return "BSL-1.0"
    if re.search(r"(?i)Permission is hereby granted, free of charge", text):
        return "MIT"
    if re.search(r"(?im)Neither the name|^\s*3\. Neither", text):
        return "BSD-3-Clause"
    if re.search(r"(?i)Redistribution and use in source and binary", text):
        return "BSD-2-Clause"
    raise ValueError(f"Could not classify SPDX license for {path}")


def expected_project_license_expression() -> str:
    expressions = {
        expression
        for expression in LICENSE_FILE_SPDX.values()
        if expression not in SPDX_OMIT_FROM_PROJECT_LICENSE
    }
    return " AND ".join(sorted(expressions))


def audit_repo_license_files(repo_root: Path) -> list[str]:
    """Return human-readable errors; empty means the tree matches policy."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return [f"Missing {pyproject}"]

    try:
        project = load_project(repo_root)
    except (tomllib.TOMLDecodeError, OSError, KeyError) as e:
        return [f"Could not read [project] from pyproject.toml: {e}"]

    included = project.get("license-files")
    if not isinstance(included, list) or not all(isinstance(p, str) for p in included):
        return ["[project].license-files must be a list of strings."]

    err: list[str] = []
    inc = set(included)
    if bad := inc & EXCLUDED_LICENSE_FILES:
        err.append(
            "license-files must not list paths that are in manifest excluded list: "
            + ", ".join(sorted(bad))
        )

    unknown = discover_license_files(repo_root) - inc - EXCLUDED_LICENSE_FILES
    if unknown:
        err.append(
            "New license file(s) under audit globs; add each to pyproject license-files or "
            f"manifest excluded list ({_MANIFEST_PATH.name}): " + ", ".join(sorted(unknown))
        )

    spdx_keys = set(LICENSE_FILE_SPDX)
    if spdx_keys != inc:
        if missing := sorted(inc - spdx_keys):
            err.append(
                f"{_MANIFEST_PATH.name} SPDX table missing paths for license-files entries: "
                + ", ".join(missing)
            )
        if extra := sorted(spdx_keys - inc):
            err.append(
                f"{_MANIFEST_PATH.name} SPDX table has extra paths not in license-files: "
                + ", ".join(extra)
            )
    else:
        for p in included:
            if not (repo_root / p).is_file():
                continue
            try:
                got = classify_license_spdx(repo_root, p)
            except ValueError as e:
                err.append(str(e))
                continue
            if got != LICENSE_FILE_SPDX[p]:
                err.append(
                    f"SPDX in manifest out of date for {p!r}: file classifies as {got!r}, "
                    f"manifest has {LICENSE_FILE_SPDX[p]!r}"
                )

    lic = project.get("license")
    if not isinstance(lic, str):
        err.append("[project].license must be a string.")
    elif spdx_keys == inc:
        if lic != (exp := expected_project_license_expression()):
            err.append(
                "[project].license does not match SPDX manifest (expected):\n"
                f"  expected: {exp!r}\n  actual:   {lic!r}"
            )
        try:
            canonicalize_license_expression(lic)
        except Exception as e:
            err.append(f"[project].license is not a valid SPDX expression: {e}")

    return err
