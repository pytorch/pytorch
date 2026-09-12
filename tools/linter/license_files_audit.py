# Shared by test/test_license.py and tools/linter/adapters/license_files_linter.py
# (https://github.com/pytorch/pytorch/issues/183434, PR #185813 review).
#
# Included paths: [project].license-files in pyproject.toml (only explicit list).
# Excluded paths + SPDX expressions per shipped file: license_audit_manifest.toml
# next to this file. Discovery uses git ls-files (not filesystem glob) so
# gitignored FetchContent trees (e.g. third_party/nccl/) are not walked.
# Authoritative in CI via quick-checks (_lint.yml / setup-linux).

from __future__ import annotations

import subprocess
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
_SKIP_REASON = (
    "third_party/ has no discoverable license files (submodules not checked out)"
)
# Missing-path enforcement runs only when most declared license-files exist on disk
# (full / quick-checks checkout). Sparse CPU/CUDA CI omits many submodules.
_POPULATED_CHECKOUT_MIN_PRESENT_FRACTION = 0.9


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
            raise ValueError(
                f"{_MANIFEST_PATH.name}: spdx[{i}].paths must be a list of strings"
            )
        for path in paths:
            spdx[path] = expression
    return frozenset(ex), spdx


def load_project(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def discover_license_files(repo_root: Path) -> set[str]:
    if not (repo_root / ".git").exists():
        return set()
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *LICENSE_GLOBS],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    found: set[str] = set()
    for rel in result.stdout.splitlines():
        rel = rel.strip()
        if rel and not _skip_discovery_path(rel):
            found.add(rel)
    return found


def _parenthesize_compound_expression(expression: str) -> str:
    if any(op in expression for op in (" OR ", " AND ", " WITH ")):
        return f"({expression})"
    return expression


def expected_project_license_expression(spdx: dict[str, str]) -> str:
    expressions = sorted(
        {
            expression
            for expression in spdx.values()
            if expression not in SPDX_OMIT_FROM_PROJECT_LICENSE
        }
    )
    return " AND ".join(_parenthesize_compound_expression(e) for e in expressions)


def _checkout_looks_populated(repo_root: Path, included: set[str]) -> bool:
    if not included:
        return True
    present = sum(1 for path in included if (repo_root / path).is_file())
    return present / len(included) >= _POPULATED_CHECKOUT_MIN_PRESENT_FRACTION


def audit_repo_license_files(repo_root: Path) -> tuple[list[str], str | None]:
    """Return (errors, skip_reason). skip_reason is set when discovery is skipped."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return ([f"Missing {pyproject}"], None)

    try:
        excluded, spdx = _load_license_audit_tables()
    except (ValueError, tomllib.TOMLDecodeError, OSError) as e:
        return ([f"Could not load {_MANIFEST_PATH.name}: {e}"], None)

    try:
        project = load_project(repo_root)
    except (tomllib.TOMLDecodeError, OSError, KeyError) as e:
        return ([f"Could not read [project] from pyproject.toml: {e}"], None)

    included = project.get("license-files")
    if not isinstance(included, list) or not all(isinstance(p, str) for p in included):
        return (["[project].license-files must be a list of strings."], None)

    discovered = discover_license_files(repo_root)
    if not any(p.startswith("third_party/") for p in discovered):
        return ([], _SKIP_REASON)

    err: list[str] = []
    inc = set(included)
    if bad := inc & excluded:
        err.append(
            "license-files must not list paths that are in manifest excluded list: "
            + ", ".join(sorted(bad))
        )

    if unknown := discovered - inc - excluded:
        err.append(
            "New license file(s) under audit globs; add each to pyproject license-files or "
            f"manifest excluded list ({_MANIFEST_PATH.name}): "
            + ", ".join(sorted(unknown))
        )

    if missing := sorted(p for p in inc if not (repo_root / p).is_file()):
        if _checkout_looks_populated(repo_root, inc):
            err.append(
                "license-files lists path(s) that do not exist in the checkout: "
                + ", ".join(missing)
            )

    spdx_keys = set(spdx)
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

    lic = project.get("license")
    if not isinstance(lic, str):
        err.append("[project].license must be a string.")
    elif spdx_keys == inc:
        if lic != (exp := expected_project_license_expression(spdx)):
            err.append(
                "[project].license does not match SPDX manifest (expected):\n"
                f"  expected: {exp!r}\n  actual:   {lic!r}"
            )
        try:
            canonicalize_license_expression(lic)
        except Exception as e:
            err.append(f"[project].license is not a valid SPDX expression: {e}")

    return (err, None)


def main() -> None:
    errors, skip_reason = audit_repo_license_files(Path("."))
    if skip_reason:
        raise SystemExit(f"license-files audit skipped: {skip_reason}")
    if errors:
        raise SystemExit("license-files audit failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
