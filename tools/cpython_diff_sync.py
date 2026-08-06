"""Helpers to keep test/cpython/v3_13 adapted tests and *.diff files in sync.

The checked-in .diff for each adapted file must be exactly the patch that turns
the pristine upstream CPython file (at the tag named in the Dynamo patch
header) into the in-tree adapted file.

Offline verification uses the git `index <before>..<after>` line written by
`git diff --full-index` (see make_unified_diff). No separate manifest file:

  1. git hash-object(adapted) == <after>   (cheap drift check)
  2. reverse_apply(adapted, diff) -> pristine
     git hash-object(pristine) == <before> (diff integrity)

Hashes are git blob IDs of CRLF-normalized (LF-only) file bytes, matching what
`git diff --full-index` embeds when regenerating.

This offline check trusts whoever last ran tools/regenerate_cpython_diffs.py;
it does not re-fetch upstream in CI. A periodic job can confirm <before> against
the CPython repo blob at the header tag.

typinganndata/ helper modules are not adapted tests and have no *.diff; they are
intentionally excluded (we only walk *.diff files).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPYTHON_DIR = REPO_ROOT / "test" / "cpython" / "v3_13"

URL_RE = re.compile(
    r"https://raw\.githubusercontent\.com/python/cpython/refs/tags/"
    r"(v[\d.]+)/(Lib/test/\S+\.py)"
)
INDEX_RE = re.compile(
    r"^index ([0-9a-f]+)\.\.([0-9a-f]+)(?:\s+\d+)?$", re.MULTILINE
)


def normalize_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n")


def git_hash_object(data: bytes) -> str:
    """Return the git blob SHA for CRLF-normalized content."""
    proc = subprocess.run(
        ["git", "hash-object", "--stdin"],
        input=normalize_bytes(data),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git hash-object failed: {err}")
    return proc.stdout.decode("utf-8").strip()


def parse_index(diff_text: str) -> tuple[str, str]:
    """Return (before_blob, after_blob) from a full-index diff header."""
    match = INDEX_RE.search(normalize_text(diff_text))
    if not match:
        raise ValueError(
            "missing git index line; regenerate with "
            "python tools/regenerate_cpython_diffs.py --force"
        )
    return match.group(1), match.group(2)


def iter_diff_paths() -> list[Path]:
    return sorted(CPYTHON_DIR.rglob("*.diff"))


def parse_header(py_path: Path) -> tuple[str, str]:
    """Return (tag, upstream_relpath) from the Dynamo patch header URL."""
    head = py_path.read_text(encoding="utf-8", errors="replace")[:5000]
    match = URL_RE.search(head)
    if not match:
        raise ValueError(f"no cpython source URL in header: {py_path}")
    tag, upstream = match.group(1), match.group(2)

    if Path(upstream).name != py_path.name:
        raise ValueError(
            f"{py_path.relative_to(CPYTHON_DIR).as_posix()}: header upstream "
            f"basename {Path(upstream).name!r} != file name {py_path.name!r}"
        )
    return tag, upstream


def fetch_pristine(tag: str, upstream_rel: str, timeout: float = 60.0) -> bytes:
    url = (
        f"https://raw.githubusercontent.com/python/cpython/refs/tags/"
        f"{tag}/{upstream_rel}"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return normalize_bytes(resp.read())


def _git_apply(
    base_content: bytes,
    diff_text: str,
    rel_in_repo: str,
    *,
    reverse: bool = False,
) -> bytes:
    """Apply (or reverse-apply) a unified diff onto base_content at rel_in_repo."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / rel_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(normalize_bytes(base_content))

        patch_path = root / "patch.diff"
        patch_path.write_text(normalize_text(diff_text), encoding="utf-8", newline="\n")

        cmd = ["git", "apply", "--verbose"]
        if reverse:
            cmd.append("-R")
        cmd.append(str(patch_path))
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"git apply{' -R' if reverse else ''} failed for {rel_in_repo}:\n{err}"
            )

        if not target.is_file():
            raise RuntimeError(f"target missing after apply: {rel_in_repo}")
        return normalize_bytes(target.read_bytes())


def reverse_apply_to_pristine(py_path: Path, diff_path: Path) -> bytes:
    rel = py_path.relative_to(REPO_ROOT).as_posix()
    adapted = normalize_bytes(py_path.read_bytes())
    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    return _git_apply(adapted, diff_text, rel, reverse=True)


def apply_diff_to_adapted(pristine: bytes, py_path: Path, diff_path: Path) -> bytes:
    """Forward-apply for regen tooling."""
    rel = py_path.relative_to(REPO_ROOT).as_posix()
    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    return _git_apply(pristine, diff_text, rel, reverse=False)


def make_unified_diff(pristine: bytes, adapted: bytes, rel_in_repo: str) -> str:
    """Create a unified diff with full-index blob hashes (kept for offline verify)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_file = root / "a" / rel_in_repo
        b_file = root / "b" / rel_in_repo
        a_file.parent.mkdir(parents=True, exist_ok=True)
        b_file.parent.mkdir(parents=True, exist_ok=True)
        a_file.write_bytes(normalize_bytes(pristine))
        b_file.write_bytes(normalize_bytes(adapted))

        proc = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--full-index",
                "--no-ext-diff",
                "--binary",
                "-U3",
                f"a/{rel_in_repo}",
                f"b/{rel_in_repo}",
            ],
            cwd=root,
            capture_output=True,
        )
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"git diff failed for {rel_in_repo}: "
                f"{proc.stderr.decode('utf-8', 'replace')}"
            )
        text = normalize_text(proc.stdout.decode("utf-8", "replace"))
        if not text.strip():
            raise RuntimeError(f"empty diff for {rel_in_repo}; files are identical?")

        lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("diff --git "):
                lines.append(f"diff --git a/{rel_in_repo} b/{rel_in_repo}")
            elif line.startswith("--- "):
                lines.append(f"--- a/{rel_in_repo}")
            elif line.startswith("+++ "):
                lines.append(f"+++ b/{rel_in_repo}")
            else:
                # Keep the full-index `index <before>..<after>` line.
                lines.append(line)
        return "\n".join(lines) + "\n"


def verify_pair(py_path: Path, diff_path: Path) -> list[str]:
    """Return a list of error strings (empty if OK)."""
    rel = py_path.relative_to(CPYTHON_DIR).as_posix()
    try:
        parse_header(py_path)
    except ValueError as e:
        return [str(e)]

    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    try:
        before, after = parse_index(diff_text)
    except ValueError as e:
        return [f"{rel}: {e}"]

    adapted = normalize_bytes(py_path.read_bytes())
    adapted_hash = git_hash_object(adapted)
    if adapted_hash != after:
        return [
            f"{rel}: adapted file hash {adapted_hash} != diff index after "
            f"hash {after} (stale .diff; regenerate with "
            f"python tools/regenerate_cpython_diffs.py --only {rel})"
        ]

    try:
        pristine = reverse_apply_to_pristine(py_path, diff_path)
    except Exception as e:
        return [f"{rel}: reverse-apply failed: {e}"]

    pristine_hash = git_hash_object(pristine)
    if pristine_hash != before:
        return [
            f"{rel}: reverse-applied pristine hash {pristine_hash} != "
            f"diff index before hash {before} (corrupt .diff; regenerate)"
        ]
    return []


def verify_all() -> list[str]:
    """Offline sync check. Errors are plain strings for the lintrunner adapter."""
    errors: list[str] = []
    for diff_path in iter_diff_paths():
        py_path = diff_path.with_suffix(".py")
        rel = diff_path.relative_to(CPYTHON_DIR).as_posix()
        if not py_path.is_file():
            # Reported as a lint error (not an exception) so lintrunner can
            # surface LintSeverity.ERROR instead of crashing the adapter.
            errors.append(f"{rel}: diff without matching .py")
            continue
        errors.extend(verify_pair(py_path, diff_path))
    return errors
