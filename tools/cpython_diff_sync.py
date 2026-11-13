"""Helpers to keep test/cpython/v3_13 adapted tests and *.diff files in sync.

The checked-in .diff for each adapted file must be exactly the patch that turns
the pristine upstream CPython file (at the tag named in the Dynamo patch
header) into the in-tree adapted file.

Offline verification uses the git `index <before>..<after>` line written by
`git diff --full-index` (see make_unified_diff). No separate manifest file:

  1. git hash-object(adapted) == <after>
  2. reverse_apply(adapted, diff) -> pristine
     git hash-object(pristine) == <before>

Both steps always run: (1) catches adapted-file drift; (2) catches a corrupt
.diff whose after-hash still matches. This is intentionally whole-tree and
subprocess-heavy (~3 git calls per pair); the lintrunner adapter is in
SLOW_LINTERS for that reason. include_patterns only need to *trigger* a run.

Hashes are git blob IDs of normalized (LF-only, no BOM) file bytes, matching
what `git diff --full-index` embeds when regenerating.

This offline check trusts whoever last ran tools/regenerate_cpython_diffs.py;
it does not re-fetch upstream in CI. A periodic job can confirm <before> against
the CPython repo blob at the header tag.

Regeneration without --pristine must not move <before>. Reverse-apply of a
stale .diff can succeed for edits outside existing hunks and would otherwise
rewrite the upstream anchor, dropping the adaptation from the .diff.

typinganndata/ helper modules are not adapted tests and have no *.diff; they are
intentionally excluded (we only walk *.diff files).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CPYTHON_DIR = REPO_ROOT / "test" / "cpython" / "v3_13"

# Isolate from ~/.gitconfig / /etc/gitconfig (diff.noprefix, core.autocrlf,
# apply.whitespace=fix, etc.) so regen and verify are machine-independent.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}
_GIT_CONFIG_ARGS = (
    "-c",
    "core.autocrlf=false",
    "-c",
    "apply.whitespace=nowarn",
)

URL_RE = re.compile(
    r"https://raw\.githubusercontent\.com/python/cpython/refs/tags/"  # @lint-ignore
    r"(v[\d.]+)/(Lib/test/\S+\.py)"
)
INDEX_RE = re.compile(r"^index ([0-9a-f]+)\.\.([0-9a-f]+)(?:\s+\d+)?$", re.MULTILINE)


def normalize_bytes(data: bytes) -> bytes:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def normalize_text(text: str) -> str:
    text = text.removeprefix("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_utf8(path: Path, text: str) -> None:
    """Write text as UTF-8 LF bytes (Py3.9-safe; avoids Path.write_text newline=)."""
    path.write_bytes(normalize_text(text).encode("utf-8"))


def _git(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *_GIT_CONFIG_ARGS, *args],
        env=_GIT_ENV,
        **kwargs,
    )


def git_hash_object(data: bytes) -> str:
    """Return the git blob SHA for normalized content."""
    proc = _git(
        ["hash-object", "--stdin"],
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
    # Read the whole file: a long preamble must not hide the source URL.
    head = py_path.read_text(encoding="utf-8", errors="strict")
    match = URL_RE.search(head)
    if not match:
        raise ValueError("no cpython source URL in header")
    tag, upstream = match.group(1), match.group(2)

    if Path(upstream).name != py_path.name:
        raise ValueError(
            f"header upstream basename {Path(upstream).name!r} != "
            f"file name {py_path.name!r}"
        )
    return tag, upstream


def upstream_raw_url(tag: str, upstream_rel: str) -> str:
    # Keep {tag} in the same literal so lint_urls does not treat a truncated
    # https://.../tags/ prefix as a fetchable URL.
    return f"https://raw.githubusercontent.com/python/cpython/refs/tags/{tag}/{upstream_rel}"


def pristine_download_hint(tag: str, upstream_rel: str) -> str:
    url = upstream_raw_url(tag, upstream_rel)
    return (
        "Could not reconstruct the pristine upstream file from the checked-in "
        ".diff (refusing to download over the network).\n"
        "Download it locally, then regenerate:\n"
        f"  curl -fsSL {url} -o /tmp/pristine.py\n"
        f"  # or: wget -qO /tmp/pristine.py {url}\n"
        "  python tools/regenerate_cpython_diffs.py --force "
        f"--pristine /tmp/pristine.py --only {Path(upstream_rel).name}"
    )


def check_pristine_anchor(
    pristine: bytes, diff_text: str, tag: str, upstream: str
) -> None:
    """Refuse reconstructed upstream bytes whose blob hash is not index <before>.

    git apply -R only requires the diff's + lines, so an edit outside existing
    hunks reverse-applies cleanly and would otherwise be absorbed into the
    checked-in before hash. A missing index line means nothing to preserve
    (first full-index write). Callers that passed a real --pristine file skip
    this check.
    """
    try:
        old_before, _ = parse_index(diff_text)
    except ValueError:
        return
    new_before = git_hash_object(pristine)
    if new_before != old_before:
        raise RuntimeError(
            f"reconstructed pristine hash {new_before} != diff index "
            f"before hash {old_before}; refusing to move the upstream "
            "anchor.\n" + pristine_download_hint(tag, upstream)
        )


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
        write_utf8(patch_path, diff_text)

        cmd = ["apply", "--verbose"]
        if reverse:
            cmd.append("-R")
        cmd.append(str(patch_path))
        proc = _git(cmd, cwd=root, capture_output=True, text=True)
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
    diff_text = diff_path.read_text(encoding="utf-8", errors="strict")
    return _git_apply(adapted, diff_text, rel, reverse=True)


def apply_diff_to_adapted(pristine: bytes, py_path: Path, diff_path: Path) -> bytes:
    """Forward-apply for regen tooling."""
    rel = py_path.relative_to(REPO_ROOT).as_posix()
    diff_text = diff_path.read_text(encoding="utf-8", errors="strict")
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

        # No --binary: a binary GIT patch would pass parse_index but defeat
        # human-readable review of the delta. Fail loudly instead.
        proc = _git(
            [
                "diff",
                "--no-index",
                "--full-index",
                "--no-ext-diff",
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
        text = normalize_text(proc.stdout.decode("utf-8", "strict"))
        if not text.strip():
            raise RuntimeError(f"empty diff for {rel_in_repo}; files are identical?")
        if "GIT binary patch" in text or "Binary files " in text:
            raise RuntimeError(
                f"binary diff for {rel_in_repo}; refusing opaque GIT binary patch"
            )

        # split("\n"), not splitlines(): keep byte-exact with exotic line endings
        # that splitlines() would also treat as separators.
        lines: list[str] = []
        for line in text.split("\n"):
            if line.startswith("diff --git "):
                lines.append(f"diff --git a/{rel_in_repo} b/{rel_in_repo}")
            elif line.startswith("--- "):
                lines.append(f"--- a/{rel_in_repo}")
            elif line.startswith("+++ "):
                lines.append(f"+++ b/{rel_in_repo}")
            else:
                # Keep the full-index `index <before>..<after>` line.
                lines.append(line)
        # text from git ends with a trailing newline; split keeps a final "".
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"


def verify_pair(py_path: Path, diff_path: Path) -> list[tuple[str, str]]:
    """Return a list of (rel_path, message) errors (empty if OK)."""
    rel = py_path.relative_to(CPYTHON_DIR).as_posix()
    try:
        parse_header(py_path)
    except ValueError as e:
        return [(rel, str(e))]

    diff_text = diff_path.read_text(encoding="utf-8", errors="strict")
    try:
        before, after = parse_index(diff_text)
    except ValueError as e:
        return [(rel, str(e))]

    adapted = normalize_bytes(py_path.read_bytes())
    adapted_hash = git_hash_object(adapted)
    if adapted_hash != after:
        return [
            (
                rel,
                f"adapted file hash {adapted_hash} != diff index after "
                f"hash {after} (stale .diff; regenerate with "
                f"python tools/regenerate_cpython_diffs.py --only {rel})",
            )
        ]

    try:
        pristine = reverse_apply_to_pristine(py_path, diff_path)
    except Exception as e:
        return [(rel, f"reverse-apply failed: {e}")]

    pristine_hash = git_hash_object(pristine)
    if pristine_hash != before:
        return [
            (
                rel,
                f"reverse-applied pristine hash {pristine_hash} != "
                f"diff index before hash {before} (corrupt .diff; regenerate)",
            )
        ]
    return []


def verify_all() -> list[tuple[str, str]]:
    """Offline sync check for the whole tree.

    Always walks every *.diff under CPYTHON_DIR regardless of which paths
    triggered the linter: include_patterns are a trigger set, not a filter.
    """
    errors: list[tuple[str, str]] = []
    for diff_path in iter_diff_paths():
        py_path = diff_path.with_suffix(".py")
        rel = diff_path.relative_to(CPYTHON_DIR).as_posix()
        if not py_path.is_file():
            # Reported as a lint error (not an exception) so lintrunner can
            # surface LintSeverity.ERROR instead of crashing the adapter.
            errors.append((rel, "diff without matching .py"))
            continue
        errors.extend(verify_pair(py_path, diff_path))
    return errors
