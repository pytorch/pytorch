"""Helpers to keep test/cpython/v3_13 adapted tests and *.diff files in sync.

The checked-in .diff for each adapted file must be exactly the patch that turns
the pristine upstream CPython file (at the tag named in the Dynamo patch
header) into the in-tree adapted file.

CI cannot fetch from GitHub, so we store sha256(pristine) in
upstream_manifest.json and verify offline via:

  reverse_apply(adapted, diff) -> pristine
  sha256(pristine) == manifest hash
  apply(pristine, diff) == adapted
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CPYTHON_DIR = REPO_ROOT / "test" / "cpython" / "v3_13"
MANIFEST_PATH = CPYTHON_DIR / "upstream_manifest.json"

URL_RE = re.compile(
    r"https://raw\.githubusercontent\.com/python/cpython/refs/tags/"
    r"(v[\d.]+)/(Lib/test/\S+\.py)"
)


def normalize_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(normalize_bytes(data)).hexdigest()


def iter_diff_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for diff_path in sorted(CPYTHON_DIR.rglob("*.diff")):
        py_path = diff_path.with_suffix(".py")
        if not py_path.is_file():
            raise FileNotFoundError(f"diff without matching .py: {diff_path}")
        pairs.append((py_path, diff_path))
    return pairs


def parse_header(py_path: Path) -> tuple[str, str]:
    """Return (tag, upstream_relpath) from the Dynamo patch header."""
    head = py_path.read_text(encoding="utf-8", errors="replace")[:5000]
    match = URL_RE.search(head)
    if match:
        tag, upstream = match.group(1), match.group(2)
    else:
        # Fallback for older adapted files missing the source URL comment.
        rel = py_path.relative_to(CPYTHON_DIR).as_posix()
        if rel.startswith("test_unittest/"):
            tag, upstream = "v3.13.5", f"Lib/test/{rel}"
        else:
            raise ValueError(f"no cpython source URL in header: {py_path}")

    # Catch copy-paste mistakes like pointing test_binop.py at test_iter.py.
    if Path(upstream).name != py_path.name:
        raise ValueError(
            f"{py_path.relative_to(CPYTHON_DIR).as_posix()}: header upstream "
            f"basename {Path(upstream).name!r} != file name {py_path.name!r}"
        )
    return tag, upstream


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {"files": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
            # Retry with whitespace tolerance for older noisy hunks.
            cmd2 = ["git", "apply", "--ignore-whitespace"]
            if reverse:
                cmd2.append("-R")
            cmd2.append(str(patch_path))
            proc2 = subprocess.run(cmd2, cwd=root, capture_output=True, text=True)
            if proc2.returncode != 0:
                err = (proc.stderr or proc.stdout or "") + "\n" + (
                    proc2.stderr or proc2.stdout or ""
                )
                raise RuntimeError(
                    f"git apply{' -R' if reverse else ''} failed for {rel_in_repo}:\n"
                    f"{err.strip()}"
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
    rel = py_path.relative_to(REPO_ROOT).as_posix()
    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    return _git_apply(pristine, diff_text, rel, reverse=False)


def make_unified_diff(pristine: bytes, adapted: bytes, rel_in_repo: str) -> str:
    """Create a stable unified diff with paths matching checked-in style."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_root = root / "a"
        b_root = root / "b"
        a_file = a_root / rel_in_repo
        b_file = b_root / rel_in_repo
        a_file.parent.mkdir(parents=True, exist_ok=True)
        b_file.parent.mkdir(parents=True, exist_ok=True)
        a_file.write_bytes(normalize_bytes(pristine))
        b_file.write_bytes(normalize_bytes(adapted))

        proc = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--no-ext-diff",
                "--binary",
                "-U3",
                f"a/{rel_in_repo}",
                f"b/{rel_in_repo}",
            ],
            cwd=root,
            capture_output=True,
        )
        # git diff --no-index returns 1 when files differ.
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"git diff failed for {rel_in_repo}: "
                f"{proc.stderr.decode('utf-8', 'replace')}"
            )
        text = normalize_text(proc.stdout.decode("utf-8", "replace"))
        if not text.strip():
            raise RuntimeError(f"empty diff for {rel_in_repo}; files are identical?")

        # Rewrite a/a/... and b/b/... (from cwd layout) into a/rel and b/rel.
        # git diff --no-index with paths a/rel b/rel already emits:
        #   diff --git a/a/rel b/b/rel
        # Normalize to a/rel b/rel.
        lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("diff --git "):
                lines.append(f"diff --git a/{rel_in_repo} b/{rel_in_repo}")
            elif line.startswith("--- "):
                lines.append(f"--- a/{rel_in_repo}")
            elif line.startswith("+++ "):
                lines.append(f"+++ b/{rel_in_repo}")
            elif line.startswith("index "):
                # Drop unstable blob hashes; apply does not need them.
                continue
            else:
                lines.append(line)
        return "\n".join(lines) + "\n"


def verify_pair(
    py_path: Path,
    diff_path: Path,
    manifest_entry: dict[str, Any] | None,
) -> list[str]:
    """Return a list of error strings (empty if OK)."""
    errors: list[str] = []
    rel = py_path.relative_to(CPYTHON_DIR).as_posix()
    try:
        tag, upstream = parse_header(py_path)
    except ValueError as e:
        return [str(e)]

    if manifest_entry is None:
        return [f"{rel}: missing upstream_manifest.json entry"]

    if manifest_entry.get("tag") != tag:
        errors.append(
            f"{rel}: manifest tag {manifest_entry.get('tag')!r} != header tag {tag!r}"
        )
    if manifest_entry.get("upstream") != upstream:
        errors.append(
            f"{rel}: manifest upstream {manifest_entry.get('upstream')!r} "
            f"!= header upstream {upstream!r}"
        )

    try:
        pristine = reverse_apply_to_pristine(py_path, diff_path)
    except Exception as e:
        errors.append(f"{rel}: reverse-apply failed: {e}")
        return errors

    got_hash = sha256_bytes(pristine)
    expected_hash = manifest_entry.get("sha256")
    if got_hash != expected_hash:
        errors.append(
            f"{rel}: pristine sha256 mismatch "
            f"(from reverse-apply={got_hash}, manifest={expected_hash}). "
            f"If you edited the adapted file, regenerate the .diff and manifest "
            f"with: python tools/regenerate_cpython_diffs.py --only {rel}"
        )

    try:
        applied = apply_diff_to_adapted(pristine, py_path, diff_path)
    except Exception as e:
        errors.append(f"{rel}: forward-apply failed: {e}")
        return errors

    adapted = normalize_bytes(py_path.read_bytes())
    if applied != adapted:
        errors.append(
            f"{rel}: apply(pristine, diff) != adapted file "
            f"(round-trip mismatch; regenerate the .diff)"
        )
    return errors


def verify_all() -> list[str]:
    manifest = load_manifest()
    files: dict[str, Any] = manifest.get("files", {})
    errors: list[str] = []
    seen: set[str] = set()

    for py_path, diff_path in iter_diff_pairs():
        rel = py_path.relative_to(CPYTHON_DIR).as_posix()
        seen.add(rel)
        errors.extend(verify_pair(py_path, diff_path, files.get(rel)))

    extra = sorted(set(files) - seen)
    for rel in extra:
        errors.append(f"{rel}: in manifest but no .py/.diff pair on disk")
    return errors
