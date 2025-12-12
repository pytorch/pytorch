"""NO_DATA_PTR: Block new usages of the legacy ``data_ptr()`` accessor."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional


LINTER_CODE = "NO_DATA_PTR"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PTR_PATTERN = re.compile(r"(?<![A-Za-z0-9_])data_ptr\s*(?:<[^>]*>)?\s*\(")
HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def _normalize_repo_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    return rel.as_posix()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _capture_git(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        logging.debug("Failed to run %s: %s", " ".join(args), err)  # noqa: G200
        return None
    return proc.stdout.strip()


@functools.lru_cache(maxsize=1)
def _determine_merge_base() -> str | None:
    # Try to find merge base with various main branch refs
    for ref in ["main", "origin/main", "upstream/main"]:
        base = _capture_git(["git", "merge-base", "HEAD", ref])
        if base:
            logging.debug("Computed merge base vs %s: %s", ref, base)
            return base

    base = _capture_git(["git", "rev-parse", "HEAD^"])
    if base:
        logging.debug("Falling back to HEAD^: %s", base)
    else:
        logging.warning(
            "Unable to determine merge base; falling back to full file scans"
        )
    return base


def _parse_diff_output(output: str) -> set[int]:
    lines: set[int] = set()
    current: Optional[int] = None
    for raw in output.splitlines():
        if raw.startswith("@@"):
            match = HUNK_RE.match(raw)
            if not match:
                current = None
                continue
            current = int(match.group("start"))
            continue
        if current is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            lines.add(current)
            current += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue
        elif raw.startswith("\\ No newline"):
            continue
        else:
            current += 1
    return lines


def _run_git_diff(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as err:
        logging.warning("Failed to run %s: %s", " ".join(args), err)  # noqa: G200
        return None
    return proc.stdout


def _is_tracked(rel_path: str) -> bool | None:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return None
    return True


def _collect_changed_lines(rel_path: str) -> tuple[bool, set[int] | None]:
    lines: set[int] = set()
    merge_base = _determine_merge_base()
    diff_commands = []
    if merge_base:
        diff_commands.append(
            [
                "git",
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                f"{merge_base}...HEAD",
                "--",
                rel_path,
            ]
        )

    diff_commands.extend(
        [
            [
                "git",
                "diff",
                "--cached",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--",
                rel_path,
            ],
            [
                "git",
                "diff",
                "--unified=0",
                "--no-color",
                "--no-ext-diff",
                "--",
                rel_path,
            ],
        ]
    )

    for diff_args in diff_commands:
        output = _run_git_diff(diff_args)
        if output is None:
            return True, None
        lines.update(_parse_diff_output(output))

    if lines:
        return True, lines

    tracked = _is_tracked(rel_path)
    if tracked is False:
        # Untracked files are treated as entirely new
        return True, None
    if tracked is None:
        logging.warning("Unable to determine git tracking status for %s", rel_path)
        return True, None

    return False, set()


def check_file(
    path: Path, rel_path: str, changed_lines: set[int] | None
) -> list[LintMessage]:
    messages: list[LintMessage] = []
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, 1):
                if changed_lines is not None and lineno not in changed_lines:
                    continue
                match = DATA_PTR_PATTERN.search(line)
                if not match:
                    continue
                messages.append(
                    LintMessage(
                        path=rel_path,
                        line=lineno,
                        char=match.start(),
                        code=LINTER_CODE,
                        severity=LintSeverity.ERROR,
                        name="data_ptr-usage",
                        original=None,
                        replacement=None,
                        description=(
                            "Do not add new uses of data_ptr(); prefer mutable_data_ptr() "
                            "or const_data_ptr()."
                        ),
                    )
                )
    except OSError as err:
        messages.append(
            LintMessage(
                path=rel_path,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="file-access-error",
                original=None,
                replacement=None,
                description=f"Error reading file: {err}",
            )
        )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checks new code for use of the legacy data_ptr() accessor.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument("filenames", nargs="+", help="paths to scan")
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    files: dict[str, Path] = {}
    for raw in args.filenames:
        resolved = _resolve_path(raw)
        rel = _normalize_repo_path(resolved)
        files[rel] = resolved

    for rel_path, abs_path in files.items():
        should_scan, changed_lines = _collect_changed_lines(rel_path)
        if not should_scan:
            logging.debug(
                "Skipping %s because no modified lines were detected", rel_path
            )
            continue
        for message in check_file(abs_path, rel_path, changed_lines):
            print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
