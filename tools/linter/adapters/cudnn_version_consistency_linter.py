# /// script
# requires-python = ">=3.10"
# ///
"""Checks that cuDNN versions in the Linux and Windows CUDA install scripts match.

Parses the cuDNN version for each CUDA version from:
  - .ci/docker/common/install_cuda.sh  (Linux)
  - .ci/pytorch/windows/internal/cuda_install.bat  (Windows)

and reports an error when they diverge.
"""

from __future__ import annotations

import argparse
import json
import re
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

LINUX_SCRIPT = REPO_ROOT / ".ci" / "docker" / "common" / "install_cuda.sh"
WINDOWS_SCRIPT = REPO_ROOT / ".ci" / "pytorch" / "windows" / "internal" / "cuda_install.bat"


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


def parse_linux_cudnn_versions(text: str) -> dict[str, tuple[str, int]]:
    """Return {cuda_short_version: (cudnn_version, line_number)} from install_cuda.sh.

    Looks for patterns like:
        function install_128 {
          CUDNN_VERSION=9.19.0.56
    """
    results: dict[str, tuple[str, int]] = {}
    func_re = re.compile(r"^function install_(\d+)\s*\{")
    cudnn_re = re.compile(r"^\s*CUDNN_VERSION=(\S+)")
    current_func = None
    for lineno, line in enumerate(text.splitlines(), 1):
        m = func_re.match(line)
        if m:
            digits = m.group(1)
            # "128" -> "12.8", "130" -> "13.0"
            current_func = digits[:-1] + "." + digits[-1]
            continue
        if current_func is not None:
            m = cudnn_re.match(line)
            if m:
                results[current_func] = (m.group(1), lineno)
                current_func = None
    return results


def parse_windows_cudnn_versions(text: str) -> dict[str, tuple[str, int]]:
    """Return {cuda_short_version: (cudnn_version, line_number)} from cuda_install.bat.

    Looks for patterns like:
        :cuda128
        ...
        set CUDNN_FOLDER=cudnn-windows-x86_64-9.19.0.56_cuda12-archive
    """
    results: dict[str, tuple[str, int]] = {}
    label_re = re.compile(r"^:cuda(\d+)\s*$")
    cudnn_re = re.compile(
        r"^set CUDNN_FOLDER=cudnn-windows-x86_64-([0-9.]+)_cuda\d+-archive"
    )
    current_label = None
    for lineno, line in enumerate(text.splitlines(), 1):
        m = label_re.match(line)
        if m:
            digits = m.group(1)
            current_label = digits[:-1] + "." + digits[-1]
            continue
        if current_label is not None:
            m = cudnn_re.match(line)
            if m:
                results[current_label] = (m.group(1), lineno)
                current_label = None
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cuDNN version consistency linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    parser.parse_args()

    linux_text = LINUX_SCRIPT.read_text()
    windows_text = WINDOWS_SCRIPT.read_text()

    linux_versions = parse_linux_cudnn_versions(linux_text)
    windows_versions = parse_windows_cudnn_versions(windows_text)

    all_cuda_versions = sorted(set(linux_versions) | set(windows_versions))
    for cuda_ver in all_cuda_versions:
        linux_entry = linux_versions.get(cuda_ver)
        windows_entry = windows_versions.get(cuda_ver)

        if linux_entry is None or windows_entry is None:
            # One platform doesn't have this CUDA version at all; not our concern.
            continue

        linux_cudnn, _ = linux_entry
        windows_cudnn, win_line = windows_entry

        if linux_cudnn != windows_cudnn:
            msg = LintMessage(
                path=str(WINDOWS_SCRIPT.relative_to(REPO_ROOT)),
                line=win_line,
                char=None,
                code="CUDNNSYNC",
                severity=LintSeverity.ERROR,
                name="cudnn-version-mismatch",
                original=None,
                replacement=None,
                description=(
                    f"cuDNN version mismatch for CUDA {cuda_ver}: "
                    f"Linux has {linux_cudnn} "
                    f"(.ci/docker/common/install_cuda.sh) "
                    f"but Windows has {windows_cudnn}. "
                    f"These should match."
                ),
            )
            print(json.dumps(msg._asdict()), flush=True)


if __name__ == "__main__":
    main()
