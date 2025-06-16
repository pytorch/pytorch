from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import black
import isort
import usort


IS_WINDOWS: bool = os.name == "nt"
REPO_ROOT = Path(__file__).absolute().parents[3]

# TODO: remove this when it gets empty and remove `black` in PYFMT
USE_BLACK_FILELIST = re.compile(
    "|".join(
        (
            r"\A\Z",  # empty string
            *map(
                fnmatch.translate,
                [
                    # **
                    # .ci/**
                    # .github/**
                    # benchmarks/**
                    # functorch/**
                    # tools/**
                    # torchgen/**
                    # test/**
                    # test/[a-h]*/**
                    # test/[i-j]*/**
                    # test/[k-m]*/**
                    # test/optim/**
                    # test/[p-z]*/**,
                    # torch/**
                    # torch/_[a-c]*/**
                    # torch/_[e-h]*/**
                    # torch/_i*/**
                    # torch/_[j-z]*/**
                    # torch/[a-c]*/**
                    # torch/d*/**
                    # torch/[e-m]*/**
                    # torch/optim/**
                    # torch/[p-z]*/**
                    "torch/[p-z]*/**",
                ],
            ),
        )
    )
)


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


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def format_error_message(filename: str, err: Exception) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="PYFMT",
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )


def run_isort(content: str, path: Path) -> str:
    isort_config = isort.Config(settings_path=str(REPO_ROOT))

    is_this_file = path.samefile(__file__)
    if not is_this_file:
        content = re.sub(r"(#.*\b)usort:\s*skip\b", r"\g<1>isort: split", content)

    content = isort.code(content, config=isort_config, file_path=path)

    if not is_this_file:
        content = re.sub(r"(#.*\b)isort: split\b", r"\g<1>usort: skip", content)

    return content


def run_usort(content: str, path: Path) -> str:
    usort_config = usort.Config.find(path)

    return usort.usort_string(content, path=path, config=usort_config)


def run_black(content: str, path: Path) -> str:
    black_config = black.parse_pyproject_toml(black.find_pyproject_toml((str(path),)))  # type: ignore[attr-defined,arg-type]
    # manually patch options that do not have a 1-to-1 match in Mode arguments
    black_config["target_versions"] = {
        black.TargetVersion[ver.upper()]  # type: ignore[attr-defined]
        for ver in black_config.pop("target_version", [])
    }
    black_config["string_normalization"] = not black_config.pop(
        "skip_string_normalization", False
    )
    black_mode = black.Mode(**black_config)
    black_mode.is_pyi = path.suffix.lower() == ".pyi"
    black_mode.is_ipynb = path.suffix.lower() == ".ipynb"

    return black.format_str(content, mode=black_mode)


def run_ruff_format(content: str, path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--config",
                str(REPO_ROOT / "pyproject.toml"),
                "--stdin-filename",
                str(path),
                "-",
            ],
            input=content,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    original = replacement = path.read_text(encoding="utf-8")

    try:
        # NB: run isort first to enforce style for blank lines
        replacement = run_isort(replacement, path=path)
        replacement = run_usort(replacement, path=path)
        if USE_BLACK_FILELIST.match(path.absolute().relative_to(REPO_ROOT).as_posix()):
            replacement = run_black(replacement, path=path)
        else:
            replacement = run_ruff_format(replacement, path=path)

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="PYFMT",
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as err:
        return [format_error_message(filename, err)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with usort + ruff-format.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
