from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import isort
from isort import Config as IsortConfig
from ufmt.core import ufmt_string
from ufmt.util import make_black_config
from usort import Config as UsortConfig


IS_WINDOWS: bool = os.name == "nt"
REPO_ROOT = Path(__file__).absolute().parents[3]
ISORT_SKIPLIST = re.compile(
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
                    # test/[a-c]*/**
                    # test/d*/**
                    # test/dy*/**
                    # test/[e-h]*/**
                    # test/i*/**
                    # test/j*/**
                    "test/j*/**",
                    # test/[k-p]*/**
                    # test/[q-z]*/**
                    # torch/**
                    # torch/_[a-c]*/**
                    # torch/_d*/**
                    "torch/_d*/**",
                    # torch/_[e-h]*/**
                    # torch/_i*/**
                    # torch/_[j-z]*/**
                    # torch/[a-c]*/**
                    "torch/[a-c]*/**",
                    # torch/d*/**
                    "torch/d*/**",
                    # torch/[e-n]*/**
                    "torch/[e-n]*/**",
                    # torch/[o-z]*/**
                ],
            ),
        )
    )
)


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


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
        code="UFMT",
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    original = path.read_text(encoding="utf-8")

    try:
        usort_config = UsortConfig.find(path)
        black_config = make_black_config(path)

        if not path.samefile(__file__) and not ISORT_SKIPLIST.match(
            path.absolute().relative_to(REPO_ROOT).as_posix()
        ):
            isorted_replacement = re.sub(
                r"(#.*\b)isort: split\b",
                r"\g<1>usort: skip",
                isort.code(
                    re.sub(r"(#.*\b)usort:\s*skip\b", r"\g<1>isort: split", original),
                    config=IsortConfig(settings_path=str(REPO_ROOT)),
                    file_path=path,
                ),
            )
        else:
            isorted_replacement = original

        # Use UFMT API to call both usort and black
        replacement = ufmt_string(
            path=path,
            content=isorted_replacement,
            usort_config=usort_config,
            black_config=black_config,
        )

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="UFMT",
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
        description="Format files with ufmt (black + usort).",
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
