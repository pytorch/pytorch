import argparse
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from typing import Any, List, NamedTuple, Optional


IS_WINDOWS: bool = os.name == "nt"


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: Optional[str]
    line: Optional[int]
    char: Optional[int]
    code: str
    severity: LintSeverity
    name: str
    original: Optional[str]
    replacement: Optional[str]
    description: Optional[str]
    bypassChangedLineFiltering: Optional[bool]


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def run_command(
    args: List[str],
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="grep wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="pattern to grep for",
    )
    parser.add_argument(
        "--linter_name",
        required=True,
        help="name of the linter",
    )
    parser.add_argument(
        "--error_name",
        required=True,
        help="human-readable description of what the error is",
    )
    parser.add_argument(
        "--error_description",
        required=True,
        help="message to display when the pattern is found",
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
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    try:
        proc = run_command(["grep", "-nPH", args.pattern, *args.filenames])
    except OSError as err:
        err_msg = LintMessage(
            path=None,
            line=None,
            char=None,
            code=args.linter_name,
            severity=LintSeverity.ERROR,
            name="command-failed",
            original=None,
            replacement=None,
            description=(
                f"Failed due to {err.__class__.__name__}:\n{err}"
            ),
            bypassChangedLineFiltering=None,
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    lines = proc.stdout.decode().splitlines()
    for line in lines:
        # tools/linter/clangtidy_linter.py:13:import foo.bar.baz
        split = line.split(":")
        msg = LintMessage(
            path=split[0],
            line=int(split[1]),
            char=None,
            code=args.linter_name,
            severity=LintSeverity.ERROR,
            name=args.error_name,
            original=None,
            replacement=None,
            description=args.error_description,
            bypassChangedLineFiltering=None,
        )
        print(json.dumps(msg._asdict()), flush=True)

if __name__ == "__main__":
    main()
