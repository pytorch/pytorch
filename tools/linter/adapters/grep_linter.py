"""
Generic linter that greps for a pattern and optionally suggests replacements.
"""

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
            capture_output=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def lint_file(
    matching_line: str,
    allowlist_pattern: str,
    replace_pattern: str,
    linter_name: str,
    error_name: str,
    error_description: str,
) -> Optional[LintMessage]:
    # matching_line looks like:
    #   tools/linter/clangtidy_linter.py:13:import foo.bar.baz
    split = matching_line.split(":")
    filename = split[0]

    if allowlist_pattern:
        try:
            proc = run_command(["grep", "-nEHI", allowlist_pattern, filename])
        except Exception as err:
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )

        # allowlist pattern was found, abort lint
        if proc.returncode == 0:
            return None

    original = None
    replacement = None
    if replace_pattern:
        with open(filename, "r") as f:
            original = f.read()

        try:
            proc = run_command(["sed", "-r", replace_pattern, filename])
            replacement = proc.stdout.decode("utf-8")
        except Exception as err:
            return LintMessage(
                path=None,
                line=None,
                char=None,
                code=linter_name,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        "COMMAND (exit code {returncode})\n"
                        "{command}\n\n"
                        "STDERR\n{stderr}\n\n"
                        "STDOUT\n{stdout}"
                    ).format(
                        returncode=err.returncode,
                        command=" ".join(as_posix(x) for x in err.cmd),
                        stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                        stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                    )
                ),
            )

    return LintMessage(
        path=split[0],
        line=int(split[1]) if len(split) > 1 else None,
        char=None,
        code=linter_name,
        severity=LintSeverity.ERROR,
        name=error_name,
        original=original,
        replacement=replacement,
        description=error_description,
    )


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
        "--allowlist-pattern",
        help="if this pattern is true in the file, we don't grep for pattern",
    )
    parser.add_argument(
        "--linter-name",
        required=True,
        help="name of the linter",
    )
    parser.add_argument(
        "--match-first-only",
        action="store_true",
        help="only match the first hit in the file",
    )
    parser.add_argument(
        "--error-name",
        required=True,
        help="human-readable description of what the error is",
    )
    parser.add_argument(
        "--error-description",
        required=True,
        help="message to display when the pattern is found",
    )
    parser.add_argument(
        "--replace-pattern",
        help=(
            "the form of a pattern passed to `sed -r`. "
            "If specified, this will become proposed replacement text."
        ),
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

    files_with_matches = []
    if args.match_first_only:
        files_with_matches = ["--files-with-matches"]

    try:
        proc = run_command(
            ["grep", "-nEHI", *files_with_matches, args.pattern, *args.filenames]
        )
    except Exception as err:
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
                if not isinstance(err, subprocess.CalledProcessError)
                else (
                    "COMMAND (exit code {returncode})\n"
                    "{command}\n\n"
                    "STDERR\n{stderr}\n\n"
                    "STDOUT\n{stdout}"
                ).format(
                    returncode=err.returncode,
                    command=" ".join(as_posix(x) for x in err.cmd),
                    stderr=err.stderr.decode("utf-8").strip() or "(empty)",
                    stdout=err.stdout.decode("utf-8").strip() or "(empty)",
                )
            ),
        )
        print(json.dumps(err_msg._asdict()), flush=True)
        exit(0)

    lines = proc.stdout.decode().splitlines()
    for line in lines:
        lint_message = lint_file(
            line,
            args.allowlist_pattern,
            args.replace_pattern,
            args.linter_name,
            args.error_name,
            args.error_description,
        )
        if lint_message is not None:
            print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
