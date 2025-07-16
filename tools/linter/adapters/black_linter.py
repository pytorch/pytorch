from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from typing import BinaryIO, NamedTuple


IS_WINDOWS: bool = os.name == "nt"


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


def _run_command(
    args: list[str],
    *,
    stdin: BinaryIO,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdin=stdin,
            capture_output=True,
            shell=IS_WINDOWS,  # So batch scripts are found.
            timeout=timeout,
            check=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: list[str],
    *,
    stdin: BinaryIO,
    retries: int,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    remaining_retries = retries
    while True:
        try:
            return _run_command(args, stdin=stdin, timeout=timeout)
        except subprocess.TimeoutExpired as err:
            if remaining_retries == 0:
                raise err
            remaining_retries -= 1
            logging.warning(
                "(%s/%s) Retrying because command failed with: %r",
                retries - remaining_retries,
                retries,
                err,
            )
            time.sleep(1)


def check_file(
    filename: str,
    retries: int,
    timeout: int,
) -> list[LintMessage]:
    try:
        with open(filename, "rb") as f:
            original = f.read()
        with open(filename, "rb") as f:
            proc = run_command(
                [sys.executable, "-mblack", "--stdin-filename", filename, "-"],
                stdin=f,
                retries=retries,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="BLACK",
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=(
                    "black timed out while trying to process a file. "
                    "Please report an issue in pytorch/pytorch with the "
                    "label 'module: lint'"
                ),
            )
        ]
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="BLACK",
                severity=LintSeverity.ADVICE,
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
        ]

    replacement = proc.stdout
    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            line=None,
            char=None,
            code="BLACK",
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="Run `lintrunner -a` to apply this patch.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with black.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out black",
    )
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for black",
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

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(os.cpu_count() or 4) // 2,
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(check_file, x, args.retries, args.timeout): x
            for x in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
