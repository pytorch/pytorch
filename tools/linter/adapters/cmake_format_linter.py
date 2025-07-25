"""Adapter for https://github.com/cheshirekow/cmake_format."""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import subprocess
import sys

from lintrunner_adapters import (
    LintMessage,
    LintSeverity,
    add_default_options,
    as_posix,
    run_command,
)

LINTER_CODE = "CMAKEFORMAT"


def check_file(
    filename: str,
    retries: int,
    timeout: int,
    config_file: str | None = None,
) -> list[LintMessage]:
    try:
        with open(filename, "rb") as f:
            original = f.read()
        with open(filename, "rb") as f:
            command = ["cmake-format", "-"]
            if config_file:
                command.extend(["-c", config_file])

            proc = run_command(
                command,
                stdin=f,
                retries=retries,
                timeout=timeout,
                check=True,
            )
    except subprocess.TimeoutExpired:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="timeout",
                original=None,
                replacement=None,
                description=f"cmake-format timed out while trying to process {filename}.",
            )
        ]
    except (OSError, subprocess.CalledProcessError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ADVICE,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                    if not isinstance(err, subprocess.CalledProcessError)
                    else (
                        f"COMMAND (exit code {err.returncode})\n"
                        f"{' '.join(as_posix(x) for x in err.cmd)}\n\n"
                        f"STDERR\n{err.stderr.decode('utf-8').strip() or '(empty)'}\n\n"
                        f"STDOUT\n{err.stdout.decode('utf-8').strip() or '(empty)'}"
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
            code=LINTER_CODE,
            severity=LintSeverity.WARNING,
            name="format",
            original=original.decode("utf-8"),
            replacement=replacement.decode("utf-8"),
            description="Run `lintrunner -a` to apply this patch.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Format files with cmake-format. Linter code: {LINTER_CODE}",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config-file",
        help="Path to cmake-format configuration file",
    )
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for cmake-format",
    )
    add_default_options(parser)
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=(
            logging.NOTSET
            if args.verbose
            else logging.DEBUG
            if len(args.filenames) < 1000
            else logging.INFO
        ),
        stream=sys.stderr,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                x,
                args.retries,
                args.timeout,
                args.config_file,
            ): x
            for x in args.filenames
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    lint_message.display()
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
