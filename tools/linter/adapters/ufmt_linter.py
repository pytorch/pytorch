import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Union


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


def _run_command(
    args: List[str],
    *,
    input: Optional[bytes],
    timeout: int,
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            input=input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=IS_WINDOWS,  # So batch scripts are found.
            timeout=timeout,
            check=True,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def run_command(
    args: List[str],
    *,
    input: Optional[bytes],
    retries: int,
    timeout: int,
) -> "subprocess.CompletedProcess[bytes]":
    remaining_retries = retries
    while True:
        try:
            return _run_command(args, input=input, timeout=timeout)
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


def format_timeout_message(filename: str) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="UFMT",
        severity=LintSeverity.ERROR,
        name="timeout",
        original=None,
        replacement=None,
        description=(
            "ufmt timed out while trying to process a file. "
            "Please report an issue in pytorch/pytorch with the "
            "label 'module: lint'"
        ),
    )


def format_error_message(
    filename: str, err: Union[OSError, subprocess.CalledProcessError]
) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="UFMT",
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


def check_file(
    filename: str,
    retries: int,
    timeout: int,
) -> List[LintMessage]:
    with open(filename, "rb") as f:
        original = f.read()

    try:
        # Note that UFMT returns error code 1 if there are differences
        # so the diff output is actually in the error
        proc = run_command(
            [sys.executable, "-mufmt", "format", filename],
            input=None,
            retries=retries,
            timeout=timeout,
        )

        # UFMT is a bit unyielding to use in lintrunner workflow because it
        # only includes options to check, generate diff, and apply the patch
        # in place
        with open(filename, "rb") as f:
            replacement = f.read()

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
                original=original.decode("utf-8"),
                replacement=replacement.decode("utf-8"),
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except subprocess.TimeoutExpired:
        return [format_timeout_message(filename)]
    except (OSError, subprocess.CalledProcessError) as err:
        return [format_error_message(filename, err)]
    finally:
        # Always revert the formatting change back to the original content
        # to fit with lintrunner -a workflow
        with open(filename, "wb") as f:
            f.write(original)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format files with ufmt (black + usort).",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out ufmt (black + usort)",
    )
    parser.add_argument(
        "--timeout",
        default=90,
        type=int,
        help="seconds to wait for ufmt (black + usort)",
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
        max_workers=os.cpu_count(),
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
