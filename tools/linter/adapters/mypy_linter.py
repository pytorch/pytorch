import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Pattern


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


# tools/linter/flake8_linter.py:15:13: error: Incompatibl...int")  [assignment]
RESULTS_RE: Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    (?:(?P<column>-?\d+):)?
    \s(?P<severity>\S+?):?
    \s(?P<message>.*)
    \s(?P<code>\[.*\])
    $
    """
)



def run_command(
    args: List[str],
    *,
    extra_env: Optional[Dict[str, str]],
    retries: int,
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


# Severity is either "error" or "note": https://git.io/JiLOP
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}

def check_file(
    filename: str,
    config: str,
    binary: str,
    retries: int,
) -> List[LintMessage]:
    try:
        proc = run_command(
            [binary, f"--config={config}", filename],
            extra_env={},
            retries=retries,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code="MYPY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                ),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code="MYPY",
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mypy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="mypy binary path",
    )
    parser.add_argument(
        "--retries",
        default=3,
        type=int,
        help="times to retry timed out mypy",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an mypy .ini config file",
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
            executor.submit(
                check_file,
                filename,
                args.config,
                args.binary,
                args.retries,
            ): filename
            for filename in args.filenames
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
