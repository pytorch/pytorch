import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from enum import Enum
from typing import Any, List, NamedTuple, Optional, Pattern
from install.clang_tidy import INSTALLATION_PATH, OUTPUT_DIR, PLATFORM_TO_HASH, PLATFORM_TO_URL, download  # type: ignore[import]


IS_WINDOWS: bool = os.name == "nt"


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str
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


# c10/core/DispatchKey.cpp:281:26: error: 'k' used after it was moved [bugprone-use-after-move]
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
) -> "subprocess.CompletedProcess[bytes]":
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


# Severity is either "error" or "note": https://git.io/JiLOP
severities = {
    "error": LintSeverity.ERROR,
    "warning": LintSeverity.WARNING,
}

def clang_search_dirs() -> List[str]:
    # Compilers are ordered based on fallback preference
    # We pick the first one that is available on the system
    compilers = ["clang", "gcc", "cpp", "cc"]
    compilers = [c for c in compilers if shutil.which(c) is not None]
    if len(compilers) == 0:
        raise RuntimeError(f"None of {compilers} were found")
    compiler = compilers[0]

    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-", "-v"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    stderr = result.stderr.decode().strip().split("\n")
    search_start = r"#include.*search starts here:"
    search_end = r"End of search list."

    append_path = False
    search_paths = []
    for line in stderr:
        if re.match(search_start, line):
            if append_path:
                continue
            else:
                append_path = True
        elif re.match(search_end, line):
            break
        elif append_path:
            search_paths.append(line.strip())

    return search_paths

include_args = []
include_dir = ["/usr/lib/llvm-11/include/openmp"] + clang_search_dirs()
for dir in include_dir:
    include_args += ["--extra-arg", f"-I{dir}"]


def check_file(
    filename: str,
    binary: str,
) -> List[LintMessage]:
    try:
        proc = run_command(
            [binary, "-p=build", *include_args, filename],
        )
    except (OSError) as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed due to {err.__class__.__name__}:\n{err}"
                ),
                bypassChangedLineFiltering=None,
            )
        ]

    return [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code="CLANGTIDY",
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
            bypassChangedLineFiltering=None,
        )
        for match in RESULTS_RE.finditer(proc.stdout.decode())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="clang-tidy wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--binary",
        required=False,
        help="clang-tidy binary path",
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

    if not args.binary:
        ok = download("clang-tidy", OUTPUT_DIR, PLATFORM_TO_URL, PLATFORM_TO_HASH)
        if not ok:
            err_msg = LintMessage(
                path="clangtidy_linte.rpy",
                line=None,
                char=None,
                code="CLANGTIDY",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(
                    f"Failed to download clang-tidy binary from {PLATFORM_TO_URL}"
                ),
                bypassChangedLineFiltering=None,
            )
            print(json.dumps(err_msg._asdict()), flush=True)
            exit(0)
        args.binary = INSTALLATION_PATH

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count(),
        thread_name_prefix="Thread",
    ) as executor:
        futures = {
            executor.submit(
                check_file,
                filename,
                args.binary,
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
