from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple


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


# tools/linter/flake8_linter.py:15:13: error: Incompatibl...int")  [assignment]
RESULTS_RE: re.Pattern[str] = re.compile(
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

# torch/_dynamo/variables/tensor.py:363: error: INTERNAL ERROR
INTERNAL_ERROR_RE: re.Pattern[str] = re.compile(
    r"""(?mx)
    ^
    (?P<file>.*?):
    (?P<line>\d+):
    \s(?P<severity>\S+?):?
    \s(?P<message>INTERNAL\sERROR.*)
    $
    """
)


def run_command(
    args: list[str],
    *,
    extra_env: dict[str, str] | None,
    retries: int,
) -> subprocess.CompletedProcess[bytes]:
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


# Severity is either "error" or "note":
# https://github.com/python/mypy/blob/8b47a032e1317fb8e3f9a818005a6b63e9bf0311/mypy/errors.py#L46-L47
severities = {
    "error": LintSeverity.ERROR,
    "note": LintSeverity.ADVICE,
}


def try_pre_req(cmd: list[str], code: str) -> list[LintMessage]:
    ret = []
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")
        ret.append(
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Could not run '{' '.join(cmd)}': {msg}",
            )
        )
    return ret


def check_mypy_installed(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "-mmypy", "-V"]
    return try_pre_req(cmd, code)


def generate_torch_version(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "-m", "tools.generate_torch_version", "--is_debug=false"]
    return try_pre_req(cmd, code)


def generate_native_functions_pyi(code: str) -> list[LintMessage]:
    cmd = [
        sys.executable,
        "-m",
        "tools.pyi.gen_pyi",
        "--native-functions-path",
        "aten/src/ATen/native/native_functions.yaml",
        "--tags-path",
        "aten/src/ATen/native/tags.yaml",
        "--deprecated-functions-path",
        "tools/autograd/deprecated.yaml",
    ]
    return try_pre_req(cmd, code)


def generate_datapipe_pyi(code: str) -> list[LintMessage]:
    cmd = [sys.executable, "torch/utils/data/datapipes/gen_pyi.py"]
    return try_pre_req(cmd, code)


def do_pre_reqs(code: str) -> list[LintMessage]:
    return [
        *check_mypy_installed(code),
        *generate_torch_version(code),
        *generate_native_functions_pyi(code),
        *generate_datapipe_pyi(code),
    ]


def in_github_actions() -> bool:
    return bool(os.getenv("GITHUB_ACTIONS"))


def check_files(
    filenames: list[str],
    config: str,
    retries: int,
    code: str,
) -> list[LintMessage]:
    # dmypy has a bug where it won't pick up changes if you pass it absolute
    # file names, see https://github.com/python/mypy/issues/16768
    filenames = [os.path.relpath(f) for f in filenames]
    try:
        mypy_commands = ["dmypy", "run", "--"]
        if in_github_actions():
            mypy_commands = ["mypy"]
        proc = run_command(
            [*mypy_commands, f"--config={config}"] + filenames,
            extra_env={},
            retries=retries,
        )
    except OSError as err:
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()
    stderr = str(proc.stderr, "utf-8").strip()
    if proc.returncode not in (0, 1):
        return [
            LintMessage(
                path=None,
                line=None,
                char=None,
                code=code,
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=stderr,
            )
        ]

    rc = [
        LintMessage(
            path=match["file"],
            name=match["code"],
            description=match["message"],
            line=int(match["line"]),
            char=int(match["column"])
            if match["column"] is not None and not match["column"].startswith("-")
            else None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in RESULTS_RE.finditer(stdout)
    ] + [
        LintMessage(
            path=match["file"],
            name="INTERNAL ERROR",
            description=match["message"],
            line=int(match["line"]),
            char=None,
            code=code,
            severity=severities.get(match["severity"], LintSeverity.ERROR),
            original=None,
            replacement=None,
        )
        for match in INTERNAL_ERROR_RE.finditer(stderr)
    ]
    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mypy wrapper linter.",
        fromfile_prefix_chars="@",
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
        "--code",
        default="MYPY",
        help="the code this lint should report as",
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

    # Use a dictionary here to preserve order. mypy cares about order,
    # tragically, e.g. https://github.com/python/mypy/issues/2015
    filenames: dict[str, bool] = {}

    # If a stub file exists, have mypy check it instead of the original file, in
    # accordance with PEP-484 (see https://www.python.org/dev/peps/pep-0484/#stub-files)
    for filename in args.filenames:
        if filename.endswith(".pyi"):
            filenames[filename] = True
            continue

        stub_filename = filename.replace(".py", ".pyi")
        if Path(stub_filename).exists():
            filenames[stub_filename] = True
        else:
            filenames[filename] = True

    lint_messages = do_pre_reqs(args.code) + check_files(
        list(filenames), args.config, args.retries, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
