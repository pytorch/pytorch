from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from typing import NamedTuple


class LintMessage(NamedTuple):
    message: str | None
    filename: str | None
    line: int | None
    column: int | None
    code: str | None


RESULTS_RE = re.compile(
    r"""(?msx)
    ^error:\s+(?P<message>`[^`]+`\s+should\s+be\s+`[^`]+`(?:,\s*`[^`]+`)*)\s*\n
    [^\n]*?╭▸\s*(?P<filename>[^\s:]+):(?P<line>\d+):(?P<column>\d+)
    """,
)


def run_command(args: list[str]) -> subprocess.CompletedProcess[bytes]:
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


def check_run_typos(filenames: list[str], config: str, code: str) -> list[LintMessage]:
    filenames = [os.path.relpath(f) for f in filenames]
    try:
        proc = run_command(
            ["typos", "--config", f"{config}"] + filenames,
        )
        # print(["typos", "--config", f"{config}"] + filenames)
    except OSError:
        return [
            LintMessage(
                message=None,
                filename=None,
                line=None,
                column=None,
                code=code,
            )
        ]
    stdout = str(proc.stdout, "utf-8").strip()

    rc = [
        LintMessage(
            message=match["message"],
            filename=match["filename"],
            line=int(match["line"]),
            column=int(match["column"]),
            code=code,
        )
        for match in RESULTS_RE.finditer(stdout)
    ]
    return rc


def check_typos_installed(code: str) -> list[LintMessage]:
    cmd = ["typos", "-V"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return []
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="replace")
        return [
            LintMessage(
                message=f"Could not run '{' '.join(cmd)}': {msg}",
                filename=None,
                line=None,
                column=None,
                code=code,
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="typos wrapper linter.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="path to an typos _typos.toml file",
    )
    parser.add_argument(
        "--code",
        default="TYPOS",
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
    # print(args.filenames)
    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )
    lint_messages = check_typos_installed(args.code) + check_run_typos(
        args.filenames, args.config, args.code
    )
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
