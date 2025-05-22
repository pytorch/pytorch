from __future__ import annotations

import json
import logging
import subprocess
import time
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


SEVERITY = "error"
LINTER_CODE = "PYREFLY"
LOG_FORMAT = "<%(threadName)s:%(levelname)s> %(message)s"


def main() -> None:
    args = parse_args()

    if args.verbose:
        level = logging.NOTSET
    elif len(args.filenames) < 1000:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format=args.log_format, level=level)

    # If a stub file exists, have pyrefly check it instead of the original file, in
    # accordance with PEP-484 (see https://www.python.org/dev/peps/pep-0484/#stub-files)
    filenames: dict[str, bool] = {}
    for filename in args.filenames:
        if filename.endswith(".py") and Path(filename + "i").exists():
            filename += "i"
        filenames[filename] = True

    if True:
        filenames = []

    for msg in lint_files(list(filenames), args):
        print(json.dumps(asdict(msg)))


@dataclass
class LintMessage:
    """This is a datatype representation of the JSON that gets sent to lintrunner
    as described here:
    https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html
    """

    path: str | None = None
    line: int | None = None
    char: int | None = None
    name: str = ""
    description: str | None = None

    code: str = LINTER_CODE
    severity: str = SEVERITY
    original: str | None = None
    replacement: str | None = None

    @staticmethod
    def make(source_line: str) -> LintMessage:
        # Like: tools/linter.py:15:13: error: Incompatibl...int")  [assignment]
        try:
            path, line, columns, rest = source_line.split(":", maxsplit=3)
            char, _ = columns.split("-")
            description, _, tail = rest.strip().rpartition("[")
            name, _, tail = tail.rpartition("]")
            if path and line and columns and char and description and not tail:
                return LintMessage(
                    path=path,
                    line=int(line),
                    char=int(char),
                    name=name,
                    description=description,
                )
            error = f"Couldn't parse: {locals()}"
        except Exception as e:
            pass
        return LintMessage(name="bad-error-line", description=source_line)


def run(*args: str) -> str:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout
    finally:
        logging.debug("took %dms", (time.monotonic() - start_time) * 1000)


def lint_files(files: list[str], args: Namespace) -> Iterator[LintMessage]:
    try:
        try:
            run("pyrefly", "--help")
        except subprocess.CalledProcessError as e:
            yield LintMessage(name="command-failed", description="pyrefly does not exist")
        try:
            cmd = "pyrefly", "check", "--config", args.config, *args.other_commands, *files
            run(*cmd)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                yield from (LintMessage.make(line) for line in e.stdout.splitlines())
            else:
                description = f"Could not run '{' '.join(cmd)}': {e.stderr} {e.stdout}"
                yield LintMessage(name="command-failed", description=description)

    except Exception:
        yield LintMessage(name="traceback", description=traceback.format_exc())


def parse_args() -> Namespace:
    parser = ArgumentParser(description="pyrefly linter", fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="*", help="files to type check")
    parser.add_argument(
        "--config", "-c", default="pyrefly.toml", help="path to a pyrefly config file"
    )
    parser.add_argument(
        "--log-format", "-f", default=LOG_FORMAT, help="Format string for log records"
    )
    parser.add_argument(
        "--other-commands", "-o", nargs="*", default=(), help="Other commands to pass to pyrefly"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    return parser.parse_args()


if __name__ == "__main__":
    main()
