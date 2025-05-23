import json
import logging
import subprocess
import sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional


SEVERITY = "error"
LINTER_CODE = "PYREFLY"
LOG_FORMAT = "<%(threadName)s:%(levelname)s> %(message)s"
IGNORE_FILENAMES = True
REPORT_ERRORS = False

_fp = open("logfile.txt", "w")
_print = partial(print, file=_fp)
_print(f"{sys.argv=}")


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
    filedict: dict[str, bool] = {}
    for filename in args.filenames:
        if filename.endswith(".py") and Path(filename + "i").exists():
            filename += "i"
        filedict[filename] = True

    filenames = [] if IGNORE_FILENAMES else list(filedict)
    for msg in lint_files(filenames, args):
        print(json.dumps(asdict(msg)))
        if not True:
            break


@dataclass
class LintMessage:
    """This is a datacass representation of the JSON that gets sent to lintrunner from:
    https://docs.rs/lintrunner/latest/lintrunner/lint_message/struct.LintMessage.html
    """

    path: Optional[str] = None
    line: Optional[int] = None
    char: Optional[int] = None
    name: str = ""
    description: Optional[str] = None

    code: str = LINTER_CODE
    severity: str = SEVERITY
    original: Optional[str] = None
    replacement: Optional[str] = None

    @staticmethod
    def make(source_line: str) -> Optional["LintMessage"]:
        # e.g. tools/linter.py:15:9-20: error: Incompatibl...int")  [assignment]
        path = ""
        try:
            path, _line, columns, rest = source_line.split(":", maxsplit=3)
        except Exception:
            pass

        if not (path.endswith((".py", ".pyi")) and "/" in path):
            if REPORT_ERRORS:
                return LintMessage(name="bad-error-line", description=source_line)
            return None

        try:
            line: Optional[int] = int(_line)
        except Exception:
            line = None
        try:
            char: Optional[int] = int(columns.partition("-")[0])
        except Exception:
            char = None

        description, _, tail = rest.strip().rpartition("[")
        name = tail.rpartition("]")[0] or "bad-error-line"
        return LintMessage(
            path=path,
            line=line,
            char=char,
            name=name,
            description=description or source_line,
        )


def run(*args: str) -> str:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout
    finally:
        logging.debug("took %dms", (time.monotonic() - start_time) * 1000)


def lint_files(files: list[str], args: Namespace) -> Iterator[LintMessage]:
    bad_lines = 0
    try:
        try:
            run("pyrefly", "--help")
        except subprocess.CalledProcessError:
            yield LintMessage(
                name="command-failed", description="pyrefly does not exist"
            )
            return

        try:
            cmd = "pyrefly", "check", "--config"
            argv = args.config, *args.other_commands, *files
            run(*cmd, *argv)
            return

        except subprocess.CalledProcessError as err:
            e = err

        if e.returncode == 1:
            for line in e.stdout.splitlines():
                if m := LintMessage.make(line):
                    yield m
                else:
                    bad_lines += 1
        else:
            description = f"Could not run '{' '.join(cmd)}': {e.stderr} {e.stdout}"
            yield LintMessage(name="command-failed", description=description)

    except Exception:
        yield LintMessage(name="traceback", description=traceback.format_exc())

    if bad_lines:
        description = (
            f"Number of pyrefly error lines which could not be parsed: {bad_lines}"
        )
        yield LintMessage(name="bad_lines", description=description)


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
        "--other-commands",
        "-o",
        nargs="*",
        default=(),
        help="Other commands to pass to pyrefly",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    return parser.parse_args()


if __name__ == "__main__":
    main()
