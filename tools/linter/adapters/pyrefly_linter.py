import json
import logging
import subprocess
import time
import traceback
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


SEVERITY = "error"
LINTER_CODE = "PYREFLY"
LOG_FORMAT = "<%(threadName)s:%(levelname)s> %(message)s"
IGNORE_FILENAMES = True


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
    for filename in args.filedict:
        if filename.endswith(".py") and Path(filename + "i").exists():
            filename += "i"
        filedict[filename] = True

    filenames = [] if IGNORE_FILENAMES else list(filedict)
    for msg in lint_files(filenames, args):
        print(json.dumps(asdict(msg)))


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
    def make(source_line: str) -> "LintMessage":
        # e.g. tools/linter.py:15:9-20: error: Incompatibl...int")  [assignment]
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
        except Exception:
            error = f"{source_line=}\n{traceback.format_exc()}"
        return LintMessage(name="bad-error-line", description=error)


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
        except subprocess.CalledProcessError:
            yield LintMessage(
                name="command-failed", description="pyrefly does not exist"
            )
        try:
            cmd = (
                "pyrefly",
                "check",
                "--config",
                args.config,
                *args.other_commands,
                *files,
            )
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
