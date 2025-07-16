from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).absolute().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DICTIONARY = REPO_ROOT / "tools" / "linter" / "dictionary.txt"

FORBIDDEN_WORDS = {
    "multipy",  # project pytorch/multipy is dead  # codespell:ignore multipy
}


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


def format_error_message(
    filename: str,
    error: Exception | None = None,
    *,
    message: str | None = None,
) -> LintMessage:
    if message is None and error is not None:
        message = (
            f"Failed due to {error.__class__.__name__}:\n{error}\n"
            "Please either fix the error or add the word(s) to the dictionary file.\n"
            "HINT: all-lowercase words in the dictionary can cover all case variations."
        )
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="CODESPELL",
        severity=LintSeverity.ERROR,
        name="spelling error",
        original=None,
        replacement=None,
        description=message,
    )


def run_codespell(path: Path) -> str:
    try:
        return subprocess.check_output(
            [
                sys.executable,
                "-m",
                "codespell_lib",
                "--toml",
                str(PYPROJECT),
                str(path),
            ],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(exc.output) from exc


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    try:
        run_codespell(path)
    except Exception as err:
        return [format_error_message(filename, err)]
    return []


def check_dictionary(filename: str) -> list[LintMessage]:
    """Check the dictionary file for duplicates."""
    path = Path(filename).absolute()
    try:
        words = path.read_text(encoding="utf-8").splitlines()
        words_set = set(words)
        if len(words) != len(words_set):
            raise ValueError("The dictionary file contains duplicate entries.")
        uncased_words = list(map(str.lower, words))
        if uncased_words != sorted(uncased_words):
            raise ValueError(
                "The dictionary file is not sorted alphabetically (case-insensitive)."
            )
        for forbidden_word in sorted(
            FORBIDDEN_WORDS & (words_set | set(uncased_words))
        ):
            raise ValueError(
                f"The dictionary file contains a forbidden word: {forbidden_word!r}. "
                "Please remove it from the dictionary file and use 'codespell:ignore' "
                "inline comment instead."
            )
    except Exception as err:
        return [format_error_message(str(filename), err)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check files for spelling mistakes using codespell.",
        fromfile_prefix_chars="@",
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
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=(os.cpu_count() or 4) // 2,
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        futures[executor.submit(check_dictionary, str(DICTIONARY))] = str(DICTIONARY)
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
