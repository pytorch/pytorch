from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import operator
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from packaging.version import Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]


REQUIRES_PYTHON_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<min_op>>=)
    \s*
    (?P<min_ver>\d+\.\d+(?:\.\d+)?)
    \s*,\s*
    (?P<max_op><(=?))
    \s*
    (?P<max_ver>\d+\.\d+(?:\.\d+)?)
    \s*$
    """,
    flags=re.VERBOSE,
)
COMPARE_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
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
        message = f"Failed due to {error.__class__.__name__}:\n{error}"
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code="PYPROJECT",
        severity=LintSeverity.ERROR,
        name="pyproject.toml consistency",
        original=None,
        replacement=None,
        description=message,
    )


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename).absolute()
    original = path.read_text(encoding="utf-8")
    try:
        pyproject = tomllib.loads(original)
    except tomllib.TOMLDecodeError as err:
        return [format_error_message(filename, err)]

    if not (isinstance(pyproject, dict) and isinstance(pyproject.get("project"), dict)):
        return [
            format_error_message(
                filename,
                message="'project' section in pyproject.toml must present and be a dictionary.",
            )
        ]
    project = pyproject["project"]

    requires_python = project.get("requires-python")
    if not isinstance(requires_python, str):
        return [
            format_error_message(
                filename,
                message="'project.requires-python' must be a string.",
            )
        ]
    match = REQUIRES_PYTHON_PATTERN.match(requires_python)
    if not match:
        return [
            format_error_message(
                filename,
                message=r"'project.requires-python' must be in the format '>={X}.{Y},<{U}.{V}'.",
            )
        ]
    min_op = COMPARE_OPS[match.group("min_op")]
    min_ver = Version(match.group("min_ver"))
    max_op = COMPARE_OPS[match.group("max_op")]
    max_ver = Version(match.group("max_ver"))
    if min_ver >= max_ver:
        return [
            format_error_message(
                filename,
                message="'project.requires-python' minimum version must be less than the maximum version.",
            )
        ]
    if min_ver.major != 3:
        return [
            format_error_message(
                filename,
                message="'project.requires-python' minimum version must be 3.x.",
            )
        ]
    if max_ver.major != 3:
        return [
            format_error_message(
                filename,
                message="'project.requires-python' maximum version must be 3.x.",
            )
        ]
    supported_versions = []
    major = 3
    for minor in range(min_ver.minor, max_ver.minor + 1):
        ver = Version(f"{major}.{minor}")
        if min_op(ver, min_ver) and max_op(ver, max_ver):
            supported_versions.append(f"{major}.{minor}")

    classifiers = project.get("classifiers")
    if not (
        isinstance(classifiers, list) and all(isinstance(c, str) for c in classifiers)
    ):
        return [
            format_error_message(
                filename,
                message="'project.classifiers' must be a list of strings.",
            )
        ]
    version_classifiers = [
        c
        for c in classifiers
        if c.startswith("Programming Language :: Python :: ")
        and not c.endswith((":: 3", ":: 3 :: Only"))
    ]
    version_classifier_set = set(version_classifiers)
    if len(set(version_classifier_set)) != len(version_classifiers):
        return [
            format_error_message(
                filename,
                message="'project.classifiers' must not contain duplicates.",
            )
        ]
    supported_version_classifier_set = {
        f"Programming Language :: Python :: {v}" for v in supported_versions
    }
    missing_classifiers = sorted(
        supported_version_classifier_set - version_classifier_set
    )
    extra_classifiers = sorted(
        version_classifier_set - supported_version_classifier_set
    )
    if missing_classifiers:
        return [
            format_error_message(
                filename,
                message=(
                    f"'project.classifiers' is missing the following classifiers: "
                    f"{missing_classifiers}."
                ),
            )
        ]
    if extra_classifiers:
        return [
            format_error_message(
                filename,
                message=(
                    f"'project.classifiers' contains extra classifiers: "
                    f"{extra_classifiers}."
                ),
            )
        ]

    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check consistency of pyproject.toml files.",
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
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
