#!/usr/bin/env python
import argparse
import collections
import re

from clang_tidy import get_changed_files, get_changed_lines

RegexLint = collections.namedtuple("RegexLint", ["error_code", "error_desc", "regex"])
LintError = collections.namedtuple(
    "LintError", ["regex_lint", "line_number", "column_number", "filename"]
)

LINTS = [
    RegexLint(
        error_code="deprecated-assert",
        error_desc="Don't use AT_XXX checks in new code, prefer TORCH_CHECK/TORCH_INTERNAL_ASSERT/TORCH_WARN",
        regex=r"AT_(ERROR|ASSERT|INDEX_ERROR|WARN)",
    ),
    RegexLint(
        error_code="include-complex",
        error_desc="Do not include the C complex number library, prefer the C++ version. See: https://git.io/Je8i2",
        regex=r'#include (<|")(complex\.h|ccomplex)(>|")',
    ),
]


def flatten_changed_lines(changed_lines):
    """Go from [(start, end), ...] to [line numbers]"""
    line_numbers = []
    for group in changed_lines:
        start, end = group
        line_numbers.extend(range(start, end))
    return line_numbers


def lint_line(filename, line, line_number):
    lint_errors = []
    for lint in LINTS:
        if re.search(lint.regex, line) is not None:
            lint_errors.append(
                LintError(
                    regex_lint=lint,
                    line_number=line_number,
                    # GitHub annotations don't do anything with columns,
                    # so don't bother setting it properly
                    column_number=0,
                    filename=filename,
                )
            )
    return lint_errors


def lint_file(filename, line_numbers):
    lint_errors = []
    with open(filename) as f:
        lines = f.readlines()
        for line_number in line_numbers:
            # -1 because lines are 1-indexed
            line = lines[line_number - 1]
            lint_errors.extend(lint_line(filename, line, line_number))
    return lint_errors


def parse_options():
    """Parses the command line options."""
    parser = argparse.ArgumentParser(description="Run Clang-Tidy (on your Git changes)")
    parser.add_argument(
        "-d",
        "--diff",
        help="Git revision to diff against to get changes",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_options()
    files = get_changed_files(options.diff, paths=["."])
    # filename => list of (start, end)
    line_filters = [get_changed_lines(options.diff, f) for f in files]

    # flatten it to filename => list of lines changed
    for line_filter in line_filters:
        line_filter["lines"] = flatten_changed_lines(line_filter["lines"])

    lint_errors = []
    for line_filter in line_filters:
        name = line_filter["name"]
        lines_changed = line_filter["lines"]
        lint_errors.extend(lint_file(name, lines_changed))

    for lint_error in lint_errors:
        # same output style as clang-tidy
        print(
            "{}:{}:{}: {} [{}]".format(
                lint_error.filename,
                lint_error.line_number,
                lint_error.column_number,
                lint_error.regex_lint.error_desc,
                lint_error.regex_lint.error_code,
            )
        )
