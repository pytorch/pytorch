"""
Linter driver script
====================

This script manages instantiating and running linters. Every linter extends the Linter base class (defined below).

Registering a new linter
------------------------

- Create a new module with an __init__.py file. The module name must be the
  linter name in snake_case.
- In __init__.py, add a class that extends the Linter base class. The class
  name must be the linter name in PascalCase
- Add the linter name (in kebab-case) to SUPPORTED_LINTERS
- Follow the instructions on extending the Linter base class below
"""


import argparse
import asyncio
import importlib
import os
import sys
from typing import List, Any
import click

# Modify sys.path so that linter modules can be found
# Here we are appending the path to the root of pytorch to sys.path
sys.path.insert(0, "")

from tools.linter.utils.filter_helpers import filter_files, filter_files_from_diff_file
from tools.linter.utils.linter import Linter
from tools.linter.utils import (
    CommandResult,
    kebab2snake,
    kebab2camel,
    Color,
    color,
    find_changed_files,
    Glob2RegexAction,
    indent,
)

# This list is used for quickly discovering linters The names provided here are
# not used when initializing the linter. Instead, they are declared in the
# corresponding class
# SUPPORTED_LINTERS = [
#     "shellcheck",
#     "flake8",
#     "mypy",
#     "clang-tidy",
#     "clang-format",
# ]

from .flake8 import Flake8
# from . import mypy
# from . import shellcheck
# from . import clang_tidy
# from . import clang_format
LINTERS = [
    Flake8,
]


@click.group()
def cli() -> None:
    """
    Lint PyTorch
    """
    # for linter in 
    pass


@cli.command()
def flake8():
    pass





def build_parser(linters) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run linters on PyTorch", add_help=False)

    # Setting default values here will prevent linters from overriding option values
    # For example, it doesn't make sense for a linter to override
    # --all set to true because this breaks the semantics of the
    # linter running on all filtered files by default
    parser.add_argument(
        "--all",
        help="Run linters without filtering for changed files",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--regex",
        action="append",
        help="Only lint files that match these regular expressions (from the start of the filename). "
        "If a pattern starts with a - the search is negated for that pattern.",
    )
    parser.add_argument(
        "-p", "--paths", nargs="+", help="Lint only the given paths (recursively)",
    )
    parser.add_argument(
        "-d",
        "--diff-file",
        default=None,
        help="Lint files filtered by a file containing a diff",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
        default=False,
    )

    subparsers = parser.add_subparsers()
    for linter in linters:
        subparser = subparsers.add_parser(
            linter.name, parents=[parser], help=f"Run {linter.name} on the codebase"
        )
        linter.build_parser(subparser)
        subparser.set_defaults(linter=linter)

    return parser


def merge_namespaces(a, b):
    a_dict = vars(a)
    b_dict = vars(b)
    c = a_dict.copy()
    for (k, v) in b_dict.items():
        if v is not None or k not in c:
            c[k] = v
    return argparse.Namespace(**c)


async def main() -> None:
    linters = [linter() for linter in LINTERS]
    base_parser = build_parser()
    for linter in linters:
        base_parser.add
    parser = build_parser(linters)
    options = parser.parse_args()

    if not options.all:
        options.paths = await find_changed_files()

    # Run linter
    if "linter" in options:
        linters = [options.linter]

    result = CommandResult(0, "", "")

    # We run the linters sequentially because the logging infrastructure for
    # linters hasn't been fully developed yet. If we run it concurrently, users
    # will see print statements from different linters interleaved.
    #
    # The semantics we need are:
    #   - If the linters are running concurrently, buffer all requests to print
    #     to console and flush at the end of the lint (to prevent interleaved
    #     outputs)
    #   - If a single linter is run, process the request to print on demand
    #
    # We can achieve these semantics if we have a logging infrastructure (like
    # for example using the Python logging module)
    for linter in linters:
        # We merge user-provided options into linter options
        # This allows the user to override linter-specified defaults
        linter_options = merge_namespaces(linter.options, options)

        files = await linter.filter_files(
            linter_options.paths, linter_options.glob, linter_options.regex
        )
        if linter_options.diff_file:
            files, line_filters = filter_files_from_diff_file(
                files, linter_options.diff_file
            )
        else:
            line_filters = []

        intermediate_result = await linter.run(files, line_filters, linter_options)
        if intermediate_result.failed():
            print(color(f"fail: {linter.name}", Color.red))
            print(indent(repr(intermediate_result), 4))
        else:
            print(color(f"pass: {linter.name}", Color.green))
        result += intermediate_result

    sys.exit(result.returncode)


if __name__ == "__main__":
    linters = [l() for l in LINTERS]
    for l in linters:
        l.init_cli(cli)
    cli()
    # asyncio.get_event_loop().run_until_complete(main())
