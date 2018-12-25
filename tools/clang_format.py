#!/usr/bin/env python
"""
A script that runs clang-format on changes detected via git. It will
report if running clang-format generated any changes.

In CI, the script considers it a failure if running clang-format makes a change.
In the pre-commit hook, the user is prompted to apply any clang-format changes.
Running tools/clang_format.py manually with no arguments should replicate the pre-commit hook behavior.

Only files that are in CLANG_FORMAT_WHITELIST are checked.
"""
import subprocess
import glob
import itertools
import os
import argparse
import difflib
import sys


# Whitelist of files to check. Takes a glob syntax. Does not support
# recursive globs ("**") because I am lazy and don't want to make that
# work with Python 2.
CLANG_FORMAT_WHITELIST = ["torch/csrc/jit/passes/alias_analysis*"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute clang-format on your working copy changes."
    )
    parser.add_argument(
        "-d",
        "--diff",
        default="HEAD",
        help="Git revision to diff against to get changes",
    )
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        default=False,
        help=(
            "If true, apply whatever changes clang-format creates. "
            "Otherwise, just print the changes and exit"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def get_whitelisted_files():
    """
    Parse CLANG_FORMAT_WHITELIST and resolve all globs.
    Returns the set of all whitelisted filenames.
    """
    paths = [glob.glob(entry) for entry in CLANG_FORMAT_WHITELIST]
    # flatten the files list
    paths = itertools.chain(*paths)
    # filter out directories
    filenames = filter(lambda path: os.path.isfile(path), paths)
    return set(filenames)


def get_changed_files(rev):
    """
    Get all changed files between the working tree and `rev`
    """
    changed_files = (
        subprocess.check_output(
            ["git", "diff-index", "--diff-filter=AMU", "--name-only", rev]
        )
        .decode()
        .split("\n")
    )
    return set(changed_files)


def get_diffs(files):
    """
    Run clang-format on all `files` and report if it changed anything.
    Returns a mapping of filename => diff generator
    """
    name_to_diffs = {}
    for f in files:
        formatted_text = subprocess.check_output(["clang-format", f]).decode()
        with open(f) as orig:
            orig_text = orig.read()
            if formatted_text != orig_text:
                orig_lines = orig_text.split("\n")
                formatted_lines = formatted_text.split("\n")
                diff = difflib.unified_diff(
                    orig_lines, formatted_lines, "original", "formatted"
                )
                name_to_diffs[f] = diff

    return name_to_diffs


def main():
    args = parse_args()

    changed_files = get_changed_files(args.diff)
    whitelisted_files = get_whitelisted_files()

    files_to_check = changed_files & whitelisted_files

    if args.verbose:
        print("Running clang-format on whitelisted files: ")
        for f in files_to_check:
            print(f)

    name_to_diffs = get_diffs(files_to_check)

    if len(name_to_diffs) == 0:
        return

    if args.accept_changes:
        # run clang-format on the necessary files
        args = ["clang-format", "-i"]
        args.extend(name_to_diffs.keys())
        subprocess.check_output(args)
    else:
        print("ERROR: Running clang-format created changes: ")
        for name, diff in name_to_diffs.items():
            print("In ", name)
            for line in diff:
                print(line)
            print("\n")


if __name__ == "__main__":
    main()
