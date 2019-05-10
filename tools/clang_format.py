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
import os
import argparse
import difflib
import re


# Whitelist of directories to check. All files that in that directory
# (recursively) will be checked.
CLANG_FORMAT_WHITELIST = ["torch/csrc/jit/", "test/cpp/jit/"]

CPP_FILE_REGEX = re.compile("^.*\\.(h|cpp|cc|c|hpp)$")


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
    parser.add_argument(
        "--check-all",
        action="store_true",
        default=False,
        help="If true, check all whitelisted files instead of just working copy changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    return parser.parse_args()


def get_whitelisted_files():
    """
    Parse CLANG_FORMAT_WHITELIST and resolve all directories.
    Returns the set of whitelist cpp source files.
    """
    matches = []
    for dir in CLANG_FORMAT_WHITELIST:
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if CPP_FILE_REGEX.match(filename):
                    matches.append(os.path.join(root, filename))
    return set(matches)


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

    whitelisted_files = get_whitelisted_files()

    if args.check_all:
        files_to_check = whitelisted_files
    else:
        changed_files = get_changed_files(args.diff)
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

        # add the changes so they will be committed
        args = ["git", "add"]
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
