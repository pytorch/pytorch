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
CPP_FILE_REGEX = re.compile(".*\\.(h|cpp|cc|c|hpp)$")
# @@ -start,count +start,count @@
CHUNK_PATTERN = r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@"


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


def get_changed_lines(filename, revision):
    """
    Given a filename and revision diff, return all the changed lines noted in the diff
    Returns a list of (start_line, end_line) tuples.
    """
    command = ["git", "diff-index", "--unified=0", revision, filename]
    output = subprocess.check_output(command).decode()
    changed_lines = []
    for chunk in re.finditer(CHUNK_PATTERN, output, re.MULTILINE):
        start = int(chunk.group(1))
        count = int(chunk.group(2) or 1)
        changed_lines.append((start, start + count))

    return changed_lines


def run_clang_format(filename, lines, in_place):
    args = ["clang-format", filename]
    line_args = ["-lines={}:{}".format(i[0], i[1]) for i in lines]
    args.extend(line_args)
    if in_place:
        args.append("-i")

    return subprocess.check_output(args).decode()


def get_clang_format_diff(filename, lines):
    """
    Return a diff of the changes that running clang-format would make (or None).
    """
    formatted_text = run_clang_format(filename, lines, in_place=False)
    with open(filename) as orig:
        orig_text = orig.read()
        if formatted_text != orig_text:
            orig_lines = orig_text.split("\n")
            formatted_lines = formatted_text.split("\n")
            return difflib.unified_diff(
                orig_lines, formatted_lines, "original", "formatted"
            )


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

    name_to_lines = {}
    for f in files_to_check:
        changed_lines = get_changed_lines(f, args.diff)
        if len(changed_lines) != 0:
            name_to_lines[f] = changed_lines

    if len(name_to_lines) == 0:
        return

    name_to_diff = {}
    for filename, lines in name_to_lines.items():
        diff = get_clang_format_diff(filename, lines)
        if diff is not None:
            name_to_diff[filename] = diff

    if args.accept_changes:
        # run clang-format on the necessary files
        for name, lines in name_to_lines.items():
            run_clang_format(name, lines, in_place=True)

        # add the changes so they will be committed
        args = ["git", "add"]
        args.extend(name_to_lines.keys())
        subprocess.check_output(args)
    else:
        if len(name_to_diff) == 0:
            return

        print("ERROR: Running clang-format created changes: ")
        for name, diff in name_to_diff.items():
            print("In " + name)
            for l in diff:
                print(l)
            print("\n")


if __name__ == "__main__":
    main()
