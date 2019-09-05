#!/usr/bin/env python
"""
A driver script to run clang-format on changes detected via git.

By default, it will report if running clang-format against lines that
were changed in the current index generated any changes. If so, it
exits with non-zero status.

In CI, the script considers it a failure if running clang-format makes
a change. In the pre-commit hook, the user is prompted to apply any
clang-format changes. Running tools/clang_format.py manually with no
arguments should replicate the pre-commit hook behavior.

Only files that are in CLANG_FORMAT_WHITELIST are checked.
"""

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys


# Whitelist of directories to check. All files that in that directory
# (recursively) will be checked.
CLANG_FORMAT_WHITELIST = [
    "torch/csrc/distributed/",
    "torch/lib/c10d/",
]

DEFAULT_FILE_PATTERN = re.compile(".*\\.(h|cpp|cc|c|cu|hpp)$")

# @@ -start,count +start,count @@
CHUNK_PATTERN = r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@"

# Set from command line arguments in main().
VERBOSE = False


def run_shell_command(arguments):
    """Executes a shell command."""
    if VERBOSE:
        print(" ".join(arguments))
    try:
        output = subprocess.check_output(arguments).decode()
    except subprocess.CalledProcessError:
        _, error, _ = sys.exc_info()
        error_output = error.output.decode().strip()
        raise RuntimeError("Error executing {}: {}".format(" ".join(arguments), error_output))

    return output


def get_whitelisted_files():
    """
    Parse CLANG_FORMAT_WHITELIST and resolve all directories.
    Returns the set of whitelist cpp source files.
    """
    matches = []
    for dir in CLANG_FORMAT_WHITELIST:
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if DEFAULT_FILE_PATTERN.match(filename):
                    matches.append(os.path.join(root, filename))
    return set(matches)


def get_changed_files(revision):
    """
    Get all changed files between the working tree and `revision`
    """
    command = "git diff-index --diff-filter=AMU --name-only"
    output = run_shell_command(shlex.split(command) + [revision])
    return set(output.split("\n"))


def get_changed_lines(revision, filename):
    """Runs git diff to get the line ranges of all file changes."""
    command = shlex.split("git diff-index --unified=0") + [revision, filename]
    output = run_shell_command(command)
    changed_lines = []
    for chunk in re.finditer(CHUNK_PATTERN, output, re.MULTILINE):
        start = int(chunk.group(1))
        count = int(chunk.group(2) or 1)
        # If count == 0, a chunk was removed and can be ignored.
        if count == 0:
            continue
        changed_lines.append([start, start + count])

    return changed_lines


def run_clang_format(filename, lines, in_place):
    command = ["clang-format", filename]
    command.extend(["-lines={}:{}".format(i[0], i[1]) for i in lines])
    if in_place:
        command.append("-i")
    return run_shell_command(command)


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


def main():
    args = parse_args()

    # This flag is pervasive enough to set it globally. It makes the code
    # cleaner compared to threading it through every single function.
    global VERBOSE
    VERBOSE = args.verbose

    whitelisted_files = get_whitelisted_files()
    if args.check_all:
        files_to_check = sorted(whitelisted_files)
    else:
        changed_files = get_changed_files(args.diff)
        files_to_check = sorted(changed_files & whitelisted_files)

    if VERBOSE:
        print("Running clang-format on whitelisted files: ")
        for f in files_to_check:
            print(f)

    # Build map with files to line ranges.
    name_to_lines = {}
    if args.check_all:
        # Include all files, with empty line range.
        for f in files_to_check:
            name_to_lines[f] = []
    else:
        # Include only changed files, with changed line ranges.
        for f in files_to_check:
            changed_lines = get_changed_lines(args.diff, f)
            if len(changed_lines) != 0:
                name_to_lines[f] = changed_lines

    # If no files in the whitelist were changed, exit early.
    if len(name_to_lines) == 0:
        print("No files detected.")
        sys.exit()

    # Build map with files to clang-format diff.
    name_to_diff = {}
    for filename, lines in name_to_lines.items():
        diff = get_clang_format_diff(filename, lines)
        if diff is not None:
            name_to_diff[filename] = diff

    # If no files produced a clang-format diff, exit early.
    if len(name_to_diff) == 0:
        sys.exit()

    if args.accept_changes:
        # Run clang-format on the necessary files.
        for name, lines in name_to_lines.items():
            run_clang_format(name, lines, in_place=True)

        # Add the changes so they will be committed.
        args = ["git", "add"]
        args.extend(name_to_lines.keys())
        subprocess.check_output(args)
    else:
        print("ERROR: Running clang-format created changes: ")
        for name, diff in name_to_diff.items():
            print("In " + name)
            for l in diff:
                print(l)
            print("\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
