#!/usr/bin/env python3
"""
A script that runs clang-format on all C/C++ files in CLANG_FORMAT_ALLOWLIST. There is
also a diff mode which simply checks if clang-format would make any changes, which is useful for
CI purposes.

If clang-format is not available, the script also downloads a platform-appropriate binary from
and S3 bucket and verifies it against a precommited set of blessed binary hashes.
"""
import argparse
import asyncio
import re
import os
import sys
from clang_format_utils import get_and_check_clang_format, CLANG_FORMAT_PATH

# Allowlist of directories to check. All files that in that directory
# (recursively) will be checked.
# If you edit this, please edit the allowlist in clang_format_ci.sh as well.
CLANG_FORMAT_ALLOWLIST = ["torch/csrc/jit/", "test/cpp/jit/", "test/cpp/tensorexpr/"]

# Only files with names matching this regex will be formatted.
CPP_FILE_REGEX = re.compile(".*\\.(h|cpp|cc|c|hpp)$")


def get_allowlisted_files():
    """
    Parse CLANG_FORMAT_ALLOWLIST and resolve all directories.
    Returns the set of allowlist cpp source files.
    """
    matches = []
    for dir in CLANG_FORMAT_ALLOWLIST:
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if CPP_FILE_REGEX.match(filename):
                    matches.append(os.path.join(root, filename))
    return set(matches)


async def run_clang_format_on_file(filename, semaphore, verbose=False):
    """
    Run clang-format on the provided file.
    """
    # -style=file picks up the closest .clang-format, -i formats the files inplace.
    cmd = "{} -style=file -i {}".format(CLANG_FORMAT_PATH, filename)
    async with semaphore:
        proc = await asyncio.create_subprocess_shell(cmd)
        _ = await proc.wait()
    if verbose:
        print("Formatted {}".format(filename))


async def file_clang_formatted_correctly(filename, semaphore, verbose=False):
    """
    Checks if a file is formatted correctly and returns True if so.
    """
    ok = True
    # -style=file picks up the closest .clang-format
    cmd = "{} -style=file {}".format(CLANG_FORMAT_PATH, filename)

    async with semaphore:
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
        # Read back the formatted file.
        stdout, _ = await proc.communicate()

    formatted_contents = stdout.decode()
    # Compare the formatted file to the original file.
    with open(filename) as orig:
        orig_contents = orig.read()
        if formatted_contents != orig_contents:
            ok = False
            if verbose:
                print("{} is not formatted correctly".format(filename))

    return ok


async def run_clang_format(max_processes, diff=False, verbose=False):
    """
    Run clang-format to all files in CLANG_FORMAT_ALLOWLIST that match CPP_FILE_REGEX.
    """
    # Check to make sure the clang-format binary exists.
    if not os.path.exists(CLANG_FORMAT_PATH):
        print("clang-format binary not found")
        return False

    # Gather command-line options for clang-format.
    args = [CLANG_FORMAT_PATH, "-style=file"]

    if not diff:
        args.append("-i")

    ok = True

    # Semaphore to bound the number of subprocesses that can be created at once to format files.
    semaphore = asyncio.Semaphore(max_processes)

    # Format files in parallel.
    if diff:
        for f in asyncio.as_completed([file_clang_formatted_correctly(f, semaphore, verbose) for f in get_allowlisted_files()]):
            ok &= await f

        if ok:
            print("All files formatted correctly")
        else:
            print("Some files not formatted correctly")
    else:
        await asyncio.gather(*[run_clang_format_on_file(f, semaphore, verbose) for f in get_allowlisted_files()])

    return ok

def parse_args(args):
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute clang-format on your working copy changes."
    )
    parser.add_argument(
        "-d",
        "--diff",
        action="store_true",
        default=False,
        help="Determine whether running clang-format would produce changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--max-processes", type=int, default=50,
                        help="Maximum number of subprocesses to create to format files in parallel")
    return parser.parse_args(args)


def main(args):
    # Parse arguments.
    options = parse_args(args)
    # Get clang-format and make sure it is the right binary and it is in the right place.
    ok = get_and_check_clang_format(options.verbose)
    # Invoke clang-format on all files in the directories in the allowlist.
    if ok:
        loop = asyncio.get_event_loop()
        ok = loop.run_until_complete(run_clang_format(options.max_processes, options.diff, options.verbose))

    # We have to invert because False -> 0, which is the code to be returned if everything is okay.
    return not ok


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
