#!/usr/bin/env python

import argparse
import json
import os.path
import re
import subprocess
import sys

DEFAULT_FILE_PATTERN = r".*\.[ch](pp)?"

# @@ -start,count +start,count @@
CHUNK_PATTERN = r"^@@\s+-\d+,\d+\s+\+(\d+)(?:,(\d+))?\s+@@"


def run_shell_command(arguments, process_name=None):
    """Executes a shell command."""
    assert len(arguments) > 0
    try:
        output = subprocess.check_output(arguments, stderr=subprocess.STDOUT)
    except OSError:
        _, e, _ = sys.exc_info()
        process_name = process_name or arguments[0]
        raise RuntimeError("Error executing {}: {}".format(process_name, e))
    else:
        return output.decode()


def transform_globs_into_regexes(globs):
    """Turns glob patterns into regular expressions."""
    return [glob.replace("*", ".*").replace("?", ".") for glob in globs]


def get_file_patterns(globs, regexes):
    """Returns a list of compiled regex objects from globs and regex pattern strings."""
    regexes += transform_globs_into_regexes(globs)
    if not regexes:
        regexes = [DEFAULT_FILE_PATTERN]
    return [re.compile(regex + "$") for regex in regexes]


def git_diff(args, verbose):
    """Executes a git diff command in the shell and returns its output."""
    # --no-pager gets us the plain output, without pagination.
    # --no-color removes color codes.
    command = ["git", "--no-pager", "diff", "--no-color"] + args
    if verbose:
        print(" ".join(command))
    return run_shell_command(command, process_name="git diff")


def filter_files(files, file_patterns):
    """Returns all files that match any of the patterns."""
    filtered = []
    for file in files:
        for pattern in file_patterns:
            if pattern.match(file):
                filtered.append(file)
    return filtered


def get_changed_files(revision, paths, verbose):
    """Runs git diff to get the paths of all changed files."""
    # --diff-filter AMU gets us files that are (A)dded, (M)odified or (U)nmerged (in the working copy).
    # --name-only makes git diff return only the file paths, without any of the source changes.
    args = ["--diff-filter", "AMU", "--ignore-all-space", "--name-only", revision]
    output = git_diff(args + paths, verbose)
    return output.split("\n")


def get_all_files(paths):
    """Yields all files in any of the given paths"""
    for path in paths:
        for root, _, files in os.walk(path):
            for file in files:
                yield os.path.join(root, file)


def get_changed_lines(revision, filename, verbose):
    """Runs git diff to get the line ranges of all file changes."""
    output = git_diff(["--unified=0", revision, filename], verbose)
    changed_lines = []
    for chunk in re.finditer(CHUNK_PATTERN, output, re.MULTILINE):
        start = int(chunk.group(1))
        count = int(chunk.group(2) or 1)
        changed_lines.append([start, start + count])

    return {"name": filename, "lines": changed_lines}


def run_clang_tidy(options, line_filters, files):
    """Executes the actual clang-tidy command in the shell."""
    command = [options.clang_tidy_exe, "-p", options.compile_commands_dir]
    if not options.config_file and os.path.exists(".clang-tidy"):
        options.config_file = ".clang-tidy"
    if options.config_file:
        import yaml

        with open(options.config_file) as config:
            # Here we convert the YAML config file to a JSON blob.
            command += ["-config", json.dumps(yaml.load(config))]
    if options.checks:
        command += ["-checks", options.checks]
    if line_filters:
        command += ["-line-filter", json.dumps(line_filters)]
    command += ["-{}".format(arg) for arg in options.extra_args]
    command += files

    if options.verbose:
        print(" ".join(command))
    if options.show_command_only:
        command = [re.sub(r"^([{[].*[]}])$", r"'\1'", arg) for arg in command]
        return " ".join(command)

    return run_shell_command(command)


def parse_options():
    parser = argparse.ArgumentParser(description="Run Clang-Tidy (on your Git changes)")
    parser.add_argument(
        "-c",
        "--clang-tidy-exe",
        default="clang-tidy",
        help="Path to clang-tidy executable",
    )
    parser.add_argument(
        "-e",
        "--extra-args",
        nargs="+",
        default=[],
        help="Extra arguments to forward to clang-tidy, without the hypen (e.g. -e 'header-filter=\"path\"')",
    )
    parser.add_argument(
        "-g",
        "--glob",
        nargs="+",
        default=[],
        help="File patterns as UNIX globs (support * and ?, not recursive **)",
    )
    parser.add_argument(
        "-x",
        "--regex",
        nargs="+",
        default=[],
        help="File patterns as regular expressions",
    )
    parser.add_argument(
        "-d",
        "--compile-commands-dir",
        default=".",
        help="Path to the folder containing compile_commands.json",
    )
    parser.add_argument("-r", "--revision", help="Git revision to get changes from")
    parser.add_argument(
        "-p", "--paths", nargs="+", default=["."], help="Lint only the given paths"
    )
    parser.add_argument(
        "-s",
        "--show-command-only",
        action="store_true",
        help="Only show the command to be executed, without running it",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--config-file",
        help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
    )
    parser.add_argument(
        "--checks", help="Appends checks to those from the config file (if any)"
    )
    return parser.parse_args()


def main():
    options = parse_options()
    if options.revision:
        files = get_changed_files(options.revision, options.paths, options.verbose)
    else:
        files = get_all_files(options.paths)
    file_patterns = get_file_patterns(options.glob, options.regex)
    files = filter_files(files, file_patterns)

    # clang-tidy error's when it does not get input files.
    if not files:
        print("No files detected.")
        sys.exit()

    line_filters = []
    if options.revision:
        for filename in files:
            changed_lines = get_changed_lines(
                options.revision, filename, options.verbose
            )
            line_filters.append(changed_lines)

    print(run_clang_tidy(options, line_filters, files))


if __name__ == "__main__":
    main()
