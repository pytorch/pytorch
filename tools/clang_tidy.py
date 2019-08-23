#!/usr/bin/env python
"""
A driver script to run clang-tidy on changes detected via git.

By default, clang-tidy runs on all files you point it at. This means that even
if you changed only parts of that file, you will get warnings for the whole
file. This script has the ability to ask git for the exact lines that have
changed since a particular git revision, and makes clang-tidy only lint those.
This makes it much less overhead to integrate in CI and much more relevant to
developers. This git-enabled mode is optional, and full scans of a directory
tree are also possible. In both cases, the script allows filtering files via
glob or regular expressions.
"""

from __future__ import print_function

import argparse
import collections
import fnmatch
import json
import os.path
import re
import shlex
import subprocess
import sys
import tempfile
import itertools
from xml.sax.saxutils import escape

try:
    from shlex import quote
except ImportError:
    from pipes import quote

Patterns = collections.namedtuple("Patterns", "positive, negative")
ErrorDescription = collections.namedtuple(
    'ErrorDescription', 'file line column error error_identifier description')


# Adapted from https://github.com/PSPDFKit-labs/clang-tidy-to-junit. License elsewhere.
class ClangTidyConverter:
    # All the errors encountered.
    errors = []

    # Parses the error.
    # Group 1: file path
    # Group 2: line
    # Group 3: column
    # Group 4: error message
    # Group 5: error identifier
    error_regex = re.compile(
        r"^([\w\/\.\-\ ]+):(\d+):(\d+): (.+) (\[[\w\-,\.]+\])$")

    # This identifies the main error line (it has a [the-warning-type] at the end)
    # We only create a new error when we encounter one of those.
    main_error_identifier = re.compile(r'\[[\w\-,\.]+\]$')

    def __init__(self, basename, input_str):
        self.basename = basename
        input_lines = input_str.split('\n')
        # Collect all lines related to one error.
        current_error = []
        for line in input_lines:
            # If the line starts with a `/`, it is a line about a file.
            if line[0] == '/':
                # Look if it is the start of a error
                if self.main_error_identifier.search(line, re.M):
                    # If so, process any `current_error` we might have
                    self.process_error(current_error)
                    # Initialize `current_error` with the first line of the error.
                    current_error = [line]
                else:
                    # Otherwise, append the line to the error.
                    current_error.append(line)
            elif len(current_error) > 0:
                # If the line didn't start with a `/` and we have a `current_error`, we simply append
                # the line as additional information.
                current_error.append(line)
            else:
                pass

        # If we still have any current_error after we read all the lines,
        # process it.
        if len(current_error) > 0:
            self.process_error(current_error)

    def print_junit_file(self, output_file):
        # Write the header.
        output_file.write("""<?xml version="1.0" encoding="UTF-8" ?>
  <testsuites id="1" name="Clang-Tidy" tests="{error_count}" errors="{error_count}" failures="0" time="0">""".format(error_count=len(self.errors)))

        sorted_errors = sorted(self.errors, key=lambda x: x.file)

        # Iterate through the errors, grouped by file.
        for file, errorIterator in itertools.groupby(sorted_errors, key=lambda x: x.file):
            errors = list(errorIterator)
            error_count = len(errors)

            # Each file gets a test-suite
            output_file.write("""\n    <testsuite errors="{error_count}" name="{file}" tests="{error_count}" failures="0" time="0">\n"""
                              .format(error_count=error_count, file=file))
            for error in errors:
                # Write each error as a test case.
                output_file.write("""
        <testcase id="{id}" name="{id}" time="0">
            <failure message="{message}">
{htmldata}
            </failure>
        </testcase>""".format(id="[{}/{}] {}".format(error.line, error.column, error.error_identifier), message=escape(error.error),
                              htmldata=escape(error.description)))
            output_file.write("\n    </testsuite>\n")
        output_file.write("</testsuites>\n")

    def process_error(self, error_array):
        if len(error_array) == 0:
            return

        result = self.error_regex.match(error_array[0])
        if result is None:
            logging.warning(
                'Could not match error_array to regex: %s', error_array)
            return

        # We remove the `basename` from the `file_path` to make prettier filenames in the JUnit file.
        file_path = result.group(1).replace(self.basename, "")
        error = ErrorDescription(file_path, int(result.group(2)), int(
            result.group(3)), result.group(4), result.group(5), "\n".join(error_array[1:]))
        self.errors.append(error)


# NOTE: Clang-tidy cannot lint headers directly, because headers are not
# compiled -- translation units are, of which there is one per implementation
# (c/cc/cpp) file.
DEFAULT_FILE_PATTERN = re.compile(r".*\.c(c|pp)?")

# @@ -start,count +start,count @@
CHUNK_PATTERN = r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@"


# Set from command line arguments in main().
VERBOSE = False


def run_shell_command(arguments):
    """Executes a shell command."""
    if VERBOSE:
        print(" ".join(arguments))
    try:
        output = subprocess.check_output(arguments).decode().strip()
    except subprocess.CalledProcessError:
        _, error, _ = sys.exc_info()
        error_output = error.output.decode().strip()
        raise RuntimeError("Error executing {}: {}".format(" ".join(arguments), error_output))

    return output


def split_negative_from_positive_patterns(patterns):
    """Separates negative patterns (that start with a dash) from positive patterns"""
    positive, negative = [], []
    for pattern in patterns:
        if pattern.startswith("-"):
            negative.append(pattern[1:])
        else:
            positive.append(pattern)

    return Patterns(positive, negative)


def get_file_patterns(globs, regexes):
    """Returns a list of compiled regex objects from globs and regex pattern strings."""
    # fnmatch.translate converts a glob into a regular expression.
    # https://docs.python.org/2/library/fnmatch.html#fnmatch.translate
    glob = split_negative_from_positive_patterns(globs)
    regexes = split_negative_from_positive_patterns(regexes)

    positive_regexes = regexes.positive + [fnmatch.translate(g) for g in glob.positive]
    negative_regexes = regexes.negative + [fnmatch.translate(g) for g in glob.negative]

    positive_patterns = [re.compile(regex) for regex in positive_regexes] or [
        DEFAULT_FILE_PATTERN
    ]
    negative_patterns = [re.compile(regex) for regex in negative_regexes]

    return Patterns(positive_patterns, negative_patterns)


def filter_files(files, file_patterns):
    """Returns all files that match any of the patterns."""
    if VERBOSE:
        print("Filtering with these file patterns: {}".format(file_patterns))
    for file in files:
        if not any(n.match(file) for n in file_patterns.negative):
            if any(p.match(file) for p in file_patterns.positive):
                yield file
                continue
        if VERBOSE:
            print("{} ommitted due to file filters".format(file))


def get_changed_files(revision, paths):
    """Runs git diff to get the paths of all changed files."""
    # --diff-filter AMU gets us files that are (A)dded, (M)odified or (U)nmerged (in the working copy).
    # --name-only makes git diff return only the file paths, without any of the source changes.
    command = "git diff-index --diff-filter=AMU --ignore-all-space --name-only"
    output = run_shell_command(shlex.split(command) + [revision] + paths)
    return output.split("\n")


def get_all_files(paths):
    """Returns all files that are tracked by git in the given paths."""
    output = run_shell_command(["git", "ls-files"] + paths)
    return output.split("\n")


def get_changed_lines(revision, filename):
    """Runs git diff to get the line ranges of all file changes."""
    command = shlex.split("git diff-index --unified=0") + [revision, filename]
    output = run_shell_command(command)
    changed_lines = []
    for chunk in re.finditer(CHUNK_PATTERN, output, re.MULTILINE):
        start = int(chunk.group(1))
        count = int(chunk.group(2) or 1)
        changed_lines.append([start, start + count])

    return {"name": filename, "lines": changed_lines}

ninja_template = """
rule do_cmd
  command = $cmd
  description = Running clang-tidy

{build_rules}
"""

build_template = """
build {i}: do_cmd
  cmd = {cmd}
"""


def run_shell_commands_in_parallel(commands):
    """runs all the commands in parallel with ninja, commands is a List[List[str]]"""
    build_entries = [build_template.format(i=i, cmd=' '.join([quote(s) for s in command]))
                     for i, command in enumerate(commands)]

    file_contents = ninja_template.format(build_rules='\n'.join(build_entries)).encode()
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        f.write(file_contents)
        f.close()
        return run_shell_command(['ninja', '-f', f.name])
    finally:
        os.unlink(f.name)


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
    command += options.extra_args

    if line_filters:
        command += ["-line-filter", json.dumps(line_filters)]

    if options.parallel:
        commands = [list(command) + [f] for f in files]
        output = run_shell_commands_in_parallel(commands)
    else:
        command += files
        if options.dry_run:
            command = [re.sub(r"^([{[].*[]}])$", r"'\1'", arg) for arg in command]
            return " ".join(command)

        output = run_shell_command(command)

    if not options.keep_going and "[clang-diagnostic-error]" in output:
        message = "Found clang-diagnostic-errors in clang-tidy output: {}"
        raise RuntimeError(message.format(output))

    return output


def parse_options():
    """Parses the command line options."""
    parser = argparse.ArgumentParser(description="Run Clang-Tidy (on your Git changes)")
    parser.add_argument(
        "-e",
        "--clang-tidy-exe",
        default="clang-tidy",
        help="Path to clang-tidy executable",
    )
    parser.add_argument(
        "-g",
        "--glob",
        action="append",
        default=[],
        help="Only lint files that match these glob patterns "
        "(see documentation for `fnmatch` for supported syntax)."
        "If a pattern starts with a - the search is negated for that pattern.",
    )
    parser.add_argument(
        "-x",
        "--regex",
        action="append",
        default=[],
        help="Only lint files that match these regular expressions (from the start of the filename). "
        "If a pattern starts with a - the search is negated for that pattern.",
    )
    parser.add_argument(
        "-c",
        "--compile-commands-dir",
        default="build",
        help="Path to the folder containing compile_commands.json",
    )
    parser.add_argument(
        "-d", "--diff", help="Git revision to diff against to get changes"
    )
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        default=["."],
        help="Lint only the given paths (recursively)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Only show the command to be executed, without running it",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--config-file",
        help="Path to a clang-tidy config file. Defaults to '.clang-tidy'.",
    )
    parser.add_argument(
        "-k",
        "--keep-going",
        action="store_true",
        help="Don't error on compiler errors (clang-diagnostic-error)",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        action="store_true",
        help="Run clang tidy in parallel per-file (requires ninja to be installed).",
    )
    parser.add_argument(
        "--junit",
        help="Output results in junit format to file",
    )
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to forward to clang-tidy"
    )
    return parser.parse_args()


def main():
    options = parse_options()

    # This flag is pervasive enough to set it globally. It makes the code
    # cleaner compared to threading it through every single function.
    global VERBOSE
    VERBOSE = options.verbose

    # Normalize the paths first.
    paths = [path.rstrip("/") for path in options.paths]
    if options.diff:
        files = get_changed_files(options.diff, paths)
    else:
        files = get_all_files(paths)
    file_patterns = get_file_patterns(options.glob, options.regex)
    files = list(filter_files(files, file_patterns))

    # clang-tidy error's when it does not get input files.
    if not files:
        print("No files detected.")
        sys.exit()

    line_filters = []
    if options.diff:
        line_filters = [get_changed_lines(options.diff, f) for f in files]

    raw_output = run_clang_tidy(options, line_filters, files)
    output = ClangTidyConverter("", raw_output)
    if len(output.errors) > 0:
        if options.junit:
            output_file = open(options.junit, "w")
            output.print_junit_file(output_file)

        print(raw_output)
        exit(1)


if __name__ == "__main__":
    main()
