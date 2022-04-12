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



import argparse
import collections
import fnmatch
import json
import os
import os.path
import re
import shlex
import subprocess
import sys
import tempfile

try:
    from shlex import quote
except ImportError:
    from pipes import quote

Patterns = collections.namedtuple("Patterns", "positive, negative")


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
            print("{} omitted due to file filters".format(file))


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
        # If count == 0, a chunk was removed and can be ignored.
        if count == 0:
            continue
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
            command += ["-config", json.dumps(yaml.load(config, Loader=yaml.SafeLoader))]
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

    pwd = os.getcwd() + "/"
    clang_tidy_output = run_clang_tidy(options, line_filters, files)
    formatted_output = []

    for line in clang_tidy_output.splitlines():
        if line.startswith(pwd):
            print(line[len(pwd):])

if __name__ == "__main__":
    main()
