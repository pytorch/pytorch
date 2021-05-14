#!/usr/bin/env python3
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
import shutil
import subprocess
import sys
import tempfile

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from typing import Any, Dict, Iterable, List, Set, Union

Patterns = collections.namedtuple("Patterns", "positive, negative")


# NOTE: Clang-tidy cannot lint headers directly, because headers are not
# compiled -- translation units are, of which there is one per implementation
# (c/cc/cpp) file.
DEFAULT_FILE_PATTERN = re.compile(r".*\.c(c|pp)?")

# @@ -start,count +start,count @@
CHUNK_PATTERN = r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@"
CLANG_WARNING_PATTERN = re.compile(r"([^:]+):(\d+):\d+:\s+warning:.*\[([^\]]+)\]")


# Set from command line arguments in main().
VERBOSE = False


# Functions for correct handling of "ATen/native/cpu" mapping
# Sources in that folder are not built in place but first copied into build folder with `.[CPUARCH].cpp` suffixes
def map_filename(build_folder: str, fname: str) -> str:
    fname = os.path.relpath(fname)
    native_cpu_prefix = "aten/src/ATen/native/cpu/"
    build_cpu_prefix = os.path.join(build_folder, native_cpu_prefix, "")
    default_arch_suffix = ".DEFAULT.cpp"
    if fname.startswith(native_cpu_prefix) and fname.endswith(".cpp"):
        return f"{build_cpu_prefix}{fname[len(native_cpu_prefix):]}{default_arch_suffix}"
    if fname.startswith(build_cpu_prefix) and fname.endswith(default_arch_suffix):
        return f"{native_cpu_prefix}{fname[len(build_cpu_prefix):-len(default_arch_suffix)]}"
    return fname


def map_filenames(build_folder: str, fnames: Iterable[str]) -> List[str]:
    return [map_filename(build_folder, fname) for fname in fnames]


def run_shell_command(arguments: List[str]) -> str:
    """Executes a shell command."""
    if VERBOSE:
        print(" ".join(arguments))
    try:
        output = subprocess.check_output(arguments).decode().strip()
    except subprocess.CalledProcessError as error:
        error_output = error.output.decode().strip()
        raise RuntimeError(f"Error executing {' '.join(arguments)}: {error_output}")

    return output


def split_negative_from_positive_patterns(patterns: Iterable[str]) -> Patterns:
    """Separates negative patterns (that start with a dash) from positive patterns"""
    positive, negative = [], []
    for pattern in patterns:
        if pattern.startswith("-"):
            negative.append(pattern[1:])
        else:
            positive.append(pattern)

    return Patterns(positive, negative)


def get_file_patterns(globs: Iterable[str], regexes: Iterable[str]) -> Patterns:
    """Returns a list of compiled regex objects from globs and regex pattern strings."""
    # fnmatch.translate converts a glob into a regular expression.
    # https://docs.python.org/2/library/fnmatch.html#fnmatch.translate
    glob = split_negative_from_positive_patterns(globs)
    regexes_ = split_negative_from_positive_patterns(regexes)

    positive_regexes = regexes_.positive + [fnmatch.translate(g) for g in glob.positive]
    negative_regexes = regexes_.negative + [fnmatch.translate(g) for g in glob.negative]

    positive_patterns = [re.compile(regex) for regex in positive_regexes] or [
        DEFAULT_FILE_PATTERN
    ]
    negative_patterns = [re.compile(regex) for regex in negative_regexes]

    return Patterns(positive_patterns, negative_patterns)


def filter_files(files: Iterable[str], file_patterns: Patterns) -> Iterable[str]:
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


def get_changed_files(revision: str, paths: List[str]) -> List[str]:
    """Runs git diff to get the paths of all changed files."""
    # --diff-filter AMU gets us files that are (A)dded, (M)odified or (U)nmerged (in the working copy).
    # --name-only makes git diff return only the file paths, without any of the source changes.
    command = "git diff-index --diff-filter=AMU --ignore-all-space --name-only"
    output = run_shell_command(shlex.split(command) + [revision] + paths)
    return output.split("\n")


def get_all_files(paths: List[str]) -> List[str]:
    """Returns all files that are tracked by git in the given paths."""
    output = run_shell_command(["git", "ls-files"] + paths)
    return output.split("\n")


def get_changed_lines(revision: str, filename: str) -> Dict[str, Union[str, List[List[int]]]]:
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


def run_shell_commands_in_parallel(commands: Iterable[List[str]]) -> str:
    """runs all the commands in parallel with ninja, commands is a List[List[str]]"""
    build_entries = [build_template.format(i=i, cmd=' '.join([quote(s) for s in command]))
                     for i, command in enumerate(commands)]

    file_contents = ninja_template.format(build_rules='\n'.join(build_entries)).encode()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_contents)
        return run_shell_command(['ninja', '-f', f.name])


def run_clang_tidy(options: Any, line_filters: Any, files: Iterable[str]) -> str:
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
        commands = [list(command) + [map_filename(options.compile_commands_dir, f)] for f in files]
        output = run_shell_commands_in_parallel(commands)
    else:
        command += map_filenames(options.compile_commands_dir, files)
        if options.dry_run:
            command = [re.sub(r"^([{[].*[]}])$", r"'\1'", arg) for arg in command]
            return " ".join(command)

        output = run_shell_command(command)

    if not options.keep_going and "[clang-diagnostic-error]" in output:
        message = "Found clang-diagnostic-errors in clang-tidy output: {}"
        raise RuntimeError(message.format(output))

    return output


def extract_warnings(output: str, base_dir: str = ".") -> Dict[str, Dict[int, Set[str]]]:
    rc: Dict[str, Dict[int, Set[str]]] = {}
    for line in output.split("\n"):
        p = CLANG_WARNING_PATTERN.match(line)
        if p is None:
            continue
        if os.path.isabs(p.group(1)):
            path = os.path.abspath(p.group(1))
        else:
            path = os.path.abspath(os.path.join(base_dir, p.group(1)))
        line_no = int(p.group(2))
        warnings = set(p.group(3).split(","))
        if path not in rc:
            rc[path] = {}
        if line_no not in rc[path]:
            rc[path][line_no] = set()
        rc[path][line_no].update(warnings)
    return rc


def apply_nolint(fname: str, warnings: Dict[int, Set[str]]) -> None:
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()

    line_offset = -1  # As in .cpp files lines are numbered starting from 1
    for line_no in sorted(warnings.keys()):
        nolint_diagnostics = ','.join(warnings[line_no])
        line_no += line_offset
        indent = ' ' * (len(lines[line_no]) - len(lines[line_no].lstrip(' ')))
        lines.insert(line_no, f'{indent}// NOLINTNEXTLINE({nolint_diagnostics})\n')
        line_offset += 1

    with open(fname, mode="w") as f:
        f.write("".join(lines))


def parse_options() -> Any:
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
    parser.add_argument("-s", "--suppress-diagnostics", action="store_true",
                        help="Add NOLINT to suppress clang-tidy violations")
    parser.add_argument(
        "extra_args", nargs="*", help="Extra arguments to forward to clang-tidy"
    )
    return parser.parse_args()


def main() -> None:
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

    clang_tidy_output = run_clang_tidy(options, line_filters, files)
    if options.suppress_diagnostics:
        warnings = extract_warnings(clang_tidy_output, base_dir=options.compile_commands_dir)
        for fname in warnings.keys():
            mapped_fname = map_filename(options.compile_commands_dir, fname)
            print(f"Applying fixes to {mapped_fname}")
            apply_nolint(fname, warnings[fname])
            if os.path.relpath(fname) != mapped_fname:
                shutil.copyfile(fname, mapped_fname)

    pwd = os.getcwd() + "/"
    for line in clang_tidy_output.splitlines():
        if line.startswith(pwd):
            print(line[len(pwd):])

if __name__ == "__main__":
    main()
