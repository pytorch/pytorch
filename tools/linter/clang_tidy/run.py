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


import collections
import fnmatch
import json
import os
import os.path
import re
import shutil
import subprocess
import sys
import asyncio
import shlex
import multiprocessing

from typing import Any, Dict, Iterable, List, Set, Tuple

Patterns = collections.namedtuple("Patterns", "positive, negative")


# NOTE: Clang-tidy cannot lint headers directly, because headers are not
# compiled -- translation units are, of which there is one per implementation
# (c/cc/cpp) file.
DEFAULT_FILE_PATTERN = re.compile(r"^.*\.c(c|pp)?$")

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
        return (
            f"{build_cpu_prefix}{fname[len(native_cpu_prefix):]}{default_arch_suffix}"
        )
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


def get_all_files(paths: List[str]) -> List[str]:
    """Returns all files that are tracked by git in the given paths."""
    output = run_shell_command(["git", "ls-files"] + paths)
    return output.split("\n")


def find_changed_lines(diff: str) -> Dict[str, List[Tuple[int, int]]]:
    # Delay import since this isn't required unless using the --diff-file
    # argument, which for local runs people don't care about
    try:
        import unidiff  # type: ignore[import]
    except ImportError as e:
        e.msg += ", run 'pip install unidiff'"  # type: ignore[attr-defined]
        raise e

    files = collections.defaultdict(list)

    for file in unidiff.PatchSet(diff):
        for hunk in file:
            start = hunk[0].target_line_no
            if start is None:
                start = 1
            end = int(hunk[-1].target_line_no or 0)
            if end == 0:
                continue

            files[file.path].append((start, end))

    return dict(files)


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

    async def run_command(cmd: List[str]) -> str:
        proc = await asyncio.create_subprocess_shell(
            " ".join(shlex.quote(x) for x in cmd),  # type: ignore[attr-defined]
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return f">>>\nstdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}\n<<<"

    async def gather_with_concurrency(n: int, tasks: List[Any]) -> Any:
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task: Any) -> Any:
            async with semaphore:
                return await task

        return await asyncio.gather(
            *(sem_task(task) for task in tasks), return_exceptions=True
        )

    async def helper() -> Any:
        coros = [run_command(cmd) for cmd in commands]
        return await gather_with_concurrency(multiprocessing.cpu_count(), coros)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(helper())
    return "\n".join(results)


def run_clang_tidy(
    options: Any, line_filters: List[Dict[str, Any]], files: Iterable[str]
) -> str:
    """Executes the actual clang-tidy command in the shell."""
    command = [options.clang_tidy_exe, "-p", options.compile_commands_dir]
    if not options.config_file and os.path.exists(".clang-tidy"):
        options.config_file = ".clang-tidy"
    if options.config_file:
        import yaml

        with open(options.config_file) as config:
            # Here we convert the YAML config file to a JSON blob.
            command += [
                "-config",
                json.dumps(yaml.load(config, Loader=yaml.SafeLoader)),
            ]
    if options.print_include_paths:
        command += ["--extra-arg", "-v"]
    if options.include_dir:
        for dir in options.include_dir:
            command += ["--extra-arg", f"-I{dir}"]

    command += options.extra_args

    if line_filters:
        command += ["-line-filter", json.dumps(line_filters)]

    if options.parallel:
        commands = [
            list(command) + [map_filename(options.compile_commands_dir, f)]
            for f in files
        ]
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


def extract_warnings(
    output: str, base_dir: str = "."
) -> Dict[str, Dict[int, Set[str]]]:
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
        nolint_diagnostics = ",".join(warnings[line_no])
        line_no += line_offset
        indent = " " * (len(lines[line_no]) - len(lines[line_no].lstrip(" ")))
        lines.insert(line_no, f"{indent}// NOLINTNEXTLINE({nolint_diagnostics})\n")
        line_offset += 1

    with open(fname, mode="w") as f:
        f.write("".join(lines))


def filter_from_diff(
    paths: List[str], diffs: List[str]
) -> Tuple[List[str], List[Dict[Any, Any]]]:
    files = []
    line_filters = []

    for diff in diffs:
        changed_files = find_changed_lines(diff)
        changed_files = {
            filename: v
            for filename, v in changed_files.items()
            if any(filename.startswith(path) for path in paths)
        }
        line_filters += [
            {"name": name, "lines": lines} for name, lines, in changed_files.items()
        ]
        files += list(changed_files.keys())

    return files, line_filters


def filter_from_diff_file(
    paths: List[str], filename: str
) -> Tuple[List[str], List[Dict[Any, Any]]]:
    with open(filename, "r") as f:
        diff = f.read()
    return filter_from_diff(paths, [diff])


def filter_default(paths: List[str]) -> Tuple[List[str], List[Dict[Any, Any]]]:
    return get_all_files(paths), []


def run(options: Any) -> int:
    # This flag is pervasive enough to set it globally. It makes the code
    # cleaner compared to threading it through every single function.
    global VERBOSE
    VERBOSE = options.verbose

    # Normalize the paths first.
    paths = [path.rstrip("/") for path in options.paths]

    if options.diff_file:
        files, line_filters = filter_from_diff_file(options.paths, options.diff_file)
    else:
        files, line_filters = filter_default(options.paths)

    file_patterns = get_file_patterns(options.glob, options.regex)
    files = list(filter_files(files, file_patterns))

    # clang-tidy error's when it does not get input files.
    if not files:
        print("No files detected.")
        sys.exit()

    clang_tidy_output = run_clang_tidy(options, line_filters, files)
    warnings = extract_warnings(
        clang_tidy_output, base_dir=options.compile_commands_dir
    )
    if options.suppress_diagnostics:
        warnings = extract_warnings(
            clang_tidy_output, base_dir=options.compile_commands_dir
        )
        for fname in warnings.keys():
            mapped_fname = map_filename(options.compile_commands_dir, fname)
            print(f"Applying fixes to {mapped_fname}")
            apply_nolint(fname, warnings[fname])
            if os.path.relpath(fname) != mapped_fname:
                shutil.copyfile(fname, mapped_fname)

    pwd = os.getcwd() + "/"
    if options.dry_run:
        print(clang_tidy_output)
    for line in clang_tidy_output.splitlines():
        if line.startswith(pwd):
            print(line[len(pwd) :])

    return len(warnings.keys())
