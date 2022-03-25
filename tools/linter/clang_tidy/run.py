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
CLANG_WARNING_PATTERN = re.compile(
    r"([^:]+):(\d+):\d+:\s+(warning|error):.*\[([^\]]+)\]"
)
# Set from command line arguments in main().
VERBOSE = False
QUIET = False


def log(*args: Any, **kwargs: Any) -> None:
    if not QUIET:
        print(*args, **kwargs)


class CommandResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout.strip()
        self.stderr = stderr.strip()

    def failed(self) -> bool:
        return self.returncode != 0

    def __add__(self, other: "CommandResult") -> "CommandResult":
        return CommandResult(
            self.returncode + other.returncode,
            f"{self.stdout}\n{other.stdout}",
            f"{self.stderr}\n{other.stderr}",
        )

    def __str__(self) -> str:
        return f"{self.stdout}"

    def __repr__(self) -> str:
        return (
            f"returncode: {self.returncode}\n"
            + f"stdout: {self.stdout}\n"
            + f"stderr: {self.stderr}"
        )


class ProgressMeter:
    def __init__(
        self, num_items: int, start_msg: str = "", disable_progress_bar: bool = False
    ) -> None:
        self.num_items = num_items
        self.num_processed = 0
        self.width = 80
        self.disable_progress_bar = disable_progress_bar

        # helper escape sequences
        self._clear_to_end = "\x1b[2K"
        self._move_to_previous_line = "\x1b[F"
        self._move_to_start_of_line = "\r"
        self._move_to_next_line = "\n"

        if self.disable_progress_bar:
            log(start_msg)
        else:
            self._write(
                start_msg
                + self._move_to_next_line
                + "[>"
                + (self.width * " ")
                + "]"
                + self._move_to_start_of_line
            )
            self._flush()

    def _write(self, s: str) -> None:
        sys.stderr.write(s)

    def _flush(self) -> None:
        sys.stderr.flush()

    def update(self, msg: str) -> None:
        if self.disable_progress_bar:
            return

        # Once we've processed all items, clear the progress bar
        if self.num_processed == self.num_items - 1:
            self._write(self._clear_to_end)
            return

        # NOP if we've already processed all items
        if self.num_processed > self.num_items:
            return

        self.num_processed += 1

        self._write(
            self._move_to_previous_line
            + self._clear_to_end
            + msg
            + self._move_to_next_line
        )

        progress = int((self.num_processed / self.num_items) * self.width)
        padding = self.width - progress
        self._write(
            self._move_to_start_of_line
            + self._clear_to_end
            + f"({self.num_processed} of {self.num_items}) "
            + f"[{progress*'='}>{padding*' '}]"
            + self._move_to_start_of_line
        )
        self._flush()

    def print(self, msg: str) -> None:
        if QUIET:
            return
        elif self.disable_progress_bar:
            print(msg)
        else:
            self._write(
                self._clear_to_end
                + self._move_to_previous_line
                + self._clear_to_end
                + msg
                + self._move_to_next_line
                + self._move_to_next_line
            )
            self._flush()


class ClangTidyWarning:
    def __init__(self, name: str, occurrences: List[Tuple[str, int]]):
        self.name = name
        self.occurrences = occurrences

    def __str__(self) -> str:
        base = f"[{self.name}] occurred {len(self.occurrences)} times\n"
        for occ in self.occurrences:
            base += f"    {occ[0]}:{occ[1]}\n"
        return base


async def run_shell_command(
    cmd: List[str], on_completed: Any = None, *args: Any
) -> CommandResult:
    """Executes a shell command and runs an optional callback when complete"""
    if VERBOSE:
        log("Running: ", " ".join(cmd))

    proc = await asyncio.create_subprocess_shell(
        " ".join(shlex.quote(x) for x in cmd),  # type: ignore[attr-defined]
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output = await proc.communicate()
    result = CommandResult(
        returncode=proc.returncode if proc.returncode is not None else -1,
        stdout=output[0].decode("utf-8").strip(),
        stderr=output[1].decode("utf-8").strip(),
    )

    if on_completed:
        on_completed(result, *args)

    return result


async def _run_clang_tidy_in_parallel(
    commands: List[Tuple[List[str], str]], disable_progress_bar: bool
) -> CommandResult:
    progress_meter = ProgressMeter(
        len(commands),
        f"Processing {len(commands)} clang-tidy jobs",
        disable_progress_bar=disable_progress_bar,
    )

    async def gather_with_concurrency(n: int, tasks: List[Any]) -> Any:
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task: Any) -> Any:
            async with semaphore:
                return await task

        return await asyncio.gather(
            *(sem_task(task) for task in tasks), return_exceptions=True
        )

    async def helper() -> Any:
        def on_completed(result: CommandResult, filename: str) -> None:
            if result.failed():
                msg = str(result) if not VERBOSE else repr(result)
                progress_meter.print(msg)
            progress_meter.update(f"Processed {filename}")

        coros = [
            run_shell_command(cmd, on_completed, filename)
            for (cmd, filename) in commands
        ]
        return await gather_with_concurrency(multiprocessing.cpu_count(), coros)

    results = await helper()
    return sum(results, CommandResult(0, "", ""))


async def _run_clang_tidy(
    options: Any, line_filters: List[Dict[str, Any]], files: Iterable[str]
) -> CommandResult:
    """Executes the actual clang-tidy command in the shell."""

    base = [options.clang_tidy_exe]

    # Apply common options
    base += ["-p", options.compile_commands_dir]
    if not options.config_file and os.path.exists(".clang-tidy"):
        options.config_file = ".clang-tidy"
    if options.config_file:
        import yaml

        with open(options.config_file) as config:
            # Here we convert the YAML config file to a JSON blob.
            base += [
                "-config",
                json.dumps(yaml.load(config, Loader=yaml.SafeLoader)),
            ]
    if options.print_include_paths:
        base += ["--extra-arg", "-v"]
    if options.include_dir:
        for dir in options.include_dir:
            base += ["--extra-arg", f"-I{dir}"]
    base += options.extra_args
    if line_filters:
        base += ["-line-filter", json.dumps(line_filters)]

    # Apply per-file options
    commands = []
    for f in files:
        command = list(base) + [map_filename(options.compile_commands_dir, f)]
        commands.append((command, f))

    if options.dry_run:
        return CommandResult(0, str([c for c, _ in commands]), "")

    return await _run_clang_tidy_in_parallel(commands, options.disable_progress_bar)


def extract_warnings(
    output: str, base_dir: str = "."
) -> Tuple[Dict[str, Dict[int, Set[str]]], List[ClangTidyWarning]]:
    warn2occ: Dict[str, List[Tuple[str, int]]] = {}
    fixes: Dict[str, Dict[int, Set[str]]] = {}
    for line in output.splitlines():
        p = CLANG_WARNING_PATTERN.match(line)
        if p is None:
            continue
        if os.path.isabs(p.group(1)):
            path = os.path.abspath(p.group(1))
        else:
            path = os.path.abspath(os.path.join(base_dir, p.group(1)))
        line_no = int(p.group(2))

        # Filter out any options (which start with '-')
        warning_names = set([w for w in p.group(4).split(",") if not w.startswith("-")])

        for name in warning_names:
            if name not in warn2occ:
                warn2occ[name] = []
            warn2occ[name].append((path, line_no))

        if path not in fixes:
            fixes[path] = {}
        if line_no not in fixes[path]:
            fixes[path][line_no] = set()
        fixes[path][line_no].update(warning_names)

    warnings = [ClangTidyWarning(name, sorted(occ)) for name, occ in warn2occ.items()]

    return fixes, warnings


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
        log("Filtering with these file patterns: {}".format(file_patterns))
    for file in files:
        if not any(n.match(file) for n in file_patterns.negative):
            if any(p.match(file) for p in file_patterns.positive):
                yield file
                continue
        if VERBOSE:
            log(f"{file} omitted due to file filters")


async def get_all_files(paths: List[str]) -> List[str]:
    """Returns all files that are tracked by git in the given paths."""
    output = await run_shell_command(["git", "ls-files"] + paths)
    return str(output).strip().splitlines()


def find_changed_lines(diff: str) -> Dict[str, List[Tuple[int, int]]]:
    # Delay import since this isn't required unless using the --diff-file
    # argument, which for local runs people don't care about
    try:
        import unidiff  # type: ignore[import]
    except ImportError as e:
        e.msg += ", run 'pip install unidiff'"  # type: ignore[attr-defined]
        raise e

    files: Any = collections.defaultdict(list)

    for file in unidiff.PatchSet(diff):
        for hunk in file:
            added_line_nos = [line.target_line_no for line in hunk if line.is_added]

            if len(added_line_nos) == 0:
                continue

            # Convert list of line numbers to ranges
            # Eg: [1, 2, 3, 12, 13, 14, 15] becomes [[1,3], [12, 15]]
            i = 1
            ranges = [[added_line_nos[0], added_line_nos[0]]]
            while i < len(added_line_nos):
                if added_line_nos[i] != added_line_nos[i - 1] + 1:
                    ranges[-1][1] = added_line_nos[i - 1]
                    ranges.append([added_line_nos[i], added_line_nos[i]])
                i += 1
            ranges[-1][1] = added_line_nos[-1]

            files[file.path] += ranges

    return dict(files)


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


async def filter_default(paths: List[str]) -> Tuple[List[str], List[Dict[Any, Any]]]:
    return await get_all_files(paths), []


async def _run(options: Any) -> Tuple[CommandResult, List[ClangTidyWarning]]:
    # These flags are pervasive enough to set it globally. It makes the code
    # cleaner compared to threading it through every single function.
    global VERBOSE
    global QUIET
    VERBOSE = options.verbose
    QUIET = options.quiet

    # Normalize the paths first
    paths = [path.rstrip("/") for path in options.paths]

    # Filter files
    if options.diff_file:
        files, line_filters = filter_from_diff_file(options.paths, options.diff_file)
    else:
        files, line_filters = await filter_default(options.paths)

    file_patterns = get_file_patterns(options.glob, options.regex)
    files = list(filter_files(files, file_patterns))

    print("<<<>>> fileslist")
    print(files)
    # clang-tidy errors when it does not get input files.
    if not files:
        log("No files detected")
        return CommandResult(0, "", ""), []

    print("<<<>>> linefilterslist")
    print(line_filters)

    result = await _run_clang_tidy(options, line_filters, files)
    fixes, warnings = extract_warnings(
        result.stdout, base_dir=options.compile_commands_dir
    )

    if options.suppress_diagnostics:
        for fname in fixes.keys():
            mapped_fname = map_filename(options.compile_commands_dir, fname)
            log(f"Applying fixes to {mapped_fname}")
            apply_nolint(fname, fixes[fname])
            if os.path.relpath(fname) != mapped_fname:
                shutil.copyfile(fname, mapped_fname)

    if options.dry_run:
        log(result)
    elif result.failed():
        # If you change this message, update the error checking logic in
        # .github/workflows/lint.yml
        msg = "Warnings detected!"
        log(msg)
        log("Summary:")
        for w in warnings:
            log(str(w))

    return result, warnings


def run(options: Any) -> Tuple[CommandResult, List[ClangTidyWarning]]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_run(options))
