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
import json
import os
import os.path
import re
import shutil
import asyncio
import multiprocessing
from typing import Any, Dict, Iterable, List, Set, Tuple

from tools.linter.utils import CommandResult, ProgressMeter, run_cmd

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


class ClangTidyWarning:
    def __init__(self, name: str, occurrences: List[Tuple[str, int]]):
        self.name = name
        self.occurrences = occurrences

    def __str__(self) -> str:
        base = f"[{self.name}] occurred {len(self.occurrences)} times\n"
        for occ in self.occurrences:
            base += f"    {occ[0]}:{occ[1]}\n"
        return base


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
            if result.failed() and not QUIET:
                msg = str(result) if not VERBOSE else repr(result)
                progress_meter.print(msg)
            progress_meter.update(f"Processed {filename}")

        coros = [
            run_cmd(cmd, on_completed=on_completed, on_completed_args=[filename])
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


async def run(
    files: List[str], line_filters: List[str], options: Any
) -> Tuple[CommandResult, List[ClangTidyWarning]]:
    # These flags are pervasive enough to set it globally. It makes the code
    # cleaner compared to threading it through every single function.
    global VERBOSE
    global QUIET
    VERBOSE = options.verbose
    QUIET = options.quiet

    # clang-tidy errors when it does not get input files.
    if not files:
        if VERBOSE:
            log("No files detected")
        return CommandResult(0, "", ""), []

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
        msg = "Warnings detected!"
        result.stdout += (
            f"\n{msg}"
            "\nSummary:"
        )
        for w in warnings:
            result.stdout += f"\n{str(w)}"

    # Reset stderr because clang-tidy outputs logs with very little signal to
    # noise like warning suppressions
    result.stderr = ""

    return result, warnings
