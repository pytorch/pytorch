#!/usr/bin/env python3

import argparse
import asyncio
import collections
import csv
import hashlib
import itertools
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Awaitable, cast, DefaultDict, Dict, List, Match, Optional, Set

from typing_extensions import TypedDict  # Python 3.11+

help_msg = """fast_nvcc [OPTION]... -- [NVCC_ARG]...

Run the commands given by nvcc --dryrun, in parallel.

All flags for this script itself (see the "optional arguments" section
of --help) must be passed before the first "--". Everything after that
first "--" is passed directly to nvcc, with the --dryrun argument added.

This script only works with the "normal" execution path of nvcc, so for
instance passing --help (after "--") doesn't work since the --help
execution path doesn't compile anything, so adding --dryrun there gives
nothing in stderr.
"""
parser = argparse.ArgumentParser(help_msg)
parser.add_argument(
    "--faithful",
    action="store_true",
    help="don't modify the commands given by nvcc (slower)",
)
parser.add_argument(
    "--graph",
    metavar="FILE.gv",
    help="write Graphviz DOT file with execution graph",
)
parser.add_argument(
    "--nvcc",
    metavar="PATH",
    default="nvcc",
    help='path to nvcc (default is just "nvcc")',
)
parser.add_argument(
    "--save",
    metavar="DIR",
    help="copy intermediate files from each command into DIR",
)
parser.add_argument(
    "--sequential",
    action="store_true",
    help="sequence commands instead of using the graph (slower)",
)
parser.add_argument(
    "--table",
    metavar="FILE.csv",
    help="write CSV with times and intermediate file sizes",
)
parser.add_argument(
    "--verbose",
    metavar="FILE.txt",
    help="like nvcc --verbose, but expanded and into a file",
)
default_config = parser.parse_args([])


# docs about temporary directories used by NVCC
url_base = "https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html"
url_vars = f"{url_base}#keeping-intermediate-phase-files"


# regex for temporary file names
re_tmp = r"(?<![\w\-/])(?:/tmp/)?(tmp[^ \"\'\\]+)"
if os.name == "nt":
    re_tmp = r"(?<![\w\-0])(?:\/Temp\/)?(tmp[^ \"\'\\]+)"


def fast_nvcc_warn(warning: str) -> None:
    """
    Warn the user about something regarding fast_nvcc.
    """
    print(f"warning (fast_nvcc): {warning}", file=sys.stderr)


def warn_if_windows() -> None:
    """
    Warn the user that using fast_nvcc on Windows might not work.
    """
    # use os.name instead of platform.system() because there is a
    # platform.py file in this directory, making it very difficult to
    # import the platform module from the Python standard library
    if os.name == "nt":
        fast_nvcc_warn("untested on Windows, might not work; see this URL:")
        fast_nvcc_warn(url_vars)


def warn_if_tmpdir_flag(args: List[str]) -> None:
    """
    Warn the user that using fast_nvcc with some flags might not work.
    """
    file_path_specs = "file-and-path-specifications"
    guiding_driver = "options-for-guiding-compiler-driver"
    scary_flags = {
        "--objdir-as-tempdir": file_path_specs,
        "-objtemp": file_path_specs,
        "--keep": guiding_driver,
        "-keep": guiding_driver,
        "--keep-dir": guiding_driver,
        "-keep-dir": guiding_driver,
        "--save-temps": guiding_driver,
        "-save-temps": guiding_driver,
    }
    for arg in args:
        for flag, frag in scary_flags.items():
            if re.match(rf"^{re.escape(flag)}(?:=.*)?$", arg):
                fast_nvcc_warn(f"{flag} not supported since it interacts with")
                fast_nvcc_warn("TMPDIR, so fast_nvcc may break; see this URL:")
                fast_nvcc_warn(f"{url_base}#{frag}")


class DryunData(TypedDict):
    env: Dict[str, str]
    commands: List[str]
    exit_code: int


def nvcc_dryrun_data(binary: str, args: List[str]) -> DryunData:
    """
    Return parsed environment variables and commands from nvcc --dryrun.
    """
    result = subprocess.run(  # type: ignore[call-overload]
        [binary, "--dryrun"] + args,
        capture_output=True,
        encoding="ascii",  # this is just a guess
    )
    print(result.stdout, end="")
    env = {}
    commands = []
    output = result.stderr
    if os.name == "nt":
        output = result.stdout
    for line in output.splitlines():
        match = re.match(r"^#\$ (.*)$", line)
        if match:
            (stripped,) = match.groups()
            mapping = re.match(r"^(\w+)=(.*)$", stripped)
            if mapping:
                name, val = mapping.groups()
                env[name] = val
            else:
                commands.append(stripped)
        else:
            print(line, file=sys.stderr)
    return {"env": env, "commands": commands, "exit_code": result.returncode}


def warn_if_tmpdir_set(env: Dict[str, str]) -> None:
    """
    Warn the user that setting TMPDIR with fast_nvcc might not work.
    """
    if os.getenv("TMPDIR") or "TMPDIR" in env:
        fast_nvcc_warn("TMPDIR is set, might not work; see this URL:")
        fast_nvcc_warn(url_vars)


def contains_non_executable(commands: List[str]) -> bool:
    for command in commands:
        # This is to deal with special command dry-run result from NVCC such as:
        # ```
        # #$ "/lib64/ccache"/c++ -std=c++11 -E -x c++ -D__CUDACC__ -D__NVCC__  -fPIC -fvisibility=hidden -O3 \
        #   -I ... -m64 "reduce_scatter.cu" > "/tmp/tmpxft_0037fae3_00000000-5_reduce_scatter.cpp4.ii
        # #$ -- Filter Dependencies -- > ... pytorch/build/nccl/obj/collectives/device/reduce_scatter.dep.tmp
        # ```
        if command.startswith("--"):
            return True
    return False


def module_id_contents(command: List[str]) -> str:
    """
    Guess the contents of the .module_id file contained within command.
    """
    if command[0] == "cicc":
        path = command[-3]
    elif command[0] == "cudafe++":
        path = command[-1]
    middle = pathlib.PurePath(path).name.replace("-", "_").replace(".", "_")
    # this suffix is very wrong (the real one is far less likely to be
    # unique), but it seems difficult to find a rule that reproduces the
    # real suffixes, so here's one that, while inaccurate, is at least
    # hopefully as straightforward as possible
    suffix = hashlib.md5(str.encode(middle)).hexdigest()[:8]
    return f"_{len(middle)}_{middle}_{suffix}"


def unique_module_id_files(commands: List[str]) -> List[str]:
    """
    Give each command its own .module_id filename instead of sharing.
    """
    module_id = None
    uniqueified = []
    for i, line in enumerate(commands):
        arr = []

        def uniqueify(s: Match[str]) -> str:
            filename = re.sub(r"\-(\d+)", r"-\1-" + str(i), s.group(0))
            arr.append(filename)
            return filename

        line = re.sub(re_tmp + r".module_id", uniqueify, line)
        line = re.sub(r"\s*\-\-gen\_module\_id\_file\s*", " ", line)
        if arr:
            (filename,) = arr
            if os.name == "nt":
                filename = "%TEMP%\\" + filename
            if not module_id:
                module_id = module_id_contents(shlex.split(line))
            uniqueified.append(f"echo -n '{module_id}' > \"{filename}\"")
        uniqueified.append(line)
    return uniqueified


def make_rm_force(commands: List[str]) -> List[str]:
    """
    Add --force to all rm commands.
    """
    return [f"{c} --force" if c.startswith("rm ") else c for c in commands]


def print_verbose_output(
    *,
    env: Dict[str, str],
    commands: List[List[str]],
    filename: str,
) -> None:
    """
    Human-readably write nvcc --dryrun data to stderr.
    """
    padding = len(str(len(commands) - 1))
    with open(filename, "w") as f:
        for name, val in env.items():
            print(f'#{" "*padding}$ {name}={val}', file=f)
        for i, command in enumerate(commands):
            prefix = f"{str(i).rjust(padding)}$ "
            print(f"#{prefix}{command[0]}", file=f)
            for part in command[1:]:
                print(f'#{" "*len(prefix)}{part}', file=f)


Graph = List[Set[int]]


def straight_line_dependencies(commands: List[str]) -> Graph:
    """
    Return a straight-line dependency graph.
    """
    return [({i - 1} if i > 0 else set()) for i in range(len(commands))]


def files_mentioned(command: str) -> List[str]:
    """
    Return fully-qualified names of all tmp files referenced by command.
    """
    if os.name == "nt":
        return [f"/%TEMP%/{match.group(1)}" for match in re.finditer(re_tmp, command)]
    return [f"/tmp/{match.group(1)}" for match in re.finditer(re_tmp, command)]


def nvcc_data_dependencies(commands: List[str]) -> Graph:
    """
    Return a list of the set of dependencies for each command.
    """
    # fatbin needs to be treated specially because while the cicc steps
    # do refer to .fatbin.c files, they do so through the
    # --include_file_name option, since they're generating files that
    # refer to .fatbin.c file(s) that will later be created by the
    # fatbinary step; so for most files, we make a data dependency from
    # the later step to the earlier step, but for .fatbin.c files, the
    # data dependency is sort of flipped, because the steps that use the
    # files generated by cicc need to wait for the fatbinary step to
    # finish first
    tmp_files: Dict[str, int] = {}
    fatbins: DefaultDict[int, Set[str]] = collections.defaultdict(set)
    graph = []
    for i, line in enumerate(commands):
        deps = set()
        for tmp in files_mentioned(line):
            if tmp in tmp_files:
                dep = tmp_files[tmp]
                deps.add(dep)
                if dep in fatbins:
                    for filename in fatbins[dep]:
                        if filename in tmp_files:
                            deps.add(tmp_files[filename])
            if tmp.endswith(".fatbin.c") and not line.startswith("fatbinary"):
                fatbins[i].add(tmp)
            else:
                tmp_files[tmp] = i
        if (line.startswith("rm ") or line.startswith("erase ")) and not deps:
            if os.name == "nt":
                commands[i] = line.replace("/", "\\")
            deps.add(i - 1)
        graph.append(deps)
    return graph


def is_weakly_connected(graph: Graph) -> bool:
    """
    Return true iff graph is weakly connected.
    """
    if not graph:
        return True
    neighbors: List[Set[int]] = [set() for _ in graph]
    for node, predecessors in enumerate(graph):
        for pred in predecessors:
            neighbors[pred].add(node)
            neighbors[node].add(pred)
    # assume nonempty graph
    stack = [0]
    found = {0}
    while stack:
        node = stack.pop()
        for neighbor in neighbors[node]:
            if neighbor not in found:
                found.add(neighbor)
                stack.append(neighbor)
    return len(found) == len(graph)


def warn_if_not_weakly_connected(graph: Graph) -> None:
    """
    Warn the user if the execution graph is not weakly connected.
    """
    if not is_weakly_connected(graph):
        fast_nvcc_warn("execution graph is not (weakly) connected")


def print_dot_graph(
    *,
    commands: List[List[str]],
    graph: Graph,
    filename: str,
) -> None:
    """
    Print a DOT file displaying short versions of the commands in graph.
    """

    def name(k: int) -> str:
        return f'"{k} {os.path.basename(commands[k][0])}"'

    with open(filename, "w") as f:
        print("digraph {", file=f)
        # print all nodes, in case it's disconnected
        for i in range(len(graph)):
            print(f"    {name(i)};", file=f)
        for i, deps in enumerate(graph):
            for j in deps:
                print(f"    {name(j)} -> {name(i)};", file=f)
        print("}", file=f)


class Result(TypedDict, total=False):
    exit_code: int
    stdout: bytes
    stderr: bytes
    time: float
    files: Dict[str, int]


async def run_command(
    command: str,
    *,
    env: Dict[str, str],
    deps: Set[Awaitable[Result]],
    gather_data: bool,
    i: int,
    save: Optional[str],
) -> Result:
    """
    Run the command with the given env after waiting for deps.
    """
    for task in deps:
        dep_result = await task
        # abort if a previous step failed
        if "exit_code" not in dep_result or dep_result["exit_code"] != 0:
            return {}
    if gather_data:
        t1 = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    code = cast(int, proc.returncode)
    results: Result = {"exit_code": code, "stdout": stdout, "stderr": stderr}
    if gather_data:
        t2 = time.monotonic()
        results["time"] = t2 - t1
        sizes = {}
        for tmp_file in files_mentioned(command):
            if os.path.exists(tmp_file):
                sizes[tmp_file] = os.path.getsize(tmp_file)
            else:
                sizes[tmp_file] = 0
        results["files"] = sizes
    if save:
        dest = pathlib.Path(save) / str(i)
        dest.mkdir()
        for src in map(pathlib.Path, files_mentioned(command)):
            if src.exists():
                shutil.copy2(src, dest / (src.name))
    return results


async def run_graph(
    *,
    env: Dict[str, str],
    commands: List[str],
    graph: Graph,
    gather_data: bool = False,
    save: Optional[str] = None,
) -> List[Result]:
    """
    Return outputs/errors (and optionally time/file info) from commands.
    """
    if os.name == "nt":
        env.update(os.environ.copy())
    tasks: List[Awaitable[Result]] = []
    for i, (command, indices) in enumerate(zip(commands, graph)):
        deps = {tasks[j] for j in indices}
        tasks.append(
            asyncio.create_task(
                run_command(  # type: ignore[attr-defined]
                    command,
                    env=env,
                    deps=deps,
                    gather_data=gather_data,
                    i=i,
                    save=save,
                )
            )
        )
    return [await task for task in tasks]


def print_command_outputs(command_results: List[Result]) -> None:
    """
    Print captured stdout and stderr from commands.
    """
    for result in command_results:
        sys.stdout.write(result.get("stdout", b"").decode("ascii"))
        sys.stderr.write(result.get("stderr", b"").decode("ascii"))


def write_log_csv(
    command_parts: List[List[str]],
    command_results: List[Result],
    *,
    filename: str,
) -> None:
    """
    Write a CSV file of the times and /tmp file sizes from each command.
    """
    tmp_files: List[str] = []
    for result in command_results:
        tmp_files.extend(result.get("files", {}).keys())
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["command", "seconds"] + list(dict.fromkeys(tmp_files))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(command_results):
            command = f"{i} {os.path.basename(command_parts[i][0])}"
            row = {"command": command, "seconds": result.get("time", 0)}
            writer.writerow({**row, **result.get("files", {})})


def exit_code(results: List[Result]) -> int:
    """
    Aggregate individual exit codes into a single code.
    """
    for result in results:
        code = result.get("exit_code", 0)
        if code != 0:
            return code
    return 0


def wrap_nvcc(
    args: List[str],
    config: argparse.Namespace = default_config,
) -> int:
    return subprocess.call([config.nvcc] + args)


def fast_nvcc(
    args: List[str],
    *,
    config: argparse.Namespace = default_config,
) -> int:
    """
    Emulate the result of calling the given nvcc binary with args.

    Should run faster than plain nvcc.
    """
    warn_if_windows()
    warn_if_tmpdir_flag(args)
    dryrun_data = nvcc_dryrun_data(config.nvcc, args)
    env = dryrun_data["env"]
    warn_if_tmpdir_set(env)
    commands = dryrun_data["commands"]
    if not config.faithful:
        commands = make_rm_force(unique_module_id_files(commands))

    if contains_non_executable(commands):
        return wrap_nvcc(args, config)

    command_parts = list(map(shlex.split, commands))
    if config.verbose:
        print_verbose_output(
            env=env,
            commands=command_parts,
            filename=config.verbose,
        )
    graph = nvcc_data_dependencies(commands)
    warn_if_not_weakly_connected(graph)
    if config.graph:
        print_dot_graph(
            commands=command_parts,
            graph=graph,
            filename=config.graph,
        )
    if config.sequential:
        graph = straight_line_dependencies(commands)
    results = asyncio.run(
        run_graph(  # type: ignore[attr-defined]
            env=env,
            commands=commands,
            graph=graph,
            gather_data=bool(config.table),
            save=config.save,
        )
    )
    print_command_outputs(results)
    if config.table:
        write_log_csv(command_parts, results, filename=config.table)
    return exit_code([dryrun_data] + results)  # type: ignore[arg-type, operator]


def our_arg(arg: str) -> bool:
    return arg != "--"


if __name__ == "__main__":
    argv = sys.argv[1:]
    us = list(itertools.takewhile(our_arg, argv))
    them = list(itertools.dropwhile(our_arg, argv))
    sys.exit(fast_nvcc(them[1:], config=parser.parse_args(us)))
