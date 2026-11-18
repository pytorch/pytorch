"""Lists the tests of many test files for worker.py, as forks of one process. What
makes that slow in a container is importing: each file imports torch and much the
same modules of it, and a container is slow at many imports at once. So this process
imports the modules of torch that most of the files import, one after the other, and
forks a child for each file, so many at a time. A child runs its file as python
would: as __main__, with its own arguments, its directory first on sys.path, and
what it prints in a file of its own.

A file's fork must find nothing imported that the file would not have imported by
then: a file can set the environment that a module reads as it is imported. So a
file is forked when this process has imported those of the modules that the file
itself imports before it does anything else, and no other.

A fork can't use CUDA that its parent has started. So nothing here may start it:
torch counts the GPUs through NVML while a module is imported, and a module that
needs CUDA to import is left to each child.

The work is in the JSON file this is given: the files, each with its arguments and
the file for what it prints, how many to run at once, and the seconds a file gets.
As a file's child ends, a line of JSON on stdout says how: the file, its exit code
and its seconds."""

import ast
import collections
import contextlib
import importlib
import json
import os
import signal
import sys
import time
import tokenize
import types
from pathlib import Path
from typing import Any, TextIO
from unittest import mock


# Set, torch.cuda.is_available asks NVML for the GPUs, which starts no CUDA
NVML_CHECK = "PYTORCH_NVML_BASED_CUDA_CHECK"
POLL_SECONDS = 0.02  # between two looks at the children


class NeedsCuda(BaseException):
    """A module needs CUDA to import. Not an Exception, so that no handler of the
    module's own takes it for a machine without a GPU."""


def prelude(file: str) -> list[ast.stmt]:
    """A file's first statements, as long as they are imports or its docstring. No
    more of the file is read than those: the rest of a test file is long."""
    lines: list[bytes] = []
    with open(file, "rb") as source:

        def read() -> bytes:
            lines.append(source.readline())
            return lines[-1]

        begins = True  # the next token begins a statement
        for token in tokenize.tokenize(read):
            if token.type in (tokenize.ENCODING, tokenize.COMMENT, tokenize.NL):
                continue
            first = token.type == tokenize.STRING or token.string in ("import", "from")
            if begins and not first:
                del lines[token.start[0] - 1 :]
                break
            begins = token.type == tokenize.NEWLINE
    return ast.parse(b"".join(lines)).body


def leads(file: str) -> set[str]:
    """The modules of torch that a file imports before it does anything else: before
    any statement that is no import, and before it imports a module beside it, which
    can do anything too. Each with its packages, which are imported with it."""
    named: set[str] = set()
    with contextlib.suppress(OSError, SyntaxError, ValueError, tokenize.TokenError):
        for node in prelude(file):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue  # its docstring
            else:
                break
            for module in modules:
                parts = module.split(".")
                beside = Path(file).with_name(parts[0])
                # A directory beside the file gives way to the standard library's module
                package = beside.is_dir() and parts[0] not in sys.stdlib_module_names
                if parts[0] == "torch":
                    named.update(".".join(parts[:n]) for n in range(1, len(parts) + 1))
                elif package or beside.with_suffix(".py").exists():
                    return named
    return named


def bring(name: str) -> None:
    """Import a module of torch without starting CUDA. If it fails to import, or
    needs CUDA to, it is left to each child."""
    with mock.patch.dict(os.environ, {NVML_CHECK: "1"}):
        torch = importlib.import_module("torch")
        # Where NVML can't count the GPUs, torch asks CUDA, which starts it
        if torch.cuda._device_count_nvml() < 0 and torch.cuda.is_available():
            sys.exit("NVML can't count the GPUs: CUDA did, so no fork can use it")
        refused = mock.patch.object(torch.cuda, "_lazy_init", side_effect=NeedsCuda)
        with refused, contextlib.suppress(Exception, NeedsCuda):
            importlib.import_module(name)


def run(job: dict[str, Any], said: TextIO) -> None:
    """In a child: run the job's file as python would, and exit as it would. Why it
    fails is for the process that then runs the file alone to say."""
    code = 1
    try:
        said.close()
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        os.dup2(os.open(job["out"], flags, 0o600), 1)
        file = os.path.abspath(job["argv"][0])
        sys.argv = job["argv"]
        sys.path.insert(0, os.path.dirname(os.path.realpath(file)))
        scope = vars(sys.modules["__main__"])
        scope["__file__"] = file
        exec(compile(Path(file).read_bytes(), file, "exec", dont_inherit=True), scope)
        code = 0
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else int(stop.code is not None)
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(code)  # without the cleanup of what this process's parent imported


def main() -> None:
    said = os.fdopen(os.dup(1), "w")  # what this says: the rest goes where 2 does
    os.dup2(2, 1)
    sys.path.pop(0)  # this file's directory, which is no test file's
    # What the files run as: a module imported here may hold it, as common_utils does
    sys.modules["__main__"] = types.ModuleType("__main__")
    work = json.loads(Path(sys.argv[1]).read_text())
    led = {job["file"]: leads(job["argv"][0]) for job in work["jobs"]}
    counts = collections.Counter(name for names in led.values() for name in names)
    # The modules to import: those that more than half of the files lead with, the
    # most led with first. A file's turn is once those before the first it doesn't
    # lead with are imported
    most = [name for name, n in counts.items() if 2 * n > len(led)]
    shared = sorted(most, key=lambda name: (-counts[name], name))
    turns = {}
    for file, names in led.items():
        without = (at for at, name in enumerate(shared) if name not in names)
        turns[file] = next(without, len(shared))
    by_turn = sorted(work["jobs"], key=lambda job: turns[job["file"]])
    waiting = collections.deque(by_turn)
    running: dict[int, tuple[dict[str, Any], float]] = {}
    brought = 0  # of the shared modules
    while waiting or running:
        if waiting and turns[waiting[0]["file"]] > brought:
            bring(shared[brought])  # every file left leads with it
            brought += 1
        elif waiting and len(running) < work["at_once"]:
            job = waiting.popleft()
            sys.stdout.flush()  # or the child would write it too
            sys.stderr.flush()
            pid = os.fork()
            if not pid:
                run(job, said)
            running[pid] = job, time.monotonic()
            continue
        else:
            time.sleep(POLL_SECONDS)
        for pid, (job, start) in list(running.items()):
            seconds = time.monotonic() - start
            if seconds > work["seconds"]:
                os.kill(pid, signal.SIGKILL)  # which its exit code then says
            ended, status = os.waitpid(pid, os.WNOHANG)
            if ended:
                del running[pid]
                told = job["file"], os.waitstatus_to_exitcode(status), seconds
                print(json.dumps(told), file=said, flush=True)


if __name__ == "__main__":
    main()
