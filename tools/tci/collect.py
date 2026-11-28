"""List the tests of every test file, and which of them are marked serial (@serialTest
on a test, serialTestFile() for a file): those run with no other test beside them.
Processes that have imported what files share fork a child for each file (forks.py).
A file that fails there is listed again by a process of its own, so that a file's
failure is always that of its own process. A run starts a process for every file, so
how long one takes to import torch and the test utilities is timed for the plan."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.testing.discover_tests import TESTS


if TYPE_CHECKING:
    import argparse


TEST_DIR = Path(__file__).resolve().parents[2] / "test"
FORKS = str(Path(__file__).with_name("forks.py"))
# What a test file gets to list its tests. It is launched as itself, so its __main__
# guard decides as it does in a run, and it lists through pytest, whose ids a run
# uses. Without pytest's cache, which every file would write.
LIST = ["--use-pytest", "--collect-only", "--verbosity=-1", "-p", "no:cacheprovider"]
# The fields the ci database files a test run under. The process that prints them
# imports torch and the test utilities, as a test file's own process does first.
ENVIRONMENT = (
    "import json; from torch.testing._internal.torchci.environment import capture; "
    "print('tci-environment', json.dumps(capture().identity()))"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="*", help="relative to test/ (default: all)")
    parser.add_argument("--output", required=True, help="gzipped JSON to write")
    cpus = len(os.sched_getaffinity(0))
    parser.add_argument("--jobs", type=int, default=cpus, help="files at once")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per file")


def fresh(env: dict[str, str]) -> tuple[float, dict[str, Any]]:
    """Start a process of its own that imports torch and the test utilities: how long
    it takes, and the fields the ci database files a test run of this machine under."""
    began = time.monotonic()
    where: dict[str, Any] = {"cwd": TEST_DIR, "env": env, "text": True}
    probe = [sys.executable, "-c", ENVIRONMENT]
    done = subprocess.run(probe, stdout=subprocess.PIPE, **where)
    said = done.stdout.rsplit("tci-environment ", 1)
    if done.returncode or len(said) != 2:
        sys.exit("cannot import the installed torch from test/")
    return time.monotonic() - began, json.loads(said[1].split("\n", 1)[0])


def tests(file: str, printed: str) -> list[str]:
    # pytest prints a node id per line, relative to its rootdir, the repository root.
    ids = [line for line in printed.split("\n") if line.startswith(f"test/{file}::")]
    return [nodeid.split("::", 1)[1] for nodeid in ids]


def alone(argv: list[str], timeout: int, env: dict[str, str]) -> dict[str, Any]:
    """List a file's tests with a process of its own."""
    start, printed, error = time.monotonic(), "", ""
    try:
        done = subprocess.run(
            [sys.executable, "-bb", *argv],
            cwd=TEST_DIR,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        if done.returncode:
            tail = done.stdout[-2000:] + done.stderr[-2000:]
            error = f"exit code {done.returncode}\n{tail}"
        else:
            printed = done.stdout
    except subprocess.TimeoutExpired:
        error = f"no result after {timeout}s"
    seconds = round(time.monotonic() - start, 2)
    result: dict[str, Any] = {"seconds": seconds, "tests": tests(argv[0], printed)}
    if error:
        result["error"] = error
    return result


def run(args: argparse.Namespace) -> None:
    # A copy of CPython's tests is for its version of Python only, as in run_test.py.
    cpython = f"cpython/v{sys.version_info[0]}_{sys.version_info[1]}/"
    names = [t for t in TESTS if not t.startswith("cpython/") or t.startswith(cpython)]
    # This file fails on purpose: with it run_test.py checks that CI sees a failure.
    names = [t for t in names if t != "test_ci_sanity_check_fail"]
    listed = sorted(f"{t}.py" for t in names if (TEST_DIR / f"{t}.py").is_file())
    files = args.files or listed
    # Listing tests computes nothing: one thread each, or the processes fight over
    # the CPUs with their thread pools.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # In test/, like the tests themselves: at the repository root `import torch`
    # finds the source tree instead of the installed wheel.
    where: dict[str, Any] = {"cwd": TEST_DIR, "env": env, "text": True}
    start = time.monotonic()
    forked = 0

    def listing(files: list[str], marker: str) -> dict[str, dict[str, Any]]:
        """List the tests of `files` that pytest's -m `marker` keeps. `left_out` says
        whether it left tests marked serial out."""
        nonlocal forked
        if not files:
            return {}
        results: dict[str, dict[str, Any]] = {}
        retries: dict[str, Future[dict[str, Any]]] = {}
        folder = tempfile.mkdtemp(dir=scratch)
        out = {file: f"{folder}/{n}.out" for n, file in enumerate(files)}
        # conftest.py writes there how many tests marked serial -m leaves out.
        count = {file: f"{folder}/{n}.count" for n, file in enumerate(files)}
        keep = {f: ["-m", marker, f"--serial-report-file={count[f]}"] for f in files}
        argv = {file: [file, *LIST, *keep[file]] for file in files}
        jobs = [{"file": f, "argv": argv[f], "out": out[f]} for f in files]
        work = {"jobs": jobs, "at_once": args.jobs, "seconds": args.timeout}
        Path(folder, "work.json").write_text(json.dumps(work))
        command = [sys.executable, "-bb", FORKS, f"{folder}/work.json"]
        # A session of its own, so that its forks can be ended with it. What the
        # forks say on stderr is dropped: a file that fails is listed again alone.
        quiet = {"stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
        forks = subprocess.Popen(command, start_new_session=True, **quiet, **where)

        def end(*_: object) -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(forks.pid, signal.SIGKILL)

        # The forks say how a file ended within its seconds. Silent for longer, they
        # are stuck: they are ended, and the files left are listed alone.
        signal.signal(signal.SIGALRM, end)
        try:
            signal.alarm(args.timeout + 30)
            for line in forks.stdout or ():
                signal.alarm(args.timeout + 30)
                file, code, seconds = json.loads(line)
                if code:
                    retries[file] = pool.submit(alone, argv[file], args.timeout, env)
                    continue
                ids = tests(file, Path(out[file]).read_text(errors="replace"))
                results[file] = {"seconds": round(seconds, 2), "tests": ids}
                if len(results) % 100 == 0:
                    elapsed = time.monotonic() - start
                    print(f"{len(results)} files, {elapsed:.0f}s", flush=True)
        finally:
            signal.alarm(0)
            end()
            forks.wait()
        forked += len(results)
        for file in files:
            if file not in results and file not in retries:
                retries[file] = pool.submit(alone, argv[file], args.timeout, env)
        results.update({file: retry.result() for file, retry in retries.items()})
        for file, result in results.items():
            counted = Path(count[file])
            lines = counted.read_text().split() if counted.exists() else []
            result["left_out"] = any(map(int, lines))
        return results

    with (
        tempfile.TemporaryDirectory() as scratch,
        ThreadPoolExecutor(args.jobs) as pool,
    ):
        # What a run pays for every file and a listing does not: almost every fork
        # finds torch and the test utilities imported, a process of its own imports
        # them first.
        started = pool.submit(fresh, env)
        beside = listing(files, "not serial")
        marked = listing([f for f in files if beside[f]["left_out"]], "serial")
    results = {}
    for file in files:
        first, second = beside[file], marked.get(file, {"seconds": 0, "tests": []})
        # A file is loaded once for its tests, whichever of them a run keeps.
        seconds = max(first["seconds"], second["seconds"])
        results[file] = {"seconds": seconds, "tests": first["tests"] + second["tests"]}
        if second["tests"]:
            results[file]["serial"] = second["tests"]
        if error := first.get("error") or second.get("error"):
            results[file]["error"] = error
    took, environment = started.result()
    content = {"environment": environment, "start": round(took, 2), "files": results}
    with gzip.open(args.output, "wt") as f:
        json.dump(content, f)

    total = sum(len(result["tests"]) for result in results.values())
    elapsed = time.monotonic() - start
    print(f"{total} tests in {len(files)} files, {elapsed:.0f}s, {args.jobs} at a time")
    print(f"a new process imports torch and the test utilities in {took:.1f}s")
    serial = sum(len(result.get("serial", ())) for result in results.values())
    print(f"{serial} of the tests are marked serial, in {len(marked)} files")
    listings = len(files) + len(marked)
    print(
        f"{forked} of the {listings} listings by a fork, the others by a process of their own"
    )
    slowest = sorted(files, key=lambda file: -results[file]["seconds"])[:5]
    print("slowest: " + ", ".join(f"{f} {results[f]['seconds']:.0f}s" for f in slowest))
    errors = {file: r["error"] for file, r in results.items() if "error" in r}
    if errors:
        print(f"{len(errors)} files could not be collected:")
        for file, error in errors.items():
            lines = [line for line in error.split("\n") if line.strip()]
            # pytest's summary line of a collection error, else the last thing said.
            summary = [line for line in lines if line.startswith("ERROR ")] or lines
            print(f"  {file}: {summary[-1][:200]}")
    if not total:
        sys.exit("no tests collected")
