"""Run the tests of one shard of the plan: those that run alone, a file after the
other, then the shard's lanes side by side, a file after the other in each. A file
is launched as itself, as collect lists it. When the plan has only some of a file's
tests for it, the file gets them through --nodeid-file, an option of
test/conftest.py. One piece of such a file also runs the tests that no piece names,
those that exist here and not where they were listed: it gets the tests of the other
pieces through --except-nodeid-file. Every file writes a report of its test runs,
for the job to upload."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.stats.import_test_stats import get_disabled_tests
from tools.tci.collect import fresh
from tools.tci.shard import minutes


if TYPE_CHECKING:
    import argparse


TEST_DIR = Path(__file__).resolve().parents[2] / "test"
# What a test file gets to run its tests through pytest, with a line per test. Without
# xdist, as on main: with it loaded the plugin for reruns starts a thread, and the
# tests of the profiler count the threads of the process.
RUN = ["--use-pytest", "-v", "-p", "no:xdist"]
# The lists of the tests that main skips: the slow ones (test/slow_tests.json) and
# those an issue disabled. The times of the plan are main's, so they assume both.
SKIP = ["--import-slow-tests", "--import-disabled-tests"]
# Where each file writes the report of its test runs (torch/testing/_internal/torchci).
# The HUD's test pages show them once the job has uploaded them, under its id in JOB_ID.
REPORTS = f"--save-torchci-reports={TEST_DIR / 'torchci-reports'}"
# What main's test jobs set and tests rely on: test_serialization.py checks the first,
# and the runner's cache of Hugging Face models is read-only, so nothing is downloaded.
ENVIRONMENT = {"TORCH_SERIALIZATION_DEBUG": "1", "TRANSFORMERS_OFFLINE": "1"}
# Held to print, so that the lines of a file stay together.
PRINTING = threading.Lock()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, help="the plan that shard wrote")
    parser.add_argument("--shard", type=int, required=True, help="its number, from 0")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per file")


def run(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text())
    shard = plan["shards"][args.shard]
    alone, lanes = shard["alone"], shard["lanes"]
    # The tests an issue disabled, fetched into test/ as run_test.py fetches them.
    get_disabled_tests(str(TEST_DIR))
    # As .ci/pytorch/test.sh sets it on main, where the plan's times are from: with a
    # thread per CPU in each, files side by side keep each other waiting. With fewer
    # than 4 threads sums are added in another order, which changes results.
    threads = max(4, len(os.sched_getaffinity(0)) // (len(lanes) + 1))
    env = {"OMP_NUM_THREADS": str(threads), **os.environ, **ENVIRONMENT}
    # Unbuffered, so that a file that crashes or hangs leaves its last lines.
    env["PYTHONUNBUFFERED"] = "1"
    # A plan for a machine with a GPU would pass on one without: the tests of the
    # device are not there to fail.
    planned_for = plan["times"]["configurations"][0]["environment"]
    if planned_for["device_count"] and not fresh(env)[1]["device_count"]:
        sys.exit(f"the plan is for {planned_for['device_name']}: no device here")
    failed: list[str] = []
    start = time.monotonic()
    parts = [part for s in plan["shards"] for part in (s["alone"], *s["lanes"])]

    def named(file: str) -> list[str]:
        """The tests of a file that its pieces in the plan name."""
        pieces = [part[file] for part in parts if part.get(file) is not None]
        pieces = [p["and_unnamed"] if isinstance(p, dict) else p for p in pieces]
        return [test for piece in pieces for test in piece]

    def launch(file: str, tests: Any, env: dict[str, str]) -> None:
        """Run a file's tests: all of them, those the plan names, or all but those
        that the plan names for the file's other pieces."""
        scratch = Path(tempfile.mkdtemp(dir=scratches))
        command = [sys.executable, "-bb", file, *RUN, REPORTS, *SKIP]
        what = "all its tests"
        if tests is not None:
            if isinstance(tests, dict):
                mine = set(tests["and_unnamed"])
                tests = [test for test in named(file) if test not in mine]
                option, what = (
                    "--except-nodeid-file",
                    f"all but {len(tests)} of its tests",
                )
            else:
                option, what = "--nodeid-file", f"{len(tests)} of its tests"
            ids = scratch / "ids"
            ids.write_text("".join(f"test/{file}::{test}\n" for test in tests))
            command.append(f"{option}={ids}")
        what = f"{file}, {what}"
        with PRINTING:
            print(f"started: {what}", flush=True)
        began = time.monotonic()
        with open(scratch / "printed", "wb") as printed:
            where: dict[str, Any] = {"cwd": TEST_DIR, "env": env}
            where |= {"stdout": printed, "stderr": subprocess.STDOUT}
            # A session of its own, so that what the file started ends with it.
            process = subprocess.Popen(command, start_new_session=True, **where)
            try:
                code = process.wait(timeout=args.timeout)
                how = "passed" if code == 0 else f"FAILED with exit code {code}"
            except subprocess.TimeoutExpired:
                how = f"FAILED, no end after {args.timeout}s"
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        took = minutes(time.monotonic() - began)
        with PRINTING:
            # GitHub folds the lines between these two in the job's log.
            print(f"::group::{what}", flush=True)
            lines = (scratch / "printed").read_bytes()
            sys.stdout.buffer.write(lines if lines.endswith(b"\n") else lines + b"\n")
            print("::endgroup::")
            print(f"{how} in {took}: {what}", flush=True)
        if how != "passed":
            failed.append(what)

    def lane(files: dict[str, Any], env: dict[str, str]) -> None:
        for file, tests in files.items():
            launch(file, tests, env)

    # As main sets it for the files it runs side by side: where there is a GPU, a
    # file's process then takes no more than its share of the GPU's memory.
    shared = {**env, "NUM_PARALLEL_PROCS": str(len(lanes))}
    with tempfile.TemporaryDirectory() as scratches:
        lane(alone, env)
        with ThreadPoolExecutor(len(lanes)) as side_by_side:
            list(side_by_side.map(lambda files: lane(files, shared), lanes))
    took, planned = minutes(time.monotonic() - start), minutes(shard["seconds"])
    count = len(alone) + sum(len(files) for files in lanes)
    print(f"shard {args.shard}: {count} files in {took}, planned {planned}")
    if failed:
        sys.exit("failed: " + "; ".join(failed))
