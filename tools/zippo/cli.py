"""Command-line interface for zippo."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Optional

from tools.zippo.coordinator import Coordinator, TestEntry
from tools.zippo.sharding import assign_shards, Entry, filter_by_marker


# Marker -> runner configuration used by `zippo plan` to build the GHA matrix.
# `max_shards` caps how many parallel jobs we ask GHA for per marker; tune it
# to roughly match the concurrent-runner availability for each instance type.
# Asking for more shards than the pool can serve just adds queueing time
# without any throughput gain. All other fields are passed through to the
# matrix entry, so the workflow can reference them via `${{ matrix.<field> }}`.
# Keep `DEFAULT_MARKER` as one of the keys so plain TestCase tests (tagged
# with it at collect time) have a runner.
_RUNNER_MAP_FIELD_MAX_SHARDS = "max_shards"
RUNNER_MAP: dict[str, dict[str, object]] = {
    "cpu": {
        "runner": "linux.4xlarge",
        "container_options": "",
        _RUNNER_MAP_FIELD_MAX_SHARDS: 30,
    },
    "cuda": {
        "runner": "linux.g4dn.4xlarge.nvidia.gpu",
        "container_options": "--gpus all",
        _RUNNER_MAP_FIELD_MAX_SHARDS: 30,
    },
}

# Tests whose iter_markers() doesn't intersect RUNNER_MAP get tagged with this
# at collect time so they still route somewhere. Must be a key in RUNNER_MAP.
DEFAULT_MARKER = "cpu"


def _default_workers(num_files: int) -> int:
    """Pick a reasonable default worker count.

    `3.32 * sqrt(cpu)` capped by core count and file count. Square-root
    scaling tapers smoothly across the whole range — every 4x in cores
    yields only 2x in workers — so I/O contention from simultaneous
    `import torch` is held in check on big machines without saturating
    small ones. The constant 3.32 is fit to one data point: ~64 workers
    at 384 cores measured to be near-optimal (3.32 ~= 64 / sqrt(384)).
    Below ~11 cores the formula exceeds the core count and we cap at it.
    """
    cpu = getattr(os, "process_cpu_count", None)
    n = cpu() if cpu is not None else os.cpu_count()
    n = n or 1
    scaled = round(3.32 * (n**0.5))
    return max(1, min(num_files, n, scaled))


def _max_tasks_per_worker(num_files: int, num_workers: int) -> int:
    """Heuristic for how many files a worker handles before recycling.

    Goals:
      - Amortize the worker's startup cost (`import torch`) across enough files
        to be worth it: floor at 20.
      - Bound peak RSS by recycling after enough files have accumulated
        imports: ceiling at 100.
      - In between, target one worker lifetime per (files / workers) so each
        worker handles roughly its even share without wasted recycles.
    """
    even_share = (num_files + num_workers - 1) // num_workers
    return max(20, min(100, even_share))


def _resolve_paths(inputs: list[str]) -> list[str]:
    """Expand file/dir inputs into a deduplicated, sorted list of test files.

    Files are taken as-is (caller's responsibility). Directories are walked
    recursively for both `test_*.py` and `*_test.py` — pytest's default
    `python_files` patterns.
    """
    files: set[str] = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for pattern in ("test_*.py", "*_test.py"):
                for f in p.rglob(pattern):
                    if f.is_file():
                        files.add(str(f))
        elif p.is_file():
            files.add(str(p))
        else:
            raise FileNotFoundError(f"path does not exist: {raw}")
    return sorted(files)


def _apply_default_markers(results: dict[str, list[TestEntry]]) -> None:
    """Tag tests with no routable marker so every test routes to a runner.

    `instantiate_device_type_tests` already appends the device_type marker to
    each instantiated test. Plain TestCase tests and tests carrying only
    non-routing markers (`serial`, `parametrize`, etc.) would otherwise be
    excluded from every per-marker shard. Append `DEFAULT_MARKER` to those
    so they land on its runner.
    """
    routable = frozenset(RUNNER_MAP)
    for path, entries in results.items():
        results[path] = [
            (
                nid,
                marks
                if any(m in routable for m in marks)
                else marks + (DEFAULT_MARKER,),
            )
            for nid, marks in entries
        ]


def _write_results(results: dict[str, list[TestEntry]], output: str) -> None:
    """Write the collected tests as JSON.

    Format: {"tests": {file: [{"test": node_id, "markers": [...]}, ...]}}.
    Node IDs are full pytest IDs (e.g. `<file>::TestClass::test_method`).
    Markers are taken from `item.iter_markers()` (function + class + module).
    """
    by_file = {
        path: [{"test": nid, "markers": list(marks)} for nid, marks in results[path]]
        for path in sorted(results)
    }
    with open(output, "w") as f:
        json.dump({"tests": by_file}, f, indent=2)
        f.write("\n")


def _write_summary(
    results: dict[str, list[TestEntry]],
    failures: list[tuple[str, str]],
    total: int,
    elapsed_s: float,
) -> None:
    succeeded = total - len(failures)
    num_tests = sum(len(entries) for entries in results.values())
    sys.stderr.write(
        f"\n[zippo] {num_tests} tests in {elapsed_s:.1f}s ~ {succeeded}/{total} OK {len(failures)}/{total} FAILED\n"
    )


def _print_failure(path: str, msg: str) -> None:
    # `\r\033[K` clears the in-line progress so the failure prints on its own
    # line without being garbled by the next progress overwrite. Color the
    # FAILED prefix only when stderr is a TTY, so piped/redirected output
    # stays clean.
    label = "\033[31mFAILED\033[0m" if sys.stderr.isatty() else "FAILED"
    sys.stderr.write(f"\r\033[K{label} {path}\t{msg}\n")
    sys.stderr.flush()


class _ProgressPrinter:
    def __init__(self, stream: IO[str], total: int) -> None:
        self._stream = stream
        self._total = total

    def __call__(self, processed: int, failed: int, total: int, working: int) -> None:
        # Trailing `\033[K` clears any leftover characters from a previously
        # longer line (e.g. "10 working" -> "9 working" leaving a stray "g").
        self._stream.write(
            f"\r[zippo] {processed}/{total} → {processed - failed} OK {failed} FAILED ~ {working} workers\033[K"
        )
        self._stream.flush()
        if processed == total:
            # Wipe the progress line so the summary is the only thing left.
            self._stream.write("\r\033[K")


def _cmd_collect(args: argparse.Namespace) -> int:
    """Implementation of the `collect` subcommand."""
    paths = _resolve_paths(args.paths)
    if not paths:
        print("[zippo] no test files found", file=sys.stderr)
        return 0

    num_workers = max(
        1, args.workers if args.workers is not None else _default_workers(len(paths))
    )
    coord = Coordinator(
        num_workers=num_workers,
        max_tasks_per_worker=_max_tasks_per_worker(len(paths), num_workers),
        progress_cb=_ProgressPrinter(sys.stderr, len(paths)),
        failure_cb=_print_failure,
    )
    started = time.monotonic()
    coord.run(paths)
    elapsed = time.monotonic() - started

    _apply_default_markers(coord.results)
    _write_results(coord.results, args.output)
    _write_summary(coord.results, coord.failures, len(paths), elapsed)

    return 0


def _load_tests(path: str) -> dict[str, list[Entry]]:
    """Load the JSON produced by `collect`.

    Returns `{file: [{"test": node_id, "markers": [...]}, ...]}`.
    """
    with open(path) as f:
        data = json.load(f)
    return data.get("tests", {})


def _cmd_plan(args: argparse.Namespace) -> int:
    """Implementation of the `plan` subcommand."""
    tests_by_file = _load_tests(args.tests)
    include: list[dict[str, object]] = []
    for marker, cfg in RUNNER_MAP.items():
        subset = filter_by_marker(tests_by_file, marker)
        if not subset:
            print(
                f"[zippo] no tests for marker {marker!r}; skipping",
                file=sys.stderr,
            )
            continue
        count = sum(len(entries) for entries in subset.values())
        max_shards = cfg[_RUNNER_MAP_FIELD_MAX_SHARDS]
        assert isinstance(max_shards, int) and max_shards > 0
        n = min(max_shards, count)
        per_shard = (count + n - 1) // n
        print(
            f"[zippo] marker {marker!r}: {count} tests across {n} shards "
            f"(~{per_shard}/shard, cap {max_shards})",
            file=sys.stderr,
        )
        entry_cfg = {k: v for k, v in cfg.items() if k != _RUNNER_MAP_FIELD_MAX_SHARDS}
        entry_template = {**entry_cfg, "marker": marker}
        for shard in range(1, n + 1):
            include.append({**entry_template, "shard": shard, "total": n})
    matrix = json.dumps({"include": include}, separators=(",", ":"))

    if args.github_output:
        gh_out = os.environ.get("GITHUB_OUTPUT")
        if not gh_out:
            print(
                "[zippo] --github-output requires $GITHUB_OUTPUT to be set",
                file=sys.stderr,
            )
            return 1
        with open(gh_out, "a") as f:
            f.write(f"matrix={matrix}\n")
    else:
        print(matrix)
    return 0


def _cmd_run_shard(args: argparse.Namespace) -> int:
    """Implementation of the `run-shard` subcommand."""
    tests_by_file = _load_tests(args.tests)
    if args.marker:
        tests_by_file = filter_by_marker(tests_by_file, args.marker)
    if not (1 <= args.shard <= args.total):
        print(
            f"[zippo] --shard must be in [1, {args.total}], got {args.shard}",
            file=sys.stderr,
        )
        return 2

    shards = assign_shards(tests_by_file, args.total)
    node_ids = shards[args.shard - 1]

    # Group node IDs by their immediate parent directory. We then invoke pytest
    # once per directory: within a single invocation pytest's prepend mode adds
    # that one directory to sys.path (so `from common_utils import X` style
    # sibling imports resolve) and all files share a unique-within-dir basename
    # (so `test_utils.py` from sibling dirs don't collide with each other).
    # Running one big pytest with the full argfile hits both failure modes.
    by_dir: dict[str, list[str]] = {}
    for nid in node_ids:
        file_path = nid.split("::", 1)[0]
        d = os.path.dirname(file_path) or "."
        by_dir.setdefault(d, []).append(nid)

    argfile = Path(args.argfile)
    argfile.parent.mkdir(parents=True, exist_ok=True)
    argfile.write_text("".join(f"{nid}\n" for nid in node_ids))

    print(
        f"[zippo] shard {args.shard}/{args.total}: {len(node_ids)} tests across "
        f"{len(by_dir)} directories -> {argfile}",
        file=sys.stderr,
    )

    if not node_ids:
        return 0
    if args.dry_run:
        return 0

    # pytest exit codes: 0 ok, 1 failures, 2 interrupted, 3 internal error,
    # 4 usage error, 5 no tests collected. We surface the worst (max) across
    # per-directory invocations; "no tests collected" (5) is benign here
    # because we explicitly enumerate node IDs, so treat it like 0.
    worst = 0
    for d, nids in sorted(by_dir.items()):
        chunk_argfile = argfile.with_suffix(
            f".{d.replace('/', '_').replace('.', '_')}.args"
        )
        chunk_argfile.write_text("".join(f"{nid}\n" for nid in nids))
        print(
            f"[zippo]   running {len(nids)} tests in {d}",
            file=sys.stderr,
        )
        cmd = ["pytest", f"@{chunk_argfile}", *args.pytest_args]
        rc = subprocess.run(cmd).returncode
        if rc == 5:
            rc = 0
        if rc > worst:
            worst = rc
    return worst


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.zippo",
        description="Parallel pytest test collection.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser(
        "collect",
        help="Collect test node IDs across the given files/directories.",
    )
    collect.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="Test files or directories. Directories are globbed for test_*.py.",
    )
    collect.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count ** (2/3), capped by file count).",
    )
    collect.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        help="JSON file to write collected tests to.",
    )
    collect.set_defaults(func=_cmd_collect)

    plan = sub.add_parser(
        "plan",
        help="Compute a GitHub Actions matrix definition from collected tests.",
    )
    plan.add_argument(
        "--tests",
        metavar="PATH",
        required=True,
        help="JSON file produced by `zippo collect`.",
    )
    plan.add_argument(
        "--github-output",
        action="store_true",
        help="Write `matrix=<json>` to $GITHUB_OUTPUT instead of stdout.",
    )
    plan.set_defaults(func=_cmd_plan)

    run_shard = sub.add_parser(
        "run-shard",
        help="Run pytest against this shard's slice of the collected tests.",
    )
    run_shard.add_argument(
        "--tests",
        metavar="PATH",
        required=True,
        help="JSON file produced by `zippo collect`.",
    )
    run_shard.add_argument(
        "--shard",
        type=int,
        required=True,
        help="1-indexed shard number to run.",
    )
    run_shard.add_argument(
        "--total",
        type=int,
        required=True,
        help="Total number of shards for this marker (from the matrix entry).",
    )
    run_shard.add_argument(
        "-m",
        "--marker",
        metavar="NAME",
        default=None,
        help="Filter to tests whose markers contain NAME before sharding. "
        "Must match the marker that `zippo plan` used so partitions agree.",
    )
    run_shard.add_argument(
        "--argfile",
        metavar="PATH",
        default="/tmp/zippo-shard.args",
        help="Path to write the pytest argfile (one node ID per line).",
    )
    run_shard.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the argfile but do not invoke pytest.",
    )
    run_shard.add_argument(
        "pytest_args",
        nargs="*",
        metavar="PYTEST_ARG",
        help="Extra args passed to pytest (use `--` to separate).",
    )
    run_shard.set_defaults(func=_cmd_run_shard)

    args = parser.parse_args(argv)
    return args.func(args)
