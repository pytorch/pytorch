"""
Dynamo profiling implementation.

This module provides profiling functionality for Dynamo, including:
- ProfileMetrics: Class for collecting and aggregating performance metrics like
  execution time, operator counts, and fusion statistics
- ProfileResult: Class for analyzing and reporting profiling results
- Utilities for tracking missed/uncaptured operations
- Functions for instrumenting FX graphs with profiling capabilities

The profiler helps measure and optimize the performance of Dynamo-compiled code
by tracking both captured and total operations, timing, and graph statistics.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any
from typing_extensions import Self

import torch

from .utils import print_once


@dataclasses.dataclass
class ProfileMetrics:
    microseconds: float = 0.0
    operators: int = 0
    fusions: int = 0
    graphs: int = 0

    def __iadd__(self, other: Self) -> Self:
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: ProfileMetrics) -> ProfileMetrics:
        assert isinstance(other, ProfileMetrics)
        return ProfileMetrics(
            self.microseconds + other.microseconds,
            self.operators + other.operators,
            self.fusions + other.fusions,
        )

    def __truediv__(self, other: Any) -> ProfileMetrics:
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(
            self.microseconds / max(1, other.microseconds),
            # pyrefly: ignore [bad-argument-type]
            self.operators / max(1, other.operators),
            # pyrefly: ignore [bad-argument-type]
            self.fusions / max(1, other.fusions),
        )

    def __str__(self) -> str:
        return f"{self.operators:4.0%} ops {self.microseconds:4.0%} time"

    def tocsv(self) -> list[float]:
        return [self.operators, self.microseconds]


class ProfileResult:
    def __init__(
        self, captured: ProfileMetrics, total: ProfileMetrics, unique_graphs: int
    ) -> None:
        self.captured: ProfileMetrics = captured or ProfileMetrics()
        self.total: ProfileMetrics = total or ProfileMetrics()
        self.unique_graphs: int = unique_graphs

    def __iadd__(self, other: Self) -> Self:
        self.captured += other.captured
        self.total += other.total
        self.unique_graphs += other.unique_graphs
        return self

    def percent(self) -> ProfileMetrics:
        return self.captured / self.total

    def __str__(self) -> str:
        return (
            f"{self.unique_graphs:2} graphs {self.captured.graphs:2} graph calls "
            f"{self.captured.operators:4}/{self.total.operators:4} = "
            + str(self.percent())
        )

    def tocsv(self) -> list[Any]:
        return [
            self.unique_graphs,
            self.captured.graphs,
            self.captured.operators,
            self.total.operators,
        ] + self.percent().tocsv()


def should_print_missing() -> bool:
    return os.environ.get("TORCHDYNAMO_PRINT_MISSING") == "1"


def print_missing(stack: list[str]) -> None:
    if any("/torch/autograd/profiler.py" in x for x in stack):
        return
    stack = [
        x for x in stack if ("<built-in" not in x and "site-packages/torch/" not in x)
    ]
    print_once("MISSING", " >> ".join(stack[-3:]))


class Profiler:
    unique_graphs: int = 0

    def __init__(self) -> None:
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_stack=should_print_missing(),
        )

    def results(self) -> ProfileResult:
        captured_regions = 0
        captured_ops = 0
        captured_microseconds = 0
        total_ops = 0
        total_microseconds = 0

        last_op_end_time = -1
        captured_region_end_time = -1
        events = sorted(self.prof.events(), key=lambda x: x.time_range.start)
        for e in events:
            if e.name == "TORCHDYNAMO":
                captured_region_end_time = e.time_range.end
                captured_regions += 1
                # ignore `handle = torch.zeros(1)` in record_function.__init__()
                total_ops -= 1
            elif e.time_range.start >= last_op_end_time:
                last_op_end_time = e.time_range.end
                if e.time_range.end <= captured_region_end_time:
                    captured_ops += 1
                    captured_microseconds += e.time_range.elapsed_us()
                elif should_print_missing():
                    print_missing(e.stack)
                total_ops += 1
                total_microseconds += e.time_range.elapsed_us()
            else:
                pass  # ops recursively called from other ops (ignored)

        unique_graphs = Profiler.unique_graphs
        Profiler.unique_graphs = 0
        # we counted one extra op that is part of the profiler setup code
        total_ops -= 1

        return ProfileResult(
            captured=ProfileMetrics(
                microseconds=captured_microseconds,
                operators=captured_ops,
                fusions=captured_ops - captured_regions,
                graphs=captured_regions,
            ),
            total=ProfileMetrics(
                microseconds=total_microseconds,
                operators=total_ops,
                fusions=total_ops - 1,
            ),
            unique_graphs=unique_graphs,
        )


def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: list[Any]) -> Any:
    def _wrapped(*args: Any) -> Any:
        with torch.profiler.record_function("TORCHDYNAMO"):
            return gm.forward(*args)

    Profiler.unique_graphs += 1
    return _wrapped


# =============================================================================
# Dynamo Tracing Profiler
#
# Tracks where Dynamo spends time during compilation, showing per-function
# cumtime (inclusive) and tottime (exclusive) in a cProfile-compatible format.
# The output can be visualized with tools like snakeviz.
#
# Usage:
#     # Enable via config (prints pstats output):
#     torch._dynamo.config.dynamo_profiler = True
#
#     # Or save to file for snakeviz:
#     torch._dynamo.config.dynamo_profiler = "/tmp/dynamo.prof"
#     # Then: snakeviz /tmp/dynamo.prof
# =============================================================================

import logging
import sys
from typing import TYPE_CHECKING

from torch._guards import FunctionTraceTiming, TracingContext


if TYPE_CHECKING:
    import pstats

_profiler_log = logging.getLogger(__name__)


def get_function_trace_timings() -> list[FunctionTraceTiming] | None:
    """
    Get timing data for all functions traced during compilation.

    Returns a list of FunctionTraceTiming objects containing:
    - func_name: Name of the function
    - filename: Source file path
    - firstlineno: First line number in source
    - cumtime_ns: Inclusive time in nanoseconds
    - tottime_ns: Exclusive time in nanoseconds
    - bytecode_count: Size of function bytecode
    - inline_depth: Nesting depth when traced

    Returns None if no TracingContext is active.
    """
    return TracingContext.get_function_trace_timings()


def _shorten_filename(filepath: str, max_len: int = 35) -> str:
    """
    Shorten a filepath to include parent directory and filename.

    Examples:
        /data/users/foo/pytorch/torch/_dynamo/utils.py -> _dynamo/utils.py
        /very/long/path/to/some/module/file.py -> module/file.py
    """
    if "/" not in filepath:
        return (
            filepath if len(filepath) <= max_len else ".." + filepath[-(max_len - 2) :]
        )

    parts = filepath.rsplit("/", 2)
    if len(parts) >= 2:
        # Get parent_dir/filename
        short = "/".join(parts[-2:])
    else:
        short = parts[-1]

    if len(short) <= max_len:
        return short
    # If still too long, truncate from the left
    return ".." + short[-(max_len - 2) :]


def format_function_trace_timings(
    timings: list[FunctionTraceTiming] | None = None,
    sort_by: str = "cumtime",
    top_n: int | None = None,
) -> str:
    """
    Format function trace timing data as a human-readable string.

    Args:
        timings: List of FunctionTraceTiming objects. If None, fetches from current TracingContext.
        sort_by: How to sort results. Options: "cumtime", "tottime", "bytecode", "depth"
        top_n: If provided, only show top N entries

    Returns:
        Formatted string with timing information, sorted by the specified metric.
    """
    if timings is None:
        timings = get_function_trace_timings()

    if not timings:
        return "No function trace timing data available."

    # Sort based on requested metric
    if sort_by == "cumtime" or sort_by == "trace_time":
        timings = sorted(timings, key=lambda x: x.cumtime_ns, reverse=True)
    elif sort_by == "tottime":
        timings = sorted(timings, key=lambda x: x.tottime_ns, reverse=True)
    elif sort_by == "bytecode":
        timings = sorted(timings, key=lambda x: x.bytecode_count, reverse=True)
    elif sort_by == "depth":
        timings = sorted(timings, key=lambda x: x.inline_depth, reverse=True)

    if top_n is not None:
        timings = timings[:top_n]

    # Format output - similar to pstats format
    lines = ["Function Trace Times:"]
    lines.append("-" * 130)
    lines.append(
        f"{'Function':<35} {'File:Line':<30} {'cumtime':>10} {'tottime':>10} {'Bytecode':>8} {'Depth':>5} {'Caller':<25}"
    )
    lines.append("-" * 130)

    for t in timings:
        func_name = t.func_name[:33] + ".." if len(t.func_name) > 35 else t.func_name
        short_path = _shorten_filename(t.filename, max_len=22)
        file_loc = f"{short_path}:{t.firstlineno}"
        if len(file_loc) > 30:
            file_loc = ".." + file_loc[-28:]

        caller_str = ""
        if t.caller_func_name:
            caller_str = (
                t.caller_func_name[:23] + ".."
                if len(t.caller_func_name) > 25
                else t.caller_func_name
            )

        lines.append(
            f"{func_name:<35} {file_loc:<30} {t.cumtime_ms:>10.2f} {t.tottime_ms:>10.2f} "
            f"{t.bytecode_count:>8} {t.inline_depth:>5} {caller_str:<25}"
        )

    lines.append("-" * 130)
    lines.append(f"Total functions traced: {len(timings)}")
    total_cumtime_ms = sum(t.cumtime_ns for t in timings) / 1e6
    total_tottime_ms = sum(t.tottime_ns for t in timings) / 1e6
    lines.append(
        f"Total cumtime: {total_cumtime_ms:.2f}ms, Total tottime: {total_tottime_ms:.2f}ms"
    )

    return "\n".join(lines)


def format_function_trace_timings_aggregated(
    timings: list[FunctionTraceTiming] | None = None,
    sort_by: str = "cumtime",
    top_n: int | None = None,
) -> str:
    """
    Format function trace timing data aggregated by unique function.

    This combines multiple calls to the same function into a single entry,
    showing total cumtime, tottime, call count, and average time per call.
    Similar to pstats aggregated view.

    Args:
        timings: List of FunctionTraceTiming objects. If None, fetches from current TracingContext.
        sort_by: How to sort results. Options: "cumtime", "tottime", "count", "avg_time"
        top_n: If provided, only show top N entries

    Returns:
        Formatted string with aggregated timing information.
    """
    if timings is None:
        timings = get_function_trace_timings()

    if not timings:
        return "No function trace timing data available."

    # Aggregate by (func_name, filename, firstlineno)
    aggregated: dict[tuple[str, str, int], dict[str, Any]] = {}
    for t in timings:
        key = (t.func_name, t.filename, t.firstlineno)
        if key not in aggregated:
            aggregated[key] = {
                "func_name": t.func_name,
                "filename": t.filename,
                "firstlineno": t.firstlineno,
                "cumtime_ns": 0,
                "tottime_ns": 0,
                "bytecode_count": t.bytecode_count,
                "call_count": 0,
                "prim_count": 0,  # primitive (non-recursive) calls
                "min_depth": t.inline_depth,
                "max_depth": t.inline_depth,
            }
        agg = aggregated[key]
        agg["tottime_ns"] += t.tottime_ns  # tottime always added (exclusive)
        agg["call_count"] += 1
        agg["min_depth"] = min(agg["min_depth"], t.inline_depth)
        agg["max_depth"] = max(agg["max_depth"], t.inline_depth)

        # Use the is_primitive_call flag to detect recursion (including indirect)
        if t.is_primitive_call:
            agg["prim_count"] += 1
            # Only add cumtime for primitive calls to avoid double counting
            agg["cumtime_ns"] += t.cumtime_ns

    # Convert to list
    agg_list = list(aggregated.values())

    # Sort based on requested metric
    if sort_by == "cumtime" or sort_by == "total_time":
        agg_list.sort(key=lambda x: x["cumtime_ns"], reverse=True)
    elif sort_by == "tottime":
        agg_list.sort(key=lambda x: x["tottime_ns"], reverse=True)
    elif sort_by == "count":
        agg_list.sort(key=lambda x: x["call_count"], reverse=True)
    elif sort_by == "avg_time":
        agg_list.sort(key=lambda x: x["cumtime_ns"] / x["call_count"], reverse=True)

    if top_n is not None:
        agg_list = agg_list[:top_n]

    # Format output - similar to pstats
    lines = ["Function Trace Times (Aggregated):"]
    lines.append("-" * 130)
    lines.append(
        f"{'ncalls':>9} {'tottime':>10} {'percall':>10} {'cumtime':>10} {'percall':>10}  {'filename:lineno(function)':<58}"
    )
    lines.append("-" * 130)

    for agg in agg_list:
        ncalls = agg["call_count"]
        pcalls = agg["prim_count"]
        tottime_s = agg["tottime_ns"] / 1e9
        cumtime_s = agg["cumtime_ns"] / 1e9
        # percall for cumtime uses primitive calls (like pstats)
        percall_tot = tottime_s / ncalls if ncalls > 0 else 0
        percall_cum = cumtime_s / pcalls if pcalls > 0 else 0

        # Format ncalls as "total/prim" if there are recursive calls
        if ncalls != pcalls:
            ncalls_str = f"{ncalls}/{pcalls}"
        else:
            ncalls_str = str(ncalls)

        short_path = _shorten_filename(agg["filename"], max_len=35)
        func_loc = f"{short_path}:{agg['firstlineno']}({agg['func_name']})"
        if len(func_loc) > 58:
            func_loc = ".." + func_loc[-56:]

        lines.append(
            f"{ncalls_str:>9} {tottime_s:>10.6f} {percall_tot:>10.6f} {cumtime_s:>10.6f} {percall_cum:>10.6f}  {func_loc:<58}"
        )

    lines.append("-" * 130)
    prim_calls = sum(agg["prim_count"] for agg in aggregated.values())
    lines.append(
        f"Unique functions: {len(aggregated)}, Total calls: {len(timings)} ({prim_calls} primitive)"
    )
    total_tottime_ms = sum(agg["tottime_ns"] for agg in aggregated.values()) / 1e6
    lines.append(f"Total tottime: {total_tottime_ms:.2f}ms")

    return "\n".join(lines)


def _generate_pstats_with_call_paths(
    timings: list[FunctionTraceTiming],
    output_file: str | None = None,
) -> "pstats.Stats":
    """
    Generate pstats with call paths encoded in function names.

    This creates separate entries for each unique call path, making snakeviz
    correctly show per-caller timing when drilling down.
    """
    import cProfile
    import io
    import pstats

    # Build entries keyed by full call path
    # Key: tuple of func names in the call stack + current func
    aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
    # Track caller->callee edges based on call paths
    caller_edges: dict[tuple[str, ...], dict[tuple[str, ...], list[float]]] = {}

    for t in timings:
        # Build the full path: call_stack + current function
        stack_names = tuple(entry[0] for entry in t.call_stack)  # func names only
        full_path = stack_names + (t.func_name,)

        if full_path not in aggregated:
            aggregated[full_path] = {
                "ncalls": 0,
                "pcalls": 0,
                "tottime": 0.0,
                "cumtime": 0.0,
                "filename": _shorten_filename(t.filename, max_len=50),
                "lineno": t.firstlineno,
                "func_name": t.func_name,
            }
            caller_edges[full_path] = {}

        agg = aggregated[full_path]
        agg["ncalls"] += 1
        agg["tottime"] += t.tottime_ns / 1e9

        if t.is_primitive_call:
            agg["pcalls"] += 1
            agg["cumtime"] += t.cumtime_ns / 1e9

        # Build caller edge (parent path -> this path)
        if stack_names:
            if stack_names not in caller_edges[full_path]:
                caller_edges[full_path][stack_names] = [0, 0, 0.0, 0.0]
            edge = caller_edges[full_path][stack_names]
            edge[0] += 1
            if t.is_primitive_call:
                edge[1] += 1
            edge[2] += t.tottime_ns / 1e9
            edge[3] += t.cumtime_ns / 1e9

    # Build the stats dict in pstats format
    # Use path-encoded function names for proper drill-down
    stats_dict: dict[
        tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]
    ] = {}

    for path, agg in aggregated.items():
        # Encode the path in the function name
        if len(path) > 1:
            path_prefix = "->".join(path[:-1]) + "->"
            display_name = path_prefix + agg["func_name"]
        else:
            display_name = agg["func_name"]

        key = (agg["filename"], agg["lineno"], display_name)

        # Build callers dict for this entry
        callers: dict[tuple[str, int, str], tuple[int, int, float, float]] = {}
        for caller_path, edge_data in caller_edges[path].items():
            if caller_path in aggregated:
                caller_agg = aggregated[caller_path]
                if len(caller_path) > 1:
                    caller_prefix = "->".join(caller_path[:-1]) + "->"
                    caller_display = caller_prefix + caller_agg["func_name"]
                else:
                    caller_display = caller_agg["func_name"]
                caller_key = (
                    caller_agg["filename"],
                    caller_agg["lineno"],
                    caller_display,
                )
                callers[caller_key] = tuple(edge_data)  # type: ignore[assignment]

        stats_dict[key] = (
            agg["pcalls"],
            agg["ncalls"],
            agg["tottime"],
            agg["cumtime"],
            callers,
        )

    # Create a pstats.Stats object
    # We need to create a dummy profile to get a valid Stats object
    dummy_profile = cProfile.Profile()
    dummy_profile.enable()
    dummy_profile.disable()
    stats = pstats.Stats(dummy_profile, stream=io.StringIO())

    stats.stats = stats_dict
    stats.total_calls = sum(s[1] for s in stats_dict.values())
    stats.prim_calls = sum(s[0] for s in stats_dict.values())
    stats.total_tt = sum(s[2] for s in stats_dict.values())

    if output_file:
        stats.dump_stats(output_file)
        _profiler_log.info(
            "Saved pstats (with call paths) to %s. Visualize with: snakeviz %s",
            output_file,
            output_file,
        )

    return stats


def generate_pstats_from_timings(
    timings: list[FunctionTraceTiming],
    output_file: str | None = None,
    use_call_paths: bool = False,
) -> "pstats.Stats":
    """
    Generate a pstats.Stats-compatible object from function trace timings.

    This allows visualization with tools like snakeviz, pyprof2calltree, gprof2dot, etc.
    Includes proper tottime (exclusive), cumtime (inclusive), and caller-callee edges.

    Args:
        timings: List of FunctionTraceTiming from Dynamo tracing
        output_file: If provided, save the stats to this file (can be loaded with pstats.Stats(file))
        use_call_paths: If True, generate separate entries for each unique call path.
            This makes snakeviz correctly show per-caller timing when drilling down.
            Function names will be prefixed with their call path (e.g., "main->caller_a->common_fn").

    Returns:
        A pstats.Stats object that can be used for visualization

    Usage:
        stats = generate_pstats_from_timings(timings, "dynamo_trace.prof")
        # Visualize with: snakeviz dynamo_trace.prof
        # Or print stats:
        stats.sort_stats('cumulative').print_stats()
        stats.print_callers()  # Show who called each function
        stats.print_callees()  # Show what each function called
    """
    import cProfile
    import pstats

    if use_call_paths:
        return _generate_pstats_with_call_paths(timings, output_file)

    # Aggregate timings by function and build caller edges
    # pstats key format: (filename, lineno, funcname)
    aggregated: dict[tuple[str, int, str], dict[str, Any]] = {}
    # Track caller->callee edges: callers[callee_key][caller_key] = [nc, cc, tt, ct]
    caller_edges: dict[
        tuple[str, int, str], dict[tuple[str, int, str], list[float]]
    ] = {}

    for t in timings:
        short_filename = _shorten_filename(t.filename, max_len=50)
        key = (short_filename, t.firstlineno, t.func_name)

        if key not in aggregated:
            aggregated[key] = {
                "ncalls": 0,  # total calls
                "pcalls": 0,  # primitive (non-recursive) calls
                "tottime": 0.0,
                "cumtime": 0.0,
            }
            caller_edges[key] = {}

        agg = aggregated[key]
        agg["ncalls"] += 1
        agg["tottime"] += t.tottime_ns / 1e9  # tottime is always added (exclusive)

        # Use the is_primitive_call flag to detect recursion (including indirect)
        if t.is_primitive_call:
            agg["pcalls"] += 1
            # Only add cumtime for primitive calls to avoid double counting
            # The primitive call's cumtime already includes all recursive calls
            agg["cumtime"] += t.cumtime_ns / 1e9

        # Build caller edge if we have caller info
        if t.caller_func_name is not None:
            caller_short = _shorten_filename(t.caller_filename or "", max_len=50)
            caller_key = (caller_short, t.caller_firstlineno or 0, t.caller_func_name)

            if caller_key not in caller_edges[key]:
                caller_edges[key][caller_key] = [0, 0, 0.0, 0.0]  # nc, cc, tt, ct

            edge = caller_edges[key][caller_key]
            edge[0] += 1  # nc (total calls from this caller)
            if t.is_primitive_call:
                edge[1] += 1  # cc (primitive calls)
            edge[2] += t.tottime_ns / 1e9  # tt
            edge[3] += t.cumtime_ns / 1e9  # ct

    # Build the stats dict in pstats format
    # Format: {(filename, lineno, funcname): (cc, nc, tottime, cumtime, callers)}
    # cc = primitive calls, nc = total calls
    # When cc != nc, pstats displays as "nc/cc" (e.g., "50/5" for recursive)
    # callers format: {caller_key: (nc, cc, tt, ct)}
    stats_dict: dict[
        tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]
    ] = {}

    for key, agg in aggregated.items():
        # Convert caller edge lists to tuples
        callers_dict = {
            caller_key: tuple(edge_data)
            for caller_key, edge_data in caller_edges[key].items()
        }
        stats_dict[key] = (
            agg["pcalls"],  # cc (primitive/non-recursive calls)
            agg["ncalls"],  # nc (total calls)
            agg["tottime"],  # tottime (exclusive)
            agg["cumtime"],  # cumtime (inclusive)
            callers_dict,
        )

    # Create a Stats object
    pr = cProfile.Profile()
    pr.enable()
    pr.disable()
    stats = pstats.Stats(pr, stream=sys.stdout)

    # Replace the stats with our data
    stats.stats = stats_dict
    stats.total_calls = sum(s[1] for s in stats_dict.values())
    stats.prim_calls = sum(s[0] for s in stats_dict.values())
    stats.total_tt = sum(s[2] for s in stats_dict.values())

    if output_file:
        stats.dump_stats(output_file)
        _profiler_log.info(
            "Saved pstats to %s. Visualize with: snakeviz %s", output_file, output_file
        )

    return stats


def generate_flamegraph_from_timings(
    timings: list[FunctionTraceTiming],
    output_file: str | None = None,
) -> str:
    """
    Generate flamegraph-compatible collapsed stack format from function trace timings.

    This format can be used with tools like:
    - speedscope (https://speedscope.app) - just paste the output
    - flamegraph.pl (https://github.com/brendangregg/FlameGraph)

    Args:
        timings: List of FunctionTraceTiming from Dynamo tracing
        output_file: If provided, save to this file

    Returns:
        String in collapsed stack format: "func1;func2;func3 time_us"
    """
    # Group by inline_depth to reconstruct call stacks
    # Sort by the order they were recorded (which reflects call order)
    lines = []

    # Build a simple representation: each function with its depth
    for t in timings:
        # Simplified stack: just the function at its depth
        short_filename = (
            t.filename.rsplit("/", 1)[-1] if "/" in t.filename else t.filename
        )
        func_id = f"{short_filename}:{t.func_name}"

        # Create a pseudo-stack based on depth
        stack = ";".join(["<trace>"] * t.inline_depth + [func_id])
        time_us = t.trace_time_ns // 1000

        lines.append(f"{stack} {time_us}")

    output = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        _profiler_log.info("Saved flamegraph data to %s", output_file)

    return output


def dump_dynamo_profiler_stats() -> None:
    """Dump dynamo profiler stats if enabled."""
    from torch._dynamo import config

    timings = TracingContext.get_function_trace_timings()
    if not timings:
        return

    # Determine output file path if config is a string
    output_file = None
    if isinstance(config.dynamo_profiler, str):
        output_file = config.dynamo_profiler

    # Generate and print pstats
    stats = generate_pstats_from_timings(timings, output_file)
    print("\n=== Dynamo Profiler ===")
    stats.sort_stats("cumulative").print_stats(30)

    if output_file:
        print(f"\nProfile saved to: {output_file}")
        print(f"Visualize with: snakeviz {output_file}")
