"""
Dynamo Profiler - tracks where Dynamo spends time during compilation.

This module provides profiling functionality for Dynamo tracing, showing per-function
cumtime (inclusive) and tottime (exclusive) in a cProfile-compatible format.
The output can be visualized with tools like snakeviz.

Usage:
    # Enable via config (prints pstats output):
    torch._dynamo.config.dynamo_profiler = True

    # Or save to file for snakeviz:
    torch._dynamo.config.dynamo_profiler = "/tmp/dynamo.prof"
    # Then: snakeviz /tmp/dynamo.prof
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import pstats


@dataclass
class FunctionTraceTiming:
    """
    Timing data for a single inlined function trace.

    Follows cProfile conventions:
    - cumtime: total time in function including all subcalls (inclusive)
    - tottime: time in function excluding subcalls (exclusive)
    - caller info: who called this function (for building call graph)
    """

    # Function identification
    func_name: str
    filename: str
    firstlineno: int
    # Timing data (in nanoseconds) - cProfile-style
    cumtime_ns: int  # Inclusive time (includes subcalls)
    tottime_ns: int  # Exclusive time (excludes subcalls)
    # Code stats (for comparing tracing overhead vs function complexity)
    bytecode_count: int
    # Nesting depth when this function was traced
    inline_depth: int
    # Caller information (for building call graph edges)
    caller_func_name: str | None = None
    caller_filename: str | None = None
    caller_firstlineno: int | None = None
    # Whether this is a primitive (non-recursive) call
    # A call is primitive if the function doesn't appear anywhere in the call stack
    is_primitive_call: bool = True
    # Full call stack at the time of this call (for proper snakeviz drill-down)
    # Each entry is (func_name, filename, firstlineno)
    call_stack: tuple[tuple[str, str, int], ...] = ()

    # Backwards compatibility alias
    @property
    def trace_time_ns(self) -> int:
        return self.cumtime_ns

    @property
    def trace_time_ms(self) -> float:
        return self.cumtime_ns / 1e6

    @property
    def cumtime_ms(self) -> float:
        return self.cumtime_ns / 1e6

    @property
    def tottime_ms(self) -> float:
        return self.tottime_ns / 1e6

    @property
    def caller_key(self) -> tuple[str, str, int] | None:
        """Return caller as a pstats-compatible key tuple."""
        if self.caller_func_name is not None:
            return (
                self.caller_filename or "",
                self.caller_firstlineno or 0,
                self.caller_func_name,
            )
        return None

    @property
    def func_key(self) -> tuple[str, int, str]:
        """Return this function as a pstats-compatible key tuple."""
        return (self.filename, self.firstlineno, self.func_name)

    def __repr__(self) -> str:
        return (
            f"FunctionTraceTiming({self.func_name} at {self.filename}:{self.firstlineno}, "
            f"cumtime={self.cumtime_ms:.2f}ms, tottime={self.tottime_ms:.2f}ms, "
            f"bytecode={self.bytecode_count}, depth={self.inline_depth})"
        )


@dataclass
class ProfilerStackEntry:
    """Stack entry for tracking function timing in the Dynamo profiler."""

    func_name: str
    filename: str
    firstlineno: int
    start_time_ns: int
    child_time_ns: int  # Accumulated time spent in children


class DynamoProfilerState:
    """State for Dynamo profiler tracking function trace timings."""

    def __init__(self) -> None:
        self.timings: list[FunctionTraceTiming] = []
        self.stack: list[ProfilerStackEntry] = []

    def record_timing(self, timing: FunctionTraceTiming) -> None:
        """Record timing data for a traced function."""
        self.timings.append(timing)

    def get_timings(self) -> list[FunctionTraceTiming]:
        """Get all recorded timings."""
        return self.timings

    def push(
        self, func_name: str, filename: str, firstlineno: int, start_time_ns: int
    ) -> bool:
        """Push a new entry onto the timing stack.

        Returns True if this is a primitive (non-recursive) call, i.e., the function
        does not already appear anywhere in the call stack.
        """
        # Check if this function already exists in the stack (indirect recursion)
        is_primitive = not any(
            entry.func_name == func_name
            and entry.filename == filename
            and entry.firstlineno == firstlineno
            for entry in self.stack
        )
        self.stack.append(
            ProfilerStackEntry(
                func_name=func_name,
                filename=filename,
                firstlineno=firstlineno,
                start_time_ns=start_time_ns,
                child_time_ns=0,
            )
        )
        return is_primitive

    def pop(self) -> ProfilerStackEntry | None:
        """Pop the top entry from the timing stack."""
        if self.stack:
            return self.stack.pop()
        return None

    def add_child_time(self, child_cumtime_ns: int) -> None:
        """Add the child's cumulative time to the parent's child_time accumulator."""
        if self.stack:
            self.stack[-1].child_time_ns += child_cumtime_ns

    def get_current_caller(self) -> tuple[str, str, int] | None:
        """Get the current caller (top of stack) as (func_name, filename, firstlineno)."""
        if self.stack:
            entry = self.stack[-1]
            return (entry.func_name, entry.filename, entry.firstlineno)
        return None

    def get_call_stack(self) -> tuple[tuple[str, str, int], ...]:
        """Get the full current call stack as tuple of (func_name, filename, firstlineno)."""
        return tuple(
            (entry.func_name, entry.filename, entry.firstlineno) for entry in self.stack
        )

    def generate_pstats(self, output_file: str | None = None) -> pstats.Stats:
        """Generate pstats.Stats object from recorded timings with call path encoding."""
        import cProfile
        import io
        import logging
        import pstats

        log = logging.getLogger(__name__)

        # Build entries keyed by full call path
        aggregated: dict[tuple[str, ...], dict[str, Any]] = {}
        caller_edges: dict[tuple[str, ...], dict[tuple[str, ...], list[float]]] = {}

        for t in self.timings:
            # Build the full path: call_stack + current function
            stack_names = tuple(entry[0] for entry in t.call_stack)
            full_path = stack_names + (t.func_name,)

            if full_path not in aggregated:
                aggregated[full_path] = {
                    "ncalls": 0,
                    "pcalls": 0,
                    "tottime": 0.0,
                    "cumtime": 0.0,
                    "filename": t.filename,
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
            log.info(
                "Saved pstats to %s. Visualize with: snakeviz %s",
                output_file,
                output_file,
            )

        return stats

    def dump_stats(self, output_file: str | None = None) -> None:
        """Print profiler stats to stdout and optionally save to file."""
        import sys

        if not self.timings:
            return

        stats = self.generate_pstats(output_file)
        print("\n=== Dynamo Profiler ===")
        stats.stream = sys.stdout
        stats.sort_stats("cumulative").print_stats()

        if output_file:
            print(f"\nProfile saved to: {output_file}")
            print(f"Visualize with: snakeviz {output_file}")
