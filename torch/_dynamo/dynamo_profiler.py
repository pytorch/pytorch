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
    def caller_key(self) -> tuple[str, int, str] | None:
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
    child_time_ns: int  # Accumulated time spent in traced children
    is_primitive_call: bool = True  # Whether this is a non-recursive call


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
    ) -> None:
        """Push a new entry onto the timing stack."""
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
                is_primitive_call=is_primitive,
            )
        )

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

    def generate_pstats(
        self, output_file: str | None = None, print_raw: bool = False
    ) -> pstats.Stats:
        """Generate pstats.Stats object from recorded timings.

        Args:
            output_file: Optional file path to save the stats.
            print_raw: If True, print raw aggregate timings before returning.
        """
        import cProfile
        import io
        import logging
        import pstats

        log = logging.getLogger(__name__)

        # Aggregate by (filename, lineno, func_name)
        aggregated: dict[tuple[str, int, str], dict[str, Any]] = {}
        # caller_edges[callee_key][caller_key] -> edge stats
        caller_edges: dict[
            tuple[str, int, str], dict[tuple[str, int, str], dict[str, Any]]
        ] = {}

        for t in self.timings:
            key = (t.filename, t.firstlineno, t.func_name)

            if key not in aggregated:
                aggregated[key] = {
                    "ncalls": 0,
                    "pcalls": 0,
                    "tottime": 0.0,
                    "cumtime": 0.0,
                }
                caller_edges[key] = {}

            agg = aggregated[key]
            agg["ncalls"] += 1
            agg["tottime"] += t.tottime_ns / 1e9

            if t.is_primitive_call:
                agg["pcalls"] += 1
                agg["cumtime"] += t.cumtime_ns / 1e9

            # Build caller edge
            if t.caller_filename is not None:
                caller_key = (
                    t.caller_filename,
                    t.caller_firstlineno or 0,
                    t.caller_func_name or "",
                )
                if caller_key not in caller_edges[key]:
                    caller_edges[key][caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                edge = caller_edges[key][caller_key]
                edge["ncalls"] += 1
                edge["tottime"] += t.tottime_ns / 1e9
                # Always add cumtime to edges for visualization (gprof2dot)
                # Function-level cumtime is already correct (only primitive calls)
                edge["cumtime"] += t.cumtime_ns / 1e9
                if t.is_primitive_call:
                    edge["pcalls"] += 1

        if print_raw:
            sorted_items = sorted(
                aggregated.items(), key=lambda x: x[1]["cumtime"], reverse=True
            )
            print("\n=== Aggregate Timings (raw) ===")
            print(
                f"{'ncalls':>8} {'pcalls':>8} {'tottime':>12} {'cumtime':>12}  function"
            )
            print("-" * 80)
            total_cumtime = 0.0
            total_tottime = 0.0
            for (filename, lineno, func_name), agg in sorted_items:
                ncalls = agg["ncalls"]
                pcalls = agg["pcalls"]
                tottime = agg["tottime"] * 1000  # convert to ms
                cumtime = agg["cumtime"] * 1000
                total_cumtime += cumtime
                total_tottime += tottime
                short_file = filename.split("/")[-1] if "/" in filename else filename
                print(
                    f"{ncalls:>8} {pcalls:>8} {tottime:>10.2f}ms {cumtime:>10.2f}ms  "
                    f"{func_name} ({short_file}:{lineno})"
                )
            print("-" * 80)
            print(
                f"Total timings: {len(self.timings)}, unique functions: {len(aggregated)}"
            )
            print(
                f"Sum tottime: {total_tottime:.2f}ms, Sum cumtime: {total_cumtime:.2f}ms"
            )

        # Ensure caller-only functions have a top-level entry.
        # gprof2dot expects every function referenced as a caller to also
        # exist as a top-level entry in the stats dict with timing data.
        for key in list(caller_edges.keys()):
            for caller_key in caller_edges[key]:
                if caller_key not in aggregated:
                    aggregated[caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                    caller_edges[caller_key] = {}

        # Build the stats dict in pstats format
        stats_dict: dict[
            tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]
        ] = {}

        for key, agg in aggregated.items():
            callers: dict[tuple[str, int, str], tuple[int, int, float, float]] = {}
            for caller_key, edge in caller_edges[key].items():
                callers[caller_key] = (
                    edge["ncalls"],
                    edge["pcalls"],
                    edge["tottime"],
                    edge["cumtime"],
                )

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

        stats.stats = stats_dict  # type: ignore[attr-defined]
        stats.total_calls = sum(s[1] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.prim_calls = sum(s[0] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.total_tt = sum(s[2] for s in stats_dict.values())  # type: ignore[attr-defined]

        if output_file:
            stats.dump_stats(output_file)
            log.info(
                "Saved pstats to %s. Visualize with: snakeviz %s",
                output_file,
                output_file,
            )

        return stats

    def generate_svg(
        self, profile_file: str, svg_file: str | None = None
    ) -> str | None:
        """Generate an SVG call graph from a profile file using gprof2dot and graphviz.

        Args:
            profile_file: Path to the pstats profile file.
            svg_file: Optional path for the output SVG. If not provided, uses
                profile_file with .svg extension.

        Returns:
            Path to the generated SVG file, or None if generation failed.
        """
        import os
        import shutil
        import subprocess

        if not shutil.which("gprof2dot"):
            print("gprof2dot not found. Install with: pip install gprof2dot")
            return None

        if not shutil.which("dot"):
            print("graphviz 'dot' not found. Install graphviz package.")
            return None

        if svg_file is None:
            svg_file = profile_file.rsplit(".", 1)[0] + ".svg"

        try:
            # gprof2dot -f pstats profile.prof | dot -Tsvg -o profile.svg
            gprof2dot = subprocess.Popen(
                [
                    "gprof2dot",
                    "-f",
                    "pstats",
                    "--node-label=total-time-percentage",
                    "--node-label=self-time-percentage",
                    "--node-label=total-time",
                    profile_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dot = subprocess.Popen(
                ["dot", "-Tsvg", "-o", svg_file],
                stdin=gprof2dot.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            gprof2dot.stdout.close()  # type: ignore[union-attr]
            _, dot_err = dot.communicate()
            _, gprof2dot_err = gprof2dot.communicate()

            if gprof2dot.returncode != 0:
                print(
                    f"gprof2dot failed: {gprof2dot_err.decode()}"  # noqa: B950
                )
                return None

            if dot.returncode != 0:
                print(f"graphviz dot failed: {dot_err.decode()}")
                return None

            if not os.path.isfile(svg_file):
                print(f"SVG file was not created: {svg_file}")
                return None

            print(f"SVG call graph saved to: {svg_file}")
            return svg_file

        except Exception as e:
            print(f"Failed to generate SVG: {e}")
            return None

    def dump_stats(
        self, output_file: str | None = None, generate_svg: bool = True
    ) -> None:
        """Print profiler stats to stdout and optionally save to file.

        Args:
            output_file: Optional path to save the pstats profile.
            generate_svg: If True and output_file is provided, also generate an SVG
                call graph using gprof2dot and graphviz.
        """
        import sys

        if not self.timings:
            return

        stats = self.generate_pstats(output_file, print_raw=True)
        print("\n=== Dynamo Profiler (pstats) ===")
        stats.stream = sys.stdout  # type: ignore[attr-defined]
        stats.sort_stats("cumulative").print_stats()

        if output_file:
            print(f"\nProfile saved to: {output_file}")
            print(f"Visualize with: snakeviz {output_file}")

            if generate_svg:
                self.generate_svg(output_file)
