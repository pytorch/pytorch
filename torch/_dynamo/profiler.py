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
# Trace Timing Profiler API
#
# This section provides APIs for measuring what percentage of compile time
# is spent tracing each user function.
# =============================================================================

import collections
import json
import threading
from typing import Literal, Optional


@dataclasses.dataclass
class TraceTiming:
    """Timing information for a single compiled function."""

    compile_id: str
    function_name: str
    filename: str
    lineno: int
    tracing_time_s: float
    total_compile_time_s: float
    tracing_percent: float
    attempt_count: int
    cache_size: int


# Thread-local scratchpad for collecting timing events during a single compile
_trace_profiler_tls = threading.local()

# Global storage for trace timing entries (bounded deque)
_trace_timing_entries: collections.deque[TraceTiming] = collections.deque(maxlen=256)
_trace_timing_lock = threading.Lock()


def _get_trace_profiler_scratchpad() -> dict[str, int]:
    """Get the thread-local scratchpad for collecting timing events."""
    if not hasattr(_trace_profiler_tls, "scratchpad"):
        _trace_profiler_tls.scratchpad = {}
    return _trace_profiler_tls.scratchpad


def _clear_trace_profiler_scratchpad() -> None:
    """Clear the thread-local scratchpad."""
    _trace_profiler_tls.scratchpad = {}


def _record_trace_timing(event_name: str, duration_us: int) -> None:
    """
    Record a timing event during compilation.

    Called from dynamo_timed for allowlisted events.
    """
    scratchpad = _get_trace_profiler_scratchpad()
    if event_name not in scratchpad:
        scratchpad[event_name] = 0
    scratchpad[event_name] += duration_us


def _flush_trace_timing(
    compile_id: str,
    co_name: str,
    co_filename: str,
    co_firstlineno: int,
    cache_size: int = 0,
    attempt_count: int = 1,
) -> None:
    """
    Flush the scratchpad into the global trace timing entries.

    Called from record_compilation_metrics at the end of compilation.
    """
    scratchpad = _get_trace_profiler_scratchpad()
    if not scratchpad:
        return

    tracing_us = scratchpad.get("bytecode_tracing", 0)
    total_us = scratchpad.get("entire_frame_compile", 0)

    # Compute percentage (avoid division by zero)
    if total_us > 0:
        tracing_percent = (tracing_us / total_us) * 100.0
    else:
        tracing_percent = 0.0

    entry = TraceTiming(
        compile_id=compile_id,
        function_name=co_name,
        filename=co_filename,
        lineno=co_firstlineno,
        tracing_time_s=tracing_us / 1_000_000.0,
        total_compile_time_s=total_us / 1_000_000.0,
        tracing_percent=tracing_percent,
        attempt_count=attempt_count,
        cache_size=cache_size,
    )

    with _trace_timing_lock:
        _trace_timing_entries.append(entry)

    _clear_trace_profiler_scratchpad()


def trace_breakdown(
    aggregate_by: Literal["compile_id", "frame"] = "compile_id",
) -> list[TraceTiming]:
    """
    Get per-function breakdown of tracing time vs total compile time.

    Args:
        aggregate_by: How to group results
            - "compile_id": One row per compilation (includes recompiles separately)
            - "frame": Aggregate all compiles of the same function

    Returns:
        List of TraceTiming entries, sorted by tracing_percent descending
    """
    with _trace_timing_lock:
        entries = list(_trace_timing_entries)

    if aggregate_by == "frame":
        # Aggregate by (filename, lineno, function_name)
        aggregated: dict[tuple[str, int, str], TraceTiming] = {}
        for e in entries:
            key = (e.filename, e.lineno, e.function_name)
            if key not in aggregated:
                aggregated[key] = TraceTiming(
                    compile_id=e.compile_id,  # Keep first compile_id
                    function_name=e.function_name,
                    filename=e.filename,
                    lineno=e.lineno,
                    tracing_time_s=0.0,
                    total_compile_time_s=0.0,
                    tracing_percent=0.0,
                    attempt_count=0,
                    cache_size=0,
                )
            agg = aggregated[key]
            agg.tracing_time_s += e.tracing_time_s
            agg.total_compile_time_s += e.total_compile_time_s
            agg.attempt_count += e.attempt_count
            agg.cache_size = max(agg.cache_size, e.cache_size)

        # Recompute percentages
        for agg in aggregated.values():
            if agg.total_compile_time_s > 0:
                agg.tracing_percent = (
                    agg.tracing_time_s / agg.total_compile_time_s
                ) * 100.0

        entries = list(aggregated.values())

    # Sort by tracing_percent descending
    entries.sort(key=lambda e: e.tracing_percent, reverse=True)
    return entries


def print_trace_breakdown(
    top_n: int = 20,
    aggregate_by: Literal["compile_id", "frame"] = "frame",
) -> None:
    """
    Print a formatted table of tracing time breakdown.

    Args:
        top_n: Maximum number of entries to print
        aggregate_by: How to group results ("compile_id" or "frame")
    """
    entries = trace_breakdown(aggregate_by=aggregate_by)[:top_n]

    if not entries:
        print("No trace timing data collected.")
        return

    # Calculate totals
    total_tracing = sum(e.tracing_time_s for e in entries)
    total_compile = sum(e.total_compile_time_s for e in entries)
    overall_percent = (total_tracing / total_compile * 100) if total_compile > 0 else 0

    print(f"\nDynamo Tracing Time Breakdown (top {top_n} by % time in tracing):\n")
    print(
        f"{'Function':<25} {'File:Line':<30} {'Tracing':>10} {'Total':>10} {'%':>8} {'Att':>5}"
    )
    print("-" * 95)

    for e in entries:
        # Shorten filename for display
        short_filename = e.filename.split("/")[-1] if "/" in e.filename else e.filename
        file_line = f"{short_filename}:{e.lineno}"
        if len(file_line) > 28:
            file_line = "..." + file_line[-25:]

        func_name = e.function_name
        if len(func_name) > 23:
            func_name = func_name[:20] + "..."

        print(
            f"{func_name:<25} {file_line:<30} {e.tracing_time_s:>9.3f}s {e.total_compile_time_s:>9.3f}s {e.tracing_percent:>7.1f}% {e.attempt_count:>5}"
        )

    print("-" * 95)
    print(
        f"{'Total':<25} {'':<30} {total_tracing:>9.3f}s {total_compile:>9.3f}s {overall_percent:>7.1f}%"
    )
    print()


def export_trace_breakdown(
    filepath: str,
    aggregate_by: Literal["compile_id", "frame"] = "compile_id",
) -> None:
    """
    Export trace breakdown to a JSON file for offline analysis.

    Args:
        filepath: Path to write JSON output
        aggregate_by: How to group results
    """
    entries = trace_breakdown(aggregate_by=aggregate_by)

    total_tracing = sum(e.tracing_time_s for e in entries)
    total_compile = sum(e.total_compile_time_s for e in entries)
    overall_percent = (total_tracing / total_compile * 100) if total_compile > 0 else 0

    data = {
        "version": 1,
        "entries": [dataclasses.asdict(e) for e in entries],
        "summary": {
            "total_tracing_s": total_tracing,
            "total_compile_s": total_compile,
            "overall_tracing_percent": overall_percent,
            "entry_count": len(entries),
        },
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def clear() -> None:
    """Clear all collected trace timing data."""
    with _trace_timing_lock:
        _trace_timing_entries.clear()
    _clear_trace_profiler_scratchpad()
