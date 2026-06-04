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

import cProfile
import dataclasses
import functools
import logging
import os
import pstats
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec, Self

import torch
from torch._guards import CompileContext
from torch._utils_internal import maybe_upload_prof_stats_to_manifold

from . import config
from .utils import print_once


if TYPE_CHECKING:
    from collections.abc import Callable


log = logging.getLogger(__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _callable_name(func: Callable[..., Any]) -> str:
    if isinstance(func, functools.partial):
        return _callable_name(func.func)
    return getattr(func, "__name__", type(func).__name__)


def maybe_cprofile(func: Callable[_P, _T]) -> Callable[_P, _T]:
    if config.cprofile:
        return cprofile_wrapper(func)
    return func


def cprofile_wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]:
    func_name = _callable_name(func)

    @functools.wraps(func)
    def profile_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        trace_id = CompileContext.current_trace_id()
        if not trace_id:
            raise AssertionError("Trace id is None")
        profile_path = Path(
            os.path.join(
                tempfile.gettempdir(),
                f"{func_name}_{str(trace_id).replace('/', '_')}.profile",
            )
        )
        prof = cProfile.Profile()
        try:
            start_ts = time.time()
            # runcall calls prof.enable() and prof.disable(), so do NOT call
            # enable outside. This leads to issues like
            # ValueError: Another profiling tool is already active
            # pyrefly: ignore [bad-argument-type]
            retval = prof.runcall(func, *args, **kwargs)
            profile_latency = time.time() - start_ts
        except ValueError:
            log.exception("failed to enable cProfile")
            profile_latency = 0
            retval = func(*args, **kwargs)
        log.warning(
            "### Cprofile for %s trace id [%s] took %.3f seconds ###",
            func_name,
            trace_id,
            profile_latency,
        )
        ps = pstats.Stats(prof)
        try:
            prof.dump_stats(profile_path)
        except OSError:
            log.exception("Cannot write to %s", profile_path)
        log.warning("Raw profile at %s", profile_path)
        svg_path = profile_path.with_suffix(".svg")
        try:
            with subprocess.Popen(
                [
                    "gprof2dot",
                    "-f",
                    "pstats",
                    "--node-label=total-time-percentage",
                    "--node-label=self-time-percentage",
                    "--node-label=total-time",
                    str(profile_path),
                ],
                stdout=subprocess.PIPE,
            ) as gprof2dot_process:
                subprocess.check_call(
                    ["dot", "-Tsvg", "-o", str(svg_path)],
                    stdin=gprof2dot_process.stdout,
                )
                log.warning("Generated SVG from profile at %s", svg_path)
        except FileNotFoundError:
            log.warning(
                "Failed to generate SVG from profile -- dumping stats instead."
                "Try installing gprof2dot and dot for a better visualization"
            )
            ps.sort_stats(pstats.SortKey.TIME).print_stats(20)
            ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

        if manifold_link := maybe_upload_prof_stats_to_manifold(
            str(profile_path)
        ):  # fb-only
            torch._logging.trace_structured(
                "link",
                lambda: {"name": "cprofile_manifold_url", "url": manifold_link},
            )
        return retval

    return profile_wrapper


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
        if not isinstance(other, ProfileMetrics):
            raise AssertionError(f"Expected ProfileMetrics, got {type(other)}")
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
