# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import functools
import os
import re
import subprocess
import sys
import types
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache, reduce
from os import sep
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hypothesis._settings import Phase, Verbosity
from hypothesis.internal.compat import PYPY
from hypothesis.internal.escalation import is_hypothesis_file

if TYPE_CHECKING:
    from typing import TypeAlias
else:
    TypeAlias = object

Location: TypeAlias = tuple[str, int]
Branch: TypeAlias = tuple[Optional[Location], Location]
Trace: TypeAlias = set[Branch]


@lru_cache(maxsize=None)
def should_trace_file(fname: str) -> bool:
    # fname.startswith("<") indicates runtime code-generation via compile,
    # e.g. compile("def ...", "<string>", "exec") in e.g. attrs methods.
    return not (is_hypothesis_file(fname) or fname.startswith("<"))


# where possible, we'll use 3.12's new sys.monitoring module for low-overhead
# coverage instrumentation; on older python versions we'll use sys.settrace.
# tool_id = 1 is designated for coverage, but we intentionally choose a
# non-reserved tool id so we can co-exist with coverage tools.
MONITORING_TOOL_ID = 3
if sys.version_info[:2] >= (3, 12):
    MONITORING_EVENTS = {sys.monitoring.events.LINE: "trace_line"}


class Tracer:
    """A super-simple branch coverage tracer."""

    __slots__ = ("branches", "_previous_location")

    def __init__(self) -> None:
        self.branches: Trace = set()
        self._previous_location: Optional[Location] = None

    @staticmethod
    def can_trace() -> bool:
        return (
            (sys.version_info[:2] < (3, 12) and sys.gettrace() is None)
            or (
                sys.version_info[:2] >= (3, 12)
                and sys.monitoring.get_tool(MONITORING_TOOL_ID) is None
            )
        ) and not PYPY

    def trace(self, frame, event, arg):
        try:
            if event == "call":
                return self.trace
            elif event == "line":
                # manual inlining of self.trace_line for performance.
                fname = frame.f_code.co_filename
                if should_trace_file(fname):
                    current_location = (fname, frame.f_lineno)
                    self.branches.add((self._previous_location, current_location))
                    self._previous_location = current_location
        except RecursionError:
            pass

    def trace_line(self, code: types.CodeType, line_number: int) -> None:
        fname = code.co_filename
        if should_trace_file(fname):
            current_location = (fname, line_number)
            self.branches.add((self._previous_location, current_location))
            self._previous_location = current_location

    def __enter__(self):
        assert self.can_trace()  # caller checks in core.py

        if sys.version_info[:2] < (3, 12):
            sys.settrace(self.trace)
            return self

        sys.monitoring.use_tool_id(MONITORING_TOOL_ID, "scrutineer")
        for event, callback_name in MONITORING_EVENTS.items():
            sys.monitoring.set_events(MONITORING_TOOL_ID, event)
            callback = getattr(self, callback_name)
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, callback)

        return self

    def __exit__(self, *args, **kwargs):
        if sys.version_info[:2] < (3, 12):
            sys.settrace(None)
            return

        sys.monitoring.free_tool_id(MONITORING_TOOL_ID)
        for event in MONITORING_EVENTS:
            sys.monitoring.register_callback(MONITORING_TOOL_ID, event, None)


UNHELPFUL_LOCATIONS = (
    # There's a branch which is only taken when an exception is active while exiting
    # a contextmanager; this is probably after the fault has been triggered.
    # Similar reasoning applies to a few other standard-library modules: even
    # if the fault was later, these still aren't useful locations to report!
    # Note: The list is post-processed, so use plain "/" for separator here.
    "/contextlib.py",
    "/inspect.py",
    "/re.py",
    "/re/__init__.py",  # refactored in Python 3.11
    "/warnings.py",
    # Quite rarely, the first AFNP line is in Pytest's internals.
    "/_pytest/_io/saferepr.py",
    "/_pytest/assertion/*.py",
    "/_pytest/config/__init__.py",
    "/_pytest/pytester.py",
    "/pluggy/_*.py",
    "/reprlib.py",
    "/typing.py",
    "/conftest.py",
)


def _glob_to_re(locs: Iterable[str]) -> str:
    """Translate a list of glob patterns to a combined regular expression.
    Only the * wildcard is supported, and patterns including special
    characters will only work by chance."""
    # fnmatch.translate is not an option since its "*" consumes path sep
    return "|".join(
        loc.replace("*", r"[^/]+")
        .replace(".", re.escape("."))
        .replace("/", re.escape(sep))
        + r"\Z"  # right anchored
        for loc in locs
    )


def get_explaining_locations(traces):
    # Traces is a dict[interesting_origin | None, set[frozenset[tuple[str, int]]]]
    # Each trace in the set might later become a Counter instead of frozenset.
    if not traces:
        return {}

    unions = {origin: set().union(*values) for origin, values in traces.items()}
    seen_passing = {None}.union(*unions.pop(None, set()))

    always_failing_never_passing = {
        origin: reduce(set.intersection, [set().union(*v) for v in values])
        - seen_passing
        for origin, values in traces.items()
        if origin is not None
    }

    # Build the observed parts of the control-flow graph for each origin
    cf_graphs = {origin: defaultdict(set) for origin in unions}
    for origin, seen_arcs in unions.items():
        for src, dst in seen_arcs:
            cf_graphs[origin][src].add(dst)
        assert cf_graphs[origin][None], "Expected start node with >=1 successor"

    # For each origin, our explanation is the always_failing_never_passing lines
    # which are reachable from the start node (None) without passing through another
    # AFNP line.  So here's a whatever-first search with early stopping:
    explanations = defaultdict(set)
    for origin in unions:
        queue = {None}
        seen = set()
        while queue:
            assert queue.isdisjoint(seen), f"Intersection: {queue & seen}"
            src = queue.pop()
            seen.add(src)
            if src in always_failing_never_passing[origin]:
                explanations[origin].add(src)
            else:
                queue.update(cf_graphs[origin][src] - seen)

    # The last step is to filter out explanations that we know would be uninformative.
    # When this is the first AFNP location, we conclude that Scrutineer missed the
    # real divergence (earlier in the trace) and drop that unhelpful explanation.
    filter_regex = re.compile(_glob_to_re(UNHELPFUL_LOCATIONS))
    return {
        origin: {loc for loc in afnp_locs if not filter_regex.search(loc[0])}
        for origin, afnp_locs in explanations.items()
    }


LIB_DIR = str(Path(sys.executable).parent / "lib")
EXPLANATION_STUB = (
    "Explanation:",
    "    These lines were always and only run by failing examples:",
)


def make_report(explanations, cap_lines_at=5):
    report = defaultdict(list)
    for origin, locations in explanations.items():
        report_lines = [f"        {fname}:{lineno}" for fname, lineno in locations]
        report_lines.sort(key=lambda line: (line.startswith(LIB_DIR), line))
        if len(report_lines) > cap_lines_at + 1:
            msg = "        (and {} more with settings.verbosity >= verbose)"
            report_lines[cap_lines_at:] = [msg.format(len(report_lines[cap_lines_at:]))]
        if report_lines:  # We might have filtered out every location as uninformative.
            report[origin] = list(EXPLANATION_STUB) + report_lines
    return report


def explanatory_lines(traces, settings):
    if Phase.explain in settings.phases and sys.gettrace() and not traces:
        return defaultdict(list)
    # Return human-readable report lines summarising the traces
    explanations = get_explaining_locations(traces)
    max_lines = 5 if settings.verbosity <= Verbosity.normal else float("inf")
    return make_report(explanations, cap_lines_at=max_lines)


# beware the code below; we're using some heuristics to make a nicer report...


@functools.lru_cache
def _get_git_repo_root() -> Path:
    try:
        where = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            timeout=10,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except Exception:  # pragma: no cover
        return Path().absolute().parents[-1]
    else:
        return Path(where)


def tractable_coverage_report(trace: Trace) -> dict[str, list[int]]:
    """Report a simple coverage map which is (probably most) of the user's code."""
    coverage: dict = {}
    t = dict(trace)
    for file, line in set(t.keys()).union(t.values()) - {None}:  # type: ignore
        # On Python <= 3.11, we can use coverage.py xor Hypothesis' tracer,
        # so the trace will be empty and this line never run under coverage.
        coverage.setdefault(file, set()).add(line)  # pragma: no cover
    stdlib_fragment = f"{os.sep}lib{os.sep}python3.{sys.version_info.minor}{os.sep}"
    return {
        k: sorted(v)
        for k, v in coverage.items()
        if stdlib_fragment not in k
        and (p := Path(k)).is_relative_to(_get_git_repo_root())
        and "site-packages" not in p.parts
    }
