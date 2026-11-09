"""Seeded exploration of the tests TD skips.

On top of TD's own picks, a job spends EXPLORATION_SHARE of the test time of
those picks on skipped tests, drawn without replacement with probability that
grows with TD's score. The draw depends only on its seed and on inputs every
shard of a job shares, so all shards draw the same tests and each runs its own
pieces of them. Each explored test's inclusion probability is estimated by
repeating the draw REPLICATES times.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import statistics
import sys
import traceback
from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING

from tools.testing.target_determination.heuristics.utils import get_pr_number
from tools.testing.test_selections import get_duration


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tools.testing.target_determination.heuristics.interface import (
        TestPrioritizations,
    )
    from tools.testing.test_run import TestRun


EXPLORATION_SHARE = 0.05
OVERHEAD_SECONDS = 15.0
MAX_FILE_SECONDS = 600.0
REPLICATES = 1000
WEIGHT_SCHEME = "score_half_uniform_half"


@dataclass(frozen=True)
class Candidate:
    test: TestRun
    score: float
    position: int  # 0-based rank in TD's order


@dataclass(frozen=True)
class ExploredTest:
    test: TestRun
    score: float
    position: int
    weight: float
    cost: float
    count: int  # replicates, the real draw included, that picked this test
    propensity: float


@dataclass(frozen=True)
class ExplorationSplit:
    call: str
    top_seconds: float
    budget_seconds: float
    frame: int
    out_of_frame: int
    out_of_frame_seconds: float
    skipped_seconds: float
    replicates: int
    explored: list[ExploredTest]  # in TD's order

    def tests(self) -> list[TestRun]:
        return [explored.test for explored in self.explored]


def rank_candidates(
    exclude: Sequence[TestRun], ranked: Sequence[tuple[float, TestRun]]
) -> list[Candidate]:
    """The tests in `exclude`, in the order given, with their score and
    position in `ranked` (TD's order)."""
    by_name = {
        str(test): (position, score) for position, (score, test) in enumerate(ranked)
    }
    candidates = []
    for test in exclude:
        position, score = by_name[str(test)]
        candidates.append(Candidate(test, score, position))
    return candidates


def estimate_costs(
    tests: Sequence[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
) -> list[float] | None:
    """Estimated seconds per test: its recorded time, or the median of the
    known times when it has none, plus OVERHEAD_SECONDS. None when no test has
    a recorded time."""
    durations = []
    for test in tests:
        duration = get_duration(test, test_file_times, test_class_times)
        durations.append(None if duration is None else max(0.0, duration))
    known = [duration for duration in durations if duration is not None]
    if not known:
        return None
    median = statistics.median(known)
    return [
        (median if duration is None else duration) + OVERHEAD_SECONDS
        for duration in durations
    ]


def weights(scores: Sequence[float]) -> list[float]:
    """Half TD's score, half uniform: n * s+ / S+ + 1, where s+ is a score
    clipped at 0 (non-finite scores count as 0) and S+ their sum. Uniform when
    no score is positive."""
    positive = [
        score if math.isfinite(score) and score > 0 else 0.0 for score in scores
    ]
    total = math.fsum(positive)
    if not total > 0 or not math.isfinite(total):
        return [1.0] * len(scores)
    return [len(scores) * score / total + 1.0 for score in positive]


def call_id(candidates: Iterable[TestRun]) -> str:
    """Tells apart the run_test.py invocations of one job: each has its own
    test list, and every shard of the invocation computes the same value."""
    names = "\n".join(sorted(str(test) for test in candidates))
    return hashlib.sha256(names.encode()).hexdigest()[:12]


def draw(
    seed: str,
    replicate: int,
    frame_weights: Sequence[float],
    frame_costs: Sequence[float],
    budget: float,
) -> list[int]:
    """One weighted draw without replacement (Efraimidis-Spirakis keys) that
    keeps each test, in key order, whose cost fits what is left of `budget`.
    Returns indices into the frame, in pick order."""
    rng = random.Random(f"{seed}:{replicate}")
    keys = [math.log(1.0 - rng.random()) / weight for weight in frame_weights]
    order = sorted(range(len(keys)), key=lambda i: (-keys[i], i))
    cheapest = min(frame_costs, default=math.inf)
    left = budget
    picks = []
    for i in order:
        if left < cheapest:
            break
        if frame_costs[i] <= left:
            picks.append(i)
            left -= frame_costs[i]
    return picks


def plan_exploration(
    top: Sequence[TestRun],
    candidates: Sequence[Candidate],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
    seed_prefix: str,
    share: float = EXPLORATION_SHARE,
    replicates: int = REPLICATES,
) -> ExplorationSplit | None:
    """Draws the tests to explore from `candidates` (TD's skipped tests, in
    TD's order) with a budget of `share` of the seconds `top` takes. Returns
    None when there is nothing to draw or no recorded test times."""
    if share <= 0 or not top or not candidates:
        return None
    costs = estimate_costs(
        [*top, *(c.test for c in candidates)], test_file_times, test_class_times
    )
    if costs is None:
        return None
    top_seconds = math.fsum(costs[: len(top)])
    candidate_costs = costs[len(top) :]
    budget = share * top_seconds
    cap = min(budget, MAX_FILE_SECONDS)
    in_frame = [i for i, cost in enumerate(candidate_costs) if cost <= cap]
    frame = [candidates[i] for i in in_frame]
    frame_costs = [candidate_costs[i] for i in in_frame]
    frame_weights = weights([c.score for c in frame])
    call = call_id(c.test for c in candidates)
    seed = f"{seed_prefix}:{call}"

    counts = [0] * len(frame)
    real: list[int] = []
    for replicate in range(replicates):
        picks = draw(seed, replicate, frame_weights, frame_costs, budget)
        if replicate == 0:
            real = picks
        for i in picks:
            counts[i] += 1

    explored = [
        ExploredTest(
            test=frame[i].test,
            score=frame[i].score,
            position=frame[i].position,
            weight=frame_weights[i],
            cost=frame_costs[i],
            count=counts[i],
            propensity=counts[i] / replicates,
        )
        for i in sorted(real)
    ]
    in_frame_set = set(in_frame)
    return ExplorationSplit(
        call=call,
        top_seconds=top_seconds,
        budget_seconds=budget,
        frame=len(frame),
        out_of_frame=len(candidates) - len(frame),
        out_of_frame_seconds=math.fsum(
            cost for i, cost in enumerate(candidate_costs) if i not in in_frame_set
        ),
        skipped_seconds=math.fsum(candidate_costs),
        replicates=replicates,
        explored=explored,
    )


def check_disjoint(
    explored: Sequence[TestRun],
    include: Sequence[TestRun],
    exclude: Sequence[TestRun],
) -> None:
    """Raises unless every explored test is one TD skips and none is one TD
    runs."""
    names = {str(test) for test in explored}
    if not names <= {str(test) for test in exclude}:
        raise ValueError("explored tests are not all in TD's excluded tests")
    if names & {str(test) for test in include}:
        raise ValueError("explored tests overlap TD's selected tests")


def run_explored(td_failed: bool, keep_going: bool) -> bool:
    """Whether the explored batch runs after TD's batch. Without keep-going, a
    failure in TD's batch stops the job's tests, so exploration stops too."""
    return keep_going or not td_failed


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def exploration_info(
    split: ExplorationSplit,
    include: Sequence[TestRun],
    *,
    sha: str,
    pr_number: int | None,
    num_tests: int,
) -> dict[str, Any]:
    """The td_exploration metric: one row per run_test.py invocation."""
    top_files = {test.test_file for test in include}
    return {
        "sha": sha,
        "pr_number": pr_number,
        "call": split.call,
        "num_tests": num_tests,
        "top_tests": len(include),
        "top_seconds": split.top_seconds,
        "budget_seconds": split.budget_seconds,
        "frame": split.frame,
        "out_of_frame": split.out_of_frame,
        "out_of_frame_seconds": split.out_of_frame_seconds,
        "skipped_seconds": split.skipped_seconds,
        "weight_scheme": WEIGHT_SCHEME,
        "replicates": split.replicates,
        "explored": [
            {
                "test": str(explored.test),
                "test_file": explored.test.test_file,
                "score": _finite_or_none(explored.score),
                "weight": explored.weight,
                "position": explored.position,
                "cost": explored.cost,
                "count": explored.count,
                "propensity": explored.propensity,
                "top_part": explored.test.test_file in top_files,
            }
            for explored in split.explored
        ],
    }


class Exploration(NamedTuple):
    enabled: bool
    explored: list[TestRun]
    skipped: list[TestRun]  # exclude without the explored tests
    info: dict[str, Any] | None  # the td_exploration metric, built on shard 1 only


def select_exploration(
    prioritizations: TestPrioritizations,
    include: list[TestRun],
    exclude: list[TestRun],
    test_file_times: dict[str, float],
    test_class_times: dict[str, dict[str, float]],
    *,
    td_enabled: bool,
    which_shard: int,
) -> Exploration:
    """The tests to explore on top of TD's `include`. Exploration is on when
    TD is, SHA1 is set and EXPLORATION_SHARE > 0; any error is logged and
    leaves TD's selection as it is."""
    sha = os.environ.get("SHA1", "")
    enabled = td_enabled and bool(sha) and EXPLORATION_SHARE > 0
    nothing = Exploration(enabled, [], exclude, None)
    if not enabled:
        return nothing
    try:
        seed_prefix = ":".join(
            [
                sha,
                os.environ.get("GITHUB_WORKFLOW", ""),
                os.environ.get("BUILD_ENVIRONMENT", ""),
                os.environ.get("TEST_CONFIG", ""),
            ]
        )
        split = plan_exploration(
            include,
            rank_candidates(exclude, prioritizations.get_scored_tests()),
            test_file_times,
            test_class_times,
            seed_prefix,
            EXPLORATION_SHARE,
        )
        if split is None:
            return nothing
        explored = split.tests()
        check_disjoint(explored, include, exclude)
        names = {str(test) for test in explored}
        skipped = [test for test in exclude if str(test) not in names]
        info = None
        if which_shard == 1:
            info = exploration_info(
                split,
                include,
                sha=sha,
                pr_number=get_pr_number(),
                num_tests=len(include) + len(exclude),
            )
        print(
            f"TD exploration: budget {split.budget_seconds:.0f}s, "
            f"frame {split.frame}, out of frame {split.out_of_frame}, "
            f"explored {[str(test) for test in explored]}",
            file=sys.stderr,
        )
        return Exploration(enabled, explored, skipped, info)
    except Exception:
        print(
            f"TD exploration failed, running TD's selection only:\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )
        return nothing
