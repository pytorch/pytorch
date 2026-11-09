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
import random
import statistics
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from tools.testing.test_selections import get_duration


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from tools.testing.test_run import ShardedTest, TestRun


EXPLORATION_SHARE = 0.05
OVERHEAD_SECONDS = 15.0
MAX_FILE_SECONDS = 600.0
REPLICATES = 1000
WEIGHT_SCHEME = "score_half_uniform_half"

FAILED = "failed"
PASSED = "passed"
NOT_RUN = "not_run"


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
    durations = [
        get_duration(test, test_file_times, test_class_times) for test in tests
    ]
    known = [max(0.0, duration) for duration in durations if duration is not None]
    if not known:
        return None
    median = statistics.median(known)
    return [
        (median if duration is None else max(0.0, duration)) + OVERHEAD_SECONDS
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


class ExploredBatchRun:
    """Runs the explored batch after TD's own batch and labels each piece
    scheduled on this shard: failed, passed, or not_run when the batch never
    started or may have stopped before reaching it."""

    def __init__(self, pieces: Sequence[ShardedTest]) -> None:
        self.pieces = pieces
        self.keep_going = False
        self.returned = False

    def run_after(
        self, top_failed: bool, keep_going: bool, run: Callable[[], None]
    ) -> None:
        if not self.pieces or (top_failed and not keep_going):
            return
        self.keep_going = keep_going
        run()
        self.returned = True

    def outcomes(self, failed: Iterable[TestRun]) -> list[str]:
        failed_names = {str(test) for test in failed}
        # Without keep-going, a failing file terminates the parallel pool and
        # the batch still returns, so the files it never reached look clean.
        completed = self.returned and (self.keep_going or not failed_names)
        return [
            FAILED
            if str(piece.test) in failed_names
            else PASSED
            if completed
            else NOT_RUN
            for piece in self.pieces
        ]


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def exploration_info(
    split: ExplorationSplit,
    pieces: Sequence[ShardedTest],
    outcomes: Sequence[str],
    *,
    sha: str,
    pr_number: int | None,
    num_tests: int,
    top_tests: int,
) -> dict[str, Any]:
    """The td_exploration metric for the explored pieces on this shard."""
    by_name = {str(explored.test): explored for explored in split.explored}
    explored_pieces = []
    for piece, outcome in zip(pieces, outcomes, strict=True):
        explored = by_name[str(piece.test)]
        explored_pieces.append(
            {
                "test": str(piece.test),
                "test_file": piece.test.test_file,
                "score": _finite_or_none(explored.score),
                "weight": explored.weight,
                "position": explored.position,
                "cost": explored.cost,
                "count": explored.count,
                "propensity": explored.propensity,
                "piece": [piece.shard, piece.num_shards],
                "outcome": outcome,
            }
        )
    return {
        "sha": sha,
        "pr_number": pr_number,
        "call": split.call,
        "num_tests": num_tests,
        "top_tests": top_tests,
        "top_seconds": split.top_seconds,
        "budget_seconds": split.budget_seconds,
        "frame": split.frame,
        "out_of_frame": split.out_of_frame,
        "out_of_frame_seconds": split.out_of_frame_seconds,
        "weight_scheme": WEIGHT_SCHEME,
        "replicates": split.replicates,
        "explored": explored_pieces,
    }
