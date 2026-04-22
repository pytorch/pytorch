from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .config import (
    forced_timing_rounds,
    initial_threshold,
    max_samples_per_launcher,
    min_samples_before_filter,
    sampling_rate,
    threshold_decay_exp,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._launcher import Launcher


# Precomputed threshold scale factors for sample counts 1..max_samples_per_launcher.
# Index i corresponds to sample_count == i+1.  Avoids repeated pow() calls.
_THRESHOLD_MULTIPLIERS: tuple[float, ...] = tuple(
    1.0 - ((n - 1) / (max_samples_per_launcher - 1)) ** threshold_decay_exp
    for n in range(1, max_samples_per_launcher + 1)
)


class IncrementalAutotuneState:
    """Drives incremental autotuning across a set of ``Launcher`` candidates.

    Call the instance to dispatch one kernel invocation; over many calls
    the state collects timings, discards underperformers, and converges
    on a single best launcher.
    """

    def __init__(
        self,
        pre_launch_fn: Callable | None = None,
        post_launch_fn: Callable[[], None] | None = None,
        on_discard_fn: Callable[[Launcher], None] | None = None,
        on_convergence_fn: Callable[[IncrementalAutotuneState], None] | None = None,
    ) -> None:
        # Queue holds launchers that either still need timing runs or have
        # launched enough but are waiting for in-flight timings to land.
        # ``_best_launcher`` only holds a launcher once that launcher has
        # finished its full sample budget — it's the stable answer used
        # at convergence and as the threshold-check baseline.
        self._queue: deque[Launcher] = deque()
        self._best_launcher: Launcher | None = None

        self._pre_launch_fn = pre_launch_fn
        self._post_launch_fn = post_launch_fn
        self._on_discard_fn = on_discard_fn
        self._on_convergence_fn = on_convergence_fn

        self._dispatch_count: int = 0

    def add_launcher(self, launcher: Launcher) -> None:
        """Register ``launcher`` as a candidate."""
        self._queue.append(launcher)

    @property
    def best_launcher(self) -> Launcher | None:
        # Only meaningful at convergence: the stable, definitively best
        # launcher. Use ``_temp_best_launcher`` for mid-autotune lookups.
        assert self.converged, "best_launcher is only valid while converged"
        return self._best_launcher

    @property
    def best_timing(self) -> float:
        assert self.converged, "best_timing is only valid while converged"
        if self._best_launcher is None:
            return float("inf")
        return self._best_launcher.timing

    @property
    def converged(self) -> bool:
        # The queue drains as launchers either become best or get
        # discarded — empty queue means we're done.
        return not self._queue

    @property
    def _temp_best_launcher(self) -> Launcher | None:
        """Best-known launcher right now, valid mid-autotune.

        Picks the faster of the cached ``_best_launcher`` (if set) and
        the fastest launcher still in the queue. Used for the dispatch
        fallback when every queued launcher is currently waiting on
        in-flight timings.
        """
        candidates: list[Launcher] = list(self._queue)
        if self._best_launcher is not None:
            candidates.append(self._best_launcher)
        if not candidates:
            return None
        return min(candidates, key=lambda l: l.timing)

    def _discard(self, launcher: Launcher) -> None:
        """Notify ``on_discard_fn`` that ``launcher`` was dropped."""
        if self._on_discard_fn is not None:
            self._on_discard_fn(launcher)

    def _next_launcher(self) -> Launcher | None:
        """Pop the next launcher to dispatch, or ``None`` if every queued
        launcher is currently waiting for in-flight timings to land.

        Walks the queue once. For each launcher, in order:

        1. If it's reached the sample budget, decide between it and any
           cached ``_best_launcher`` (both are fully sampled, so the
           comparison is on equal footing). The loser is discarded.
        2. Else, if it's already significantly slower than the temp
           best per the decaying threshold, discard it early.
        3. Else, if its in-flight launches alone have already pushed it
           past the budget, requeue and skip — we'll revisit when the
           in-flight timings resolve. Cycle detection avoids an infinite
           loop when every queued launcher is in this state.
        4. Otherwise, return it for dispatch.
        """
        seen_waiting: OrderedSet[int] = OrderedSet()
        # Snapshot the temp best once — recomputing per iteration would be
        # O(n) per loop step. Slight staleness across the loop is fine; the
        # next ``_next_launcher`` call recomputes.
        temp_best = self._temp_best_launcher
        while self._queue:
            launcher = self._queue.popleft()
            available = launcher.num_available_timings

            # 1) Done — promote against the cached best, discard the loser.
            if available >= max_samples_per_launcher:
                if self._best_launcher is None:
                    self._best_launcher = launcher
                elif launcher.timing < self._best_launcher.timing:
                    old_best = self._best_launcher
                    self._best_launcher = launcher
                    self._discard(old_best)
                else:
                    self._discard(launcher)
                continue

            # 2) Threshold-based early discard against the snapshotted temp
            # best. Each side gets its own permissiveness based on its
            # sample count; we multiply so a noisy candidate or a noisy
            # baseline both relax the bar. Skip the baseline itself so we
            # never discard our reference point, and require at least one
            # sample on the baseline so the multiplier index is in range.
            if (
                temp_best is not None
                and launcher is not temp_best
                and available >= min_samples_before_filter
                and temp_best.num_available_timings >= 1
            ):
                cand_mult = max(
                    1.0,
                    1.0
                    + (initial_threshold - 1.0) * _THRESHOLD_MULTIPLIERS[available - 1],
                )
                best_mult = max(
                    1.0,
                    1.0
                    + (initial_threshold - 1.0)
                    * _THRESHOLD_MULTIPLIERS[temp_best.num_available_timings - 1],
                )
                if launcher.timing > cand_mult * best_mult * temp_best.timing:
                    self._discard(launcher)
                    continue

            # 3) Waiting — launched enough but not all resolved yet.
            if launcher.num_total_timings >= max_samples_per_launcher:
                if id(launcher) in seen_waiting:
                    self._queue.append(launcher)
                    return None
                seen_waiting.add(id(launcher))
                self._queue.append(launcher)
                continue

            # 4) Ready to dispatch.
            return launcher
        return None

    # -- Dispatch ----------------------------------------------------------

    def __call__(self, *args: object, stream: object, **kwargs: object) -> object:
        if self.converged:
            # Clear before invoking so the callback fires exactly once even
            # if subsequent dispatches re-enter this branch.
            if (on_convergence_fn := self._on_convergence_fn) is not None:
                self._on_convergence_fn = None
                on_convergence_fn(self)
            best = self.best_launcher
            assert best is not None, (
                "converged with no best launcher — every candidate was discarded"
            )
            return self._launch(
                best,
                *args,
                stream=stream,
                timed=False,
                requeue=False,
                **kwargs,
            )

        if (launcher := self._next_launcher()) is not None:
            self._dispatch_count += 1
            # Time iff the launcher is warm AND either we're still inside
            # the forced-timing window for this launcher OR this dispatch
            # lands on a sampling beat. Either way we requeue: more timings
            # may still be needed and ``_next_launcher`` will pull the
            # launcher back out when it's done.
            timed = launcher.is_warm and (
                launcher.num_total_timings < forced_timing_rounds
                or self._dispatch_count % sampling_rate == 0
            )
            try:
                return self._launch(
                    launcher,
                    *args,
                    stream=stream,
                    timed=timed,
                    requeue=True,
                    **kwargs,
                )
            except RuntimeError as e:
                # Triton surfaces invalid kernel configs (e.g. requested
                # block size exceeds device limits) as a RuntimeError with
                # "invalid configuration" in the message. Drop the launcher
                # and retry on the same dispatch.
                if "invalid configuration" not in str(e).lower():
                    raise
                self._discard(launcher)
                return self(*args, stream=stream, **kwargs)

        # Every queued launcher is waiting; pick the best we know so far.
        if (fallback := self._temp_best_launcher) is None:
            raise RuntimeError("No active launchers available for incremental autotune")
        return self._launch(
            fallback,
            *args,
            stream=stream,
            timed=False,
            requeue=False,
            **kwargs,
        )

    def _launch(
        self,
        launcher: Launcher | None,
        *args: object,
        stream: object,
        timed: bool,
        requeue: bool,
        **kwargs: object,
    ) -> object:
        """Run ``launcher`` (timed or untimed), optionally re-queueing it."""
        assert launcher is not None
        try:
            if self._pre_launch_fn is not None:
                self._pre_launch_fn(launcher, *args, stream=stream, **kwargs)
            result = launcher(*args, stream=stream, timed=timed, **kwargs)
        finally:
            if self._post_launch_fn is not None:
                self._post_launch_fn()
        if requeue:
            self._queue.append(launcher)
        return result
