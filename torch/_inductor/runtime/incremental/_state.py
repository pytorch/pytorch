from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING

import torch

from ._launcher import Launcher
from ._resolver import submit_event
from .config import (
    _FORCED_TIMING_ROUNDS,
    _INITIAL_THRESHOLD,
    _MAX_SAMPLES_PER_LAUNCHER,
    _MIN_SAMPLES_BEFORE_FILTER,
    _SAMPLING_RATE,
    _THRESHOLD_MULTIPLIERS,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class IncrementalAutotuneState:
    """Central state for incremental autotuning.

    Manages a round-robin queue of Launcher candidates, CUDA event timing
    via a daemon thread, progressive filtering, and convergence detection.

    All filtering decisions (max dispatches, threshold, invalid config) are
    made lazily in _next_launcher.  Launchers with fewer than
    _FORCED_TIMING_ROUNDS dispatches are always timed; once past that
    threshold, only 1 in _SAMPLING_RATE dispatches is timed.
    """

    def __init__(
        self,
        pre_launch_fn: Callable | None = None,
        post_launch_fn: Callable[[], None] | None = None,
        on_convergence_fn: Callable[[IncrementalAutotuneState], None] | None = None,
        on_cleanup_fn: Callable[[IncrementalAutotuneState], None] | None = None,
    ) -> None:
        self._launchers: list[Launcher] = []
        self._queue: deque[Launcher] = deque()
        self.best_launcher: Launcher | None = None

        self._lock = threading.RLock()
        self._pre_launch_fn = pre_launch_fn
        self._post_launch_fn = post_launch_fn
        self._on_convergence_fn = on_convergence_fn
        self._on_cleanup_fn = on_cleanup_fn

        self._pending_events: int = 0
        self._total_dispatch_count: int = 0

        self._background_error: Exception | None = None

    def init_fresh(self, launchers: list[Launcher]) -> None:
        """Populate with launchers, seeding ``best_launcher`` from any
        existing timings (so a launcher reused from another state with prior
        samples is immediately the running best).
        """
        self._launchers = list(launchers)
        best_timing = float("inf")
        for launcher in launchers:
            launcher.attach(self)
            self._queue.append(launcher)
            if (timing := launcher.timing) < best_timing:
                self.best_launcher = launcher
                best_timing = timing

    def _should_skip(self, launcher: Launcher) -> bool:
        """Check if a launcher should be skipped for timing."""
        if launcher.dispatch_count >= _MAX_SAMPLES_PER_LAUNCHER:
            return True
        if self.best_launcher is not None and launcher is not self.best_launcher:
            sample_count = launcher.sample_count
            if sample_count >= _MIN_SAMPLES_BEFORE_FILTER:
                clamped = min(sample_count, _MAX_SAMPLES_PER_LAUNCHER)
                threshold = 1.0 + (_INITIAL_THRESHOLD - 1.0) * _THRESHOLD_MULTIPLIERS[clamped - 1]
                if launcher.timing > threshold * self.best_timing:
                    return True
        return False

    # -- Round-robin ----------------------------------------------------------

    def _next_launcher(self) -> Launcher | None:
        """Pop and return the next launcher eligible for timing, or None."""
        while self._queue:
            launcher = self._queue.popleft()
            if not self._should_skip(launcher):
                return launcher
        return None

    # -- Dispatch ----------------------------------------------------------

    def dispatch(self, *args: object, stream: object, **kwargs: object) -> object:
        with self._lock:
            if self._background_error is not None:
                raise self._background_error

            if self.converged:
                if self._on_convergence_fn is not None:
                    self._on_convergence_fn(self)
                if self.converged:
                    self.shutdown()
                return self._launch(self.best_launcher, *args, stream=stream, **kwargs)

            # If we have a best launcher and it's past warmup, only time
            # 1 in _SAMPLING_RATE dispatches — run the best untimed otherwise.
            self._total_dispatch_count += 1
            if (
                self.best_launcher is not None
                and self.best_launcher.dispatch_count >= _FORCED_TIMING_ROUNDS
                and self._total_dispatch_count % _SAMPLING_RATE != 0
            ):
                return self._launch(
                    self.best_launcher, *args, stream=stream, **kwargs
                )

            if (launcher := self._next_launcher()) is not None:
                try:
                    result = self._timed_launch(launcher, *args, stream=stream, **kwargs)
                except RuntimeError as e:
                    # Triton surfaces invalid kernel configs (e.g. requested
                    # block size exceeds device limits) as a RuntimeError with
                    # "invalid configuration" in the message. Drop the launcher
                    # and retry on the same dispatch.
                    if "invalid configuration" not in str(e).lower():
                        raise
                    self._launchers.remove(launcher)
                    return self.dispatch(*args, stream=stream, **kwargs)
                if self.best_launcher is None:
                    self.best_launcher = launcher
                return result

            if self.best_launcher is not None:
                return self._launch(
                    self.best_launcher, *args, stream=stream, **kwargs
                )
            raise RuntimeError(
                "No active launchers available for incremental autotune"
            )

    def _launch(self, launcher: Launcher | None, *args: object, stream: object, **kwargs: object) -> object:
        """Launch a kernel without timing."""
        assert launcher is not None
        try:
            if self._pre_launch_fn is not None:
                self._pre_launch_fn(launcher, *args, stream=stream, **kwargs)
            return launcher(*args, **kwargs, stream=stream)
        finally:
            if self._post_launch_fn is not None:
                self._post_launch_fn()

    def _timed_launch(self, launcher: Launcher, *args: object, stream: object, **kwargs: object) -> object:
        """Launch a kernel with CUDA event timing."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        try:
            if self._pre_launch_fn is not None:
                self._pre_launch_fn(launcher, *args, stream=stream, **kwargs)
            start_event.record()
            result = launcher(*args, **kwargs, stream=stream)
            end_event.record()
        finally:
            if self._post_launch_fn is not None:
                self._post_launch_fn()
        self._queue.append(launcher)
        self._pending_events += 1
        submit_event(self, launcher, start_event, end_event)
        return result

    # -- Event resolution -------------------------------------------------

    @property
    def best_timing(self) -> float:
        if self.best_launcher is None:
            return float("inf")
        return self.best_launcher.timing

    def maybe_update_best(self, launcher: Launcher) -> None:
        """Update best_launcher if this launcher is now faster."""
        with self._lock:
            if launcher.timing < self.best_timing:
                self.best_launcher = launcher

    def decrement_pending(self) -> None:
        with self._lock:
            self._pending_events -= 1

    def set_background_error(self, error: Exception) -> None:
        """Stamp an error from the resolver daemon; the next dispatch raises it."""
        with self._lock:
            if self._background_error is None:
                self._background_error = error

    # -- Convergence -------------------------------------------------------

    @property
    def converged(self) -> bool:
        if self.best_launcher is not None and all(
            self._should_skip(launcher)
            for launcher in self._launchers
            if launcher is not self.best_launcher
        ):
            return True
        if self._pending_events > 0:
            return False
        if not self._queue:
            assert self.best_launcher is not None, (
                "queue empty with no best launcher — all configs were rejected"
            )
            return True
        return False

    # -- Cleanup -----------------------------------------------------------

    def __del__(self) -> None:
        # Finalizers must not raise — log and continue. We narrow to Exception
        # here (rather than letting it propagate) only because exceptions from
        # __del__ are silently swallowed by CPython anyway, and a debug log is
        # strictly more useful than the default.
        try:
            if self._on_cleanup_fn is not None:
                self._on_cleanup_fn(self)
                self._on_cleanup_fn = None
        except Exception:
            from ._utils import log

            log.debug(
                "IncrementalAutotuneState.__del__ cleanup callback raised",
                exc_info=True,
            )

    def shutdown(self) -> None:
        pass
