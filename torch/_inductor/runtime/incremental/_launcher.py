from __future__ import annotations

import bisect
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._state import IncrementalAutotuneState


class Launcher:
    """A callable kernel wrapper with timing stats and arbitrary metadata.

    Each call increments dispatch_count automatically.  Timing data is
    accumulated via resolve_timing() (called by the resolver daemon), which
    notifies all attached IncrementalAutotuneState instances.

    Filtering (removing slow launchers) is handled per-state, not on the
    launcher — different states may have different filtering thresholds.

    Any extra keyword arguments to __init__ become metadata entries
    (e.g. CachingAutotuner passes ``config=...`` and ``cache_hash=...``).
    """

    def __init__(self, fn: Callable[..., object], **metadata: object) -> None:
        self._fn = fn
        self.metadata: dict[str, object] = metadata

        self._timings: list[float] = []
        self.dispatch_count: int = 0
        self._lock: threading.Lock = threading.Lock()
        self._attached_states: weakref.WeakSet[IncrementalAutotuneState] = (
            weakref.WeakSet()
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            self.dispatch_count += 1
        return self._fn(*args, **kwargs)

    def attach(self, state: IncrementalAutotuneState) -> None:
        self._attached_states.add(state)

    def _add_timing(self, elapsed_ms: float) -> None:
        with self._lock:
            bisect.insort(self._timings, elapsed_ms)

    def resolve_timing(self, elapsed_ms: float) -> None:
        """Add a timing sample and notify all attached states."""
        self._add_timing(elapsed_ms)
        for state in list(self._attached_states):
            state.maybe_update_best(self)

    @property
    def timing(self) -> float:
        """Representative timing (median of all samples)."""
        with self._lock:
            if not self._timings:
                return float("inf")
            n = len(self._timings)
            mid = n // 2
            if n % 2 == 1:
                return self._timings[mid]
            return (self._timings[mid - 1] + self._timings[mid]) / 2

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._timings)
