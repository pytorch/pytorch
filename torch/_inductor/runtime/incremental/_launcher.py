from __future__ import annotations

import bisect
import threading
import weakref
from typing import TYPE_CHECKING

import torch

from ._resolver import submit_event
from .config import timing_aggregation, TimingAggregation


if TYPE_CHECKING:
    from collections.abc import Callable


class Launcher:
    """A callable kernel wrapper that collects timing samples.

    Holds a weak reference to the wrapped function ``fn``, a metadata
    dict, and a list of timing samples. Call the instance to dispatch
    the kernel; pass ``timed=True`` to record a CUDA-event timed
    dispatch (the daemon calls ``add_timing`` later when the events
    resolve). Read ``timing`` for the aggregated representative latency
    (median or mean, selected via ``config.timing_aggregation``).

    fn lifecycle: ``fn`` is held via a weakref because Launchers are
    cached for reuse across compilations. A single ``fn`` holds a
    trivial amount of device memory, but a full set of fns for an
    entire compiled graph adds up — keeping them pinned in the cache
    after autotuning has converged would hold substantial device memory
    indefinitely. The weakref lets ``fn`` be garbage-collected as soon
    as nothing outside the Launcher references it. The contract is that
    ``IncrementalAutotuneState`` owns ``fn``'s lifecycle: it holds the
    strong reference while the launcher is in active use and drops it
    once the launcher is filtered or convergence is reached.
    ``set_fn`` swaps in a fresh ``fn`` when an existing cached launcher
    is reused after its previous ``fn`` has been collected.

    Threading: ``__call__`` and ``set_fn`` are only ever invoked from
    the main (dispatch) thread, so they don't take ``_lock`` to read or
    write ``_fn_ref`` / ``_warm``. ``add_timing`` and the timing-count
    properties may be called concurrently from the resolver daemon
    thread, so they take ``_lock`` to protect ``_timings`` and
    ``_num_timings_in_flight``.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._fn_ref: weakref.ref[Callable[..., object]] = weakref.ref(fn)
        self.metadata: dict[str, object] = metadata if metadata is not None else {}

        self._timings: list[float] = []
        self._num_timings_in_flight: int = 0
        self._warm: bool = False
        self._lock: threading.Lock = threading.Lock()

    def set_fn(self, fn: Callable[..., object]) -> None:
        """Refresh the underlying callable; resets the warm flag.

        Assumes the new ``fn`` is interchangeable with the old (e.g., a
        re-compiled equivalent of the same kernel/config), so the
        accumulated timing samples remain meaningful and only ``_warm``
        is reset to force the new fn to warm up before its next timed
        dispatch. No lock — see the threading note in the class
        docstring.
        """
        self._fn_ref = weakref.ref(fn)
        self._warm = False

    def __call__(
        self,
        *args: object,
        timed: bool = False,
        **kwargs: object,
    ) -> object:
        fn = self._fn_ref()
        assert fn is not None, "Launcher's fn has been garbage-collected"
        self._warm = True
        if not timed:
            return fn(*args, **kwargs)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # Two failure modes for fn under timed dispatch:
        #   A) Launch failure (e.g., invalid configuration argument):
        #      raises synchronously here, before we increment
        #      _num_timings_in_flight, so the counter stays consistent
        #      and the caller observes the error normally.
        #   B) Execution failure (e.g., illegal memory access): is
        #      asynchronous and only surfaces on the next device
        #      sync (typically far away, on a different code path). We
        #      don't reconcile the counter for this case — such failures
        #      are assumed fatal to the entire program, so a small
        #      bookkeeping desync on a Launcher that is about to be
        #      discarded is moot.
        result = fn(*args, **kwargs)
        end_event.record()
        with self._lock:
            self._num_timings_in_flight += 1
        submit_event(self.add_timing, start_event, end_event)
        return result

    def add_timing(self, elapsed_ms: float) -> None:
        """Record a timing sample and decrement the in-flight counter."""
        with self._lock:
            bisect.insort(self._timings, elapsed_ms)
            self._num_timings_in_flight -= 1

    @property
    def is_warm(self) -> bool:
        return self._warm

    @property
    def num_available_timings(self) -> int:
        with self._lock:
            return len(self._timings)

    @property
    def num_in_flight_timings(self) -> int:
        with self._lock:
            return self._num_timings_in_flight

    @property
    def num_total_timings(self) -> int:
        with self._lock:
            return len(self._timings) + self._num_timings_in_flight

    @property
    def _median(self) -> float:
        with self._lock:
            if not self._timings:
                return float("inf")
            n = len(self._timings)
            mid = n // 2
            if n % 2 == 1:
                return self._timings[mid]
            return (self._timings[mid - 1] + self._timings[mid]) / 2

    @property
    def _mean(self) -> float:
        with self._lock:
            if not self._timings:
                return float("inf")
            return sum(self._timings) / len(self._timings)

    @property
    def timing(self) -> float:
        """Representative timing aggregated per the configured strategy."""
        match timing_aggregation:
            case TimingAggregation.MEAN:
                return self._mean
            case TimingAggregation.MEDIAN:
                return self._median
            case _:
                # Defensive: TimingAggregation is exhaustively matched
                # above so this branch is currently unreachable. Keep it
                # so that adding a new TimingAggregation member without
                # a matching case fails loudly at dispatch time instead
                # of silently returning the wrong aggregation.
                raise ValueError(f"Unknown timing aggregation: {timing_aggregation!r}")
