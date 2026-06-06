# mypy: allow-untyped-defs
"""Base class for CUPTI activity-monitor observers.

An observer registers the activity kinds it wants with the shared CUPTI monitor
(``torch.profiler.cupti.monitor.instance()``) and, on the monitor's worker
thread, gets handed the columns the monitor demuxed from each completed buffer
(``{ActivityKind: {field_id: column}}``) sliced to its selection. What it does
with them is up to the subclass's ``_on_activities`` hook. This base handles
registration, availability, teardown, and the clock passthroughs.
"""

from __future__ import annotations

from typing import Any


class CuptiMonitorObserver:
    """Base for observers backed by the shared CUPTI monitor.

    Subclasses set up their state, then call ``super().__init__(activities)`` with
    the activity kinds they want -- a set of kinds (meaning "all fields"), or a
    field map ``{kind: iterable of field ids | "all"}``. Registration happens last
    so the state is ready before the worker thread can deliver buffers. The monitor
    demuxes every completed buffer to columns (whether v1 or v2), so subclasses
    implement a single hook, ``_on_activities(columns)`` -- where ``columns`` is
    ``{ActivityKind: {field_id: column}}`` already sliced to this observer's
    selection -- and typically a ``drain()`` for callers. ``version`` only selects
    the underlying monitor (1 = classic activities, 2 = user-defined records); the
    column contract is identical either way.
    """

    def __init__(self, activities: Any) -> None:
        # frozenset of the requested kinds (a field map collapses to its keys) for
        # the observer's own "is this kind mine?" checks; the full request (incl.
        # any field selection) is handed to the monitor. The monitor is the
        # process-wide singleton, whose version is fixed by the
        # $TORCH_CUPTI_MONITOR_USE_V2_API env var.
        self._activities: frozenset[int] = frozenset(activities)
        self._monitor: Any = None
        self._obs = None
        try:
            from torch.profiler.cupti.monitor import instance

            self._monitor = instance()
        except Exception:
            return
        self._version = self._monitor.version
        self._obs = self._monitor.register(activities, self._on_activities)

    @property
    def available(self) -> bool:
        """True when the monitor was available and this observer registered."""
        return self._obs is not None

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        """Worker-thread hook: ``{ActivityKind: {field_id: column}}`` demuxed by the
        monitor and sliced to this observer's selection. Implemented by subclasses."""
        raise NotImplementedError

    def now_ns(self) -> int:
        """Current time on the same unix-epoch clock as record timestamps --
        passthrough to the monitor."""
        return self._monitor.now_unix_ns() if self._monitor is not None else 0

    def convert_time(self, value: int) -> int:
        """Convert a CUPTI-clock timestamp to unix-epoch ns -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        return self._monitor.convert_time(value) if self._monitor is not None else value

    def close(self) -> None:
        """Unregister from the monitor. Idempotent."""
        if self._obs is not None and self._monitor is not None:
            self._monitor.unregister(self._obs)
            self._obs = None
