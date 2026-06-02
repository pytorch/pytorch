# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Base class for CUPTI activity-mux observers.

An observer consumes demuxed records from the shared in-process mux
(``torch.profiler.cupti.instance()``). ``MuxObserver`` handles the parts
every observer shares -- registering a field selection, availability,
teardown, and passthroughs to the mux's clocks -- so subclasses only
implement ``_on_records(kind, fields)`` plus whatever aggregation/drain
surface they expose.
"""

from __future__ import annotations

from typing import Any

from torch.profiler.cupti.mux import instance


class MuxObserver:
    """Base for observers backed by the shared CUPTI activity mux.

    Subclasses set up any aggregation state, then call
    ``super().__init__(wants)`` with their per-kind field selection
    (``{kind: {field_ids} | "all"}``); registration happens last so the
    state is ready before the mux poll thread can deliver records. They
    implement ``_on_records(kind, {field_id: ndarray})`` (invoked on the
    mux poll thread) and typically a ``drain()`` to hand results to callers.
    """

    def __init__(self, wants: "dict[int, set[int] | str]") -> None:
        self._mux = instance()
        self._obs = None
        if self._mux.available:
            self._obs = self._mux.register(wants, self._on_records)

    @property
    def available(self) -> bool:
        """True when the mux was available and this observer is registered."""
        return self._obs is not None

    def _on_records(self, kind: int, fields: dict[int, Any]) -> None:
        raise NotImplementedError

    def now_ns(self) -> int:
        """Current CUPTI timestamp (ns), in the same clock domain as the
        record START/END fields -- passthrough to the mux. Use to stamp
        wall-clock boundaries comparable to record timestamps."""
        return self._mux.now_ns()

    def convert_time(self, value: int) -> int:
        """Convert a record timestamp (CUPTI clock) to unix-epoch ns --
        passthrough to the mux (identity if clock alignment is unavailable)."""
        return self._mux.convert_time(value)

    def close(self) -> None:
        """Unregister from the mux. Idempotent."""
        if self._obs is not None:
            self._mux.unregister(self._obs)
            self._obs = None
