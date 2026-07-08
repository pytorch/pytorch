# mypy: allow-untyped-defs
"""Base class for CUPTI activity-monitor observers.

An observer registers the activity kinds it wants with the shared CUPTI monitor and, on
the monitor's worker thread, is handed the demuxed columns (``{ActivityKind: {field_id:
column}}``) sliced to its selection -- what it does with them is the subclass's
``_on_activities`` hook. This base handles registration, availability, teardown, the
clock passthroughs, and the user-annotation push/pop for naming regions.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


# graph_node_id -> annotation name (or None). The graph naming mechanism shared by
# observers: map a CUDA-graph node id to its registered region name (survives graph
# replay, needs no extra record kinds).
GraphAnnotationResolver = Callable[[int], "Any | None"]


def default_graph_annotation_resolver(graph_node_id: int) -> Any | None:
    """Default resolver: map a CUDA-graph node id to its registered annotation."""
    if graph_node_id == 0:
        return None
    try:
        from torch.cuda._graph_annotations import get_kernel_annotations

        annotations = get_kernel_annotations()
    except Exception:
        return None
    return annotations.get(graph_node_id)


@dataclass(frozen=True)
class ObserverAnnotationSettings:
    """How an observer attributes activity to named regions. Each source enforces its own
    fields when enabled (folded into the selection) and contributes nothing when disabled.

    - ``graph_annotation_resolver`` -- graph-node naming (``graph_node_id -> name``) from the
      recorded CUDA-graph registry (survives replay); collection-free beyond graph_node_id.
      Pluggable; defaults to ``None`` (disabled). Pass ``default_graph_annotation_resolver``
      (or a custom resolver) to enable.
    - ``support_eager_annotations`` -- eager ``record_function`` naming via the built-in
      external-correlation join (``correlation_id -> external_id -> name``). Folds in
      EXTERNAL_CORRELATION + RUNTIME (-> slower per-record decode), so off by default.
    """

    graph_annotation_resolver: GraphAnnotationResolver | None = None
    support_eager_annotations: bool = False


class CuptiMonitorObserver:
    """Base for observers backed by the shared CUPTI monitor.

    Subclasses set up state, then call ``super().__init__(activities)`` with the kinds they
    want (a set, or a field map ``{kind: field ids | "all"}``) -- registration last, so
    state is ready before the worker thread delivers buffers. They implement
    ``_on_activities(columns)`` (``{ActivityKind: {field_id: column}}`` sliced to their
    selection) and typically a ``drain()``.

    Observers can also bracket regions with ``push_annotation``/``pop_annotation`` (or
    ``annotate``): each push registers a global external-correlation id mapped here to a
    name, attributing activity until the pop via ``correlation_id -> external_id -> name``
    (eager only -- external ids don't survive graph capture; under graphs use
    ``graph_node_id``)."""

    def __init__(
        self,
        activities: Any,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        # Region naming (see ObserverAnnotationSettings): an enabled source folds its
        # required fields into the selection (graph: just graph_node_id; eager: extra kinds).
        if annotations is None:
            self._resolver: GraphAnnotationResolver | None = None
            self._eager = False
        else:
            self._resolver = annotations.graph_annotation_resolver
            self._eager = annotations.support_eager_annotations
        if self._resolver is not None:
            activities = self._with_graph_fields(activities)
        if self._eager:
            activities = self._with_eager_fields(activities)
        # frozenset of requested kinds (a field map collapses to keys) for the observer's
        # own "is this kind mine?" checks; the full request goes to the monitor singleton.
        self._activities: frozenset[int] = frozenset(activities)
        self._monitor: Any = None
        self._obs = None
        # external_id -> annotation name for the monitor's global pushes. Guarded by
        # _ann_lock (push on the caller's thread; a drain may read/reset from another).
        self._ann_lock = threading.Lock()
        self._ext_names: dict[int, str] = {}
        # Degrade gracefully (available == False) if the monitor can't be reached or
        # registration fails (CUPTI subscribe rejected, libcupti lacks v2)
        try:
            from torch.profiler._cupti.monitor import instance

            self._monitor = instance()
            self._obs = self._monitor.register(activities, self._on_activities)
        except Exception:
            self._obs = None

    @property
    def available(self) -> bool:
        """True when the monitor was available and this observer registered."""
        return self._obs is not None

    def _on_activities(self, columns: dict[Any, dict[int, Any]]) -> None:
        """Worker-thread hook: ``{ActivityKind: {field_id: column}}`` demuxed by the
        monitor and sliced to this observer's selection. Implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def _with_eager_fields(activities: Any) -> dict[int, set[int]]:
        """Augment a field map for the eager join: add each kind's CORRELATION_ID plus the
        EXTERNAL_CORRELATION and RUNTIME records (CUPTI only emits the former when RUNTIME
        is enabled; RUNTIME is just the carrier). Expects a ``{kind: fields}`` map."""
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.records import CORRELATION_FIELD, ExternalCorrelation

        aug: dict[int, set[int]] = {}
        for kind, sel in dict(activities).items():
            k = int(kind)
            fields = {int(f) for f in sel}
            if k in CORRELATION_FIELD:
                fields.add(CORRELATION_FIELD[k])
            aug[k] = fields
        aug[int(ActivityKind.EXTERNAL_CORRELATION)] = {
            int(ExternalCorrelation.EXTERNAL_ID),
            int(ExternalCorrelation.CORRELATION_ID),
        }
        aug[int(ActivityKind.RUNTIME)] = {CORRELATION_FIELD[int(ActivityKind.RUNTIME)]}
        return aug

    @staticmethod
    def _with_graph_fields(activities: Any) -> dict[int, set[int]]:
        """Augment a field map so the graph resolver can name nodes: add each GPU-op kind's
        GRAPH_NODE_ID. Collection-free (it's a normal record field, no extra kinds, stays on
        the vectorized path). Expects a ``{kind: fields}`` map."""
        from torch.profiler._cupti.records import GRAPH_NODE_FIELD

        aug: dict[int, set[int]] = {}
        for kind, sel in dict(activities).items():
            k = int(kind)
            fields = {int(f) for f in sel}
            if k in GRAPH_NODE_FIELD:
                fields.add(GRAPH_NODE_FIELD[k])
            aug[k] = fields
        return aug

    def push_annotation(self, name: str) -> int | None:
        """Push a global external-correlation id (mapped here to ``name``) so activity
        until the pop is attributed via ``correlation_id -> external_id -> name``. Eager
        only. No-op returning None when the monitor is unavailable."""
        if not self.available or self._monitor is None:
            return None
        ext_id = self._monitor.push_external_correlation_id()
        if ext_id is not None:
            with self._ann_lock:
                self._ext_names[ext_id] = name
        return ext_id

    def pop_annotation(self) -> int | None:
        """Pop the most recent external-correlation id (balances push_annotation)."""
        if not self.available or self._monitor is None:
            return None
        return self._monitor.pop_external_correlation_id()

    @contextlib.contextmanager
    def annotate(self, name: str) -> Iterator[int | None]:
        """Context-manager form of push_annotation/pop_annotation."""
        ext_id = self.push_annotation(name)
        try:
            yield ext_id
        finally:
            self.pop_annotation()

    def annotation_names(self, *, reset: bool = False) -> dict[int, str]:
        """Snapshot of the ``external_id -> name`` map pushed so far; pass
        ``reset=True`` to also clear it (e.g. when closing a window)."""
        with self._ann_lock:
            snapshot = dict(self._ext_names)
            if reset:
                self._ext_names = {}
        return snapshot

    def now_ns(self) -> int:
        """Current time on the same unix-epoch clock as record timestamps --
        passthrough to the monitor."""
        return self._monitor.now_unix_ns() if self._monitor is not None else 0

    def now_record_ns(self) -> int:
        """Current value of CUPTI's native record clock -- the unconverted timebase of
        decoded record START/END. Use this to stamp a window boundary compared against
        raw record timestamps (see NodeTimerObserver bucketing). 0 if unavailable."""
        return self._monitor.now_record_ns() if self._monitor is not None else 0

    def convert_time(self, value: int) -> int:
        """Convert a CUPTI-clock timestamp to unix-epoch ns -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        return self._monitor.convert_time(value) if self._monitor is not None else value

    def convert_time_array(self, values: Any) -> Any:
        """Vectorized :meth:`convert_time` over a whole column -- passthrough to the
        monitor (identity if clock alignment is unavailable)."""
        if self._monitor is None:
            return values
        return self._monitor.convert_time_array(values)

    def close(self) -> None:
        """Unregister from the monitor. Idempotent."""
        if self._obs is not None and self._monitor is not None:
            self._monitor.unregister(self._obs)
            self._obs = None
