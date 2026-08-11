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
import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.hooks import RemovableHandle


# graph_node_id -> annotation name (or None). The graph naming mechanism shared by
# observers: map a CUDA-graph node id to its registered region name (survives graph
# replay, needs no extra record kinds).
GraphAnnotationResolver = Callable[[int], "Any | None"]

# graph_node_id -> (logical_lane, lane_name), or None to leave the op on its CUDA stream:
# which display lane a graphed op renders on and how that lane is named (the op's CUDA stream
# is preserved as ``original_stream`` when the lane differs). The resolver itself is optional
# (see ``ObserverAnnotationSettings.graph_lane_resolver`` -- unset means no reassignment); when
# set it is called once per distinct graph_node_id and returns that node's logical lane (a
# graph node's annotation, hence its lane, is stable once baked), or None to keep the op on its
# CUDA stream. Keyed on graph_node_id alone, so it is wrapped in functools.cache like the
# annotation resolver; it reads the node's name via the graph annotation registry.
LaneResolver = Callable[[int], "tuple[int, str] | None"]

# graph_node_id -> predecessor graph_node_ids (or None when the node has no recorded
# dependencies): the CUDA-graph node->node edges the export draws as dependency arrows, read
# from the observer's own dependency map (recorded at graph instantiate, keyed on graph_node_id
# like the annotation resolver). See ObserverAnnotationSettings.record_graph_dependencies.
GraphDependencyResolver = Callable[[int], "list[int] | None"]


def default_graph_annotation_resolver(graph_node_id: int) -> Any | None:
    """Default resolver: map a CUDA-graph node id to its registered annotation, or None when
    it has none."""
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
    # Pluggable graphed-op lane assignment (see LaneResolver). None -> ops render on their
    # CUDA stream lane (no reassignment). Independent of graph_annotation_resolver, though a
    # consumer's implementation typically maps the node's annotation to a lane.
    graph_lane_resolver: LaneResolver | None = None
    # Record each graph's node->node dependency topology at instantiate into the observer's own
    # map (the source for dependency arrows). Off by default (extra work at graph instantiate +
    # arrow rendering); the observer registers an instantiate hook + a resolver over its map
    # only when this is set.
    record_graph_dependencies: bool = False


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

    # Both graph resolvers are keyed on graph_node_id: a node's annotation and lane are stable
    # once its graph is baked, so each resolves once for this observer's lifetime (reused across
    # every buffer delivery). Both take the int graph_node_id and are wrapped in functools.cache
    # on assignment. The caches grow with distinct graph_node_ids (each recapture mints new ids);
    # the per-resolver graph-destroy hooks (_register_graph_destroy_hooks) invalidate them
    # when a graph is destroyed, bounding growth in long runs.
    @property
    def _annotation_resolver(self) -> GraphAnnotationResolver | None:
        return self._annotation_resolver_cached

    @_annotation_resolver.setter
    def _annotation_resolver(self, fn: GraphAnnotationResolver | None) -> None:
        self._annotation_resolver_cached = (
            functools.cache(fn) if fn is not None else None
        )

    @property
    def _lane_resolver(self) -> LaneResolver | None:
        return self._lane_resolver_cached

    @_lane_resolver.setter
    def _lane_resolver(self, fn: LaneResolver | None) -> None:
        self._lane_resolver_cached = functools.cache(fn) if fn is not None else None

    @property
    def _dependency_resolver(self) -> GraphDependencyResolver | None:
        return self._dependency_resolver_cached

    @_dependency_resolver.setter
    def _dependency_resolver(self, fn: GraphDependencyResolver | None) -> None:
        self._dependency_resolver_cached = (
            functools.cache(fn) if fn is not None else None
        )

    def __init__(
        self,
        activities: Any,
        *,
        annotations: ObserverAnnotationSettings | None = None,
    ) -> None:
        record_deps = annotations is not None and annotations.record_graph_dependencies
        # Graph node->node dependency map (graph_node_id -> predecessor graph_node_ids), read
        # by our dependency resolver. When recording, share the process-global recorder's map
        # by reference: it is armed early (before graphs are captured) and persists across
        # windows, so it holds topology this per-window observer -- created at prepare_trace,
        # long after warm-up capture, and torn down each window -- would otherwise never see.
        if record_deps:
            from torch.profiler._cupti._graph_deps import _GraphDependencyRecorder

            rec = _GraphDependencyRecorder()
            rec.arm()
            self._graph_dependencies: dict[int, list[int]] = rec.deps
            # event-record node graph_node_id -> cudaEvent_t handle, recorded by the same
            # recorder (its CUDA_EVENT record has no event handle). Shared by reference.
            self._graph_event_record_events: dict[int, int] = rec.event_record_events
            # graph_node_id -> (host_fn_name, host_fn_addr), recorded by the same recorder
            # (host nodes carry no name in the CUPTI record). Shared by reference.
            self._graph_host_fns: dict[int, tuple[str | None, int]] = rec.host_fns
        else:
            self._graph_dependencies = {}
            self._graph_event_record_events = {}
            self._graph_host_fns = {}
        # Region naming (see ObserverAnnotationSettings): an enabled source folds its
        # required fields into the selection (graph: just graph_node_id; eager: extra kinds).
        if annotations is None:
            self._annotation_resolver = None
            self._lane_resolver = None
            self._dependency_resolver = None
            self._eager = False
        else:
            self._annotation_resolver = annotations.graph_annotation_resolver
            self._lane_resolver = annotations.graph_lane_resolver
            self._dependency_resolver = (
                self._make_dependency_resolver() if record_deps else None
            )
            self._eager = annotations.support_eager_annotations
        if self._annotation_resolver is not None:
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
            from torch.profiler._cupti.monitor import CuptiMonitor

            self._monitor = CuptiMonitor()
            self._obs = self._monitor.register(activities, self._on_activities)
        except Exception:
            self._obs = None
        # Register a graph-destroy hook per installed graph-node resolver so a
        # destroyed CUDA graph purges that resolver's registry and invalidates
        # its cache. Registering any hook is also the "monitor active" gate
        # torch.cuda.graphs checks before arming its destroy callback. Handles are
        # removed in close() so nothing leaks this observer.
        self._destroy_hook_handles: list[RemovableHandle] = []
        if self.available:
            self._register_graph_destroy_hooks()

    def _make_dependency_resolver(self) -> GraphDependencyResolver:
        """A resolver over the shared graph dependency map: graph_node_id -> predecessor
        graph_node_ids (None when the node has none). Captures the map, not self, so the
        cached resolver never pins the observer."""
        deps = self._graph_dependencies

        def resolve(graph_node_id: int) -> list[int] | None:
            if graph_node_id == 0:
                return None
            return deps.get(graph_node_id)

        return resolve

    @property
    def available(self) -> bool:
        """True when the monitor was available and this observer registered."""
        return self._obs is not None

    def _register_graph_destroy_hooks(self) -> None:
        """Register a graph-destroy hook per installed graph-node resolver. Each hook
        purges its backing store (the annotation module registry, or this observer's own
        dependency map) and clears its resolver cache (cache_clear is global per resolver
        -- it drops every graph's cached lookups, not just the destroyed graph's --
        acceptable on the infrequent destroy path and what bounds cache growth over a
        long run). Hooks capture only the cache wrapper + purge fn (never self, so they
        cannot pin this observer -- the dependency purge closes over the map dict, not the
        observer); the destroy fan-out also swallows any error they raise
        (finalizer-safe, since a destroy may fire from a GC/finalizer thread). The lane
        resolver is externally backed (nothing to purge), so its hook only clears the
        cache."""
        from torch.cuda._graph_annotations import remove_kernel_annotations
        from torch.cuda.graphs import register_graph_destroy_hook

        handles = self._destroy_hook_handles
        deps = self._graph_dependencies
        event_record_events = self._graph_event_record_events
        host_fns = self._graph_host_fns

        def purge_deps(ids: set[int]) -> None:
            for key in [k for k in deps if k >> 32 in ids]:
                del deps[key]
            for key in [k for k in event_record_events if k >> 32 in ids]:
                del event_record_events[key]
            for key in [k for k in host_fns if k >> 32 in ids]:
                del host_fns[key]

        def add(cache: Any, purge: Any) -> None:
            def hook(ids: set[int]) -> None:
                if purge is not None:
                    purge(ids)
                cache.cache_clear()

            handles.append(register_graph_destroy_hook(hook))

        if self._annotation_resolver_cached is not None:
            add(self._annotation_resolver_cached, remove_kernel_annotations)
        if self._dependency_resolver_cached is not None:
            add(self._dependency_resolver_cached, purge_deps)
        if self._lane_resolver_cached is not None:
            add(self._lane_resolver_cached, None)

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
        for handle in self._destroy_hook_handles:
            handle.remove()
        self._destroy_hook_handles = []
        if self._obs is not None and self._monitor is not None:
            self._monitor.unregister(self._obs)
            self._obs = None
