# mypy: allow-untyped-defs
"""Passive bridge: map a CUDA_EVENT record back to the CUDA-graph event-record node.

CUPTI emits a ``CUDA_EVENT`` activity record for each event-record node executed in a
graph (e.g. the nodes NCCL inserts under ``NCCL_GRAPH_MIXING_SUPPORT=1``), but the record
carries only an ``event_id`` (keyed to the CUDA event object) -- never a ``graph_node_id``,
so it cannot be joined to the graph node the way kernel/memcpy/memset records are.

There is no API to query a CUevent's ``event_id``, and the events are owned by whoever built
the graph (NCCL), so eagerly re-recording them to learn the mapping is unsafe. Instead this
learns the ``{event_id -> graph_node_id}`` map PASSIVELY from records that already flow:

  * A graph launch's graphed kernel/memset records share the launch's ``correlation_id`` and
    carry the exec graph id in the upper 32 bits of their ``graph_node_id``. So per launch,
    ``correlation_id -> exec_graph_id``.
  * The same launch's ``CUDA_EVENT`` records share that ``correlation_id`` and appear in
    execution order (increasing ``cuda_event_sync_id``).
  * :func:`arm_event_node_recording` records, at graph instantiate, each graph's ordered
    ``event_record`` node ids (from ``get_graph_data()``).

Matching the k-th ``CUDA_EVENT`` record of a launch to the k-th ordered event node yields
``event_id -> graph_node_id``. ``event_id`` is object-keyed and stable, so one unambiguous
launch teaches it for the whole run. Learning is conservative: it fires only when a launch's
event-record count matches the graph's and the graph's event nodes are totally ordered by
dependency depth -- otherwise the launch is skipped, never guessed.
"""

from __future__ import annotations

from typing import Any


def order_event_nodes(nodes: list[dict[str, Any]]) -> list[int] | None:
    """Return one graph's ``event_record`` node ``tools_id``s in execution order, or ``None``
    when they are not totally ordered (so the caller must not learn from that graph).

    ``nodes`` is ``get_graph_data()["nodes"]``. Order is by longest-dependency-path depth over
    the whole DAG; distinct depths for every event node is the (conservative) proxy for "these
    execute in a fixed serial order", which is what lets a launch's records match by position.
    """
    n = len(nodes)
    depth = [-1] * n

    def node_depth(i: int) -> int:
        d = depth[i]
        if d >= 0:
            return d
        deps = nodes[i]["dependencies"]
        d = 0 if not deps else 1 + max(node_depth(j) for j in deps)
        depth[i] = d
        return d

    event_idx = [i for i, nd in enumerate(nodes) if nd["node_type"] == "event_record"]
    if not event_idx:
        return []
    depths = [node_depth(i) for i in event_idx]
    if len(set(depths)) != len(event_idx):
        return None  # not totally ordered -> ambiguous, refuse to learn
    return [nodes[i]["tools_id"] for i in sorted(event_idx, key=lambda i: depth[i])]


class _EventNodeRecorder:
    """Process-global recorder + learned map for CUDA-graph event-record nodes.

    Armed once via :meth:`arm` (before graphs are captured), like the graph-dependency
    recorder. :attr:`graph_event_nodes` maps ``exec_graph_id -> ordered event-node tools_ids``
    (``None`` for graphs whose event nodes are not totally ordered). :attr:`event_id_to_node`
    is the learned ``event_id -> graph_node_id`` map, filled lazily from launches.
    """

    def __init__(self) -> None:
        self.graph_event_nodes: dict[int, list[int] | None] = {}
        self.event_id_to_node: dict[int, int] = {}
        self._handle: Any = None

    def arm(self) -> None:
        """Register the graph-instantiate hook once (idempotent)."""
        if self._handle is not None:
            return
        from torch.cuda.graphs import register_graph_instantiate_hook

        self._handle = register_graph_instantiate_hook(self._on_instantiate)

    def _on_instantiate(self, torch_cuda_graph: Any) -> None:
        # Template is live here (both keep_graph modes), so get_graph_data() works. Degrade to
        # no record when cuda.bindings / a recent driver is unavailable.
        try:
            data = torch_cuda_graph.get_graph_data()
            nodes = data["nodes"]
            exec_graph_id = data["exec_graph_id"]
        except (RuntimeError, AttributeError, KeyError):
            return
        ordered = order_event_nodes(nodes)
        if ordered == []:  # no event nodes -> nothing to track
            return
        # ordered is a list (learnable) or None (event nodes present but not totally ordered,
        # so we refuse to learn from this graph). Track the exec id for destroy-time purge.
        self.graph_event_nodes[exec_graph_id] = ordered
        torch_cuda_graph._recorded_exec_ids.add(exec_graph_id)

    def purge_exec_ids(self, exec_ids: set[int]) -> None:
        """Drop learned state for destroyed graphs (called from the graph-destroy hook)."""
        for eid in exec_ids:
            ordered = self.graph_event_nodes.pop(eid, None)
            if ordered:
                stale = set(ordered)
                for k in [k for k, v in self.event_id_to_node.items() if v in stale]:
                    del self.event_id_to_node[k]

    def learn(self, exec_graph_id: int, event_ids_in_order: list[int]) -> None:
        """Learn ``event_id -> graph_node_id`` from one launch's ordered event records.

        Fires only when the launch's event-record count matches the graph's ordered event
        nodes; otherwise the launch is dropped rather than mislearned.
        """
        ordered = self.graph_event_nodes.get(exec_graph_id)
        if not ordered or len(ordered) != len(event_ids_in_order):
            return
        for event_id, tools_id in zip(event_ids_in_order, ordered):
            self.event_id_to_node.setdefault(event_id, tools_id)

    def resolve(self, event_id: int) -> int | None:
        """The learned ``graph_node_id`` for a CUDA_EVENT record's ``event_id`` (or None)."""
        return self.event_id_to_node.get(event_id)


def learn_and_resolve_window(
    recorder: _EventNodeRecorder,
    corr_exec_pairs: Any,
    event_rows: list[tuple[int, int, int]],
) -> list[int | None]:
    """Learn from one export window's records and resolve its CUDA_EVENT records to nodes.

    ``corr_exec_pairs`` is ``(correlation_id, graph_node_id)`` for the window's graphed work
    records (kernels/memsets); ``event_rows`` is ``(correlation_id, cuda_event_sync_id,
    event_id)`` per CUDA_EVENT record. Returns the resolved ``graph_node_id`` (or None) aligned
    to ``event_rows``. Pure: the numpy marshalling stays in the observer.
    """
    corr_to_exec: dict[int, int] = {}
    for corr, gnid in corr_exec_pairs:
        if gnid:
            corr_to_exec.setdefault(corr, gnid >> 32)
    by_corr: dict[int, list[tuple[int, int]]] = {}
    for corr, sid, eid in event_rows:
        by_corr.setdefault(corr, []).append((sid, eid))
    for corr, rows in by_corr.items():
        exec_id = corr_to_exec.get(corr)
        if exec_id is not None:
            rows.sort()  # by sync id == execution order within the launch
            recorder.learn(exec_id, [eid for _, eid in rows])
    return [recorder.resolve(eid) for _, _, eid in event_rows]


_recorder = _EventNodeRecorder()


def arm_event_node_recording() -> None:
    """Arm the process-global recorder (idempotent). Call before graphs are captured."""
    _recorder.arm()


def event_node_recorder() -> _EventNodeRecorder:
    """The shared recorder (its ``event_id_to_node`` map is read by the observer)."""
    return _recorder
