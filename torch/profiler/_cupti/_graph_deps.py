# mypy: allow-untyped-defs
"""Process-global CUDA-graph node->node dependency recorder for the CUPTI profiler.

The topology of a CUDA graph is only readable (via ``get_graph_data()``) while its
template is live, which -- for ``keep_graph=False`` graphs -- is only during the
``register_graph_instantiate_hook`` callback fired inside ``CUDAGraph.instantiate()``.
A workload typically captures/instantiates its graphs ONCE (during warm-up) and then
only replays them, so the recording hook must already be armed before that one-time
capture. The per-window ``ProfilerObserver`` cannot do this: it is created (and its
hook registered) at ``prepare_trace`` -- long after warm-up -- and torn down between
windows, so its hook never observes an ``instantiate()`` and its map stays empty.

This recorder decouples recording from the observer lifecycle. It is armed once, early
(from ``torch.profiler`` when ``enable_graph_dependencies`` is set, i.e. at profiler
construction, before the training loop), and its map persists for the process. Each
observer shares this map by reference, so its dependency resolver reads topology
recorded before that observer existed. ``keep_graph`` is irrelevant: the hook runs
during ``instantiate()`` while the template is still live.
"""

from __future__ import annotations

from typing import Any


class _GraphDependencyRecorder:
    """Persistent recorder of graph_node_id -> predecessor graph_node_ids edges.

    Edges are keyed by ``tools_id`` (== the CUPTI ``graph_node_id`` that joins to
    profiler kernel records). Armed at most once via :meth:`arm`; :attr:`deps` is the
    shared map read (by reference) by every observer's dependency resolver.
    """

    def __init__(self) -> None:
        self.deps: dict[int, list[int]] = {}
        self._handle: Any = None

    def arm(self) -> None:
        """Register the graph-instantiate hook once (idempotent)."""
        if self._handle is not None:
            return
        from torch.cuda.graphs import register_graph_instantiate_hook

        self._handle = register_graph_instantiate_hook(self._on_instantiate)

    def _on_instantiate(self, torch_cuda_graph: Any) -> None:
        # We hold the live graph here, so get_graph_data() works during instantiate()
        # for both keep_graph modes (the template is destroyed only afterwards). Raises
        # when cuda.bindings / a recent driver is unavailable -- degrade to no edges.
        try:
            nodes = torch_cuda_graph.get_graph_data()["nodes"]
        except (RuntimeError, AttributeError, KeyError):
            return
        recorded = {
            n["tools_id"]: [nodes[i]["tools_id"] for i in n["dependencies"]]
            for n in nodes
            if n["dependencies"]
        }
        if not recorded:
            return
        self.deps.update(recorded)
        # Track the exec graph id (tools_id >> 32, shared by all the graph's nodes) so the
        # observer's graph-destroy hook can purge these edges from the map on destruction.
        torch_cuda_graph._recorded_exec_ids.add(next(iter(recorded)) >> 32)


_recorder = _GraphDependencyRecorder()


def arm_graph_dependency_recording() -> None:
    """Arm the process-global recorder (idempotent). Call before graphs are captured."""
    _recorder.arm()


def graph_dependencies() -> dict[int, list[int]]:
    """The persistent, shared graph_node_id -> predecessor graph_node_ids map."""
    return _recorder.deps
