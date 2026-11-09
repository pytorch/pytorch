"""Helpers for testing the symmetric-memory stream-ordering guard
(GroupStreamGuard): read the dependencies of a captured graph, and observe
eager ordering with stream gates. Both need cuda.bindings."""

import os
import time
import unittest
from collections.abc import Callable

import torch


def captured_graph_kernels_and_edges(
    graph: torch.cuda.CUDAGraph,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Kernel nodes and edges of a graph captured with keep_graph=True, read
    back through cuda.bindings. Requires cuda.bindings."""
    from cuda.bindings import runtime as rt

    from torch.cuda._utils import _check_cuda_bindings as check

    raw = graph.raw_cuda_graph()
    _, num_nodes = check(rt.cudaGraphGetNodes(raw, numNodes=0))
    nodes, _ = check(rt.cudaGraphGetNodes(raw, numNodes=num_nodes))
    kind = rt.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernels = [int(n) for n in nodes if check(rt.cudaGraphNodeGetType(n)) == kind]
    # cuda-bindings 13 inserts an edge-data value before the count, so index
    # from the ends to accept 12 and 13.
    num_edges = check(rt.cudaGraphGetEdges(raw, numEdges=0))[-1]
    if num_edges == 0:
        return kernels, []
    srcs, dsts = check(rt.cudaGraphGetEdges(raw, numEdges=num_edges))[:2]
    return kernels, [(int(a), int(b)) for a, b in zip(srcs, dsts, strict=True)]


def graph_path_exists(edges: list[tuple[int, int]], src: int, dst: int) -> bool:
    """Whether dst is reachable from src. Reachability rather than a search
    for event nodes: a captured dependency may be lowered to a plain edge."""
    successors: dict[int, list[int]] = {}
    for a, b in edges:
        successors.setdefault(a, []).append(b)
    seen: set[int] = set()
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(successors.get(node, ()))
    return False


def captured_signal_pad_order(
    stream: torch.cuda.Stream,
    launches_a: list[Callable[[], object]],
    launches_b: list[Callable[[], object]],
) -> tuple[bool, bool]:
    r"""Report the orderings the launches insert, read from a CUDA graph that
    is captured but never replayed::

        stream: fork -> [launches_a] -> marker
                  \-> side: [launches_b]

    Returns ``(after_a, after_marker)``: whether the first of ``launches_b``
    depends on the last of ``launches_a``, and on the marker. The fork comes
    first, so any such dependency was inserted by the launches. Launches take
    no arguments and must be warmed up on ``stream``. Requires cuda.bindings."""
    from cuda.bindings import runtime as rt

    from torch.cuda._utils import _check_cuda_bindings as check

    def tail(s: torch.cuda.Stream) -> int:
        # The captured node the next launch on s will depend on.
        # (status, id, graph, deps, [edge data in cuda-bindings 13], count)
        info = check(rt.cudaStreamGetCaptureInfo(s.cuda_stream))
        deps, n = info[3], info[-1]
        if n != 1:
            raise AssertionError(f"expected one node at the capture tail, got {n}")
        return int(deps[0])

    side = torch.cuda.Stream(device=stream.device)
    marker = torch.zeros(1, device=stream.device)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        side.wait_stream(stream)
        for launch in launches_a:
            launch()
        a_last = tail(stream)
        marker.add_(1)
        marker_node = tail(stream)
        with torch.cuda.stream(side):
            launches_b[0]()
            b_first = tail(side)
            for launch in launches_b[1:]:
                launch()
        stream.wait_stream(side)

    _, edges = captured_graph_kernels_and_edges(graph)
    return (
        graph_path_exists(edges, a_last, b_first),
        graph_path_exists(edges, marker_node, b_first),
    )


# Bounds for gated_signal_pad_order's polls. A slow device can only let a
# broken guard pass (blocked probe), or fail a correct one if launches take
# over a minute (finish timeout); see its docstring.
_GATE_BLOCKED_PROBE_S = 1.0
_GATE_FINISH_TIMEOUT_S = 60.0


def gated_signal_pad_order(
    launches_a: list[Callable[[], object]],
    launches_b: list[Callable[[], object]],
    expect_ordered: bool,
    between: Callable[[], object] | None = None,
) -> tuple[bool, bool]:
    """The eager counterpart of ``captured_signal_pad_order``::

        stream_a: gate 0 -> [launches_a] -> gate 1
        stream_b: [launches_b]

    Gates are cuStreamWaitValue32 waits on device flags. Returns
    ``(while_gated, after_open)``: whether ``launches_b`` finished while gate 0
    was closed, and once it opened with gate 1 still closed. The wait that
    ``expect_ordered`` says should stay blocked gives up after
    ``_GATE_BLOCKED_PROBE_S``, the others after ``_GATE_FINISH_TIMEOUT_S``: a
    broken guard passes only if ``launches_b`` outlast the probe, a correct
    one fails only if they outlast the timeout. Launches must be warmed up:
    loading a kernel lazily waits for the whole device, which stays busy while
    a stream is gated. ``between``, if given, runs after ``launches_a`` are
    issued and before ``launches_b``. Requires cuda.bindings."""
    # With blocking launches, launching an op correctly ordered behind the
    # gated one waits on the host before the gate can open.
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        raise unittest.SkipTest("deadlocks with CUDA_LAUNCH_BLOCKING=1")
    # With one hardware queue, every stream, the gate opener's included, waits
    # behind the closed gate.
    if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1":
        raise unittest.SkipTest("deadlocks with CUDA_DEVICE_MAX_CONNECTIONS=1")
    from cuda.bindings import driver as drv

    from torch.cuda._utils import _check_cuda_bindings_driver as check

    gates = torch.zeros(2, dtype=torch.int32, device="cuda")
    stream_a, stream_b, opener = (torch.cuda.Stream() for _ in range(3))
    # Pool streams do not wait for the current stream, where gates is zeroed.
    for s in (stream_a, opener):
        s.wait_stream(torch.cuda.current_stream())
    geq = drv.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ

    def gate(i: int) -> None:
        ptr = gates[i].data_ptr()
        check(drv.cuStreamWaitValue32(stream_a.cuda_stream, ptr, 1, geq))

    def open_gate(i: int) -> None:
        ptr = gates[i].data_ptr()
        check(drv.cuStreamWriteValue32(opener.cuda_stream, ptr, 1, 0))

    def finished_within(event: torch.cuda.Event, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not event.query():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    b_done = torch.cuda.Event()
    try:
        with torch.cuda.stream(stream_a):
            gate(0)
            for launch in launches_a:
                launch()
            gate(1)
        if between is not None:
            between()
        with torch.cuda.stream(stream_b):
            for launch in launches_b:
                launch()
            b_done.record()
        probe = _GATE_BLOCKED_PROBE_S if expect_ordered else _GATE_FINISH_TIMEOUT_S
        while_gated = finished_within(b_done, probe)
        open_gate(0)
        after_open = finished_within(b_done, _GATE_FINISH_TIMEOUT_S)
    finally:
        open_gate(0)
        open_gate(1)
        torch.cuda.synchronize()
    return while_gated, after_open
