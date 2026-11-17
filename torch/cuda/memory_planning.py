r"""CUDA graphs whose memory is planned offline.

A :class:`MemoryPlannedGraph` captures a function as a CUDA graph and chooses
the address of every tensor the capture allocates, instead of the caching
allocator's private pool. It sees each allocation and free of the capture at
its exact size and plans all of them into one arena (in C++, with a fixed set
of strategies), so that allocations that may be in use on the GPU at the same
time never overlap::

    step = torch.cuda.MemoryPlannedGraph(train_step)  # reads static_input
    for batch in loader:
        static_input.copy_(batch)
        loss = step()  # first call: capture; later calls: replay

The first call captures the function twice (:func:`torch.cuda.graph` with an
allocator): once to record the allocations into reserved virtual address space
(all mapped to the same scratch memory), then, once each of them has an offset,
again to place each allocation in the arena. The recording graph is
discarded; the placed one is replayed.

Sharing rule. Allocation ``j`` may reuse the memory of allocation ``i`` if ``i``
was freed before ``j`` was allocated (on the host) and every node of ``i``'s
free frontier is an ancestor of (or equal to) a node of ``j``'s allocation
frontier, so that all work issued after ``j``'s allocation depends on the work
that used ``i``. The frontiers are read from the capture
(``cudaStreamGetCaptureInfo``): at each allocation, the allocating stream's; at
each free, the allocation's stream's and that of every stream recorded for it
with :meth:`torch.Tensor.record_stream` (a stream that is not being captured
has none: its work is not in the graph). This allows sharing across streams
along synchronization edges. As in eager mode, a tensor used on another stream
must either be recorded with ``record_stream`` or not be freed before that
stream's work has been joined.
"""

from __future__ import annotations

import bisect
import dataclasses
import gc
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["MemoryPlannedGraph"]

# cudaMalloc's alignment guarantee, so placed tensors keep every alignment property
# (e.g. what cuBLASLt picks kernels by) that code can assume of a fresh allocation
_ALIGNMENT = 256


def _python_allocator(
    alloc_fn: Callable[[int, int, int], int],
    free_fn: Callable[[int, int, int, int], None],
    record_stream_fn: Callable[[int, int], None] | None = None,
) -> torch.cuda.memory._CUDAAllocator:
    return torch.cuda.memory._CUDAAllocator(
        torch._C._cuda_createPythonAllocator(alloc_fn, free_fn, record_stream_fn)
    )


_CONDITIONAL_ERROR = (
    "MemoryPlannedGraph: allocations and record_stream inside a "
    "conditional node's body are not supported"
)


@dataclasses.dataclass
class _Allocation:
    size: int
    stream: int
    alloc_time: int
    free_time: int | None = None  # None: live at the end of the capture
    used_streams: set[int] = dataclasses.field(default_factory=set)
    # capture frontiers, as graph node handles while recording, then as node
    # indices once the graph is known
    alloc_nodes: list[int] = dataclasses.field(default_factory=list)
    free_nodes: list[int] = dataclasses.field(default_factory=list)
    # the numbers of frontier nodes as recorded, compared by the placer
    alloc_width: int = 0
    free_width: int = 0


class _Recorder:
    """The recording allocator: hands out distinct, never-reused addresses from
    reserved virtual address space, all mapped to the same scratch memory, and
    logs every allocation and free of the capture. The reservation is released
    once no recorded allocation is in use (the allocator's ``free_fn``)."""

    # the first reservation of address space; each further one is at least
    # twice the previous (and the allocation), so the reservation stays within
    # about twice the recorded bytes
    _CHUNK = 1 << 30
    # The scratch memory: one chunk, mapped over every reservation (with one
    # granularity-sized chunk for the tail). Captured kernels do not run, so
    # nothing reads or writes it, but kernel launchers may check their pointer
    # arguments (Triton's with cuPointerGetAttribute, which fails on unmapped
    # addresses). It is not exportable, so NCCL cannot register it.
    _SCRATCH = 256 << 20

    def __init__(self) -> None:
        if torch.version.hip:
            raise RuntimeError("MemoryPlannedGraph is not supported on ROCm")
        self.allocations: list[_Allocation] = []
        self.num_nodes = 0
        self.edges: list[tuple[int, int]] = []
        self._reserved: list[tuple[int, int]] = []
        self._mapped: list[tuple[int, int]] = []
        self._scratch: dict[int, int] = {}  # chunk size -> handle
        self._granularity = 0
        self._top = self._end = 0
        self._recording = True
        self._live: dict[int, int] = {}
        self._time = 0
        self._released = False
        # the graph being captured (set by the capture), and whether a stream
        # of another graph was seen: a conditional node's body, whose stream is
        # destroyed when the body ends, so it is not queried again (end() raises)
        self.root = 0
        self._nested = False

    def alloc(self, size: int, device: int, stream: int) -> int:
        graph, frontier = torch._C._cuda_streamCaptureInfo(stream)
        self._nested |= graph != self.root
        aligned = (size + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        if self._top + aligned > self._end:
            self._reserve(
                max(
                    2 * self._reserved[-1][1] if self._reserved else self._CHUNK,
                    aligned,
                ),
                device,
            )
        ptr = self._top
        self._top += aligned
        t, self._time = self._time, self._time + 1
        self._live[ptr] = len(self.allocations)
        self.allocations.append(
            _Allocation(
                size, stream, t, alloc_nodes=frontier, alloc_width=len(frontier)
            )
        )
        return ptr

    def _reserve(self, nbytes: int, device: int) -> None:
        C = torch._C
        if not self._scratch:
            self._granularity = C._cuda_vmmGranularity(device)
            for n in (self._SCRATCH, self._granularity):
                self._scratch[n] = C._cuda_vmmCreate(n, device)
        g = self._granularity
        nbytes = (nbytes + g - 1) // g * g
        base = C._cuda_vmmAddressReserve(nbytes)
        self._reserved.append((base, nbytes))
        offset = 0
        while offset < nbytes:
            n = self._SCRATCH if nbytes - offset >= self._SCRATCH else g
            C._cuda_vmmMap(base + offset, n, self._scratch[n])
            self._mapped.append((base + offset, n))
            offset += n
        C._cuda_vmmSetAccess(base, nbytes, device)
        self._top, self._end = base, base + nbytes

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        try:
            index = self._live.get(ptr)
            if index is not None and self._recording and not self._nested:
                a = self.allocations[index]
                # (if a query raises, the free stays unrecorded: live to the end)
                frontier = [
                    h
                    for s in {a.stream} | a.used_streams
                    for h in torch._C._cuda_streamCaptureInfo(s)[1]
                ]
                a.free_time, self._time = self._time, self._time + 1
                a.free_nodes, a.free_width = frontier, len(frontier)
        finally:
            self._live.pop(ptr, None)
            self._maybe_release()

    def record_stream(self, ptr: int, stream: int) -> None:
        index = self._live.get(ptr)
        if index is not None:
            if torch._C._cuda_streamCaptureInfo(stream)[0] not in (0, self.root):
                self._nested = True
            self.allocations[index].used_streams.add(stream)

    def end(self, nodes: list[int], edges: list[tuple[int, int]]) -> None:
        """Ends the recording and turns the frontiers into node indices of the
        captured graph."""
        self._recording = False
        if self._nested:
            raise RuntimeError(_CONDITIONAL_ERROR)
        index = {h: i for i, h in enumerate(nodes)}
        self.num_nodes = len(nodes)
        self.edges = [(index[a], index[b]) for a, b in edges]
        try:
            for a in self.allocations:
                a.alloc_nodes = [index[h] for h in a.alloc_nodes]
                a.free_nodes = [index[h] for h in a.free_nodes]
        except KeyError:
            # a frontier in another graph: a conditional node's body
            raise RuntimeError(_CONDITIONAL_ERROR) from None

    def release(self) -> None:
        self._released = True
        self._maybe_release()

    def _maybe_release(self) -> None:
        if self._released and not self._live:
            C = torch._C
            for ptr, nbytes in self._mapped:
                C._cuda_vmmUnmap(ptr, nbytes)
            for base, nbytes in self._reserved:
                C._cuda_vmmAddressFree(base, nbytes)
            for handle in self._scratch.values():
                C._cuda_vmmRelease(handle)
            self._mapped.clear()
            self._reserved.clear()
            self._scratch.clear()


class _Placer:
    """The placing allocator of a step (the captures of a recording): hands out
    ``arena + offset`` in the recording's allocation order and checks that the
    captures repeat the recording. Each capture gets its own allocator over the
    placer (:meth:`capture`), and its graph holds the placer until
    :meth:`release`. The arena lives until every graph released the placer and
    every allocation placed in it is freed (the allocator's ``free_fn``): a
    placed tensor may outlive the graph (e.g. a ``.grad`` kept by an
    optimizer), and its memory must not be handed out again meanwhile."""

    def __init__(
        self,
        allocations: list[_Allocation],
        offsets: list[int],
        arena: torch.Tensor,
    ) -> None:
        self.allocations = allocations
        self.offsets = offsets
        self._arena: torch.Tensor | None = arena  # the planned memory
        self.base = arena.data_ptr()
        self.device = torch.cuda.current_device()  # the captures' device
        self.count = 0
        self._holders = 0
        self._released = False
        # recorded frees, by the number of allocations made before them
        alloc_times = [a.alloc_time for a in allocations]
        self._frees = sorted(
            (bisect.bisect_left(alloc_times, a.free_time), k)
            for k, a in enumerate(allocations)
            if a.free_time is not None
        )
        self._next_free = 0
        self._freed = [False] * len(allocations)
        self._live: dict[int, int] = {}
        self._mismatch: str | None = None  # found by free(), raised later

    def _check_frees(self, before: int) -> None:
        """Every allocation that the recording freed before allocation
        ``before`` must have been freed by now."""
        while (
            self._next_free < len(self._frees)
            and self._frees[self._next_free][0] <= before
        ):
            k = self._frees[self._next_free][1]
            if not self._freed[k]:
                where = (
                    f"before allocation {before}"
                    if before < len(self.allocations)
                    else "by the end of the capture"
                )
                raise RuntimeError(
                    f"MemoryPlannedGraph: allocation {k} "
                    f"({self.allocations[k].size} bytes) was freed {where} when "
                    "recorded, but not when captured again; the captured function "
                    "must free deterministically"
                )
            self._next_free += 1

    def alloc(self, size: int, device: int, stream: int) -> int:
        width = len(torch._C._cuda_streamCaptureInfo(stream)[1])
        k = self.count
        if k >= len(self.allocations) or (
            self.allocations[k].size,
            self.allocations[k].stream,
        ) != (size, stream):
            recorded = (
                f"{self.allocations[k].size} bytes on stream "
                f"{self.allocations[k].stream:#x} were recorded"
                if k < len(self.allocations)
                else f"only {k} allocations were recorded"
            )
            raise RuntimeError(
                "MemoryPlannedGraph: the second capture made a different allocation "
                f"than the first (allocation {k}: {size} bytes on stream "
                f"{stream:#x}, but {recorded}); the captured function must "
                "allocate deterministically"
            )
        if width != self.allocations[k].alloc_width and self._mismatch is None:
            # (raised by finish(), once the capture has ended: raising here
            # leaves side streams unjoined, which hides the error)
            self._mismatch = (
                f"allocation {k} depends on {width} nodes, but on "
                f"{self.allocations[k].alloc_width} when recorded"
            )
        self._check_frees(k)
        ptr = self.base + self.offsets[k]
        self._live[ptr] = k
        self.count = k + 1
        return ptr

    def free(self, ptr: int, size: int, device: int, stream: int) -> None:
        k = self._live.pop(ptr, None)
        try:
            if k is not None:
                self._freed[k] = True
                a = self.allocations[k]
                if a.free_time is not None and self._mismatch is None:
                    width = sum(
                        len(torch._C._cuda_streamCaptureInfo(s)[1])
                        for s in {a.stream} | a.used_streams
                    )
                    if width != a.free_width:
                        # (an exception would be lost in the free callback)
                        self._mismatch = (
                            f"the free of allocation {k} depends on {width} "
                            f"nodes, but on {a.free_width} when recorded"
                        )
        finally:
            self._maybe_release()

    def _check_mismatch(self) -> None:
        if self._mismatch is not None:
            raise RuntimeError(
                f"MemoryPlannedGraph: the second capture's dependencies differ from "
                f"the first's ({self._mismatch}); the captured function must be "
                "deterministic, including its stream synchronization"
            )

    def record_stream(self, ptr: int, stream: int) -> None:
        # the arena is returned to its allocator once every placed tensor is
        # freed: its side-stream uses are those of the placed tensors. A use
        # inside a capture is part of the graph, which completes on the
        # replay's stream (recorded by replay()); recording it would also
        # leave a stale entry in the caching allocator's capture-use map
        if torch._C._cuda_streamCaptureInfo(stream)[0] == 0:
            device = torch._C._cuda_streamDevice(stream)
            self._arena.record_stream(torch.cuda.ExternalStream(stream, device=device))

    def capture(self) -> torch.cuda.memory._CUDAAllocator:
        """The allocator of one capture of the step."""
        self._holders += 1
        return _python_allocator(self.alloc, self.free, self.record_stream)

    def finish(self) -> None:
        """Checks, after the step's last capture, that it made every allocation
        and free of the recording."""
        if self.count != len(self.allocations):
            raise RuntimeError(
                "MemoryPlannedGraph: the second capture made fewer allocations "
                f"({self.count}) than the first ({len(self.allocations)}); the "
                "captured function must allocate deterministically"
            )
        self._check_frees(len(self.allocations))
        self._check_mismatch()

    def release(self) -> None:
        """A graph placed with the placer is destroyed (or its capture failed)."""
        self._holders -= 1
        self._released = self._holders <= 0
        self._maybe_release()

    def _maybe_release(self) -> None:
        if self._released and not self._live:
            self._arena = None


def _check_recording_dropped(recorder: _Recorder) -> None:
    """After the placing captures: no tensor of the recording is referenced."""
    if recorder._live:  # (cyclic garbage may still hold some)
        gc.collect()
    if recorder._live:
        raise RuntimeError(
            f"MemoryPlannedGraph: {len(recorder._live)} tensors created by "
            "the first capture (scratch memory) are still referenced after "
            "the second: fn keeps tensors it creates in Python state (e.g. "
            "appends them to a list, or creates state on first use: warm up "
            "fn first)"
        )


def _plan(recorder: _Recorder) -> tuple[list[int], torch.Tensor]:
    """Plans the recorded allocations (the C++ planner): their offsets, and the
    arena they are relative to, from the caching allocator."""
    allocations = recorder.allocations
    alloc_ptr, alloc_nodes, free_ptr, free_nodes = [0], [], [0], []
    for a in allocations:
        alloc_nodes.extend(a.alloc_nodes)
        alloc_ptr.append(len(alloc_nodes))
        free_nodes.extend(a.free_nodes)
        free_ptr.append(len(free_nodes))
    problem = torch._C._cuda_PlanningProblem(
        [a.size for a in allocations],
        [a.alloc_time for a in allocations],
        [-1 if a.free_time is None else a.free_time for a in allocations],
        alloc_ptr,
        alloc_nodes,
        free_ptr,
        free_nodes,
        recorder.num_nodes,
        [v for e in recorder.edges for v in e],
    )
    _, arena_bytes, offsets = problem.plan()
    del problem  # (its conflict matrix)
    arena = torch.empty(max(arena_bytes, 1), dtype=torch.uint8, device="cuda")
    return offsets, arena


class MemoryPlannedGraph:
    r"""A CUDA graph of ``fn`` whose memory is planned offline.

    The first call captures ``fn()`` twice, as described in
    :mod:`torch.cuda.memory_planning` (record, plan, place),
    then replays the graph; later calls replay it. Every call returns the same
    outputs, which live in the arena and are overwritten by each replay. As with
    :func:`torch.cuda.graph`, ``fn`` takes no arguments: its inputs are static
    tensors it closes over, and anything else it reads is fixed at capture.

    Warm up ``fn`` eagerly before the first call, as for any capture. ``fn`` must
    be deterministic, including its stream synchronization: the second capture
    must repeat the first, which is checked in part (allocation sizes, order and
    streams, frees, frontier widths, and the graph's size). Python state is the
    caller's: state ``fn`` updates across calls (e.g. ``.grad``) must exist
    before the first call or be dropped by ``fn`` before it is created again,
    and ``fn`` must not carry tensors it creates across calls in Python state
    (update them in place); a tensor of the first capture still referenced
    after the second raises. After a failed capture, drop state the recording
    created (e.g. ``.grad``) before running eagerly again. A capture's
    allocations must be made by one thread at a time (the calling thread, or
    autograd's while it waits). Conditional nodes inside ``fn`` are not
    supported and raise.

    Example::

        >>> # xdoctest: +SKIP("requires CUDA")
        >>> step = torch.cuda.MemoryPlannedGraph(train_step)
        >>> for batch in loader:
        ...     static_input.copy_(batch)
        ...     loss = step()

    Args:
        fn (callable): the function to capture.

    Attributes:
        graph (torch.cuda.CUDAGraph): the placed graph, after the first call.
            Owned by the MemoryPlannedGraph: replay it through :meth:`replay`
            (which orders the arena's release after the replay), and not after
            :meth:`release`, when its memory may be in use elsewhere.
        outputs: what ``fn`` returned when placed; :meth:`replay` returns them.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: Any = None
        self._placer: _Placer | None = None
        self._topology = (0, 0)  # numbers of nodes and edges of the recording
        self._replay_stream: torch.cuda.Stream | None = None

    def __call__(self) -> Any:
        if self.graph is None:
            self.capture()
        return self.replay()

    def capture(self) -> Any:
        r"""Captures ``fn()`` and returns its outputs. Capturing
        again re-plans: the previous graph is destroyed first, and its arena is
        freed once no tensor placed in it is alive."""
        if self.graph is not None:
            self.release()
        recorder = _Recorder()
        try:
            # the recording's outputs refer to scratch memory: dropped
            self._record(recorder)
            offsets, memory = _plan(recorder)
            placer = _Placer(recorder.allocations, offsets, memory)
            del memory
            self._place(placer)
            placer.finish()
            _check_recording_dropped(recorder)
        except BaseException:
            if self.graph is not None:  # placed, but the step's checks failed
                self.release()
            raise
        finally:
            recorder.release()  # (once no recorded tensor is alive)
        return self.outputs

    def _record(self, recorder: _Recorder) -> Any:
        """The first capture: records ``fn`` with ``recorder`` and returns the
        recording's outputs, which refer to scratch memory."""
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        allocator = _python_allocator(
            recorder.alloc, recorder.free, recorder.record_stream
        )
        with torch.cuda.graph(graph, allocator=allocator):
            recorder.root = torch._C._cuda_streamCaptureInfo(
                torch.cuda.current_stream().cuda_stream
            )[0]
            outputs = self.fn()
        nodes, edges = torch._C._cuda_graphTopology(graph.raw_cuda_graph())
        recorder.end(nodes, edges)
        self._topology = (len(nodes), len(edges))
        # (tensors live after the capture carry the allocator, which holds the
        # recorder; the recorder must not hold the allocator in turn)
        return outputs

    def _place(self, placer: _Placer) -> None:
        """The second capture: places ``fn``'s allocations with ``placer``,
        checks that the capture repeats the recording's topology and
        instantiates the graph, which holds the placer until :meth:`release`."""
        allocator = placer.capture()
        try:
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, allocator=allocator):
                outputs = self.fn()
            del allocator
            # (counts only: fetching the placed graph's nodes and edges took
            # 0.25-0.36 s for an 8B FSDP step, 2 ms for its recording)
            size = torch._C._cuda_graphSize(graph.raw_cuda_graph())
            if size != self._topology:
                raise RuntimeError(
                    f"MemoryPlannedGraph: the second capture has {size[0]} nodes "
                    f"and {size[1]} edges, the first {self._topology[0]} and "
                    f"{self._topology[1]}; the captured function must be "
                    "deterministic"
                )
            graph.instantiate()
        except BaseException:
            placer.release()
            raise
        self.graph, self.outputs, self._placer = graph, outputs, placer
        self._replay_stream = None

    def replay(self) -> Any:
        r"""Replays the graph and returns the outputs of :meth:`capture`."""
        if self.graph is None:
            raise RuntimeError("MemoryPlannedGraph.replay() called before capture()")
        # (CUDAGraph.replay launches on the capture device's current stream)
        stream = torch.cuda.current_stream(
            self._placer.device if self._placer else None
        )
        if stream != self._replay_stream and self._placer is not None:
            # the arena is returned to its allocator stream-ordered after the
            # replays on every stream they used
            self._placer._arena.record_stream(stream)
            self._replay_stream = stream
        self.graph.replay()
        return self.outputs

    def release(self) -> None:
        r"""Destroys the graph. The arena is freed once no tensor placed in it is
        alive, ordered after the graph's replays (on any stream); the outputs
        must not be used afterwards."""
        self.graph = self.outputs = None
        placer, self._placer = self._placer, None
        if placer is not None:
            placer.release()

    def __del__(self) -> None:
        if getattr(self, "_placer", None) is not None:
            self.release()
