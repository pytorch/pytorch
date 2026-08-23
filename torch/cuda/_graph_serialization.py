"""Serialize a captured CUDA graph so a later process can replay it without capturing.

Graph capture dominates cold start for large models, and a captured graph is not
portable: its kernel nodes hold ``CUfunction`` handles into modules this process
loaded, and their arguments embed device pointers verbatim -- including inside
opaque packed structs (cuBLASLt hands its kernels a 2560-byte blob with pointers
at undocumented offsets). Nothing can be relocated after the fact.

So reproduce the environment instead of rewriting the graph. Three pieces travel
in the archive, each recoverable exactly:

* device code, captured on the way in by :mod:`torch.cuda._graph_kernel_capture`
  and matched back by mangled name -- there is no API to read a module's image
  back out of the driver, and JIT-generated kernels (cuBLASLt's ``nvjet_*``)
  exist in no file on disk;
* memory *layout* -- the allocator's expandable segments and the blocks inside
  them, re-reserved at the virtual addresses they were recorded at (see Note
  [Expandable Segment Reserved Address]);
* the graph template itself -- node parameters, launch attributes and topology --
  replayed byte-for-byte because the memory underneath it did not move.

What that buys is that no argument is ever rewritten, which is the only reason
opaque argument blobs are serializable at all.

Contents are deliberately not carried. Parameters keep training after a graph is
captured, so bytes written at capture time would be stale by the time anyone
replays, and in serving they are already being loaded from a checkpoint -- copying
them into the archive would make it model-sized to duplicate work the caller is
doing anyway. So the archive is structure and code, tens of megabytes rather than
gigabytes, and the caller fills the restored addresses at load. What that leaves
open is memory whose contents matter but belong to no tensor the caller can name;
graph-safe RNG state is the known instance, tracked separately.

Not everything is serializable, and the refusals are properties rather than a
denylist of libraries. A node type must be one that can be reproduced -- pure-value
parameters, or an event whose identity is all that matters -- and every device
pointer the graph references must live in memory the caching allocator owns, so the
archive can carry it.

Between them those reject the cases that hold state we cannot rebuild. NCCL
collectives fail the pointer check on either transport, since their kernel
arguments embed ``ncclDevComm`` state that NCCL allocated itself, and on the
network path they also bring a host node. Symmetric memory reserves its own address
space and fails the same way. Legacy cuBLAS is refused because its workspace is not
on expandable segments (cuBLASLt is fine).
"""

from __future__ import annotations

import bisect
import ctypes
import json
import warnings
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.cuda.graphs import CUDAGraph


ARCHIVE_VERSION = 1

# Where the last load() spent its time, phase name -> seconds. Diagnostic only:
# the phases have very different scaling (reserving address space is per segment,
# building nodes is per node, loading cubins is per unique module), so a single
# total hides which one a slow load is actually in.
_LAST_LOAD_PROFILE: dict[str, float] = {}

MANIFEST_PATH = "manifest.json"
CUBIN_DIR = "cubins"
SEGMENT_DIR = "segments"

# Node types that can be reproduced. Kernel/memcpy/memset/empty parameters are
# pure values, so replaying them needs nothing but the same addresses.
#
# Event nodes are different: a CUevent handle is process-local and cannot be
# recreated. What is recorded instead is which event each node refers to, so a
# record and the waits on it stay paired, and load creates one fresh event per
# distinct handle. That reproduces the ordering *within* the graph, which is what
# these nodes are almost always there for, and deliberately does not reproduce any
# interaction with an event outside it -- a wait on an event some other stream
# records will simply be satisfied by the graph's own record.
#
# HOST nodes stay refused: their payload is a host function pointer plus an opaque
# userData, which cannot be rebound from outside the library that created it.
# EXT_SEMAS_* are OS-level handles, likewise.
_SERIALIZABLE_NODE_TYPES = (
    "CU_GRAPH_NODE_TYPE_KERNEL",
    "CU_GRAPH_NODE_TYPE_MEMCPY",
    "CU_GRAPH_NODE_TYPE_MEMSET",
    "CU_GRAPH_NODE_TYPE_EMPTY",
    "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
    "CU_GRAPH_NODE_TYPE_WAIT_EVENT",
)

# CUfunction attributes that a freshly loaded module does not inherit and that
# cuFuncSetAttribute can put back. MAX_DYNAMIC_SHARED_SIZE_BYTES is derivable from
# the node's sharedMemBytes, but NON_PORTABLE_CLUSTER_SIZE_ALLOWED is host-side
# state the node does not carry at all -- cuBLASLt sets it, and without it a
# kernel with a large cluster fails to launch.
_SETTABLE_FUNC_ATTRS = (
    "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES",
    "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED",
    "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT",
    "CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT",
    "CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH",
)

# Launch attributes stored on a kernel node. Read and replayed as the raw union
# bytes, so a value we do not interpret still round-trips.
_KERNEL_NODE_ATTRS = (
    "CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION",
    "CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE",
    "CU_LAUNCH_ATTRIBUTE_COOPERATIVE",
    "CU_LAUNCH_ATTRIBUTE_PRIORITY",
    "CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN",
)

# Cheap pre-filter before asking the driver whether a word is a pointer: device
# virtual addresses are 48-bit (measured -- 128 TiB reserves succeed, 255 TiB
# cumulatively, 256 TiB fails) and no allocation lives in the bottom 4 GiB.
_MIN_DEVICE_ADDRESS = 1 << 32
_MAX_DEVICE_ADDRESS = 1 << 48


class UnserializableGraph(RuntimeError):
    """A captured graph cannot be serialized. The message names the reason."""


def _driver() -> Any:
    from cuda.bindings import driver  # pyrefly: ignore[missing-import]

    return driver


def _chk(result: Any) -> Any:
    from torch.cuda._utils import _check_cuda_bindings_driver

    return _check_cuda_bindings_driver(result)


"""
Note [Edge data]
~~~~~~~~~~~~~~~~
An edge carries a from_port, a to_port and a type, not just endpoints, and a
from_port of PROGRAMMATIC is what lets a consumer kernel start before its producer
has retired. Losing that is not a wrong answer, it is a graph that replays more
slowly than the one that was saved -- and getting it wrong in the other direction,
applying PROGRAMMATIC where it was never asked for, would be worse.

Two hazards, one on each side.

Reading needs cuda.bindings 13.3 or newer. Before that, cuGraphGetEdges returned
CUgraphEdgeData objects wrapping a pointer it had already freed, so the ports read
back as whatever happened to be in that memory. 13.3 deep-copies (the values
survive heavy heap churn). Rather than work around a use-after-free, save refuses
on older bindings.

Writing must issue one call per distinct edge data, because
cuGraphAddDependencies_v2 applies edgeData[0] to every edge in the call and ignores
the rest. That is the driver, not the binding: it reproduces through raw ctypes
with a byte-verified contiguous array, on edges sharing no endpoints, on drivers
13000, 13030 and 13040, and both cuGraphGetEdges_v2 and
cuGraphNodeGetDependencies_v2 agree the stored data really is uniform. Edges added
in separate calls do keep distinct data, so grouping by edge data is enough, and an
ordinary graph still pays a single call.
"""

# cuda.bindings older than this returns freed memory for CUgraphEdgeData.
_MIN_EDGE_DATA_BINDINGS = (13, 3)


def _check_edge_data_readable() -> None:
    import cuda.bindings  # pyrefly: ignore[missing-import]

    version = getattr(cuda.bindings, "__version__", "0")
    try:
        parsed = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return  # unparsable, e.g. a dev build; assume it is new enough
    if parsed < _MIN_EDGE_DATA_BINDINGS:
        raise UnserializableGraph(
            f"cuda-bindings {version} returns freed memory for graph edge data, so "
            "an edge's ports cannot be read reliably and a saved graph could replay "
            "with dependency semantics it never had. Saving needs cuda-bindings "
            f"{'.'.join(str(part) for part in _MIN_EDGE_DATA_BINDINGS)} or newer."
        )


def _kernel_args(driver: Any, params: Any) -> tuple[list[bytes], bytes | None]:
    """A kernel node's arguments, as either one blob per parameter or the single
    packed buffer libraries pass through ``extra``.

    cuBLASLt uses the latter (``CU_LAUNCH_PARAM_BUFFER_POINTER``), so both have to
    be handled; the bytes are copied out verbatim either way. A kernel taking no
    arguments has neither, which reads as an empty argument list rather than as a
    packed blob to be walked.
    """
    if not int(params.kernelParams) and not int(params.extra):
        return [], None

    if int(params.kernelParams):
        sizes = []
        while True:
            err, _offset, size = driver.cuFuncGetParamInfo(params.func, len(sizes))
            if err != driver.CUresult.CUDA_SUCCESS:
                break
            sizes.append(size)
        slots = ctypes.cast(int(params.kernelParams), ctypes.POINTER(ctypes.c_void_p))
        return [ctypes.string_at(slots[i], sz) for i, sz in enumerate(sizes)], None

    # extra[] = {BUFFER_SIZE, &nbytes, BUFFER_POINTER, buffer, END}
    extra = ctypes.cast(int(params.extra), ctypes.POINTER(ctypes.c_void_p))
    tagged: dict[int, Any] = {}
    i = 0
    while extra[i]:
        tagged[int(extra[i])] = extra[i + 1]
        i += 2
    nbytes = ctypes.cast(tagged[2], ctypes.POINTER(ctypes.c_size_t))[0]
    return [], ctypes.string_at(tagged[1], nbytes)


def _referenced_pointers(
    driver: Any, blob: bytes, memo: dict[int, str | None] | None = None
) -> set[tuple[int, str]]:
    """The words in ``blob`` the driver recognizes as memory, each tagged
    ``"device"`` or ``"host"``.

    Asking cuPointerGetAttribute is what makes this usable: no byte pattern
    distinguishes a pointer from a pair of adjacent 32-bit scalars, and guessing
    rejects valid graphs. A cuBLASLt argument blob contains 0x2000000020 (two tile
    dimensions of 32) and 0xffffffc0000000, both of which look like plausible
    addresses and neither of which the driver recognizes.

    Every byte offset is examined, not just aligned ones, so a pointer inside a
    packed struct that does not honour the usual alignment is still found. Doing
    that a byte at a time in Python is what costs -- 343k iterations over a
    350 KB graph, ~84 ms, against 0.7 ms for the driver calls it feeds. So the
    candidates are extracted with eight overlapping uint64 views, one per byte
    alignment, and only the distinct survivors reach the driver. ``memo`` carries
    the driver's answers across nodes, which matters because the same weight
    address recurs in every layer.
    """
    import numpy as np

    if memo is None:
        memo = {}
    attribute = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE
    host_type = int(driver.CUmemorytype.CU_MEMORYTYPE_HOST)
    candidates: set[int] = set()
    for shift in range(8):
        tail = blob[shift:]
        count = len(tail) // 8
        if count == 0:
            continue
        words = np.frombuffer(tail, dtype="<u8", count=count)
        selected = words[(words >= _MIN_DEVICE_ADDRESS) & (words < _MAX_DEVICE_ADDRESS)]
        candidates.update(selected.tolist())

    found = set()
    for word in candidates:
        known = memo.get(word)
        if known is None:
            err, kind = driver.cuPointerGetAttribute(attribute, word)
            # Registered (pinned) host memory answers this too, and reporting it as
            # a stray device pointer would send the reader looking in the wrong
            # place -- so it is recorded distinctly.
            known = (
                None
                if err != driver.CUresult.CUDA_SUCCESS
                else ("host" if int(kind) == host_type else "device")
            )
            memo[word] = known
        if known is not None:
            found.add((word, known))
    return found


def _segment_ranges(segments: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(s["address"], s["address"] + s["total_size"]) for s in segments]


def _collect_nodes(
    driver: Any, raw_graph: int
) -> tuple[list[dict], list[list[int]], list[list[int]], int]:
    """Serialize every node's parameters, the topology connecting them, which of
    those edges are not ordinary, and how many distinct events the event nodes
    refer to."""
    *_, count = _chk(driver.cuGraphGetNodes(raw_graph, 0))
    nodes, *_ = _chk(driver.cuGraphGetNodes(raw_graph, count))
    index = {int(node): i for i, node in enumerate(nodes)}

    # Only the edges that are not ordinary are recorded, as
    # [edge index, from_port, to_port, type], which leaves the manifest unchanged
    # for the graphs that have none. See Note [Edge data].
    _check_edge_data_readable()
    *_, edge_count = _chk(driver.cuGraphGetEdges(raw_graph, 0))
    edges: list[list[int]] = []
    edge_data: list[list[int]] = []
    if edge_count:
        src, dst, data, _ = _chk(driver.cuGraphGetEdges(raw_graph, edge_count))
        for position, (parent, child, datum) in enumerate(zip(src, dst, data)):
            edges.append([index[int(parent)], index[int(child)]])
            ports = [int(datum.from_port), int(datum.to_port), int(datum.type)]
            if any(ports):
                edge_data.append([position, *ports])

    node_type = driver.CUgraphNodeType
    func_attr = driver.CUfunction_attribute
    launch_attr = driver.CUlaunchAttributeID

    # Distinct CUevent handles, in first-seen order. Only the identity matters:
    # nodes sharing a handle must share one recreated event.
    events: dict[int, int] = {}

    out: list[dict[str, Any]] = []
    for node in nodes:
        kind = _chk(driver.cuGraphNodeGetType(node))
        if kind.name not in _SERIALIZABLE_NODE_TYPES:
            raise UnserializableGraph(
                f"graph contains a {kind.name} node, which holds process-local state "
                "(a host function pointer, or an event or external-semaphore handle) "
                "and cannot be serialized"
            )
        if kind == node_type.CU_GRAPH_NODE_TYPE_KERNEL:
            # Bind the params object: reading .func off a temporary leaves a
            # dangling handle, since the value borrows the struct's storage.
            params = _chk(driver.cuGraphKernelNodeGetParams(node))
            args, packed = _kernel_args(driver, params)
            attrs = {}
            for attr_name in _SETTABLE_FUNC_ATTRS:
                attr = getattr(func_attr, attr_name, None)
                if attr is None:
                    continue
                err, value = driver.cuFuncGetAttribute(attr, params.func)
                if err == driver.CUresult.CUDA_SUCCESS:
                    attrs[attr_name] = value
            node_attrs = {}
            for attr_name in _KERNEL_NODE_ATTRS:
                attr = getattr(launch_attr, attr_name, None)
                if attr is None:
                    continue
                err, value = driver.cuGraphKernelNodeGetAttribute(node, attr)
                if err == driver.CUresult.CUDA_SUCCESS:
                    node_attrs[attr_name] = bytes(value.pad).hex()
            out.append(
                {
                    "type": "kernel",
                    "name": _chk(driver.cuFuncGetName(params.func)).decode(),
                    "grid": [params.gridDimX, params.gridDimY, params.gridDimZ],
                    "block": [params.blockDimX, params.blockDimY, params.blockDimZ],
                    "shared_mem_bytes": params.sharedMemBytes,
                    "args": [a.hex() for a in args] if args else None,
                    "packed_args": packed.hex() if packed is not None else None,
                    "func_attrs": attrs,
                    "node_attrs": node_attrs,
                }
            )
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMCPY:
            params = _chk(driver.cuGraphMemcpyNodeGetParams(node))
            fields = {
                key: int(getattr(params, key))
                for key in (
                    "srcXInBytes",
                    "srcY",
                    "srcZ",
                    "srcLOD",
                    "srcPitch",
                    "srcHeight",
                    "dstXInBytes",
                    "dstY",
                    "dstZ",
                    "dstLOD",
                    "dstPitch",
                    "dstHeight",
                    "WidthInBytes",
                    "Height",
                    "Depth",
                )
            }
            fields["srcMemoryType"] = int(params.srcMemoryType)
            fields["dstMemoryType"] = int(params.dstMemoryType)
            fields["srcDevice"] = int(params.srcDevice)
            fields["dstDevice"] = int(params.dstDevice)
            out.append({"type": "memcpy", "params": fields})
        elif kind == node_type.CU_GRAPH_NODE_TYPE_MEMSET:
            params = _chk(driver.cuGraphMemsetNodeGetParams(node))
            out.append(
                {
                    "type": "memset",
                    "params": {
                        "dst": int(params.dst),
                        "value": int(params.value),
                        "elementSize": int(params.elementSize),
                        "width": int(params.width),
                        "height": int(params.height),
                        "pitch": int(params.pitch),
                    },
                }
            )
        elif kind in (
            node_type.CU_GRAPH_NODE_TYPE_EVENT_RECORD,
            node_type.CU_GRAPH_NODE_TYPE_WAIT_EVENT,
        ):
            is_record = kind == node_type.CU_GRAPH_NODE_TYPE_EVENT_RECORD
            getter = (
                driver.cuGraphEventRecordNodeGetEvent
                if is_record
                else driver.cuGraphEventWaitNodeGetEvent
            )
            handle = int(_chk(getter(node)))
            out.append(
                {
                    "type": "event_record" if is_record else "event_wait",
                    "event": events.setdefault(handle, len(events)),
                }
            )
        else:
            out.append({"type": "empty"})
    return out, edges, edge_data, len(events)


def _select_cubins(driver: Any, names: set[str]) -> dict[str, int]:
    """Map each kernel name to the id of a captured module containing it.

    Loading the captured images and asking each for the name gives the exact set
    the graph needs, so the archive carries only those rather than every module the
    process happened to load.
    """
    from torch.cuda import _graph_kernel_capture

    modules = _graph_kernel_capture.captured_modules()
    if not modules:
        raise UnserializableGraph(
            "no cubins were captured, so the graph's kernels could not be saved. "
            "Call torch.cuda._graph_kernel_capture.start() before any CUDA work: "
            "modules load lazily and are announced once."
        )
    resolved: dict[str, int] = {}
    libraries = []
    try:
        for module_id, image in modules.items():
            if len(resolved) == len(names):
                break
            err, library = driver.cuLibraryLoadData(image, [], [], 0, [], [], 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                continue
            libraries.append(library)
            for name in names - resolved.keys():
                if (
                    driver.cuLibraryGetKernel(library, name.encode())[0]
                    == driver.CUresult.CUDA_SUCCESS
                ):
                    resolved[name] = module_id
    finally:
        for library in libraries:
            driver.cuLibraryUnload(library)
    missing = sorted(names - resolved.keys())
    if missing:
        from torch.cuda._graph_kernel_capture import skipped_modules

        # Blaming only the usual cause sends the reader looking in the wrong place
        # when the module was announced without a usable image.
        skipped = skipped_modules()
        cause = (
            f"{skipped} module load(s) were announced without a usable cubin image"
            if skipped
            else "capture was probably armed after they were loaded"
        )
        raise UnserializableGraph(
            "no captured module contains these kernels, so they cannot be recovered "
            f"in another process: {missing}. {cause}."
        )
    return resolved


def _check_reproducible(segments: list[dict[str, Any]]) -> None:
    """Refuse any segment whose address a later process could not reserve again.

    Only expandable segments qualify: a cudaMalloc address cannot be reclaimed by
    cuMemAddressReserve at all. The address itself did not have to be predictable
    when it was recorded, since load asks for exactly the one recorded here.
    """
    for segment in segments:
        if not segment["is_expandable"]:
            raise UnserializableGraph(
                f"segment at {segment['address']:#x} is not an expandable segment, so "
                "its address cannot be reproduced. Set "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the whole process."
            )


def save(
    cuda_graph: CUDAGraph,
    path: str,
    *,
    tensors: dict[str, Any] | list[Any] | None = None,
    save_fn: Callable[[list[Any]], None] | None = None,
) -> None:
    """Write ``cuda_graph`` and the state it needs to ``path``.

    See :meth:`torch.cuda.CUDAGraph.save`.
    """
    import torch

    driver = _driver()
    raw_graph = cuda_graph.raw_cuda_graph()

    nodes, edges, edge_data, num_events = _collect_nodes(driver, raw_graph)

    # A mapping is preferred: the archive outlives the code that wrote it, and
    # positional order is the kind of coupling that breaks quietly across a process
    # boundary. A list is still accepted and named by position.
    if isinstance(tensors, dict):
        named = list(tensors.items())
    else:
        named = [(str(i), t) for i, t in enumerate(tensors or [])]
    tensors = [t for _name, t in named]
    tensor_records = []
    for name, tensor in named:
        if not tensor.is_cuda:
            raise UnserializableGraph(
                "tensors to save alongside the graph must be CUDA tensors"
            )
        tensor_records.append(
            {
                "name": name,
                "address": tensor.data_ptr(),
                "nbytes": tensor.untyped_storage().nbytes(),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
            }
        )

    # Every device address the graph touches. A graph routinely reads memory
    # outside its own pool -- static inputs are allocated before capture -- so the
    # segments to carry are decided by what the nodes actually reference, not by
    # pool membership.
    referenced: set[int] = set()
    host_referenced: set[int] = set()
    pointer_memo: dict[int, str | None] = {}
    for node in nodes:
        if node["type"] == "kernel":
            blobs = [bytes.fromhex(a) for a in (node["args"] or [])]
            if node["packed_args"] is not None:
                blobs.append(bytes.fromhex(node["packed_args"]))
            for blob in blobs:
                for address, kind in _referenced_pointers(driver, blob, pointer_memo):
                    (referenced if kind == "device" else host_referenced).add(address)
        elif node["type"] == "memcpy":
            referenced |= {
                addr
                for addr in (node["params"]["srcDevice"], node["params"]["dstDevice"])
                if addr
            }
        elif node["type"] == "memset":
            referenced.add(node["params"]["dst"])

    # The graph's own pools (so the restored pool has the same layout), plus every
    # segment holding something the caller asked for or the graph references.
    pool_ids = {tuple(pool) for pool in cuda_graph.pools()}
    segments = []
    referenced_only = []
    for segment in torch.cuda.memory_snapshot(include_traces=False):
        start, end = segment["address"], segment["address"] + segment["total_size"]
        in_pool = tuple(segment["segment_pool_id"]) in pool_ids
        holds_tensor = any(start <= r["address"] < end for r in tensor_records)
        holds_reference = any(start <= addr < end for addr in referenced)
        if not (in_pool or holds_tensor or holds_reference):
            continue
        segments.append(segment)
        if holds_reference and not (in_pool or holds_tensor):
            referenced_only.append(segment)
    _check_reproducible(segments)

    # Memory the graph reads that the caller did not list is carried anyway, so the
    # archive is at least complete, but it is worth saying out loud: normally such
    # memory is a static input, and passing it in `tensors` is what records the
    # metadata to place it and (unless save_fn takes over) its contents. A segment
    # reaching this branch is scratch space living outside the graph's pool, which
    # usually means a library workspace that wants handling of its own.
    if referenced_only:
        warnings.warn(
            "saving "
            + ", ".join(
                f"{s['total_size']} bytes at {s['address']:#x}" for s in referenced_only
            )
            + " because the graph reads it, though no tensor passed in `tensors` lives "
            "there. If its contents matter, pass those tensors so they are restored "
            "explicitly; if it is scratch space, it is being carried unnecessarily.",
            stacklevel=3,
        )

    if host_referenced:
        raise UnserializableGraph(
            "the graph references pinned host memory at "
            + ", ".join(f"{address:#x}" for address in sorted(host_referenced))
            + ". Restoring it would need the same host virtual address, and pinned "
            "memory is allocated with cudaHostAlloc rather than through the VMM "
            "reserve/map path, so there is no way to place it today."
        )

    # Anything still unaccounted for is memory no saved segment covers, so nothing
    # brings it back. Whether the allocator knows about it at all is the useful
    # distinction for the caller, so name the owning pool when there is one: a
    # MemPool built on a custom allocator holds real allocator memory that is still
    # unrestorable, which reads as a contradiction without being told.
    ranges = _segment_ranges(segments)
    for address in sorted(referenced):
        if any(lo <= address < hi for lo, hi in ranges):
            continue
        owner = next(
            (
                segment
                for segment in torch.cuda.memory_snapshot(include_traces=False)
                if segment["address"]
                <= address
                < segment["address"] + segment["total_size"]
            ),
            None,
        )
        if owner is not None and not owner["is_expandable"]:
            raise UnserializableGraph(
                f"the graph references device memory at {address:#x} in pool "
                f"{owner['segment_pool_id']}, which did not come from the expandable "
                "segment path and so cannot be reserved again. This is what a MemPool "
                "with its own allocator produces -- ncclMemAlloc for registered "
                "collective buffers, for example."
            )
        raise UnserializableGraph(
            f"the graph references device memory at {address:#x} that the caching "
            "allocator does not own, so it cannot be restored. It comes from "
            "outside the allocator -- NCCL device state, symmetric memory, or a "
            "pool with its own allocator."
        )

    kernel_names = {node["name"] for node in nodes if node["type"] == "kernel"}
    cubins = _select_cubins(driver, kernel_names)

    from torch.cuda import _graph_kernel_capture

    images = _graph_kernel_capture.captured_modules()
    manifest = {
        "version": ARCHIVE_VERSION,
        "device": torch.cuda.current_device(),
        "nodes": nodes,
        "edges": edges,
        "edge_data": edge_data,
        "num_events": num_events,
        "segments": [
            {
                **{
                    key: segment[key]
                    for key in (
                        "address",
                        "total_size",
                        "segment_type",
                        "expandable_segment_base",
                    )
                },
                # Reproducing each block's state is what keeps an address the graph
                # writes to from being handed to a later allocation. Contents are
                # not carried (see the note on data above), so only the layout.
                "blocks": [
                    {
                        "address": block["address"],
                        "size": block["size"],
                        "requested_size": block["requested_size"],
                        "state": block["state"],
                    }
                    for block in segment["blocks"]
                ],
            }
            for segment in segments
        ],
        "kernels": {name: str(module_id) for name, module_id in cubins.items()},
        "tensors": tensor_records,
    }

    writer = torch._C.PyTorchFileWriter(path)
    try:
        writer.write_record(
            MANIFEST_PATH,
            (blob := json.dumps(manifest).encode()),
            len(blob),
        )
        for module_id in sorted(set(cubins.values())):
            image = images[module_id]
            writer.write_record(f"{CUBIN_DIR}/{module_id}.cubin", image, len(image))
        writer.write_end_of_file()
    finally:
        del writer
    if save_fn is not None and tensors:
        save_fn(tensors)


def save_hook(
    path: str,
    *,
    tensors: Callable[[], dict[str, Any] | list[Any]] | None = None,
    save_fn: Callable[[list[Any]], None] | None = None,
) -> Callable[[CUDAGraph], None]:
    """Return a post-instantiate hook that saves the graph to ``path``.

    Register it with :meth:`CUDAGraph.register_post_instantiate_hook`. That point
    runs after the capture-end hooks and after anything else that could still
    modify the template, and the template is still live there in both
    ``keep_graph`` modes.

    ``tensors`` is a callable rather than a list because a hook cannot be handed
    the tensors at registration time. Note hooks fire in registration order, so
    register this last if another post-instantiate hook also mutates the graph.
    """

    def _hook(cuda_graph: CUDAGraph) -> None:
        save(cuda_graph, path, tensors=tensors() if tensors else None, save_fn=save_fn)

    return _hook


class RestoredCUDAGraph:
    """A graph rebuilt by :func:`load`, replayable but not a :class:`CUDAGraph`.

    ``CUDAGraph`` owns its ``cudaGraph_t`` in C++ and there is no way to hand it an
    externally built one, so a restored graph cannot be presented as one without a
    C++ addition. This exposes the part that matters -- :meth:`replay` -- and owns
    everything the graph's nodes point at: the loaded libraries (unloading one
    would invalidate its kernels), the recreated events, the memory pool, and a
    holder per reconstructed allocation. Dropping any of those out from under a
    live graph is a use-after-free, which is why they are kept here rather than
    left to the caller.
    """

    def __init__(
        self,
        graph: int,
        exec_graph: int,
        pool: Any,
        holders: list[Any],
        libraries: list[Any],
        events: list[Any],
    ) -> None:
        self._graph = graph
        self._exec = exec_graph
        self._pool = pool
        self._holders = holders
        self._libraries = libraries
        self._events = events

    def replay(self) -> None:
        """Launch the restored graph on the current stream."""
        import torch

        driver = _driver()
        _chk(driver.cuGraphLaunch(self._exec, torch.cuda.current_stream().cuda_stream))

    def raw_cuda_graph(self) -> int:
        return self._graph

    def raw_cuda_graph_exec(self) -> int:
        return self._exec


def _topological_order(count: int, edges: list[list[int]]) -> list[int]:
    """Node indices such that every node follows its dependencies.

    Nodes are created without dependencies and wired up afterwards, so this is not
    needed to make handles exist in time. It earns its place by rejecting a cyclic
    edge list with a message that says so, rather than leaving the driver to fail
    the bulk cuGraphAddDependencies with INVALID_VALUE, and by making the order
    nodes are built in a property of the graph rather than of the archive.
    """
    successors: dict[int, list[int]] = {}
    indegree = [0] * count
    for parent, child in edges:
        successors.setdefault(parent, []).append(child)
        indegree[child] += 1
    ready = [i for i in range(count) if indegree[i] == 0]
    order = []
    while ready:
        node = ready.pop()
        order.append(node)
        for child in successors.get(node, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != count:
        raise UnserializableGraph("the saved graph's edges contain a cycle")
    return order


def _add_edges(
    driver: Any,
    graph: Any,
    made: dict[int, Any],
    edges: list[list[int]],
    edge_data: list[list[int]],
) -> None:
    """Reconnect the saved topology, keeping each edge's ports and type.

    ``edge_data`` names only the edges that are not ordinary, as
    ``[edge index, from_port, to_port, type]``; the rest are zero, an ordinary
    dependency. One call per distinct edge data, because the driver applies the
    first entry to the whole call -- see Note [Edge data].
    """
    if not edges:
        return
    ports = {
        position: (from_port, to_port, kind)
        for position, from_port, to_port, kind in edge_data
    }
    groups: dict[tuple[int, int, int], list[int]] = {}
    for position in range(len(edges)):
        groups.setdefault(ports.get(position, (0, 0, 0)), []).append(position)

    for (from_port, to_port, kind), members in groups.items():
        datum = driver.CUgraphEdgeData()
        datum.from_port = from_port
        datum.to_port = to_port
        datum.type = kind
        _chk(
            driver.cuGraphAddDependencies(
                graph,
                [made[edges[position][0]] for position in members],
                [made[edges[position][1]] for position in members],
                [datum] * len(members),
                len(members),
            )
        )


def load(
    path: str,
    *,
    load_fn: Callable[[], dict[str, Any]] | None = None,
) -> tuple[RestoredCUDAGraph, dict[str, Any]]:
    """Rebuild a graph saved by :func:`save`, and return it with its named tensors.

    Must run before anything else allocates on the device: the whole point is to
    reclaim the exact virtual addresses the graph was captured against, and an
    allocation made first can take them.

    ``load_fn`` supplies contents for the named tensors and must return them on the
    **CPU**. The archive carries no data (parameters evolve after capture, so bytes
    written then would be stale), so anything whose contents matter comes from here
    -- and it has to be host memory, because materialising a checkpoint straight to
    the device would allocate over the addresses being reclaimed.
    """
    import time

    import torch

    profile = _LAST_LOAD_PROFILE
    profile.clear()

    class _Phase:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> None:
            self.start = time.perf_counter()

        def __exit__(self, *exc: Any) -> None:
            profile[self.name] = profile.get(self.name, 0.0) + (
                time.perf_counter() - self.start
            )

    driver = _driver()
    # PyTorchFileReader addresses records relative to the archive root, stripping
    # the zip's own directory prefix, so names go in exactly as they were written.
    reader = torch._C.PyTorchFileReader(path)

    def record(name: str) -> bytes:
        return bytes(reader.get_record(name))

    with _Phase("read_manifest"):
        manifest = json.loads(record(MANIFEST_PATH))
    if manifest["version"] != ARCHIVE_VERSION:
        raise UnserializableGraph(
            f"archive version {manifest['version']} is not {ARCHIVE_VERSION}"
        )

    # Contents first, while nothing has touched the device yet.
    with _Phase("load_fn"):
        contents = load_fn() if load_fn is not None else {}

    device = manifest["device"]
    with _Phase("restore_segments"):
        pool = torch.cuda.MemPool()
        torch.cuda.memory._restore_expandable_segments(
            [{**segment, "is_expandable": True} for segment in manifest["segments"]],
            pool.id,
            device,
        )

    # Put each allocation back exactly as it was: a block that was allocated has to
    # be allocated again, or a later allocation could be handed an address the graph
    # writes to.
    holders: list[Any] = []
    blocks: list[tuple[int, int, Any]] = []
    phase_blocks = _Phase("reproduce_blocks")
    phase_blocks.__enter__()
    with torch.cuda.use_mem_pool(pool):
        for segment in manifest["segments"]:
            for block in segment["blocks"]:
                if not block["state"].startswith("active"):
                    continue
                holder = torch.empty(0, dtype=torch.uint8, device=device)
                holder.untyped_storage()._resize_with_addr_(
                    block["size"], block["address"]
                )
                holders.append(holder)
                blocks.append((block["address"], block["size"], holder))
    phase_blocks.__exit__()

    # Sorted once and bisected per tensor: scanning every block for each named
    # tensor is quadratic, and a graph with thousands of parameters has thousands
    # of each.
    blocks.sort(key=lambda entry: entry[0])
    starts = [entry[0] for entry in blocks]

    def view_for(rec: dict[str, Any]) -> Any:
        dtype = getattr(torch, str(rec["dtype"]).rsplit(".", 1)[-1])
        index = bisect.bisect_right(starts, rec["address"]) - 1
        if index >= 0:
            address, size, holder = blocks[index]
            if rec["address"] < address + size:
                offset = rec["address"] - address
                itemsize = torch._utils._element_size(dtype)
                if offset % itemsize:
                    raise UnserializableGraph(
                        f"tensor at {rec['address']:#x} is not aligned to its dtype"
                    )
                out = torch.empty(0, dtype=dtype, device=device)
                out.set_(
                    holder.untyped_storage(),
                    offset // itemsize + rec["storage_offset"],
                    tuple(rec["shape"]),
                    tuple(rec["stride"]),
                )
                return out
        raise UnserializableGraph(
            f"no restored allocation covers the tensor at {rec['address']:#x}"
        )

    tensors: dict[str, Any] = {}
    with _Phase("bind_views"):
        for index, rec in enumerate(manifest["tensors"]):
            tensors[rec.get("name", str(index))] = view_for(rec)
    # Copying contents in is checkpoint-restore cost, paid whichever way the graph
    # was obtained, so it is timed apart from the work load itself does.
    with _Phase("copy_contents"):
        for name, tensor in tensors.items():
            if name in contents:
                tensor.copy_(contents[name])
        torch.cuda.synchronize()

    # Kernels come back by name out of the archived cubins.
    libraries: list[Any] = []
    functions: dict[str, int] = {}
    wanted = set(manifest["kernels"])
    phase_kernels = _Phase("load_kernels")
    phase_kernels.__enter__()
    for module_id in sorted(set(manifest["kernels"].values())):
        image = record(f"{CUBIN_DIR}/{module_id}.cubin")
        err, library = driver.cuLibraryLoadData(image, [], [], 0, [], [], 0)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise UnserializableGraph(f"could not load cubin {module_id}: {err.name}")
        libraries.append(library)
        for name in list(wanted):
            found, kernel = driver.cuLibraryGetKernel(library, name.encode())
            if found == driver.CUresult.CUDA_SUCCESS:
                functions[name] = int(_chk(driver.cuKernelGetFunction(kernel)))
                wanted.discard(name)
    phase_kernels.__exit__()
    if wanted:
        raise UnserializableGraph(f"cubins do not contain kernels {sorted(wanted)}")

    # Fresh events: only the identity was recorded, so ordering inside the graph is
    # reproduced and any interaction with an outside event is not. Timing is
    # disabled because the flags of the original cannot be recovered and nothing
    # can observe these.
    events = [
        _chk(driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DISABLE_TIMING))
        for _ in range(manifest.get("num_events", 0))
    ]

    phase_nodes = _Phase("build_nodes")
    phase_nodes.__enter__()
    graph = _chk(driver.cuGraphCreate(0))
    func_attr = driver.CUfunction_attribute
    launch_attr = driver.CUlaunchAttributeID
    made: dict[int, Any] = {}
    keep: list[Any] = []
    # Nodes are created unattached and wired up in one pass at the end, because a
    # dependency's ports and type can only be supplied to cuGraphAddDependencies,
    # not to the cuGraphAdd*Node calls.
    deps: list[Any] = []
    for index in _topological_order(len(manifest["nodes"]), manifest["edges"]):
        node = manifest["nodes"][index]
        kind = node["type"]
        if kind == "kernel":
            function = functions[node["name"]]
            for attr_name, value in node["func_attrs"].items():
                attr = getattr(func_attr, attr_name, None)
                if attr is not None:
                    driver.cuFuncSetAttribute(function, attr, value)
            params = driver.CUDA_KERNEL_NODE_PARAMS()
            params.func = function
            params.gridDimX, params.gridDimY, params.gridDimZ = node["grid"]
            params.blockDimX, params.blockDimY, params.blockDimZ = node["block"]
            params.sharedMemBytes = node["shared_mem_bytes"]
            if node["args"]:
                blobs = [
                    ctypes.create_string_buffer(bytes.fromhex(a), len(a) // 2)
                    for a in node["args"]
                ]
                slots = (ctypes.c_void_p * len(blobs))(
                    *[ctypes.cast(b, ctypes.c_void_p) for b in blobs]
                )
                keep += [blobs, slots]
                params.kernelParams = ctypes.addressof(slots)
            elif node["packed_args"] is not None:
                packed = bytes.fromhex(node["packed_args"])
                blob = ctypes.create_string_buffer(packed, len(packed))
                size = ctypes.c_size_t(len(packed))
                extra = (ctypes.c_void_p * 5)(
                    ctypes.c_void_p(2),
                    ctypes.cast(ctypes.byref(size), ctypes.c_void_p),
                    ctypes.c_void_p(1),
                    ctypes.cast(blob, ctypes.c_void_p),
                    ctypes.c_void_p(0),
                )
                keep += [blob, size, extra]
                params.extra = ctypes.addressof(extra)
            # a kernel taking no arguments leaves kernelParams and extra null
            handle = _chk(driver.cuGraphAddKernelNode(graph, deps, 0, params))
            for attr_name, pad in node["node_attrs"].items():
                attr = getattr(launch_attr, attr_name, None)
                if attr is None:
                    continue
                value = driver.CUkernelNodeAttrValue()
                raw = bytes.fromhex(pad)
                ctypes.memmove(int(value.getPtr()), raw, len(raw))
                driver.cuGraphKernelNodeSetAttribute(handle, attr, value)
        elif kind == "memcpy":
            fields = node["params"]
            copy = driver.CUDA_MEMCPY3D()
            for key, value in fields.items():
                if key in ("srcMemoryType", "dstMemoryType"):
                    setattr(copy, key, driver.CUmemorytype(value))
                elif key in ("srcDevice", "dstDevice"):
                    setattr(copy, key, driver.CUdeviceptr(value))
                else:
                    setattr(copy, key, value)
            context = _chk(driver.cuCtxGetCurrent())
            handle = _chk(driver.cuGraphAddMemcpyNode(graph, deps, 0, copy, context))
        elif kind == "memset":
            fields = node["params"]
            fill = driver.CUDA_MEMSET_NODE_PARAMS()
            fill.dst = driver.CUdeviceptr(fields["dst"])
            fill.value = fields["value"]
            fill.elementSize = fields["elementSize"]
            fill.width = fields["width"]
            fill.height = fields["height"]
            fill.pitch = fields["pitch"]
            context = _chk(driver.cuCtxGetCurrent())
            handle = _chk(driver.cuGraphAddMemsetNode(graph, deps, 0, fill, context))
        elif kind == "event_record":
            handle = _chk(
                driver.cuGraphAddEventRecordNode(graph, deps, 0, events[node["event"]])
            )
        elif kind == "event_wait":
            handle = _chk(
                driver.cuGraphAddEventWaitNode(graph, deps, 0, events[node["event"]])
            )
        else:
            handle = _chk(driver.cuGraphAddEmptyNode(graph, deps, 0))
        made[index] = handle

    _add_edges(driver, graph, made, manifest["edges"], manifest["edge_data"])
    phase_nodes.__exit__()
    with _Phase("instantiate"):
        exec_graph = _chk(driver.cuGraphInstantiate(graph, 0))
    del keep
    return (
        RestoredCUDAGraph(
            int(graph), int(exec_graph), pool, holders, libraries, events
        ),
        tensors,
    )
