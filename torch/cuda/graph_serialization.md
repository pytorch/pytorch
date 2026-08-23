# CUDA graph serialization

Saving a captured CUDA graph so a later process can replay it without capturing.

Capture is a large part of cold start for a model served with CUDA graphs: an
engine captures one graph per batch size it supports, and each capture has to warm
up first so the kernels it launches are compiled, loaded and selected. Restoring
skips all of that.

Experimental and incomplete. See "What is not supported" below; anything unsupported
is refused at save time rather than producing an archive that misbehaves, with the
exception noted under "Known gaps".

## Why a captured graph is not portable

A `cudaGraph_t` is not a description of work, it is a set of bound handles. Its
kernel nodes hold `CUfunction` pointers into modules *this* process loaded, and
their arguments are opaque byte blobs with device pointers embedded at offsets only
the producing library knows. cuBLASLt, for instance, hands its kernels a single
2560-byte packed struct.

That rules out the obvious approach. You cannot rewrite the pointers, because you
cannot reliably find them: no byte pattern distinguishes a pointer from adjacent
scalar fields. A real cuBLASLt argument blob contains `0x2000000020` (two tile
dimensions of 32, side by side) and `0xffffffc0000000`, both of which look exactly
like plausible device addresses. Guessing rejects valid graphs; guessing wrong
corrupts memory.

## The approach: reproduce the environment, never relocate the graph

Instead of making the graph fit a new process, make the new process match the
graph. Three things are reproduced exactly, and then the graph's bytes are replayed
verbatim:

1. **Device code** — the cubins its kernels live in.
2. **Memory addresses** — the allocator's segments, reclaimed at the same virtual
   addresses, with the same blocks allocated inside them.
3. **Graph structure** — node parameters, launch attributes and topology.

Because the memory underneath does not move, no argument is ever rewritten. That is
the whole reason an opaque argument blob is serializable at all.

## What travels, and what does not

The archive is a zip (`torch._C.PyTorchFileWriter`) holding a JSON manifest, one
record per cubin, and nothing else. It is tens of megabytes, dominated by device
code — 22 MB for a 352-node graph over a model with 3.2 GB of parameters.

**Data does not travel.** Two reasons. A graph is serialized at capture time, but
parameters keep training afterwards, so bytes written then are stale by the time
anyone replays. And in serving they are already being loaded from a checkpoint, so
embedding them would make the archive model-sized in order to duplicate work the
caller is doing anyway.

So the caller supplies contents at load, through `load_fn`, which must return them
**on the CPU**. That is not a stylistic preference: materialising a checkpoint
straight onto the device would allocate over the very addresses being reclaimed.
For the same reason, `load()` has to run before anything else allocates.

## Reclaiming addresses

Restoring an address means reserving it again, which only works for memory the
allocator obtained through the VMM path. A `cudaMalloc` address cannot be reclaimed
by `cuMemAddressReserve` at all — the runtime's suballocator owns that range — so
saving requires `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the whole
process for the memory the graph reaches, and refuses any segment backing it
that is non-expandable.

The refusal is scoped to what the graph references, not to the process. A job can
hold `cudaMalloc` memory in a `MemPool` built on a custom allocator and still be
serializable, so long as no captured kernel points into it; conversely a graph that
does point there is refused by the argument scan below, which is the check that
actually decides the question.

Expandable segments let the driver choose where to reserve, so addresses vary run
to run. That turns out not to matter: an address does not have to be *predictable*,
only *reproducible*, and recording each segment's base at save gives that for free.
Load asks for the recorded address (see
`Note [Expandable Segment Reserved Address]` in `c10/cuda/CUDACachingAllocator.cpp`),
which the driver honours because the reservation is large — a segment reserves 9/8
of device memory. Nothing about the normal allocation path changes, which is the
main reason to prefer this over imposing a fixed address layout on every process.

An earlier design placed segments on a fixed grid so that an address was a pure
function of `(bucket, slot)`. That was dropped: it made every expandable-segments
user pay for a property only serialization needed, and recording the address covers
the same ground. Its one residual benefit was steering reservations away from the
low region that `cudaMalloc` also uses; if a recorded address is ever found taken at
load, that is the reason to revisit it.

Two behaviours to know about:

- `cuMemAddressReserve` returns `CUDA_SUCCESS` while **silently ignoring** an
  address hint it cannot honour, and only honours one above roughly 34-40 MiB. The
  returned pointer must always be compared against the request.
- Restoring is all-or-nothing: a requested address that the driver declines raises,
  rather than falling back to a different address that would leave every kernel
  argument in the graph pointing at the wrong memory.

## Reproducing allocations, not just mappings

Restoring a segment maps its ranges and leaves them as *free* blocks. That is not
enough: an output whose storage nobody holds would sit free at an address the graph
writes to, and the next allocation could be handed it. So each block recorded as
allocated is allocated again — via `UntypedStorage._resize_with_addr_`, which
exists to place a storage at an exact address inside a mapped-but-inactive block.

The rule is to reproduce the allocation state that existed at save, not to hold
everything: blocks recorded `active_allocated` are allocated, blocks recorded
`inactive` are left free. Ownership then falls out rather than being invented —
tensors the caller named are handed back, and other allocated blocks are held by
the restored graph, because something held them at save and the graph is what
outlives load. A cuBLASLt workspace is exactly this: allocated, 32 MiB, nobody's
tensor.

The restored graph also owns the loaded libraries (unloading one invalidates its
kernels) and the recreated events. Dropping any of those under a live graph is a
use-after-free.

## Recovering device code

There is no API that returns a module's binary image: the whole
`cuModule`/`cuLibrary`/`cuFunc`/`cuKernel` surface can report a function's name,
module and attributes but never its code. Recovering kernels by name from the
fatbins in the host `.so` files covers most of ATen, but not code generated at
runtime — cuBLASLt's `nvjet_*` kernels on Blackwell exist in no fatbin on disk and
are not written to `CUDA_CACHE_PATH` either.

So the images are captured on the way in, from CUPTI's `RESOURCE` /
`MODULE_LOADED` callback (`torch.cuda._graph_kernel_capture`), which hands over the
cubin the driver was given. That covers every producer without depending on where
the code came from. Modules load lazily and are announced once, so capture must be
armed **before any CUDA work**; arming late is a refusal at save time, naming the
kernels it could not find.

Only the modules the graph's kernels actually live in are written to the archive,
found by loading the captured images and asking each for the names the graph uses.

A freshly loaded module does not inherit the host-side attributes the original
`CUfunction` had, so the settable ones are recorded and replayed.
`MAX_DYNAMIC_SHARED_SIZE_BYTES` is derivable from the node, but
`NON_PORTABLE_CLUSTER_SIZE_ALLOWED` is not carried by the node at all — cuBLASLt
sets it, and without it a kernel with a large cluster fails to launch.

## The graph itself

Nodes are rebuilt one `cuGraphAdd*Node` call at a time, in topological order:
`cuGraphGetNodes` makes no promise about order, and adding a node requires its
dependencies to exist already.

Event nodes are supported rather than refused. A `CUevent` handle is process-local,
so what travels is *which* event each node refers to; load creates one fresh event
per distinct handle, which reproduces the ordering inside the graph. It deliberately
does not reproduce interaction with an event outside the graph — a wait on an event
another stream records will be satisfied by the graph's own record. The original
event's flags cannot be recovered (there is no `cuEventGetFlags`), so restored
events have timing disabled, which is what pure ordering wants.

Note that stream capture lowers a `torch.cuda.Event` recorded during capture to a
dependency *edge*, not a node, so event nodes in practice come only from libraries
using the raw APIs during capture.

## What is not supported, and how that is decided

The refusals are properties rather than a denylist of libraries, so a case nobody
anticipated still fails loudly:

- **A node type must be reproducible.** Kernel, memcpy, memset, empty and event
  nodes are; `HOST` nodes are not, because their payload is a host function pointer
  plus an opaque `userData` that cannot be rebound from outside the library that
  created it. External-semaphore nodes are OS-level handles, likewise.
- **Every device pointer must live in memory the caching allocator owns**, so the
  archive can reproduce it. This is checked by asking `cuPointerGetAttribute` about
  each candidate word — the driver is the authority on what is a pointer, which is
  what makes the check usable at all. Because the driver decides, every byte offset
  can be scanned rather than only aligned ones, so a pointer inside a packed struct
  that ignores the usual alignment is still found.

Between them those reject:

- **NCCL**, on either transport: its kernel arguments embed `ncclDevComm` state
  NCCL allocated itself, and the network path additionally brings a host node.
  Restoring a collective is a reconnection problem, not a serialization one.
- **Symmetric memory**, which reserves its own address space.
- **Legacy cuBLAS**, whose workspace is not on expandable segments (cuBLASLt is
  fine).
- **Pinned host memory**, which is allocated with `cudaHostAlloc` rather than
  through the VMM reserve/map path, so there is no way to place it at the same host
  address.

One case is a warning rather than a refusal: memory the graph reads that the caller
did not list in `tensors`. It is carried anyway so the archive is complete, but
reaching that branch normally means scratch space living outside the graph's pool,
which usually wants handling of its own.

## Using it

```python
from torch.cuda import _graph_kernel_capture as kernel_capture

# before any CUDA work: modules load lazily and are announced once
kernel_capture.start()

... build the model, warm up ...

graph = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(graph, for_save=True):
    static_out = model(static_in)

kernel_capture.stop()     # disarms, keeps the images so saving still works
graph.save("graph.ptcg", tensors={"x": static_in, "y": static_out})
kernel_capture.clear()    # returns the host memory the images held
```

and in a fresh process, before anything else allocates:

```python
from torch.cuda._graph_serialization import load

graph, tensors = load("graph.ptcg", load_fn=lambda: torch.load("state.pt", map_location="cpu"))
tensors["x"].copy_(real_input)
graph.replay()
result = tensors["y"]
```

`tensors` accepts a mapping and the names are recorded, because an archive outlives
the code that wrote it and positional order breaks quietly across a process
boundary. Naming matters more than it looks: a name is the only way an address
becomes reachable again after load, so an output allocated *during* capture must be
listed or nothing can reach it.

`for_save=True` clears the per-stream cuBLAS workspaces before capture. Workspaces
are per stream, so a stream used for cuBLAS outside the pool already has one, and
capturing on it reuses that allocation — leaving the graph reading memory the pool
does not own. Warming up on the capture stream with the pool already active avoids
the problem without it. It is opt-in because clearing discards warmup work for
captures nobody intends to save.

Saving can also be automated with
`torch.cuda.graphs.save_graph_hook(path, tensors=...)` registered as a
post-instantiate hook, which fires after the capture-end hooks and after any later
modification, with the template still live in both `keep_graph` modes.

## Costs

Measured on a GB200. Save adds roughly 20 ms plus 46 us/node, on top of ~50 ms to
arm capture once and ~15 ms of callback during module loads; the captured images
also hold host memory until dropped (31 MB for a workload loading 31 modules).

Load is roughly 150 ms fixed plus 86 us/node, and where it goes is worth knowing,
because the phases scale differently:

| phase | 5 000 nodes | 10 000 nodes | scales with |
| --- | --- | --- | --- |
| read manifest | 41 ms | 82 ms | nodes (JSON) |
| restore segments | 279 ms | 327 ms | segments — nearly fixed |
| reproduce blocks | 12 ms | 20 ms | blocks |
| bind views | 25 ms | 54 ms | named tensors |
| load kernels | 15 ms | 15 ms | unique modules — flat |
| build nodes | 125 ms | 353 ms | nodes |
| instantiate | 9 ms | 19 ms | nodes |

Restoring the memory state dominates and is nearly fixed, which is why the
advantage over recapturing is smallest for small graphs. Recovering the kernels, the
part one would expect to be expensive, is 15 ms and does not grow.
`_LAST_LOAD_PROFILE` in `_graph_serialization.py` records this per load.

Against reconstructing the model, warming up and recapturing — both paths taking
weights from the same CPU state file — restoring reached a replayable graph 2.7x to
6.4x faster, improving with graph size. Note capture itself was never the cost: it
is ~30 ms of a 1.6 s baseline at 320 nodes. Most of the saving is that the weight
copy lands straight into restored addresses, so there is no separate model
materialization, and warmup's module loads and kernel selection never happen.

## Known gaps

- **Graph-safe RNG state is not restored.** `captured_generator_states_` holds
  philox seed/offset in graph-pool memory whose contents matter but which belongs to
  no tensor the caller can name, so it is neither carried nor detected. A restored
  graph containing dropout would replay with garbage state. This is the one known
  way to get a wrong result rather than a refusal.
- One archive per graph repeats the whole cubin set, so the case that motivates
  this — an engine with hundreds of graphs across batch sizes — should share one set
  and one segment snapshot.
- The manifest hex-encodes argument bytes into JSON, which costs both size
  (~3.9 KB/node) and parse time.
- `load()` returns a `RestoredCUDAGraph` exposing `replay()`, not a `CUDAGraph`:
  `CUDAGraph` owns its `cudaGraph_t` in C++ and there is no way to hand it an
  externally built one.
