# CUDA streams, the caching allocator, and CUDA graph capture

**Applies when** a feature defers, moves, shares, or reuses GPU memory (lazy
copies, stealing an allocation, buffer pools, caches of data pointers); adds
hooks on `data_ptr()`, `const_data_ptr()`, or storage materialization for
CUDA; adds cross-stream synchronization; or has to work during CUDA graph
capture. For whether a particular `record_stream` call is avoidable, use the
`cuda-streams` skill instead.

None of this applies to CPU-only features. Don't change CPU behavior just to
match CUDA; say in the design that the two differ.

## The correctness model

For every asynchronous access to some memory, two orderings must hold:

1. The producer of the data happens before the access starts.
2. The access finishes before the memory is freed **or mutated**.

Freeing and mutating are equivalent here: either way the accessed data is
gone. CUDA Sanitizer (`torch.cuda._sanitizer`) checks the same model, so it is
useful for testing a design.

Then list every way the feature lets memory be freed or mutated. Watch for
new ones, such as a write that secretly releases a shared allocation, or the
last reference stealing memory and writing to it in place. Users won't sync
for those unless something tells them to.

## Lifetime belongs to the allocation stream

The caching allocator reuses a block only on the stream that allocated it, and
users already sync their side-stream uses back to that stream before freeing a
tensor. Build on that contract:

- If the feature creates a replacement allocation for existing data (a
  materialized copy, say), allocate it on the original allocation stream, so
  the same rule protects it. Get the stream from the allocator rather than
  assuming the current stream. `c10::Allocator` has no public query for it
  today (`Allocator::allocation_stream_info` is proposed in #200800, not
  landed); until one exists, take the fallback below.
- If work runs on another stream, order it in both directions: the working
  stream waits for the allocation stream before starting, and the allocation
  stream waits for the work afterwards.
- If an allocator can't report the allocation stream (`cudaMallocAsync`,
  pluggable allocators), fall back to the simple path rather than guessing.
- Memory in a private pool (CUDA graph pools, user memory pools) isn't tied to
  a stream, and graph replays write it behind your back. Treat it as
  unshareable.

## Synchronization mechanics

- Enqueue copies with `cudaMemcpyAsync` on the current stream. A plain
  `cudaMemcpy` runs on the legacy default stream: it isn't ordered with
  PyTorch's non-blocking streams and isn't captured into CUDA graphs.
- Express cross-stream dependencies as event record plus wait. The same code
  is then correct in eager mode and under capture, where CUDA turns it into
  graph edges. Don't skip a sync because "replay is ordered after eager work
  anyway"; that drops ordering between streams inside the same capture. Check
  that both streams are outside capture or in the same capture.
- A wait on a whole stream at the point of a free or steal also waits for all
  the unrelated work enqueued on it since the last use. Prefer an event
  recorded at the point of the last use, supplied by the user or recorded by
  an explicit API, over syncing at the free point.
- Don't add `recordStream` to a new mechanism. It defers reuse
  nondeterministically and doesn't order other writes to the memory.
- Don't record the current stream in `data_ptr()` / `const_data_ptr()` to
  track readers. The current stream at that moment may not be the stream the
  work runs on (custom ops can do anything). Pointers are also taken for
  incidental reasons like alignment checks. And you can only store a stream
  there, not an event, so a later sync on it is late (see above). It is fine
  to use pointer access as a signal in the safe direction, for example to
  invalidate a user's earlier declaration that they were done with the data.
- `Tensor.record_stream()` is silently ignored for any DataPtr whose deleter
  isn't the caching allocator's own. If the feature wraps DataPtrs, decide
  what `record_stream` should do on them and document it.
- Host-side metadata shared between aliases still needs host-thread
  synchronization, even when all GPU work is correctly ordered.
- Set a device guard for the storage's device before allocating or getting
  the current stream. The current device may be a different GPU.

## CUDA graph capture

- **Once vs every replay.** Captured work runs on every replay; work done
  eagerly at capture time runs once. A copy forced inside a capture for data
  created outside it re-copies on every replay, and a copy forced after
  capture for data created inside it happens only once. Record which capture
  (or none) each shared resource belongs to, and reject operations that cross
  the boundary.
- **Baked-in addresses.** A graph keeps using the addresses it captured.
  Anything that moves a tensor to new memory after its address was captured
  (materialization, reallocation) silently splits it from the graph. Stealing
  keeps the address and is safe in this respect; copying is not.
- **Capture fixes the executed path.** If the captured path needs a copy, the
  graph needs that copy on every replay anyway, so asking the user for an
  explicit `clone()` at the call site moves a necessary copy earlier rather
  than adding one. Prefer an actionable error over supporting the copy inside
  capture.
- **Unknown lifetimes.** An optimization that depends on knowing a tensor
  outlives the graph (static inputs, parameters) needs a reliable way to know
  that. Inside a capture you usually don't, so either have the user declare
  it or don't do the optimization.
- **External pointer caches.** NCCL registrations, symmetric memory, and saved
  raw pointers have the same address problem, and the feature can't see them.
  Say in the design what the user is responsible for.
