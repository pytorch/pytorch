---
name: cuda-streams
description: Manage tensor lifetimes across CUDA streams and decide whether record_stream is avoidable. Use when using a tensor on a stream other than the one it was allocated on, adding or reviewing Tensor.record_stream / CUDACachingAllocator::recordStream, side streams, wait_stream, or CUDA events for cross-stream ordering, or designing APIs that hand side-stream tensors to callers.
---

# CUDA Streams and record_stream

The caching allocator only reuses a block on the stream that allocated it. If a
tensor is used on some other stream, then before it is freed you need either
(a) a sync from that stream back to the creation stream (an event, or
`creation_stream.wait_stream(side_stream)`), or (b) `record_stream()` /
`CUDACachingAllocator::recordStream()`. (a) is deterministic. (b) is not: on
later allocations the allocator polls the recorded streams, so whether memory
gets reused depends on how far the GPU has gotten, and memory usage can blow up
when the CPU runs ahead. Prefer (a) where possible.

When writing or reviewing code that needs one of these, work out which of the
following cases applies:

- **Avoidable:** the code doing the cross-stream use also controls when the
  tensor is freed, or can hold a reference until a point it controls. Record
  an event after the last side-stream use and make the creation stream wait on
  it before dropping the reference. Don't use `record_stream`. FSDP2 works
  this way (see `docs/source/distributed.fsdp.fully_shard.md`).
- **Unavoidable under the current API:** the tensor goes to arbitrary user
  code, which frees it at an unknown time, and nothing we control runs in
  between where we could insert the sync. Here `record_stream` is the correct
  tool and shouldn't be flagged.
- **But we design the APIs.** If the API is new or can still change, ask
  whether a different shape would give us a point to sync. Examples: return a
  handle whose `wait()` does the sync and releases internal references;
  allocate outputs on the caller's stream and sync the side stream back before
  returning; keep ownership of buffers internally. c10d shows both sides:
  collectives keep their tensors alive in the `Work` object until `wait()`
  syncs the streams, with no `recordStream`, but `isend` doesn't require
  `wait()`, so it falls back to `recordStream` (see `ProcessGroupNCCL.cpp`).
  Weigh the design cost against the determinism benefit, and point out the
  option rather than insisting on it.
- **Hard to avoid:** sometimes avoiding it would take a redesign. For
  example, the autograd engine moves gradients between nodes whose streams
  come from the forward pass (`torch/csrc/autograd/input_buffer.cpp`). The
  engine has the graph and could delay frees to a point where it syncs, but
  arbitrary user hooks can see and hold on to gradients, so it doesn't know
  when the last use happens. Don't treat such cases as settled: say what
  information is missing for a sync point and what it would take to get it.

When `record_stream` is the right call, leave a comment explaining why the
free point can't be controlled. See also the `record_stream` docstring in
`torch/_tensor_docs.py`.
