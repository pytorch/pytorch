Copy-on-write storage
=====================
This library adds support for copy-on-write storage, i.e. lazy copies,
to tensors. The design maintains the PyTorch invariant that tensors
alias if and only if they share a storage. Thus, tensors that are lazy
copies of one another will have distinct storages that share a data
allocation.

Thread-safety
-------------
The correctness of this design hinges on the pre-existing PyTorch user
requirement (and general default programming assumption) that users
are responsible for guaranteeing that writes do not take place
concurrently with reads and other writes.

Lazily copied tensors add a complication to this programming model
because users are not required to know if lazy copies exist and are
not required to serialize writes across lazy copies. For example: two
tensors with distinct storages that share a copy-on-write data context
may be given to different threads that may do whatever they wish to
them, and the runtime is required to guarantee its safety.

Lazy-clone is a host read; materialization and `cow::record_done` are host writes. Serialize either
write with all host accesses through the same storage, including aliases. Distinct storages sharing
a COW context may operate concurrently. Each holds its reference through copying, so a unique
reference can steal the allocation safely. Concurrent CPU materializations may both copy.

CUDA streams and capture
------------------------
A lazy clone can share memory in eager execution or within one CUDA capture when
created on the allocation stream, where its input must already be ready. The
allocation stream remains the stream on which the allocator can reuse memory.
Cloning on another stream or with unknown allocation metadata (cudaMallocAsync,
pluggable allocators) immediately copies. Private-pool allocations, including
MemPool allocations, immediately copy unless created in the current capture.

To materialize on operation stream B, allocate and copy on allocation stream A, then make B wait for
A. Both allocations belong to A; the immutable source lets the copy overlap earlier reads on B.
Events provide the same ordering in eager execution and capture, without host blocking or
`recordStream`. Deferred copies can introduce later dependencies, so `_lazy_clone()` does not
promise the stream dependencies of `clone()` at the original call site.

The caller must join ordinary side-stream uses back onto A before releasing a storage; COW's copy
dependency does not cover them. `record_stream()` does not provide this lifetime guarantee for COW
tensors. An external allocation stream must remain valid while COW may enqueue work on it.

### Retention and explicit completion

Finishing a materializing copy does not finish earlier asynchronous reads, including reads submitted
before COW existed. Therefore materialization retains the old COW reference in
`StorageExtraMeta::cow_retired_data`, preventing reuse or stealing until storage release or explicit
completion. A materialized storage can keep twice its memory for its entire lifetime. While
retaining that one old allocation, subsequent lazy clones use eager copies to bound retention. If
both siblings materialize without completion, both may copy.

`torch._C._cow_record_done(x)` (C++: `cow::record_done(storage)`) records events on the current
stream and releases the retained reference, enabling sharing of the new allocation again. It leaves
current references, values, and pointers unchanged.

Before completion, join **all earlier uses** onto the current stream, including uses through storage
aliases, views, saved pointers, uses before the first lazy clone, and materializing copies.
Serialize the call with all host access to that storage. These are caller obligations; violating
them can corrupt data.

Completion is final for each reference; repeated calls are no-ops. Subsequent data access or lazy
cloning through a completed current reference raises, including access through storage aliases.
Do not submit further work through saved pointers to that reference. Completing a retained reference
leaves the materialized tensor usable on its new allocation. Lazy siblings remain independent.

Each `DataPtr` owns a `COWReference` pointing to the shared `COWDeleterContext`. Release transfers
its completion event to the context before decrementing the refcount. The last reference waits on
these events before stealing, avoiding unrelated later work on A. If any departed reference was
incomplete, it instead takes a fresh wait on A, relying on the ordinary lifetime contract.
Completion also makes A wait immediately, protecting reuse without work in a deleter. One event per
completed sibling remains until the context dies. Completion covers aliases, not lazy siblings.

### Capture boundaries

Both streams must be outside capture or belong to the same active capture. Check this before
allocation, copying, or completion. Events become graph dependencies and copies run on every replay.

| Situation | Behavior |
| --- | --- |
| Existing COW tensor used as graph input | Reject even read-only access. Materialize before capture. |
| External graph input lazily cloned during capture | Reject creating a COW context. Omit the lazy clone for reads, or use `clone()`. Allocators without metadata still eagerly copy. |
| Allocation and COW context created in this capture | Sharing, shared materialization, and completion are permitted within it. |
| Shared COW output materialized after capture | Reject; materialize during capture or release other references (including retained ones) first. |
| Unique COW output materialized after capture | Steal without changing the graph's address. Use a fresh stream wait, not captured completion events. The caller orders graph replay and external uses. |
| Private-pool output lazily cloned after capture | Immediately copy. |

Completion cannot cross the context's capture boundary, even for retained references. Raw-pointer
writes, external registrations, or graphs using addresses saved before COW remain the caller's
responsibility: resolve lazy copies before exposing a writable address and keep the allocation
stable while it is used.
