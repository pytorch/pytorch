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

It turns out that this is not that difficult to protect because, due
to the copy-on-write requirement, we just need to materialize a tensor
upon writing. This could be done entirely without synchronization if
we materialized each copy, however, we have a common-sense
optimization to elide the copy for the last remaining reference. This
requires waiting for any pending copies.

### Thread-safety detailed design
There are three operations that affect the copy-on-write details of a
tensor:

1) lazy-clone
2) materialization (i.e. requesting a mutable data pointer)
3) explicit completion (`cow::record_done`)

Lazy-clone is logically a read operation; materialization and explicit completion
are logically write operations. For a given set of tensors that share a storage, if
materialization or completion is taking place, no other host access, including
lazy-clone or another completion call, can be concurrent with it. This exclusion
applies to all aliases of that storage; GPU stream waits do not serialize host
threads. Distinct storages sharing a COW context may be completed concurrently.

However, this insight only applies within a set of tensors that share
a storage. We also have to be concerned with tensors with different
storages that share a copy-on-write context. In this world,
materialization can race with lazy-clone or even other
materializations. _However_, in order for this to be the case, there
must be _at least_ two references to the context. This means that the
context _can not_ vanish out from under you if you are performing a
lazy-clone, and hence, it only requires an atomic refcount bump.

The most complicated case is that all lazy-copies are concurrently
materializing. In this case, because a write is occurring, there are
no in-flight lazy-copies taking place. We must simply ensure that all
lazy-copies are able to materialize (read the data) concurrently. If
we didn't have the aforementioned optimization where the last copy
steals the data, we could get away with no locking whatsoever: each
makes a copy and decrements the refcount. However, because of the
optimization, we require the loser of the materializing race wait for
the pending copies to finish, and then steal the data without copying
it.

Materialization holds its reference until copying succeeds, including through
allocation failures. A unique reference can take the allocation without copying.

CUDA streams and capture
------------------------

CUDA operations run asynchronously. A materializing copy can finish while
earlier reads still use the original allocation. COW must protect those reads
before allowing another clone to overwrite the allocation or allowing the
allocator to reuse its memory.

The implementation therefore retains references to old allocations after
materialization. Callers can explicitly complete earlier uses to release those
references before the storage is destroyed.

### Allocation and operation streams

The **allocation stream** is the stream on which an allocation was made and
on which the allocator may reuse its memory. COW does not change this stream.
The **operation stream** is the current CUDA stream on which an operation
accessing the tensor runs.

A lazy clone can share an allocation in eager execution or within CUDA graph
capture when the clone is created on the allocation stream. The source's data
must be ready on that stream before `_lazy_clone()` is called, just as for
`clone()`. A lazy clone made on another stream, or using an allocator that cannot
supply allocation metadata, falls back to an immediate copy. During capture,
the allocation and shared materializations must belong to the same capture;
the capture restrictions are described below.

Suppose a shared allocation belongs to stream A and an operation on stream B
materializes one of its storages. COW performs these steps:

1. Allocate the replacement on A.
2. Queue the copy from the original allocation to the replacement on A.
3. Make B wait for that copy.

The operation can then use the replacement on B. The shared source is immutable,
so the copy can overlap earlier reads on B. Both allocations belong to A, and
the copy needs no reverse wait from A to B. During capture, the copy and wait
become part of the graph. Neither operation blocks the CPU.

Allocating the replacement on A preserves the original storage's allocation
stream. The caller must still order other uses on side streams back onto A
before releasing the storage. COW's internal waits cover the materializing
copy; they do not replace the caller's responsibility to order ordinary reads
and writes.

Deferring a copy can introduce stream dependencies at materialization time.
Consequently, `_lazy_clone()` does not promise the same stream dependencies as
an ordinary `clone()` performed at the original call site. CPU behavior is
unchanged.

### Why materialization retains the old allocation

Consider an earlier read of `x` queued on stream B. A later call to
`x.data_ptr()` on stream A can materialize `x` while that read is still pending.
Although `data_ptr()` requests a writable pointer, obtaining the pointer does
not itself write to the allocation. Both the earlier read and the materializing
copy read the original allocation, so they need not wait for each other.
Finishing the copy does not imply that the earlier read has finished.

After materialization, `x` therefore retains its reference to the original
allocation. This prevents the allocator from reusing that memory and prevents
another lazy clone from stealing and overwriting it. The reference remains
until `x`'s storage releases its resources or the caller explicitly completes
the earlier uses.

We distinguish two kinds of COW reference belonging to a storage:

- A **current reference** gives the storage access to its current data.
- A **retained reference** keeps an old allocation alive after materialization
  to protect earlier CUDA uses.

Retention has two costs. If both `x` and its lazy clone are modified while
their storages remain alive, both may need to copy. Also, subsequent lazy
clones of `x` use immediate copies while `x` retains the old allocation. This
fallback limits retention to one old allocation per storage.

### Explicit completion

`torch._C._cow_record_done(x)` (C++: `cow::record_done(storage)`) records a point
in the current CUDA stream after which earlier uses associated with `x`'s
storage will have finished. The call queues events; it does not wait for the
GPU to reach that point.

Before calling completion, make the current stream wait for all earlier uses
associated with the storage's current and retained COW references. Include:

- Uses through `x` and every tensor that shares its storage, including views.
- Uses through pointers obtained from that storage.
- Uses submitted before the first lazy clone.
- Copies submitted during materialization.

For example, if an earlier use ran on stream B and completion will be recorded
on A, call `A.wait_stream(B)` after submitting that use and before recording
completion on A.

After completion, do not use saved pointers covered by the call again. Obtain
a fresh pointer through the tensor or storage before submitting new work.
This gives COW an opportunity to invalidate the previous completion marker.

Completion applies to a storage and all its aliases. It does not complete
independent lazy clones: if `y = x._lazy_clone()`, completing `x` does not
complete `y`.

Serialize completion calls with other host accesses to the same storage,
including calls through aliases. GPU stream waits alone do not provide this
host-thread exclusion.

#### Effect on ownership

Completion releases the storage's retained references. This allows another
clone to steal an old allocation once it is the only remaining owner and has
waited for the recorded events. It also allows subsequent lazy clones of the
materialized tensor to share its new allocation again.

If the storage still has a current COW reference, completion leaves that
reference alive. Destroying the storage later releases the reference with its
recorded completion event. The tensor's values and current data pointer do not
change as a result of completion.

Completion has no effect on CPU storage or on storage with neither a current
nor a retained COW reference.

#### Example: avoiding a second copy

```python
import torch

A, B, C = [torch.cuda.Stream() for _ in range(3)]

with torch.cuda.stream(A):
    x = torch.ones(1024, device="cuda")
    y = x._lazy_clone()

with torch.cuda.stream(B):
    B.wait_stream(A)
    x.add_(1)  # Materializes x by copying; y keeps the original allocation.
    torch._C._cow_record_done(x)

with torch.cuda.stream(C):
    C.wait_stream(A)
    y.add_(2)  # Waits for x's completion event, then steals the original.

# Both allocations belong to A. Join their side-stream uses back to A.
A.wait_stream(B)
A.wait_stream(C)
torch.cuda.current_stream().wait_stream(A)
```

At the end, `x` contains twos and `y` contains threes. Without the completion
call, `x` would still retain its old reference, so modifying `y` would require
a second copy.

#### Using a tensor again after completion

Accessing data through a storage with a current COW reference invalidates its
completion marker. This includes read-only access through `const_data_ptr()`.
Materialization also invalidates completion because its copy is a new read of
the shared allocation.

After submitting new uses, either record completion again or order those uses
back onto the allocation stream before releasing the storage. If completion
is recorded again on a different stream, COW first makes that stream wait for
the reference's previous completion event. This preserves dependencies from
earlier calls.

New lazy clones have their own completion state. A completion marker is never
inherited from the storage that was cloned.

### Completion events and allocation lifetime

Each COW `DataPtr` owns a `COWReference`. References sharing an allocation point
to one `COWDeleterContext`, which owns the original allocation and tracks its
reference count. Each reference has its own completion state and optional
event.

CUDA materialization moves the old `DataPtr`, including its `COWReference`,
into the optional `StorageExtraMeta::cow_retired_data` slot. A storage retains
at most one old allocation. Storage destruction, `release_resources()`, and
`reset()` release retained references. Explicit
completion records their events and releases them sooner.

When a completed reference is released, it transfers its event to the shared
context before decrementing the reference count. If every previously released
reference supplied a completion event, the last remaining storage waits for
those events before stealing the allocation. This avoids waiting for unrelated
work submitted to the allocation stream after completion. If any reference was
released without completion, stealing instead waits for work already queued
on the allocation stream.

Completion also immediately queues a wait on the allocation stream. This
protects eventual allocator reuse without requiring the COW deleter to enqueue
new work. Events from different references are kept separately.

### CUDA graph capture

Allocations record the capture in which they were created. A COW context also
records its capture, or records that it was created outside capture. These
identities determine which operations are permitted:

| Situation | Required behavior |
| --- | --- |
| An existing COW tensor will be used as a graph input | Materialize it before capture. Even read-only data access during capture rejects an existing COW context from outside that capture. |
| An external graph input is lazily cloned during capture | Creating a new COW context for it is rejected. Omit `_lazy_clone()` for read-only use, or use `clone()` for independent storage. |
| An allocation is created during capture | Eligible lazy clones may share it, and shared materialization must occur within the same capture. The copies and dependencies run on every replay. |
| A COW output still has multiple references after capture | Shared materialization is rejected. Materialize during capture or release the other references first. Retained references also count. |
| A captured allocation has only one COW reference after capture | Materialization can take ownership without changing the address used by the graph. The caller must order accesses relative to graph replay. |
| An output from a graph's private memory pool is cloned after capture | The lazy clone uses an immediate copy. |

Completion must occur within the capture that created its COW context, or
entirely outside capture for a context created outside capture. A completion
call cannot cross this boundary, even when it only releases retained
references.

Taking ownership of a unique captured allocation outside capture uses the
ordinary stream wait, rather than an event from the earlier capture. The
caller remains responsible for ordering graph replay and external accesses.

The CUDA backend implements stream dependencies by recording an event on the
source stream and making the waiting stream wait for it. Both streams must be
outside capture or belong to the same active capture and graph. During
capture, these operations create graph dependencies. They do not poll events,
block the host, or call `recordStream`.

Allocators without allocation metadata use the immediate-copy fallback rather
than creating a COW context. The external-input restriction above concerns
creating a COW context, not this fallback.

COW cannot track addresses cached by external code. Raw-pointer writes,
external buffer registrations, and graph uses of addresses saved before a
tensor became COW remain the caller's responsibility. Resolve lazy copies
before exposing a writable address, and keep that allocation stable for as
long as the address may be used.
