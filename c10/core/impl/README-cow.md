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
are responsible for guaranteeing that writes do not take places
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
There are two operations that affect the copy-on-write details of a
tensor:

1) lazy-clone (e.g. an explicit call or a hidden implementation detail
   added through an operator like reshape)
2) materialization (i.e. any write to the tensor)

The key insight that we exploit is that lazy-clone is logically a read
operation and materialization is logically a write operation. This
means that, for a given set of tensors that share a storage, if
materialization is taking place, no other read operation, including
lazy-clone, can be concurrent with it.

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

CUDA allocations have an immutable reuse stream. Eager COW sharing is enabled
when this is the stream on which the lazy clone is made. Other streams and
allocators without allocation metadata fall back to an eager copy. CPU behavior
is unchanged.

A materialization can run on the current operation's stream. It allocates the
replacement on the original reuse stream, makes the operation stream depend on
allocation readiness, enqueues an asynchronous copy, and makes the reuse stream
depend on that copy. The replacement therefore keeps the lifetime contract of
the original tensor. The last reference also establishes a dependency from the
reuse stream before taking the allocation. Callers still join ordinary side
stream uses back to a tensor's creation stream before releasing it.

Copy completion is not sufficient to retire the old allocation. For example,
data_ptr() can materialize x while an earlier read of x is pending on another
stream. That read need only be joined before x is released, not before another
read-only use of x. CUDA materializations retain the old COW DataPtr in the
storage's extra metadata until release_resources() or destruction. These
references prevent both allocator reuse and premature stealing by another
clone. Without explicit completion, materialized CUDA tensors can keep old
allocations alive, and two live tensors which are both written can require two
copies. Further lazy clones of a storage which still retains an old allocation
use eager copies, bounding retention to one previous allocation per storage.

### Explicit completion

`torch._C._cow_record_done(tensor)` (C++: `cow::record_done(storage)`) records an
event on the current stream for each current or retired COW reference belonging
to the tensor's storage. The caller must first join all uses of those allocations
onto this stream, including reads before the first lazy clone, materializing
copies, and uses through views or saved pointers. This is a storage-wide boundary;
saved pointers covered by it must not be used again. CPU and ordinary storage
without retained references are unaffected.

For example, with x created and lazily cloned on stream A:

```python
y = x._lazy_clone()
with torch.cuda.stream(B):
    B.wait_stream(A)
    x.add_(1)
    torch._C._cow_record_done(x)
with torch.cuda.stream(C):
    C.wait_stream(A)
    y.add_(2)  # Steals the original allocation after waiting for x's event.
```

Completion releases retired references without changing the materialized tensor.
It also enables subsequent lazy clones to share its new allocation, instead of
taking the retention fallback. Each new sibling has an independent completion
record; completing a parent never completes its descendants.

After read-only use, completion leaves the current reference alive. Dropping the
tensor and its storage aliases then retires that reference with the recorded
event. Any subsequent data access, including const_data_ptr(), cancels this
completion and restores the normal lifetime contract. Complete again after the
new uses, or join them back to the allocation stream before releasing the tensor.
Materialization also cancels prior completion, since its copy is a new read.
Renewed completion first waits on this reference's previous event, so moving the
boundary to another stream cannot discard earlier completed uses.

Each COW DataPtr owns a COWReference; materialization moves that same reference
into retained storage metadata. On release, a completed reference transfers its
event to the shared COWDeleterContext before decrementing the count. If every
departed sibling completed explicitly, stealing waits on these fixed events and
does not fence unrelated later work on the allocation stream. Otherwise it keeps
the conservative stream wait. Completion also enqueues a wait on the allocation
stream immediately, making eventual allocator reuse safe without CUDA calls
from the deleter. Events from different siblings are never overwritten.

Completion must occur in the same capture as its COW context, or entirely in
eager execution. It cannot release a captured reference from eager code or an
eager reference during capture. A unique captured output can still keep its
address when materialized after capture, under the existing replay-ordering
contract; it uses the ordinary stream wait, not a captured completion event.

The CUDA backend uses event record/wait for both eager and captured dependencies.
During capture, CUDA turns these operations into graph dependencies. This does
not poll events, block the host, or call recordStream. The stream wait checks
that the streams are both eager or belong to the same capture and graph;
dependencies across those boundaries are rejected.

Allocations record their capture of origin. COW on capture-local allocations can
materialize inside the same capture: copies and ordering become part of every
replay. Creating a COW context for an external graph input during capture is
rejected at _lazy_clone(): omit it for read-only use, or use clone() for
independent storage. Existing eager COW inputs must be materialized before
capture. Shared COW outputs cannot materialize after capture; resolve their
copies during capture or release the other references. Unique materialization
preserves the captured address.

Private-pool outputs cloned after capture use eager copies. This does not
register every tensor used by arbitrary CUDA graphs: addresses cached before a
tensor becomes COW, raw-pointer writes, and external buffer registrations still
require callers to keep the allocation stable and resolve lazy copies at that
boundary.
