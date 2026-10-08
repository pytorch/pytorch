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

We implement this by taking a shared lock when copying the data and
taking an exclusive lock when stealing the data. The exclusive lock
acquisition ensures that all pending shared locks are finished before
we steal the data.

Streams and CUDA graphs
-----------------------
On devices with streams, a lazy clone has the semantics of a clone()
enqueued on the stream that is current when the lazy clone is made. The
user synchronizes as if that clone had happened then, so materialization
has to keep the same ordering, even though it happens later, on whatever
stream is current then:

- The copy reads the shared data. It has to come after the writes that
  produced it, and before the data's memory is reused by the allocator.
- The copy writes the new allocation. It has to come before the uses of
  the materialized tensor, and before reuse of the new memory.

The CUDA caching allocator only reuses memory for allocations on the
stream it was allocated on, so work enqueued on that stream is ordered
before any reuse. Copying on the stream the lazy clone was made on (and,
for the storage that was lazily cloned from, the stream its memory was
allocated on) satisfies all of the above by stream order, and gives the new
allocation the stream that the user expects it to have. Copying on another
stream would need synchronization (`recordStream` or host blocking), which
we avoid; such materializations raise an error instead. Therefore,
`Allocator::copy_data` must enqueue the copy on the current stream.

Until it is materialized, a lazy clone keeps using the original
allocation. If it was made on a stream other than the one that allocation
belongs to (e.g., a copy made for a communication stream, or when warming
up for CUDA graph capture on a side stream), its uses on that stream would
race with reuse of the allocation once the original is freed, which an
eager clone wouldn't. So outside of capture, lazily cloning on such a
stream clones eagerly instead, and all lazy copies of some data share the
stream its memory was allocated on.

When the last reference steals the data instead of copying it, it writes
to the data on the current stream, which has to come after copies of the
data that may still be pending on other streams. Each eager copy records
an event on its stream (one per stream suffices, since later events on the
same stream supersede earlier ones), and the stealing stream waits for
them on the device.

CUDA graph capture relaxes the invariant above for one pattern: a graph
input (or a parameter), whose memory was allocated before the capture on
its own stream, lazily cloned during the capture on the capture stream.
Cloning eagerly would add a copy to every replay, even if the lazy clone is
never written, so it stays lazy. Its lifetime is not a concern: memory that
a graph accesses must stay alive for as long as the graph can be replayed.
But a graph is a recording that is replayed at baked-in addresses, which
brings two other hazards:

- Once vs. every replay: a copy made during capture is captured and redone
  by every replay. That is what a lazy clone made during the capture stands
  for, but a lazy clone made before the capture stands for a single copy at
  that time (and vice versa, a copy made after the capture can't be redone
  by the replays of the graph a lazy clone was made in). So each reference
  records whether it was made during capture, and copying is only allowed
  if that matches whether we are capturing (we don't distinguish between
  captures).
- Baked-in addresses: copying moves the materialized storage to a new
  allocation. Moving the storage that was lazily cloned from during capture
  is wrong if its memory was allocated before the capture (e.g., a graph
  input, which the user keeps writing to the old memory). Memory allocated
  during capture comes from the graph's private pool, for which
  `Allocator::was_allocated_on_stream` returns unknown, so copying a storage
  whose allocation stream is known raises an error during capture.

Stealing doesn't move the data, so it is allowed during and after capture
(e.g., when the user refills a graph input). Copies made during capture
aren't waited for by stealing, since they only run on replay, which the
user orders with respect to writing the inputs of the graph.

Two hazards are not checked, and are the user's responsibility:

- During capture, a lazy clone counts as a use of its source's memory for
  as long as it is alive and not yet materialized (the deferred copy reads
  it too). Anything that releases that memory, e.g., an external event
  recorded mid-graph that lets another stream overwrite a graph input, must
  come after the worst-case point where the lazy clone may still be
  unmaterialized.
- Materialization moves a storage to a new allocation. If a CUDA graph (or
  anything else that caches raw pointers) already captured its address, it
  keeps using the old memory: e.g., writing a lazily cloned tensor after a
  graph that reads it was captured is silently not seen by the graph.
