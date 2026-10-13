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

Stream-ordered devices
----------------------
On devices like CUDA, reads of a tensor are enqueued on streams and run
asynchronously, and the host-side refcount of a copy-on-write context
says nothing about whether they have finished. We want a lazy clone to
be safe to use under the same synchronization a user would need for an
ordinary clone made on the current stream. Two things could break
that:

1) Stealing the data into a write. The last reference to materialize
   would normally take the data without copying it. But another
   reference, even one that has been freed, may have reads of the data
   pending on some other stream; a user of an ordinary clone would not
   have ordered those with writes to a different tensor. (Recording a
   stream for a freed tensor only defers the reuse of its memory; it
   does not order other writes to it.)

2) Freeing the data too early. When a reference materializes, it
   copies the data and drops its reference to it. If that was the last
   reference, the data is freed while reads enqueued through the
   materialized reference may still be pending. The user only promised
   to synchronize those reads before freeing that tensor, not before
   writing to it.

So on stream-ordered devices we never steal: materialization always
copies, on the current stream, into a new allocation. The materialized
storage keeps its reference to the shared data until the storage itself
is freed (`StorageExtraMeta::cow_materialized_from_`). A lazy clone of
such a storage copies eagerly instead of creating another
copy-on-write context.

Synchronizing with the lifetime of a tensor means synchronizing with
the stream it was allocated on: the caching allocator orders reuse of
its memory after the work on that stream. For the shared data to obey
the same rule, every reference to it must have the same allocation
stream. So lazy clones are only made on the stream the data was
allocated on, and the copy made by materialization is allocated on that
stream as well; otherwise, the lazy clone copies eagerly.

Lazy clones also copy eagerly when the allocator cannot tell us the
allocation stream, when the data is in a private memory pool (such as a
CUDA graph's, which graph replays write to), and during CUDA graph
capture. Materializing during capture is an error: the copy would be
replayed with every replay of the graph.

CUDA graphs can also read and write memory outside their private pool,
like graph inputs, without us knowing. A lazy clone of such memory made
after capture changes when a replay writes the memory, and a replay
keeps reading the old address of a graph input which later materializes
after being lazily cloned. Callers must not lazily clone memory that
graphs use. Freeing shared data whose allocation stream is being
captured is not supported either.

`record_stream()` on a lazy clone records the stream for the shared
data, so it is honored when the shared data is freed. On a materialized
storage, it also records the stream for the data it was materialized
from, which uses enqueued before materialization read. (Code which
records streams with the caching allocator directly, at the time of
use, records them for the data that use reads.)
