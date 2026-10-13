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

Asynchronous reads
------------------
On devices with streams (e.g., CUDA), materialization copies the data
asynchronously: the copy is enqueued on the current stream of the
storage's device and may still be running when materialization
returns. As with any other asynchronous work, we assume that all users
of the data (including the lazy copies sharing it) use the same
stream, so that work enqueued afterwards, such as a write to the
source by the copy that steals it, or the reuse of its memory after it
is freed, is ordered after the copy. This is also what lets the copy
be captured by CUDA graphs.

Stealing introduces a hazard that the usual stream discipline does not
cover. When the last remaining lazy copy steals the data, it may write
to it right away. That write is not ordered with respect to a pending
asynchronous read of the data on some other stream. `record_stream`
does not help: it only delays the reuse of memory by the allocator
after it is freed, and stealing is not a free.

Therefore, it is illegal to issue an asynchronous read directly on a
tensor that is a lazy copy, or that may be lazily copied while the read
is pending. Instead, take a lazy copy of the tensor, issue the read on
that lazy copy, and keep a reference to the lazy copy (not just a
`record_stream` on it) until the read has completed. While the lazy copy is alive, no other copy can steal the
data, so any write to one of them materializes a fresh copy instead of
writing to the data being read.

Materialization follows this rule itself: the materializing copy holds
its reference to the data (via the shared lock described above) until
its copy has been enqueued, so a steal can only happen afterwards, and
the stealer's writes are ordered after the copy on the shared stream.
