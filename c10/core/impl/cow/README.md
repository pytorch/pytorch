Copy-on-Write Storage
=====================

Motivation
----------
PyTorch inherited from NumPy an optimization on the reshape function
that produces a view as an output if it can be represented as
such. This complicates the work of the PyTorch compilation stack
because it needs to understand the representation of the input in
order to understand if the output will be a copy or an alias.

The compilation stack would rather not be concerned with such details,
motivating the Stride Agnostic PyTorch project.

To address reshape specifically, we wish to simplify the
implementation to *always* copy, but copy lazily upon modification if
we can represent the output as a view.

Implementation plan
-------------------
We have not implemented copy-on-write tensors yet, because this is a
backward incompatible change (see Backward Incompatible section
below). But this is the design.

A copy-on-write tensor, also known as a lazily-copied tensor, is
initially going to be created by an operation like reshape which would
historically have created a view. We wish to maintain the performance
of the view but drop the aliasing aspect. Note that there is desire
for copy-on-write tensors outside of reshape, so we will likely also
want to add a public operator that can do this. It could be named
"lazy_clone" or something similar.

The core tenet of the design is that we wish to maintain the invariant
that Tensors alias if and only if they share a storage. Thus when we
create a lazy-copy, we will need to create a new storage. We also will
have to modify the source storage, since it also must now be lazily
copied.

The algorithm for creating a new copy-on-write tensor (and also
converting the source to copy-on-write) is as follows:

```
def lazy_clone(src: Tensor) -> Tensor:
    # First ensure that the source has a copy-on-write enabled storage.
    # We implement this using a custom context on the DataPtr. The
    # source tensor might already have this enabled, but if it doesn't,
    # convert it.
    if not has_copy_on_write_context(src.storage().storage_impl()):
        if not(wrap_with_copy_on_write_context(src.storage().storage_impl())):
            # For whatever reason, we weren't able to wrap the DataPtr.
            # We have to do an eager copy.
            return src.clone()

    new_storage = fork_copy_on_write_storage(src.storage()) # can't fail
    # Now just create a new tensor using the new storage.
    return new Tensor(storage=new_storage, sizes_and_strides_like=src)
```

That's the high level algorithm. The copy-on-write context that we
introduce is morally just a refcount on the underlying physical data
pointer. Each unique storage represents a set of tensors that share a
view and thus will hold a single refcount on the context.

Now we just need to intercept writes to the storage and materialize
them. We can use a few mechanisms to do this:

1) autograd knows which operators write to which tensor inputs. We can
   materialize at that point when autograd is enabled.
2) if autograd is not enabled, we can introduce a new dispatch key
   that does the same trick
3) we can also materialize whenever there's mutable access to the data
   pointer through any of `at::Tensor`, `c10::TensorImpl`,
   `c10::Storage`, `c10::StorageImpl`, or `c10::DataPtr`. With the
   current codebase, this will be too aggressive, but we will refactor
   to have a minimal set of mutable accesses.

Backwards incompatibiility
--------------------------
Changing reshape to produce a copy is a backwards incompatible change,
because users could be relying on the aliasing behavior, intentionally
or not.

For one release, rather than returning copy-on-write tensors, we
instead warn when users have triggered behavior in their program that
relies on the aliasing of the output and input.

To do this, we must simulate the behavior in a backward compatible
way. To remain backward compatible, the aliases must preserve the
invariant that they have the same storage. This is a big deviation
from the design detailed above. To get closer to the real design and
implementation, we introduce a new `c10::TensorImpl` level concept
called "Shadow Storage". The shadow storage represents what the
storage would have looked like the view actually been a lazy copy.

In the instrumented world we thus maintain the formal invariant that
tensors that alias share a storage. But we have a new invariant:
tensors that are lazy-copies of each other will share a shadow
storage.

So what do we warn on? We warn if there is a write to a tensor in a
set that shares a shadow storage followed by a read or a write to a
different tensor that shares a physical storage but has a different
shadow storage. In the real implementation, the first write would have
triggered a copy, forever cleaving the two sets of tensors, but in the
current world we instead had behavior that relied on the view-ness of
the output of reshape.

We can track these violations simply by adding a generation number to
the shadow and physical storages, updating them both on writes and
observing if a read or write ever encounters values that are out of
sync.

We have a few mechanisms for tracking reads and writes:
 * reads can be tracked by const accesses to the data pointer
 * writes can be tracked by mutable accesses to the data pointer
 * writes may also be tracked via autograd, using the same mechanism
   to bump version numbers

Note that we presently are only checking via autograd, since we don't
have const access to the data pointer, so we would be way too
aggressive if we assumed every access was a real write.

### Optimizations to the instrumentation
Technically, every tensor will require a shadow storage, because if we
were to create a view of a tensor and then create a lazy-copy, both
the original tensor and the view would have to share the shadow
storage, and thus it has to be created and shared before we ever even
know we needed it.

But we don't want to pay the memory cost of this since we don't expect
it to be that common. We can get around this by saving the original
shadow storage on the physical storage itself. We will have an
asymmetrical rule that states that any tensor that has a null shadow
storage will instead get its shadow storage from the physical
storage. This allows us to avoid refcount bumps on the shadow storage
as well as deferring any generation number bumps until we actually
have an outstanding copy on write.

The simulation instrumentation itself will be unnecessary to maintain
once we can transition to enabling lazy copies. This may happen after
the first that goes out which contains the instrumentation and the
user warning.

Future work
-----------
* enrich the warning by flagging reads/writes to data pointer after a
  big refactoring
* analyze violations of the warning and make a decision about whether
  we require any coordination about the BC change or if we should just
  let the warning run its course
* implement the actual copy on write
* simplify the compiler stack to no longer concern itself with this
