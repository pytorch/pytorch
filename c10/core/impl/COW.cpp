#include <c10/core/impl/COW.h>

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/COWDeleter.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/ParallelGuard.h>
#include <c10/util/UniqueVoidPtr.h>

#include <memory>
#include <optional>

namespace c10::impl::cow {

namespace {

// Wraps a DataPtr with a copy-on-write DataPtr.
at::DataPtr make_data_ptr(
    at::DataPtr const& data_ptr,
    cow::COWDeleterContext& ctx) {
  return at::DataPtr(data_ptr.get(), &ctx, cow::cow_deleter, data_ptr.device());
}

/// Copies a copy-on-write DataPtr.
at::DataPtr copy_data_ptr(at::DataPtr const& data_ptr) {
  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->increment_refcount();
  return make_data_ptr(data_ptr, *ctx);
}

// Returns the current stream of `device` if COW tracks streams for it, or
// nullopt otherwise. This is only done on CUDA, whose allocators enqueue
// `copy_data` on the current stream and whose streams support events; other
// backends keep the stream-agnostic behavior.
std::optional<c10::Stream> current_stream(c10::Device device) {
  if (!device.is_cuda()) {
    return std::nullopt;
  }
  return c10::impl::getDeviceGuardImpl(device.type())->getStream(device);
}

// Clones `storage` eagerly, on the current stream.
c10::intrusive_ptr<StorageImpl> eager_clone(StorageImpl& storage) {
  c10::DeviceGuard device_guard(storage.data_ptr().device());
  return make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      storage.allocator()->clone(storage.data_ptr().get(), storage.nbytes()),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
}

// Raises an error if data shared through `ctx` may not be materialized by
// copying it on `stream`: the copy must be ordered like the clone() that the
// lazy clone stands for, and we raise an error rather than synchronizing when
// it would not be.
void check_copy_allowed(
    const cow::COWDeleterContext& ctx,
    c10::Stream stream,
    bool capturing) {
  // During graph capture, a copy would become part of the graph and be redone
  // by every replay, which is wrong for a lazy clone made before the capture,
  // and copying the storage that was lazily cloned from would move it to new
  // memory inside of the graph (e.g., a graph input, which the user keeps
  // writing to). Rather than distinguish these cases, we don't copy during
  // capture at all: if the captured code needs a copy, the graph needs one on
  // every replay anyway, so making the lazy clone a clone() costs nothing.
  TORCH_CHECK(
      !capturing,
      "Cannot write to this tensor (or take its data_ptr()) during graph "
      "capture: it shares memory with a lazy copy (from _lazy_clone()), and "
      "writing to it requires copying it, which isn't supported during "
      "capture. If you only read the tensor, use const_data_ptr() instead of "
      "data_ptr(). Otherwise, use clone() instead of _lazy_clone() where the "
      "lazy copy is made.");
  // A lazy clone made during capture stands for a copy that every replay of
  // the graph makes, which a copy after the capture can't provide.
  TORCH_CHECK(
      !ctx.captured(),
      "Cannot write to this tensor (or take its data_ptr()) outside of graph "
      "capture: it shares memory with a lazy copy (from _lazy_clone()) made "
      "during a graph capture, and writing to it requires copying it, which "
      "the replays of the graph would not redo. If you only read the tensor, "
      "use const_data_ptr() instead of data_ptr(). Otherwise, use clone() "
      "instead of _lazy_clone() where the lazy copy is made, or free the lazy "
      "copy first.");
  // The copy must be enqueued on the stream the lazy copies were made on,
  // which is also the stream their memory was allocated on (when the
  // allocator can tell), and the stream the new allocation belongs to.
  const std::optional<c10::Stream> shared_stream = ctx.stream();
  TORCH_CHECK(
      shared_stream == stream,
      "Cannot write to this tensor (or take its data_ptr()) on ",
      stream,
      ": it shares memory with a lazy copy (from _lazy_clone()) made on ",
      shared_stream.has_value() ? c10::str(*shared_stream)
                                : std::string("another stream"),
      ", and writing to it requires copying it, which can't be done safely on "
      "another stream without synchronizing the streams. If you only read the "
      "tensor, use const_data_ptr() instead of data_ptr(). Otherwise, write "
      "to it on the stream the lazy copy was made on, or use clone() instead "
      "of _lazy_clone() where the lazy copy is made.");
}

} // namespace

bool has_simple_data_ptr(const c10::StorageImpl& storage) {
  const c10::DataPtr& data_ptr = storage.data_ptr();
  const void* ctx = data_ptr.get_context();
  const void* data = data_ptr.get();
  const c10::Allocator* allocator = storage.allocator();
  if (allocator != nullptr) {
    return allocator->is_simple_data_ptr(data_ptr);
  } else {
    return ctx == data;
  }
}

bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return reinterpret_cast<const void*>(data_ptr.get_deleter()) ==
      reinterpret_cast<const void*>(&cow::cow_deleter);
}

c10::intrusive_ptr<StorageImpl> lazy_clone_storage(StorageImpl& storage) {
  const at::DataPtr& data_ptr = storage.data_ptr();

  // There are three possible circumstances:
  //
  // 1) The storage has a normal data pointer with no out of the ordinary
  //    context. In this case we know that there are no blind aliases to the
  //    storage impl: they all will be public aliases and the user is expected
  //    to synchronize manually.
  //
  //    No locking is required in this case.
  //
  // 2) The storage already has a copy on write context. There
  //    is a potential race condition with a blind alias (i.e. an
  //    alias that the user is not required to synchronize
  //    with). Because our input storage is bound to a live reference
  //    to the data, we know that it isn't going away. A blind alias
  //    could be copying from it right now, but we will grab the
  //    context's mutex to protect us.
  //
  //    We do not need to lock in this case either, because we're just
  //    wrapping a context that we know isn't going away.
  //
  // 3) The storage has a context that is not the copy on write
  //    context. This is not supported, so we just return null.
  //
  //    No locking is required in this case.

  std::optional<DataPtr> new_data_ptr; // must be set below

  const bool simple = has_simple_data_ptr(storage);
  if (!simple && !is_cow_data_ptr(data_ptr)) {
    // Case 3) There is a context and it's not copy-on-write. Nothing
    // we can do here.
    return nullptr;
  }

  // The lazy clone has the semantics of a clone() enqueued now, on the
  // current stream. Until it is materialized, though, it keeps using the
  // original allocation, which the allocator only orders with respect to the
  // stream it was allocated on, so its uses on another stream would race with
  // reuse of that memory once the original is freed. So all references to
  // some data share one stream (and capture state, see materialize_cow), and
  // we clone eagerly instead when that's not the current one. (Except during
  // graph capture, where keeping the memory of graph inputs alive is the
  // user's responsibility anyway, and cloning eagerly would add a copy to
  // every replay of the graph.) See "Streams and CUDA graphs" in
  // README-cow.md.
  const std::optional<c10::Stream> stream = current_stream(data_ptr.device());
  const bool capturing = stream.has_value() && stream->is_capturing();
  if (stream.has_value()) {
    bool eager = false;
    if (!capturing && storage.allocator() != nullptr) {
      const std::optional<bool> allocated_on_stream =
          storage.allocator()->was_allocated_on_stream(data_ptr.get(), *stream);
      eager = allocated_on_stream.has_value() && !*allocated_on_stream;
    }
    if (!eager && !simple) {
      auto* ctx =
          data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
      TORCH_INTERNAL_ASSERT(ctx != nullptr);
      if (ctx->is_unique()) {
        // There are no other references, so we can move the data to the
        // current stream and capture state, after any pending copies.
        if (!capturing) {
          ctx->wait_for_copies(*stream);
        }
        ctx->set_stream(*stream, capturing);
      } else {
        eager = ctx->stream() != stream || ctx->captured() != capturing;
      }
    }
    if (eager) {
      return eager_clone(storage);
    }
  }

  if (simple) {
    // Case 1) We have a simple data pointer: wrap it.
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage._mutable_data_ptr_no_checks().move_context();

    // Save this for the result.
    new_data_ptr = make_data_ptr(
        data_ptr, *new cow::COWDeleterContext(std::move(original_ctx)));

    // Update this storage to the new copy on write context.
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr));
    storage.set_materializer(&materialize_cow);
    if (stream.has_value()) {
      new_data_ptr->cast_context<cow::COWDeleterContext>(cow::cow_deleter)
          ->set_stream(*stream, capturing);
    }
  } else {
    // Case 2): there is already a copy on write context. Just return a
    // new storage impl.
    TORCH_INTERNAL_ASSERT(storage.has_materializer());
    new_data_ptr = copy_data_ptr(data_ptr);
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());

  auto result = make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      *std::move(new_data_ptr),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
  result->set_materializer(&materialize_cow);
  return result;
}

void materialize_cow(StorageImpl* storage) {
  TORCH_INTERNAL_ASSERT(
      !c10::ParallelGuard::is_enabled(),
      "Materializing a storage in the loop function of at::parallel_for is forbidden");
  const at::DataPtr& data_ptr = storage->data_ptr();

  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);

  // Materialization copies the data on the current stream, or steals the data
  // if this is the last reference to it. See "Streams and CUDA graphs" in
  // README-cow.md for why the checks below are needed.
  //
  // Everything that can fail (the checks, allocating, copying, recording and
  // waiting for events) happens before this reference gives up its count on
  // the context, so that the storage remains a valid copy-on-write storage if
  // it does. While we hold the count, the data can't be stolen or freed by
  // another reference.
  //
  // Whether we copy or steal is only known once the count is decremented. If
  // this is the only reference, no other can be created concurrently (that
  // would require reading this storage while we are writing it), so we will
  // steal. Otherwise we copy, even if another thread concurrently frees the
  // last other reference, in which case the copy was unnecessary (and the
  // checks too strict), but correct.
  const c10::Device device = data_ptr.device();
  const std::optional<c10::Stream> stream = current_stream(device);
  const bool capturing = stream.has_value() && stream->is_capturing();
  std::optional<DataPtr> copy;
  if (ctx->is_unique()) {
    // Stealing the data means writing to it on the current stream, which
    // must be ordered after any copies of it that may still be pending.
    // (During capture, we can't wait for eager work, but torch.cuda.graph
    // synchronizes before capturing.)
    if (stream.has_value() && !capturing) {
      ctx->wait_for_copies(*stream);
    }
  } else {
    if (stream.has_value()) {
      check_copy_allowed(*ctx, *stream, capturing);
    }
    c10::OptionalDeviceGuard device_guard;
    if (stream.has_value()) {
      device_guard.reset_device(device);
    }
    copy = storage->allocator()->clone(data_ptr.get(), storage->nbytes());
    if (stream.has_value()) {
      ctx->record_copy_event();
    }
  }

  // Nothing below can fail.
  auto result = ctx->decrement_refcount();

  std::optional<DataPtr> new_data_ptr;
  if (copy.has_value()) {
    // If this turned out to be the last reference after all, the original
    // data is freed along with `result`.
    new_data_ptr = std::move(copy);
  } else {
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<cow::COWDeleterContext::LastReference>(result));
    std::unique_ptr<void, DeleterFnPtr> data =
        std::get<cow::COWDeleterContext::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    new_data_ptr = DataPtr(
        data.release(), data_ptr.get(), data.get_deleter(), data_ptr.device());
  }

  DataPtr old_data_ptr =
      storage->set_data_ptr_no_materialize(*std::move(new_data_ptr));
  // The refcount of the context was already decremented above. Release the
  // reference to the context so the refcount doesn't get decremented again
  old_data_ptr.release_context();
}

} // namespace c10::impl::cow
