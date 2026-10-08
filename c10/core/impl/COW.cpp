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

// Makes a copy-on-write DataPtr referencing `ctx`. The caller is responsible
// for the refcount of `ctx`.
at::DataPtr make_data_ptr(
    void* data,
    cow::COWDeleterContext& ctx,
    std::optional<c10::Stream> clone_stream,
    bool made_while_capturing,
    c10::Device device) {
  return at::DataPtr(
      data,
      new cow::COWReference{&ctx, clone_stream, made_while_capturing},
      cow::cow_deleter,
      device);
}

// Returns the current stream of `device`, or nullopt if the device does not
// have streams.
std::optional<c10::Stream> current_stream(c10::Device device) {
  if (device.is_cpu() || device.is_meta() ||
      !c10::impl::hasDeviceGuardImpl(device.type())) {
    return std::nullopt;
  }
  return c10::impl::getDeviceGuardImpl(device.type())->getStream(device);
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

  // The lazy clone has the semantics of a clone() enqueued now, i.e., on the
  // stream that is current when the lazy clone is made (not when it is later
  // materialized). We record that stream so that materialization can be
  // checked against it.
  void* data = data_ptr.get();
  const c10::Device device = data_ptr.device();
  const std::optional<c10::Stream> stream = current_stream(device);
  const bool capturing = stream.has_value() && stream->is_capturing();

  // Until it is materialized, a lazy clone keeps using the original
  // allocation, which the allocator only orders with respect to the stream it
  // was allocated on. A lazy clone made on another stream (e.g., a copy made
  // for a communication stream) would therefore need synchronization that an
  // eager clone doesn't, so we clone eagerly instead. (Except during graph
  // capture, where keeping the memory of graph inputs alive is the user's
  // responsibility anyway, and the copy would be redone by every replay.)
  if (stream.has_value() && !capturing && storage.allocator() != nullptr) {
    const std::optional<bool> allocated_on_stream =
        storage.allocator()->was_allocated_on_stream(data, *stream);
    if (allocated_on_stream.has_value() && !*allocated_on_stream) {
      c10::DeviceGuard device_guard(device);
      return make_storage_impl(
          StorageImpl::use_byte_size_t(),
          storage.sym_nbytes(),
          storage.allocator()->clone(data, storage.nbytes()),
          storage.allocator(),
          storage.resizable(),
          storage.device_type());
    }
  }

  if (simple) {
    // Case 1) We have a simple data pointer: wrap it.
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage._mutable_data_ptr_no_checks().move_context();
    auto* ctx = new cow::COWDeleterContext(std::move(original_ctx));

    // Save this for the result.
    new_data_ptr = make_data_ptr(data, *ctx, stream, capturing, device);

    // Update this storage to the new copy on write context.
    ctx->increment_refcount();
    storage.set_data_ptr_noswap(make_data_ptr(
        data, *ctx, /*clone_stream=*/std::nullopt, capturing, device));
    storage.set_materializer(&materialize_cow);
  } else {
    // Case 2): there is already a copy on write context. Just return a
    // new storage impl.
    TORCH_INTERNAL_ASSERT(storage.has_materializer());
    auto* ref = data_ptr.cast_context<cow::COWReference>(cow::cow_deleter);
    TORCH_INTERNAL_ASSERT(ref != nullptr);
    ref->context->increment_refcount();
    new_data_ptr =
        make_data_ptr(data, *ref->context, stream, capturing, device);
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

  auto* ref = data_ptr.cast_context<cow::COWReference>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ref != nullptr);
  cow::COWDeleterContext* ctx = ref->context;

  // Materialization enqueues a copy of the data on the current stream, or
  // steals the data if this is the last reference to it. See "Streams and
  // CUDA graphs" in README-cow.md for why the checks below are needed.
  //
  // A copy is only allowed where it is ordered like the clone() that the lazy
  // clone stands for, so we raise an error rather than synchronizing when it
  // would not be. This has to happen before the refcount is decremented.
  //
  // Stealing the data means writing to it on the current stream, which must
  // be ordered after any copies of it that may still be pending on other
  // streams. Those are waited for (on the device) below.
  //
  // Copies enqueued under CUDA graph capture only run when the graph is
  // replayed, which the user orders with respect to other work, so they don't
  // need to be waited for (and a capture couldn't wait for eager work).
  const c10::Device device = data_ptr.device();
  const std::optional<c10::Stream> stream = current_stream(device);
  const bool capturing = stream.has_value() && stream->is_capturing();
  if (stream.has_value() && !ctx->is_unique()) {
    // A copy made during graph capture is replayed with the graph, so it is
    // only allowed if the lazy clone was also made during the capture. (We
    // don't distinguish between captures, though.)
    if (ref->made_while_capturing != capturing) {
      const char* when = capturing ? "before the capture" : "during a capture";
      if (ref->clone_stream.has_value()) {
        TORCH_CHECK(
            false,
            "Cannot write to this tensor (or take its data_ptr()) ",
            capturing ? "during" : "outside of",
            " graph capture: it is a lazy copy (from _lazy_clone()) that was "
            "made ",
            when,
            ", and writing to it requires copying it, which ",
            capturing ? "the graph would redo on every replay"
                      : "the replays of the graph would not redo",
            ". If you only read the tensor, use const_data_ptr() instead of "
            "data_ptr(). Otherwise, clone() it ",
            capturing ? "before" : "during",
            " the capture.");
      } else {
        TORCH_CHECK(
            false,
            "Cannot write to this tensor (or take its data_ptr()) ",
            capturing ? "during" : "outside of",
            " graph capture: a lazy copy of it (from _lazy_clone()) that was "
            "made ",
            when,
            " is alive, and writing to it requires copying it, which ",
            capturing ? "the graph would redo on every replay"
                      : "the replays of the graph would not redo",
            ". If you only read the tensor, use const_data_ptr() instead of "
            "data_ptr(). Otherwise, free the lazy copies of it before writing "
            "to it.");
      }
    }
    if (ref->clone_stream.has_value()) {
      // The copy must be enqueued on the stream the clone was made on, which
      // is also the stream the new allocation belongs to.
      TORCH_CHECK(
          *ref->clone_stream == *stream,
          "Cannot write to this tensor (or take its data_ptr()) on ",
          *stream,
          ": it is a lazy copy (from _lazy_clone()) that was made on ",
          *ref->clone_stream,
          ", and while the tensor it was copied from is alive, writing to it "
          "requires copying it, which can't be done safely on another stream "
          "without synchronizing the streams. If you only read the tensor, "
          "use const_data_ptr() instead of data_ptr(). Otherwise, write to it "
          "on the stream it was made on, or write to a clone() of it made on "
          "this stream instead.");
    } else if (storage->allocator() != nullptr) {
      // For the storage that was lazily cloned from, the copy must be
      // enqueued on the stream its memory was allocated on.
      const std::optional<bool> allocated_on_stream =
          storage->allocator()->was_allocated_on_stream(
              data_ptr.get(), *stream);
      if (capturing) {
        // Memory allocated during capture comes from a private pool, for
        // which the allocation stream is unknown. Memory with a known
        // allocation stream was allocated before the capture.
        TORCH_CHECK(
            !allocated_on_stream.has_value(),
            "Cannot write to this tensor (or take its data_ptr()) during "
            "graph capture: a lazy copy of it (from _lazy_clone()) is alive, "
            "so writing to it requires moving it to new memory inside of the "
            "graph, while code outside of the graph (e.g., writing the inputs "
            "of the graph) would keep using its old memory. If you only read "
            "the tensor, use const_data_ptr() instead of data_ptr(). "
            "Otherwise, free the lazy copies of it before writing to it.");
      } else {
        TORCH_CHECK(
            allocated_on_stream.value_or(true),
            "Cannot write to this tensor (or take its data_ptr()) on ",
            *stream,
            ": a lazy copy of it (from _lazy_clone()) is alive, so writing to "
            "it requires copying it, which can't be done safely on a stream "
            "other than the one its memory was allocated on without "
            "synchronizing the streams. If you only read the tensor, use "
            "const_data_ptr() instead of data_ptr(). Otherwise, write to it "
            "on the stream its memory was allocated on, or free the lazy "
            "copies of it first.");
      }
    }
  }

  auto result = ctx->decrement_refcount();

  // This must be set by each branch below.
  std::optional<DataPtr> new_data_ptr;

  if (std::holds_alternative<cow::COWDeleterContext::LastReference>(result)) {
    // This is the only reference to the data. If there were any racing writes,
    // the context ensured they finished before giving us the result.
    auto last_reference =
        std::get<cow::COWDeleterContext::LastReference>(std::move(result));
    if (stream.has_value() && !capturing) {
      for (const auto& [copy_stream, event] : last_reference.copy_events) {
        if (copy_stream != *stream) {
          event.block(*stream);
        }
      }
    }
    std::unique_ptr<void, DeleterFnPtr> data = std::move(last_reference.data);
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    new_data_ptr = DataPtr(
        data.release(), data_ptr.get(), data.get_deleter(), data_ptr.device());
  } else {
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<cow::COWDeleterContext::NotLastReference>(
            result));
    // We don't need to consume the result, it's just a shared lock ensuring
    // that the data will remain while we copy it.
    c10::OptionalDeviceGuard device_guard;
    if (stream.has_value()) {
      device_guard.reset_device(device);
    }
    new_data_ptr =
        storage->allocator()->clone(data_ptr.get(), storage->nbytes());
    if (stream.has_value() && !capturing) {
      ctx->record_copy_event(*stream);
    }
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());
  DataPtr old_data_ptr =
      storage->set_data_ptr_no_materialize(*std::move(new_data_ptr));
  // The refcount of the context was already decremented above. Release the
  // reference to the context so the refcount doesn't get decremented again,
  // but free the reference itself.
  delete static_cast<cow::COWReference*>(old_data_ptr.release_context());
}

} // namespace c10::impl::cow
