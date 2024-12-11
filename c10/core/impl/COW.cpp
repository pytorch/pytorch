#include <c10/core/impl/COW.h>

#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/alignment.h>
#include <c10/core/impl/COWDeleter.h>
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
    cow::COWDeleterContext& ctx,
    DeleterFnPtr deleter_fn) {
  return at::DataPtr(data_ptr.get(), &ctx, deleter_fn, data_ptr.device());
}

/// Copies a copy-on-write DataPtr.
at::DataPtr copy_data_ptr(
    at::DataPtr const& data_ptr,
    DeleterFnPtr deleter_fn) {
  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(deleter_fn);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->increment_refcount();
  return make_data_ptr(data_ptr, *ctx, deleter_fn);
}

} // namespace

bool has_simple_data_ptr(const c10::StorageImpl& storage) {
  const c10::DataPtr& data_ptr = storage.data_ptr();
  const void* ctx = data_ptr.get_context();
  const void* data = data_ptr.get();
  const c10::Allocator* allocator = storage.allocator();
  if (ctx == data) {
    return true;
  } else if (allocator != nullptr) {
    // TODO: Sometimes even when this Allocator* is non-null,
    // it may still point to invalid memory, causing a segfault
    // here. I need to find out why that happens before this is
    // merged.
    return allocator->is_simple_data_ptr(data_ptr);
  } else {
    return false;
  }
}

bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return (void*)data_ptr.get_deleter() == (void*)&cow::cow_deleter;
}

bool is_cowsim_data_ptr(const c10::DataPtr& data_ptr) {
  return data_ptr.get_deleter() == &cow::cowsim_deleter;
}

c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage,
    bool future) {
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

  DeleterFnPtr deleter_fn = future ? cow_deleter : cowsim_deleter;

  if (has_simple_data_ptr(storage)) {
    // Case 1) We have a simple data pointer: wrap it.
    if (!future && get_extra_conditional_view_warnings()) {
      alert_cowsim(
          "This operation creates a conditional view. This behavior is ",
          "deprecated, and in the future it will unconditionally create a ",
          "lazy clone (semantic copy) instead.");
    }
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage._mutable_data_ptr_no_checks().move_context();

    COWDeleterContext* ctx = future
        ? new COWDeleterContext(std::move(original_ctx))
        : new COWSimDeleterContext(std::move(original_ctx));

    // Save this for the result.
    new_data_ptr = make_data_ptr(data_ptr, *ctx, deleter_fn);

    // Update this storage to the new copy on write context.
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr, deleter_fn));
  } else if (is_cow_data_ptr(data_ptr)) {
    // Case 2): there is already a copy on write context. Just return a
    // new storage impl.
    TORCH_CHECK(
        future,
        "You cannot create a simulated lazy clone of a genuine COW storage");
    new_data_ptr = copy_data_ptr(data_ptr, deleter_fn);
  } else if (is_cowsim_data_ptr(data_ptr)) {
    TORCH_CHECK(
        !future,
        "You cannot create a genuine lazy clone of a simulated COW storage");
    new_data_ptr = copy_data_ptr(data_ptr, deleter_fn);
  } else {
    // Case 3) There is a context and it's not copy-on-write. Nothing
    // we can do here.
    return nullptr;
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());

  return make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      *std::move(new_data_ptr),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
}

C10_API void materialize_cow_storage(StorageImpl& storage) {
  TORCH_INTERNAL_ASSERT(
      !c10::ParallelGuard::is_enabled(),
      "Materializing a storage in the loop function of at::parallel_for is forbidden");
  const at::DataPtr& data_ptr = storage.data_ptr();

  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);

  auto result = ctx->decrement_refcount();

  // This must be set by each branch below.
  std::optional<DataPtr> new_data_ptr;

  if (std::holds_alternative<cow::COWDeleterContext::LastReference>(result)) {
    // This is the only reference to the data. If there were any racing writes,
    // the context ensured they finished before giving us the result.
    std::unique_ptr<void, DeleterFnPtr> data =
        std::get<cow::COWDeleterContext::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    new_data_ptr = DataPtr(
        data.release(), data_ptr.get(), data.get_deleter(), data_ptr.device());
  } else {
    TORCH_INTERNAL_ASSERT(
        std::holds_alternative<cow::COWDeleterContext::NotLastReference>(
            result));
    // We don't need to consume the result, it's just a shared lock ensuring
    // that the data will remain while we copy it.
    new_data_ptr = storage.allocator()->clone(data_ptr.get(), storage.nbytes());
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());
  DataPtr old_data_ptr =
      storage.set_data_ptr_no_materialize_cow(*std::move(new_data_ptr));
  // The refcount of the context was already decremented above. Release the
  // reference to the context so the refcount doesn't get decremented again
  old_data_ptr.release_context();
}

C10_API void check_cowsim_write(StorageImpl& storage) {
  const at::DataPtr& data_ptr = storage._data_ptr_no_checks();
  auto* ctx =
      data_ptr.cast_context<cow::COWSimDeleterContext>(cow::cowsim_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->check_write(reinterpret_cast<std::uintptr_t>(&storage));
}

C10_API void check_cowsim_read(const StorageImpl& storage) {
  const at::DataPtr& data_ptr = storage._data_ptr_no_checks();
  auto* ctx =
      data_ptr.cast_context<cow::COWSimDeleterContext>(cow::cowsim_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->check_read(reinterpret_cast<std::uintptr_t>(&storage));
}

static bool _future_lazy_clone = false;

C10_API void set_future_lazy_clone(bool mode) {
  _future_lazy_clone = mode;
}

C10_API bool get_future_lazy_clone() {
  return _future_lazy_clone;
}

} // namespace c10::impl::cow
