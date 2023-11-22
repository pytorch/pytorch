#include <c10/core/impl/cow/COW.h>

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/alignment.h>
#include <c10/core/impl/cow/COWDeleter.h>
#include <c10/util/Exception.h>
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

bool is_simple_context(
    const void* context,
    const void* data,
    const at::Allocator* allocator) {
  if (allocator == c10::GetDefaultMobileCPUAllocator()) {
    return reinterpret_cast<size_t>(data) ==
        reinterpret_cast<size_t>(context) + c10::gAlignment;
  } else {
    return data == context;
  }
}

} // namespace

bool has_simple_data_ptr(const c10::StorageImpl& storage) {
  const c10::DataPtr& data_ptr = storage.data_ptr();
  return is_simple_context(
      data_ptr.get_context(), data_ptr.get(), storage.allocator());
}

bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return (void*)data_ptr.get_deleter() == (void*)&cow::cow_deleter;
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

  if (has_simple_data_ptr(storage)) {
    // Case 1) We have a simple data pointer: wrap it.
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage.mutable_data_ptr().move_context();
    TORCH_INTERNAL_ASSERT(is_simple_context(
        original_ctx.get(), data_ptr.get(), storage.allocator()));

    // Save this for the result.
    new_data_ptr = make_data_ptr(
        data_ptr, *new cow::COWDeleterContext(std::move(original_ctx)));

    // Update this storage to the new copy on write context.
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr));
  } else if (is_cow_data_ptr(data_ptr)) {
    // Case 2): there is already a copy on write context. Just return a
    // new storage impl.
    new_data_ptr = copy_data_ptr(data_ptr);
  } else {
    // Case 3) There is a context and it's not copy-on-write. Nothing
    // we can do here.
    return nullptr;
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());

  return make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      *std::move(new_data_ptr),
      storage.allocator(),
      storage.resizable());
}

} // namespace c10::impl::cow
