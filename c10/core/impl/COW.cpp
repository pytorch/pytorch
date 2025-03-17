#include <c10/core/impl/COW.h>

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
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
  return (void*)data_ptr.get_deleter() == (void*)&cow::cow_deleter;
}

static void check_clone_between_devices(
    DeviceType src_device_type,
    DeviceType dst_device_type) {
  TORCH_CHECK(
      src_device_type == dst_device_type || src_device_type == c10::kCPU ||
          dst_device_type == c10::kCPU,
      "Can only lazy clone or materialize between two different devices if they ",
      "both have the same device type or one of them is CPU. Got source '",
      src_device_type,
      "' and destionation '",
      dst_device_type,
      "'.");
}

c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage,
    c10::optional<c10::Device> device_opt,
    c10::optional<c10::Allocator*> allocator_opt) {
  TORCH_INTERNAL_ASSERT(device_opt.has_value() == allocator_opt.has_value());

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

  std::optional<DataPtr> new_data_ptr_opt; // must be set below

  if (has_simple_data_ptr(storage)) {
    // Case 1) We have a simple data pointer: wrap it.
    std::unique_ptr<void, DeleterFnPtr> original_ctx =
        storage._mutable_data_ptr_no_checks().move_context();

    // Save this for the result.
    new_data_ptr_opt = make_data_ptr(
        data_ptr,
        *new cow::COWDeleterContext(
            std::move(original_ctx), storage.device(), storage.allocator()));

    // Update this storage to the new copy on write context.
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr_opt));
  } else if (is_cow_data_ptr(data_ptr)) {
    // Case 2): there is already a copy on write context. Just return a
    // new storage impl.
    new_data_ptr_opt = copy_data_ptr(data_ptr);
  } else {
    // Case 3) There is a context and it's not copy-on-write. Nothing
    // we can do here.
    return nullptr;
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr_opt.has_value());

  c10::Allocator* allocator = storage.allocator();
  c10::DeviceType device_type = storage.device_type();

  if (device_opt.has_value()) {
    allocator = allocator_opt.value();

    DeviceGuard device_guard(device_opt.value());
    Device dst_device = device_guard.current_device();

    // If a different target device was given, then convert the data pointer to
    // that device.
    if (dst_device != storage.device()) {
      check_clone_between_devices(storage.device_type(), dst_device.type());

      DataPtr& new_data_ptr = new_data_ptr_opt.value();
      auto* ctx = new_data_ptr.cast_context<c10::impl::cow::COWDeleterContext>(
          c10::impl::cow::cow_deleter);
      device_type = dst_device.type();
      new_data_ptr.release_context();
      new_data_ptr_opt = c10::DataPtr(
          new_data_ptr.get(), ctx, c10::impl::cow::cow_deleter, dst_device);
    }
  }

  return make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      *std::move(new_data_ptr_opt),
      allocator,
      storage.resizable(),
      device_type);
}

static c10::DataPtr clone_between_devices(
    const void* data,
    std::size_t n,
    Device src_device,
    Allocator* src_allocator,
    Device dst_device,
    Allocator* dst_allocator) {
  DeviceType src_type = src_device.type();
  DeviceType dst_type = dst_device.type();
  check_clone_between_devices(src_type, dst_type);

  if (src_type == dst_type) {
    return dst_allocator->clone(data, n, /*sync=*/true);
  } else if (src_type == c10::kCPU) {
    return dst_allocator->clone_from_cpu(data, n);
  } else {
    return src_allocator->clone_to_cpu(data, n);
  }
}

C10_API void materialize_cow_storage(StorageImpl& storage) {
  TORCH_INTERNAL_ASSERT(
      !c10::ParallelGuard::is_enabled(),
      "Materializing a storage in the loop function of at::parallel_for is forbidden");
  const at::DataPtr& data_ptr = storage.data_ptr();

  auto* ctx = data_ptr.cast_context<cow::COWDeleterContext>(cow::cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  Device src_device = ctx->original_device();
  Device dst_device = storage.device();
  bool devices_match = src_device == dst_device;
  auto result = ctx->decrement_refcount();

  // This must be set by each branch below.
  std::optional<DataPtr> new_data_ptr;

  if (devices_match &&
      std::holds_alternative<cow::COWDeleterContext::LastReference>(result)) {
    // This is the only reference to the data. If there were any racing writes,
    // the context ensured they finished before giving us the result.
    std::unique_ptr<void, DeleterFnPtr> data =
        std::get<cow::COWDeleterContext::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    new_data_ptr = DataPtr(
        data.release(), data_ptr.get(), data.get_deleter(), data_ptr.device());
  } else {
    // We don't need to consume the result, it's just a shared lock ensuring
    // that the data will remain while we copy it.
    if (devices_match) {
      new_data_ptr =
          storage.allocator()->clone(data_ptr.get(), storage.nbytes());
    } else {
      new_data_ptr = clone_between_devices(
          data_ptr.get(),
          storage.nbytes(),
          src_device,
          ctx->original_allocator(),
          dst_device,
          storage.allocator());
    }
  }

  TORCH_INTERNAL_ASSERT(new_data_ptr.has_value());
  DataPtr old_data_ptr =
      storage.set_data_ptr_no_materialize_cow(*std::move(new_data_ptr));
  // The refcount of the context was already decremented above. Release the
  // reference to the context so the refcount doesn't get decremented again
  old_data_ptr.release_context();
}

} // namespace c10::impl::cow
