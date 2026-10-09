#include <c10/core/impl/COW.h>

#include <c10/core/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/COWDeleter.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <c10/util/ParallelGuard.h>
#include <c10/util/UniqueVoidPtr.h>

#include <memory>
#include <optional>

namespace c10::impl::cow {

namespace {

at::DataPtr make_data_ptr(const at::DataPtr& data_ptr, COWDeleterContext& ctx) {
  return at::DataPtr(data_ptr.get(), &ctx, cow_deleter, data_ptr.device());
}

at::DataPtr copy_data_ptr(const at::DataPtr& data_ptr) {
  auto* ctx = data_ptr.cast_context<COWDeleterContext>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  ctx->increment_refcount();
  return make_data_ptr(data_ptr, *ctx);
}

c10::intrusive_ptr<StorageImpl> eager_clone(StorageImpl& storage) {
  c10::DeviceGuard guard(storage.device());
  return make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      storage.allocator()->clone(storage.data_ptr().get(), storage.nbytes()),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
}

} // namespace

bool has_simple_data_ptr(const c10::StorageImpl& storage) {
  const auto& data_ptr = storage.data_ptr();
  const auto* allocator = storage.allocator();
  return allocator != nullptr ? allocator->is_simple_data_ptr(data_ptr)
                              : data_ptr.get_context() == data_ptr.get();
}

bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return data_ptr.get_deleter() == &cow_deleter;
}

void check_cow_read(const StorageImpl* storage) {
  const auto& data_ptr = storage->data_ptr_;
  if (!data_ptr.device().is_cuda()) {
    return;
  }
  c10::DeviceGuard guard(data_ptr.device());
  auto* ctx = data_ptr.cast_context<COWDeleterContext>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);
  const auto* impl = getDeviceGuardImpl(data_ptr.device().type());
  const auto capture_id =
      impl->getStreamCaptureId(impl->getStream(data_ptr.device()));
  TORCH_CHECK(
      capture_id == 0 || capture_id == ctx->capture_id(),
      "Materialize COW graph inputs before CUDA graph capture; an existing "
      "lazy snapshot cannot change its address inside another capture");
}

c10::intrusive_ptr<StorageImpl> lazy_clone_storage(StorageImpl& storage) {
  const auto& data_ptr = storage.data_ptr();
  const bool simple = has_simple_data_ptr(storage);
  if (!simple && !is_cow_data_ptr(data_ptr)) {
    return nullptr;
  }

  std::optional<Stream> stream;
  uint64_t capture_id = 0;
  bool capture_local = false;
  c10::OptionalDeviceGuard device_guard;
  if (data_ptr.device().is_cuda()) {
    TORCH_CHECK(
        storage.allocator() != nullptr,
        "Cannot copy CUDA COW storage without an allocator");
    device_guard.reset_device(data_ptr.device());
    // Do not accumulate an allocation history on a long-lived storage.
    if (storage.extra_meta_ && !storage.extra_meta_->cow_retired_data.empty()) {
      return eager_clone(storage);
    }
    const auto* impl = getDeviceGuardImpl(data_ptr.device().type());
    stream = impl->getStream(data_ptr.device());
    capture_id = impl->getStreamCaptureId(*stream);
    if (simple) {
      const auto allocation =
          storage.allocator()->allocation_stream_info(data_ptr.get());
      if (!allocation) {
        return eager_clone(storage);
      }
      capture_local = capture_id != 0 && allocation->capture_id == capture_id;
      const bool same_stream =
          impl->getStreamNativeHandle(allocation->stream) ==
          impl->getStreamNativeHandle(*stream);
      if ((allocation->private_pool && !capture_local) ||
          ((!capture_id || capture_local) && !same_stream) ||
          (storage.extra_meta_ && storage.extra_meta_->cow_capture_id &&
           storage.extra_meta_->cow_capture_id != capture_id)) {
        return eager_clone(storage);
      }
    } else {
      const auto* ctx = data_ptr.cast_context<COWDeleterContext>(cow_deleter);
      if (ctx->stream() != stream || ctx->capture_id() != capture_id) {
        return eager_clone(storage);
      }
    }
  }

  std::optional<DataPtr> new_data_ptr;
  if (simple) {
    if (capture_id) {
      storage.get_extra_meta().cow_capture_id = capture_id;
    }
    auto* ctx = new COWDeleterContext(
        storage._mutable_data_ptr_no_checks().move_context(),
        stream,
        capture_id,
        capture_local);
    new_data_ptr = make_data_ptr(data_ptr, *ctx);
    storage.set_data_ptr_noswap(copy_data_ptr(*new_data_ptr));
    storage.set_materializer(&materialize_cow);
  } else {
    TORCH_INTERNAL_ASSERT(storage.has_materializer());
    new_data_ptr = copy_data_ptr(data_ptr);
  }

  auto result = make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      std::move(*new_data_ptr),
      storage.allocator(),
      storage.resizable(),
      storage.device_type());
  if (capture_id) {
    result->get_extra_meta().cow_capture_id = capture_id;
  }
  result->set_materializer(&materialize_cow);
  return result;
}

void materialize_cow(StorageImpl* storage) {
  TORCH_INTERNAL_ASSERT(
      !c10::ParallelGuard::is_enabled(),
      "Materializing a storage in the loop function of at::parallel_for is forbidden");
  const auto& data_ptr = storage->data_ptr();
  auto* ctx = data_ptr.cast_context<COWDeleterContext>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ctx != nullptr);

  std::optional<DataPtr> copy;
  if (ctx->stream()) {
    c10::DeviceGuard device_guard(data_ptr.device());
    const auto* impl = getDeviceGuardImpl(data_ptr.device().type());
    const auto stream = impl->getStream(data_ptr.device());
    const auto capture_id = impl->getStreamCaptureId(stream);
    if (ctx->is_unique()) {
      impl->waitStream(stream, *ctx->stream());
    } else {
      TORCH_CHECK(
          capture_id == ctx->capture_id(),
          "Cannot materialize shared COW storage across CUDA graph capture "
          "boundaries; resolve the lazy clone before capture ends");
      TORCH_CHECK(
          !capture_id || ctx->capture_local(),
          "Cannot relocate a COW graph input during capture; use clone() for "
          "graph inputs which need independent writable storage");

      auto& retired = storage->get_extra_meta().cow_retired_data;
      retired.reserve(retired.size() + 1);
      {
        c10::StreamGuard allocation_guard(*ctx->stream());
        copy = storage->allocator()->allocate(storage->nbytes());
      }
      // Preserve the allocation stream while copying on the operation's
      // current stream. Both directions are graph edges during capture.
      impl->waitStream(stream, *ctx->stream());
      storage->allocator()->copy_data(
          copy->get(), data_ptr.get(), storage->nbytes());
      impl->waitStream(*ctx->stream(), stream);
      retired.push_back(storage->set_data_ptr_no_materialize(std::move(*copy)));
      return;
    }
  } else if (!ctx->is_unique()) {
    // Keep the reference through allocation and copying, including failures.
    copy = storage->allocator()->clone(data_ptr.get(), storage->nbytes());
  }

  auto result = ctx->decrement_refcount();
  if (!copy) {
    auto data = std::get<COWDeleterContext::LastReference>(std::move(result));
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    auto* ptr = data.release();
    copy = DataPtr(ptr, ptr, data.get_deleter(), data_ptr.device());
  }
  auto old_data_ptr = storage->set_data_ptr_no_materialize(std::move(*copy));
  old_data_ptr.release_context();
}

} // namespace c10::impl::cow
