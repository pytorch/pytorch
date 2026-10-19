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

at::DataPtr copy_data_ptr(const at::DataPtr& data_ptr) {
  auto* ref = data_ptr.cast_context<COWReference>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ref != nullptr);
  auto copy = std::make_unique<COWReference>();
  ref->ctx->increment_refcount();
  copy->ctx = ref->ctx;
  return at::DataPtr(
      data_ptr.get(), copy.release(), cow_deleter, data_ptr.device());
}

c10::intrusive_ptr<StorageImpl> eager_clone(StorageImpl& storage) {
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
  const auto& data_ptr = storage.data_ptr_;
  const auto* allocator = storage.allocator();
  return allocator != nullptr ? allocator->is_simple_data_ptr(data_ptr)
                              : data_ptr.get_context() == data_ptr.get();
}

bool is_cow_data_ptr(const c10::DataPtr& data_ptr) {
  return data_ptr.get_deleter() == &cow_deleter;
}

void check_cow_read(const StorageImpl* storage) {
  const auto& data_ptr = storage->data_ptr_;
  c10::DeviceGuard guard(data_ptr.device());
  auto* ref = data_ptr.cast_context<COWReference>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ref != nullptr);
  ref->completed = false;
  const auto* impl = getDeviceGuardImpl(data_ptr.device().type());
  const auto capture_id =
      impl->getStreamCaptureId(impl->getStream(data_ptr.device()));
  TORCH_CHECK(
      capture_id == 0 || capture_id == ref->ctx->capture_id(),
      "Materialize COW graph inputs before CUDA graph capture; an existing "
      "lazy snapshot cannot change its address inside another capture");
}

void record_done(StorageImpl& storage) {
  if (!storage.device().is_cuda()) {
    return;
  }
  if (auto* ref = storage.data_ptr_.cast_context<COWReference>(cow_deleter)) {
    ref->record_done();
  }
  if (storage.extra_meta_) {
    auto& retired = storage.extra_meta_->cow_retired_data;
    if (retired) {
      retired->cast_context<COWReference>(cow_deleter)->record_done();
      retired.reset();
    }
  }
}

c10::intrusive_ptr<StorageImpl> lazy_clone_storage(StorageImpl& storage) {
  if (storage.throw_on_immutable_data_ptr()) {
    storage.data_ptr();
  }
  const auto& data_ptr = storage.data_ptr_;
  const bool simple = has_simple_data_ptr(storage);
  if (!simple && !is_cow_data_ptr(data_ptr)) {
    return nullptr;
  }

  std::optional<Stream> stream;
  uint64_t capture_id = 0;
  c10::OptionalDeviceGuard device_guard;
  if (data_ptr.device().is_cuda()) {
    TORCH_CHECK(
        storage.allocator() != nullptr,
        "Cannot copy CUDA COW storage without an allocator");
    device_guard.reset_device(data_ptr.device());
    // Do not accumulate an allocation history on a long-lived storage.
    if (storage.extra_meta_ && storage.extra_meta_->cow_retired_data) {
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
      const bool capture_local =
          capture_id != 0 && allocation->capture_id == capture_id;
      if (allocation->private_pool && !capture_local) {
        return eager_clone(storage);
      }
      TORCH_CHECK(
          !capture_id || capture_local,
          "Cannot lazily clone a CUDA graph input during capture; omit "
          "_lazy_clone() for read-only use or use clone() for independent storage");
      const bool same_stream =
          impl->getStreamNativeHandle(allocation->stream) ==
          impl->getStreamNativeHandle(*stream);
      if (!same_stream) {
        return eager_clone(storage);
      }
    } else {
      const auto* ctx = data_ptr.cast_context<COWReference>(cow_deleter)->ctx;
      if (ctx->stream() != stream || ctx->capture_id() != capture_id) {
        return eager_clone(storage);
      }
    }
  }

  if (simple) {
    auto ref = std::make_unique<COWReference>();
    ref->ctx = new COWDeleterContext(
        storage._mutable_data_ptr_no_checks().move_context(),
        stream,
        capture_id);
    storage.set_data_ptr_noswap(at::DataPtr(
        data_ptr.get(), ref.release(), cow_deleter, data_ptr.device()));
    storage.set_materializer(&materialize_cow);
  }
  TORCH_INTERNAL_ASSERT(storage.has_materializer());
  auto result = make_storage_impl(
      StorageImpl::use_byte_size_t(),
      storage.sym_nbytes(),
      copy_data_ptr(data_ptr),
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
  const auto& data_ptr = storage->data_ptr();
  auto* ref = data_ptr.cast_context<COWReference>(cow_deleter);
  TORCH_INTERNAL_ASSERT(ref != nullptr);
  auto* ctx = ref->ctx;

  std::optional<DataPtr> copy;
  if (ctx->stream()) {
    c10::DeviceGuard device_guard(data_ptr.device());
    const auto* impl = getDeviceGuardImpl(data_ptr.device().type());
    const auto stream = impl->getStream(data_ptr.device());
    const auto capture_id = impl->getStreamCaptureId(stream);
    if (ctx->is_unique()) {
      ctx->wait_for_completion(stream);
    } else {
      TORCH_CHECK(
          capture_id == ctx->capture_id() &&
              impl->getStreamCaptureId(*ctx->stream()) == ctx->capture_id(),
          "Cannot materialize shared COW storage across CUDA graph capture "
          "boundaries; resolve the lazy clone before capture ends");
      auto& retired = storage->get_extra_meta().cow_retired_data;
      TORCH_INTERNAL_ASSERT(!retired);
      {
        c10::StreamGuard allocation_guard(*ctx->stream());
        copy = storage->allocator()->clone(data_ptr.get(), storage->nbytes());
      }
      impl->waitStream(stream, *ctx->stream());
      retired = storage->set_data_ptr_no_materialize(std::move(*copy));
      return;
    }
  } else if (!ctx->is_unique()) {
    // Keep the reference through allocation and copying, including failures.
    copy = storage->allocator()->clone(data_ptr.get(), storage->nbytes());
  }

  auto data = ref->release();
  if (!copy) {
    TORCH_INTERNAL_ASSERT(data.get() == data_ptr.get());
    auto* ptr = data.release();
    copy = DataPtr(ptr, ptr, data.get_deleter(), data_ptr.device());
  }
  auto old_data_ptr = storage->set_data_ptr_no_materialize(std::move(*copy));
}

} // namespace c10::impl::cow
