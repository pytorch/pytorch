#include <c10/core/RefcountedDeleter.h>

#include <mutex>

namespace c10 {

void refcounted_deleter(void* ctx_) {
  RefcountedDeleterContext& ctx =
      *reinterpret_cast<RefcountedDeleterContext*>(ctx_);
  ctx.refcount--;
  if (ctx.refcount == 0) {
    ctx.other_ctx = nullptr;
    delete &ctx;
  }
}

static std::mutex replace_data_ptr_mutex;

void maybeApplyRefcountedDeleter(const c10::Storage& storage) {
  std::lock_guard<std::mutex> guard(replace_data_ptr_mutex);
  c10::DataPtr& data_ptr = storage.mutable_data_ptr();

  if ((void*)data_ptr.get_deleter() == (void*)&c10::refcounted_deleter) {
    // Data pointer is already shared
    return;
  }

  void* data = data_ptr.get();
  void* other_ctx = data_ptr.get_context();
  c10::DeleterFnPtr other_deleter = data_ptr.get_deleter();
  c10::Device device = data_ptr.device();

  // Release the context of the original DataPtr so that the data doesn't
  // get deleted when the original DataPtr is replaced
  data_ptr.release_context();

  c10::RefcountedDeleterContext* refcount_ctx =
      new c10::RefcountedDeleterContext(other_ctx, other_deleter);

  c10::DataPtr new_data_ptr(
      data,
      reinterpret_cast<void*>(refcount_ctx),
      &c10::refcounted_deleter,
      device);
  storage.set_data_ptr(std::move(new_data_ptr));
}

c10::Storage newStorageImplFromRefcountedDataPtr(const c10::Storage& storage) {
  c10::maybeApplyRefcountedDeleter(storage);

  c10::StorageImpl* storage_impl = storage.unsafeGetStorageImpl();

  c10::DataPtr& data_ptr = storage.mutable_data_ptr();
  c10::DataPtr new_data_ptr(
      data_ptr.get(),
      data_ptr.get_context(),
      data_ptr.get_deleter(),
      data_ptr.device());

  // NOTE: This refcount increment should always happen immediately after
  // `new_data_ptr` is created. No other lines of code should be added between
  // them in the future, unless there's a very good reason for it, because if
  // any errors are raised and `new_data_ptr` is deleted before the refcount is
  // incremented, the refcount will get decremented and end up being one less
  // than it should be.
  reinterpret_cast<c10::RefcountedDeleterContext*>(data_ptr.get_context())
      ->refcount++;

  c10::Storage new_storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      storage_impl->nbytes(),
      std::move(new_data_ptr),
      storage_impl->allocator(),
      /*resizable=*/storage_impl->resizable());
  return new_storage;
}

} // namespace c10
