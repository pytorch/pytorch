#include <ATen/Storage.h>
#include <ATen/Context.h>
#include <iostream>

namespace at {

Storage::Storage(at::ScalarType scalar_type, size_t size, Allocator* allocator)
    : storage_impl_(new StorageImpl(
          scalar_type,
          size,
          allocator,
          /* resizable */ false)) {}

Storage::Storage(
    at::ScalarType scalar_type,
    at::DataPtr data_ptr,
    size_t size,
    const std::function<void(void*)>& deleter)
    : storage_impl_(new StorageImpl(
          scalar_type,
          size,
          std::move(data_ptr),
          /* allocator */ nullptr,
          /* resizable */ false)) {}

Storage::~Storage() {
  if (!storage_impl_) {
    return;
  }
  if (--storage_impl_->refcount == 0) {
    if (storage_impl_->finalizer) {
      (*storage_impl_->finalizer)();
    }
    storage_impl_->finalizer = nullptr;
    storage_impl_->data_ptr.clear();
    if (storage_impl_ && --storage_impl_->weakcount == 0) {
      delete storage_impl_;
    }
  }
}

} // namespace at
