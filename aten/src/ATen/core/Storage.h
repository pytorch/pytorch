#pragma once

#include <ATen/core/StorageImpl.h>

namespace at {

struct AT_API Storage {
public:
  Storage() {}
  Storage(StorageImpl* storage_impl) : storage_impl_(c10::intrusive_ptr<StorageImpl>::reclaim(storage_impl)) {}
  Storage(const c10::intrusive_ptr<StorageImpl>& ptr) : storage_impl_(ptr) {}
  Storage(c10::intrusive_ptr<StorageImpl>&& ptr) : storage_impl_(std::move(ptr)) {}
  Storage(
      at::ScalarType,
      size_t size,
      Allocator* allocator,
      bool resizable = false);
  Storage(
      at::ScalarType,
      at::DataPtr,
      size_t size,
      const std::function<void(void*)>& deleter,
      bool resizable = false);

  template <typename T>
  T* data() const { return storage_impl_->data<T>(); }

  template <typename T>
  T* unsafe_data() const { return storage_impl_->unsafe_data<T>(); }

  size_t elementSize() const { return storage_impl_->itemsize(); }
  ptrdiff_t size() const { return storage_impl_->numel(); }
  bool resizable() const { return storage_impl_->resizable(); }
  // get() use here is to get const-correctness
  void* data() const { return storage_impl_.get()->data(); }
  const at::DataType dtype() const {
    return storage_impl_->dtype();
  }
  const at::DataPtr& data_ptr() const { return storage_impl_->data_ptr(); }
  DeviceType device_type() const { return storage_impl_->device_type(); }
  at::Allocator* allocator() const { return storage_impl_.get()->allocator(); }
  at::Device device() const { return storage_impl_->device(); }

  StorageImpl* unsafeReleaseStorageImpl() {
    return storage_impl_.release();
  }
  StorageImpl* unsafeGetStorageImpl() const noexcept {
    return storage_impl_.get();
  }
  operator bool() const {
    return storage_impl_;
  }
  size_t use_count() const {
    return storage_impl_.use_count();
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

} // namespace at
