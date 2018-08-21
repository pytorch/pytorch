#pragma once

#include <ATen/StorageImpl.h>

namespace at {

struct AT_API Storage {
public:
  Storage() = delete;
  Storage(StorageImpl* storage_impl) : storage_impl_(storage_impl) {}
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
  ~Storage();
  // There are reasonable interpretations of these constructors, but they're to
  // be implemented on demand.
  Storage(Storage&) = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage(const Storage&&) = delete;
  void set_pImpl(StorageImpl* storage_impl) {
    storage_impl_ = storage_impl;
  }
  StorageImpl* pImpl() {
    return storage_impl_;
  }
  StorageImpl* pImpl() const {
    return storage_impl_;
  }
  StorageImpl* retained_pImpl() const {
    storage_impl_->retain();
    return storage_impl_;
  }

 protected:
  StorageImpl* storage_impl_;
};

} // namespace at
