#pragma once

#include <ATen/StorageImpl.h>

namespace at {

struct Storage {
public:
  Storage() = delete;
  Storage(StorageImpl* storage_impl) : storage_impl_(storage_impl) {}
  ~Storage();
  // There are reasonable interpretations of these constructors, but they're to
  // be implemented on demand.
  Storage(Storage&) = delete;
  Storage(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage(const Storage&&) = delete;
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
