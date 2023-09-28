#pragma once

#include <c10/core/StorageImpl.h>

namespace c10 {

struct C10_API Storage {
 public:
  struct use_byte_size_t {};

  Storage() = default;
  Storage(c10::intrusive_ptr<StorageImpl> ptr)
      : storage_impl_(std::move(ptr)) {}

  // Allocates memory buffer using given allocator and creates a storage with it
  Storage(
      use_byte_size_t /*use_byte_size*/,
      SymInt size_bytes,
      Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            std::move(size_bytes),
            allocator,
            resizable)) {}

  // Creates storage with pre-allocated memory buffer. Allocator is given for
  // potential future reallocations, however it can be nullptr if the storage
  // is non-resizable
  Storage(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator = nullptr,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            std::move(data_ptr),
            allocator,
            resizable)) {}

  // Legacy constructor for partially initialized (dtype or memory) storages
  // that can be temporarily created with Caffe2 APIs. See the note on top of
  // TensorImpl.h for details.
  static Storage create_legacy(at::Device device) {
    auto allocator = GetAllocator(device.type());
    return Storage(c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        0,
        allocator->allocate(0), // materialize a non-default Device.
        allocator,
        true));
  }

  // Mimic create_legacy, but without requiring a newly-created StorageImpl.
  void reset_legacy() {
    TORCH_CHECK(resizable() && allocator());
    set_nbytes(0);
    set_data_ptr_noswap(allocator()->allocate(0));
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) const {
    storage_impl_.get()->set_nbytes(size_bytes);
  }

  void set_nbytes(c10::SymInt size_bytes) const {
    storage_impl_.get()->set_nbytes(std::move(size_bytes));
  }

  bool resizable() const {
    return storage_impl_->resizable();
  }

  size_t nbytes() const {
    return storage_impl_->nbytes();
  }

  SymInt sym_nbytes() const {
    return storage_impl_->sym_nbytes();
  }
  // get() use here is to get const-correctness

  const void* data() const {
    return storage_impl_->data();
  }

  void* mutable_data() const {
    return storage_impl_->mutable_data();
  }

  at::DataPtr& mutable_data_ptr() const {
    return storage_impl_->mutable_data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const {
    return storage_impl_.get()->set_data_ptr(std::move(data_ptr));
  }

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) const {
    return storage_impl_.get()->set_data_ptr_noswap(std::move(data_ptr));
  }

  DeviceType device_type() const {
    return storage_impl_->device_type();
  }

  at::Allocator* allocator() const {
    return storage_impl_.get()->allocator();
  }

  at::Device device() const {
    return storage_impl_->device();
  }

  StorageImpl* unsafeReleaseStorageImpl() {
    return storage_impl_.release();
  }

  StorageImpl* unsafeGetStorageImpl() const noexcept {
    return storage_impl_.get();
  }

  c10::weak_intrusive_ptr<StorageImpl> getWeakStorageImpl() const {
    return c10::weak_intrusive_ptr<StorageImpl>(storage_impl_);
  }

  operator bool() const {
    return storage_impl_;
  }

  size_t use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  bool is_alias_of(const Storage& other) const {
    return storage_impl_ == other.storage_impl_;
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      size_t capacity,
      DeleterFnPtr d = nullptr) {
    if (!storage_impl_.unique()) {
      TORCH_CHECK(
          false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t capacity) {
    if (!storage_impl_.unique()) {
      TORCH_CHECK(
          false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

} // namespace c10
