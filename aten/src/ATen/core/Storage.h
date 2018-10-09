#pragma once

#include <ATen/core/StorageImpl.h>

namespace at {

struct CAFFE2_API Storage {
 public:
  Storage() {}
  Storage(c10::intrusive_ptr<StorageImpl> ptr) : storage_impl_(std::move(ptr)) {}
  Storage(
      caffe2::TypeMeta data_type,
      size_t size,
      Allocator* allocator,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            size,
            allocator,
            resizable)) {}

  Storage(
      caffe2::TypeMeta data_type,
      at::DataPtr data_ptr,
      size_t size,
      const std::function<void(void*)>& deleter,
      bool resizable = false)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            size,
            std::move(data_ptr),
            /* allocator */ nullptr,
            resizable)) {}

  Storage(at::DeviceType device_type)
      : storage_impl_(c10::make_intrusive<StorageImpl>(device_type)) {}
  Storage(at::DeviceType device_type, caffe2::TypeMeta data_type)
      : storage_impl_(
            c10::make_intrusive<StorageImpl>(device_type, data_type)) {}

  Storage(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            data_type,
            numel,
            std::move(data_ptr),
            allocator,
            resizable)) {}

  template <typename T>
  inline bool IsType() const {
    return storage_impl_->IsType<T>();
  }

  template <typename T>
  T* data() const { return storage_impl_->data<T>(); }

  template <typename T>
  T* unsafe_data() const { return storage_impl_->unsafe_data<T>(); }

  size_t elementSize() const {
    return storage_impl_->itemsize();
  }

  inline size_t itemsize() const {
    return storage_impl_->itemsize();
  }

  ptrdiff_t size() const {
    return storage_impl_->numel();
  }

  int64_t numel() const {
    return storage_impl_->numel();
  }

  // TODO: remove later
  void set_numel(int64_t numel) {
    storage_impl_->set_numel(numel);
  }

  bool resizable() const {
    return storage_impl_->resizable();
  }

  size_t capacity() const {
    return storage_impl_->capacity();
  }
  // get() use here is to get const-correctness

  void* data() {
    return storage_impl_->data();
  }

  void* data() const {
    return storage_impl_.get()->data();
  }

  const caffe2::TypeMeta& dtype() const {
    return storage_impl_->dtype();
  }

  at::DataPtr& data_ptr() {
    return storage_impl_->data_ptr();
  }

  const at::DataPtr& data_ptr() const {
    return storage_impl_->data_ptr();
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    return storage_impl_->set_data_ptr(std::move(data_ptr));
  };

  void set_dtype(const caffe2::TypeMeta& data_type) {
    storage_impl_->set_dtype(data_type);
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

  operator bool() const {
    return storage_impl_;
  }

  size_t use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      const caffe2::TypeMeta& data_type,
      size_t capacity,
      DeleterFnPtr d = nullptr) {
    if (!storage_impl_.unique()) {
      AT_ERROR(
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(
        src, data_type, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity) {
    if (!storage_impl_.unique()) {
      AT_ERROR(
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), data_type, capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

} // namespace at
