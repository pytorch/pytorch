#ifndef CAFFE2_CORE_STORAGE_H_
#define CAFFE2_CORE_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "caffe2/core/allocator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"

#include <ATen/core/Allocator.h>
#include <ATen/core/Device.h>
#include <ATen/core/DeviceType.h>
#include <ATen/core/intrusive_ptr.h>
#include <ATen/core/StorageImpl.h>

namespace caffe2 {

using StorageImpl = at::StorageImpl;

class CAFFE2_API Storage {
 public:
  Storage() {}
  Storage(at::DeviceType device_type)
      : storage_impl_(c10::make_intrusive<StorageImpl>(device_type)) {}
  Storage(at::DeviceType device_type, TypeMeta data_type)
      : storage_impl_(
            c10::make_intrusive<StorageImpl>(device_type, data_type)) {}

  Storage(
      TypeMeta data_type,
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

  void reset() {
    storage_impl_->reset();
  }

  // For debugging purpose only, please don't call it
  StorageImpl* unsafeGetStorageImp() const {
    return storage_impl_.get();
  }

  template <typename T>
  inline bool IsType() const {
    return storage_impl_->IsType<T>();
  }

  void* data() const {
    return storage_impl_->data();
  }

  void* data() {
    return storage_impl_->data();
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

  void set_dtype(const TypeMeta& data_type) {
    storage_impl_->set_dtype(data_type);
  }

  const TypeMeta& dtype() const {
    return storage_impl_->dtype();
  }

  size_t capacity() const {
    return storage_impl_->capacity();
  }

  int64_t numel() const {
    return storage_impl_->numel();
  }

  // TODO: remove later
  void set_numel(int64_t numel) {
    storage_impl_->set_numel(numel);
  }

  at::DeviceType device_type() const {
    return storage_impl_->device_type();
  }

  const at::Allocator* allocator() const {
    return storage_impl_->allocator();
  }

  inline size_t itemsize() const {
    return storage_impl_->itemsize();
  }

  inline long use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity,
      MemoryDeleter d = nullptr) {
    CAFFE_ENFORCE_WITH_CALLER(
        storage_impl_.unique(),
        "UniqueStorageShareExternalPointer can only be called when \
        use_count == 1");
    storage_impl_->UniqueStorageShareExternalPointer(
        src, data_type, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    CAFFE_ENFORCE_WITH_CALLER(
        storage_impl_.unique(),
        "UniqueStorageShareExternalPointer can only be called when \
        use_count == 1");
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), data_type, capacity);
  }

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};

} // namespace caffe2

#endif // CAFFE2_CORE_STORAGE_H_
