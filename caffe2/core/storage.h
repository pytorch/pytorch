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

namespace caffe2 {

using DataType = TypeMeta;

class StorageImpl;
using Storage = std::shared_ptr<StorageImpl>;

class StorageImpl {
 public:
  StorageImpl() = delete;
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl& operator=(const StorageImpl&) = delete;

  explicit StorageImpl(DeviceType device_type) : device_type_(device_type) {}
  StorageImpl(DeviceType device_type, TypeMeta data_type)
      : data_type_(data_type), device_type_(device_type) {}

  void reset() {
    data_ptr_.reset();
    capacity_ = 0;
  }

  template <typename T>
  inline bool IsType() const {
    return data_type_.Match<T>();
  }

  const void* data_ptr() const {
    return data_ptr_.get();
  }

  void* data_ptr() {
    return data_ptr_.get();
  }

  const DataType& dtype() const {
    return data_type_;
  }

  size_t capacity() const {
    return capacity_;
  }

  int64_t numel() const {
    return capacity_ / itemsize();
  }

  inline void set_device_type(DeviceType device_type) {
    device_type_ = device_type;
  }

  inline DeviceType device_type() const {
    return device_type_;
  }

  inline size_t itemsize() const {
    return data_type_.itemsize();
  }

  // Rule of Five
  StorageImpl(StorageImpl&&) = default;
  ~StorageImpl() = default;
  StorageImpl& operator=(StorageImpl&&) = default;

 protected:
  template <typename Deleter = MemoryDeleter>
  void ShareExternalPointer(
      void* src,
      const DataType& data_type,
      size_t capacity = 0,
      Deleter d = nullptr) {
    // Check if the deleter is a MemoryDeleter and is a simple nullptr.
    if (std::is_same<MemoryDeleter, Deleter>::value &&
        reinterpret_cast<MemoryDeleter*>(&d)[0] == nullptr) {
      // Use aliasing constructor trick to avoid calling the destructor.
      data_ptr_ = std::shared_ptr<void>(std::shared_ptr<void>(), src);
    } else {
      data_ptr_.reset(src, d);
    }
    // Sets capacity. If not specified, we will implicitly assume that
    // the capacity is the current size.
    if (capacity) {
      capacity_ = capacity;
    }
  }

  // TODO: changed to DataPtr in Aten when shared folder
  // is ready
  using DataPtr = std::shared_ptr<void>;
  int64_t capacity_ = 0;
  DataType data_type_;
  DataPtr data_ptr_;
  // allocator_ takes precedence over StaticContext from device_type_
  // Allocator* allocator_;
  DeviceType device_type_ = CPU;

  friend class TensorImpl;
};

} // namespace caffe2

#endif // CAFFE2_CORE_STORAGE_H_
