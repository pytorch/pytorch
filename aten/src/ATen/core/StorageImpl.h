#pragma once

#include <ATen/core/Allocator.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/ScalarTypeUtils.h>

#include <ATen/core/intrusive_ptr.h>

namespace at {

struct Type;

struct AT_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  StorageImpl() = delete;
  ~StorageImpl() {};
  StorageImpl(
      at::DataType data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(
      at::DataType data_type,
      int64_t numel,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl& operator=(StorageImpl&& other) = default;

  template <typename T>
  inline T* data() const {
    auto data_type_T = at::scalarTypeToDataType(at::CTypeToScalarType<T>::to());
    if (dtype() != data_type_T) {
      AT_ERROR(
          "Attempt to access StorageImpl having data type ",
          dtype(),
          " as data type ",
          data_type_T);
    }
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override {
    data_ptr_.clear();
  }

  void operator=(const StorageImpl&) = delete;

  size_t itemsize() const {
    return at::elementSize(dataTypeToScalarType(data_type_));
  }

  Type& type();

  int64_t numel() const {
    return numel_;
  };
  void set_numel(int64_t numel) {
    numel_ = numel;
  };
  bool resizable() const {
    return resizable_;
  };
  at::DataPtr& data_ptr() {
    return data_ptr_;
  };
  const at::DataPtr& data_ptr() const {
    return data_ptr_;
  };
  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    std::swap(data_ptr_, data_ptr);
    return std::move(data_ptr);
  };
  void* data() {
    return data_ptr_.get();
  };
  const void* data() const {
    return data_ptr_.get();
  };
  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }
  at::Allocator* allocator() {
    return allocator_;
  };
  const DataType dtype() const {
    return data_type_;
  }
  const at::Allocator* allocator() const {
    return allocator_;
  };
  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }
  Device device() const {
    return data_ptr_.device();
  }
  void set_resizable(bool resizable) {
    resizable_ = resizable;
  }

 private:
  at::DataType data_type_;
  at::DataPtr data_ptr_;
  int64_t numel_;
  bool resizable_;
  at::Allocator* allocator_;
};
} // namespace at
