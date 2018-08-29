#pragma once

#include <ATen/Allocator.h>
#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>

#include <ATen/core/intrusive_ptr.h>

#include <atomic>

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

namespace at {

struct Type;

struct AT_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  StorageImpl() = delete;
  ~StorageImpl() {};
  StorageImpl(
      at::DataType data_type,
      ptrdiff_t size,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(
      at::DataType data_type,
      ptrdiff_t size,
      at::Allocator* allocator,
      bool resizable);
  StorageImpl(StorageImpl&) = delete;
  StorageImpl(const StorageImpl&) = delete;
  // NB: Don't move ref count!
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl& operator=(StorageImpl&& other) = default;

  // TODO: Rename this into th_data, and move it out of the class;
  // the real data shouldn't call th::from_type
  template <typename T>
  inline T* data() const {
    auto data_type_T =
        at::scalarTypeToDataType(at::CTypeToScalarType<T>::to());
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

  size_t elementSize() const {
    return at::elementSize(dataTypeToScalarType(data_type_));
  }

  Type& type();

  // TODO: Rename to size() and size to size_
  ptrdiff_t size() const {
    return size_;
  };
  void set_size(ptrdiff_t size) {
    size_ = size;
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
  ptrdiff_t size_;
  bool resizable_;
  at::Allocator* allocator_;
};
} // namespace at
