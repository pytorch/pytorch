#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeUtils.h>

#include <c10/util/intrusive_ptr.h>

namespace c10 {

struct C10_API StorageImpl final : public c10::intrusive_ptr_target {
 public:
  StorageImpl(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);

  StorageImpl(
      caffe2::TypeMeta data_type,
      int64_t numel,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            data_type,
            numel,
            allocator->allocate(data_type.itemsize() * numel),
            allocator,
            resizable) {}

  explicit StorageImpl(at::Device device)
      : StorageImpl(device, caffe2::TypeMeta()) {}

  StorageImpl(at::Device device, caffe2::TypeMeta data_type)
      : StorageImpl(data_type, 0, at::DataPtr(nullptr, device), nullptr, true) {
  }

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl();

  void reset();

  template <typename T>
  inline bool IsType() const {
    return data_type_.Match<T>();
  }

  template <typename T>
  inline T* data() const {
    // TODO: This is bad: it means storage.data<T>() calls only work on
    // T that are valid ScalarType.  FIXME!
    auto data_type_T =
        at::scalarTypeToDataType(c10::CTypeToScalarType<T>::to());
    if (dtype().id() != data_type_T) {
      AT_ERROR(
          "Attempt to access StorageImpl having data type ",
          dtype().id(),
          " as data type ",
          data_type_T);
    }
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override;

  size_t itemsize() const {
    return data_type_.itemsize();
  }

  size_t capacity() const {
    return numel_ * itemsize();
  }

  int64_t numel() const {
    return numel_;
  };

  // TODO: remove later
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
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr);

  // XXX: TERRIBLE! DONT USE UNLESS YOU HAVE TO! AND EVEN THEN DONT, JUST DONT!
  // Setting the data_type will require you to audit many other parts of the
  // struct again to make sure it's still valid.
  void set_dtype(const caffe2::TypeMeta& data_type) {
    int64_t capacity = numel_ * data_type_.itemsize();
    data_type_ = data_type;
    numel_ = capacity / data_type_.itemsize();
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const caffe2::TypeMeta& dtype() const {
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

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      const caffe2::TypeMeta& data_type,
      size_t capacity,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), data_type, capacity);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity);

  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

 private:
  caffe2::TypeMeta data_type_;
  DataPtr data_ptr_;
  int64_t numel_;
  bool resizable_;
  Allocator* allocator_;
  bool received_cuda_;
};
} // namespace c10
