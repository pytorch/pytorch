#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>

#include <c10/util/intrusive_ptr.h>

namespace c10 {

struct C10_API StorageImpl final : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable) {
      AT_ASSERTM(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            allocator->allocate(size_bytes),
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() = default;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
  }

  template <typename T>
  inline T* data() const {
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override {
    data_ptr_.clear();
  }

  size_t nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
  }

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
    if (resizable) {
      // We need an allocator to be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;
    resizable_ = false;
  }

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

 private:
  DataPtr data_ptr_;
  size_t size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  Allocator* allocator_;
};
} // namespace c10
