#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::stable {

using DeviceIndex =
    int8_t; // this is from c10/core/Device.h and can be header only?

static inline void delete_tensor_object(AtenTensorHandle ath) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_tensor_object(ath));
}

class Tensor {
 private:
  std::shared_ptr<AtenTensorOpaque> shared_ath_;
  // The torch::stable::Tensor class is a highlevel C++ wrapper around the C
  // shim Tensor APIs. I'm starting to think this will look a lot like
  // RAIIAtenTensorHandle but use a shared pointer instead of a uniq_ptr.
 public:
  Tensor() = delete;

  // Wrap AtenTensorHandle
  explicit Tensor(AtenTensorHandle ath)
    : shared_ath_(ath, delete_tensor_object) {}

  // Copy and move constructors can be default cuz the underlying handle is a shared_ptr
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) noexcept = default;
  
  // Copy and move assignment operators can be default cuz the underlying handle is a shared_ptr
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  ~Tensor() = default;

  AtenTensorHandle get() const {
    return shared_ath_.get();
  }

  void* data_ptr() const {
    void* data_ptr;
    aoti_torch_get_data_ptr(shared_ath_.get(), &data_ptr);
    return data_ptr;
  }

  int64_t stride(int64_t dim) const {
    int64_t stride;
    aoti_torch_get_stride(shared_ath_.get(), dim, &stride);
    return stride;
  }

  template <typename T>
  T* data_ptr() const {
    // the actual implementation directs through mutable_data_ptr but like. how different is that really.
    return static_cast<T*>(this->data_ptr());
  }

  /// Returns a `Tensor`'s device index.
  DeviceIndex get_device() const {
    int32_t device_index;
    aoti_torch_get_device_index(shared_ath_.get(), &device_index);
    return static_cast<DeviceIndex>(device_index);
  }

  bool is_cuda() const {
    int32_t device_type;
    aoti_torch_get_device_type(shared_ath_.get(), &device_type);
    return device_type == aoti_torch_device_type_cuda();
  }

  int64_t size(int64_t dim) const {
    int64_t size;
    aoti_torch_get_size(shared_ath_.get(), dim, &size);
    return size;
  }

  // the below are APIs that I plan on adding to support more custom ops

  // /// Returns the `TensorOptions` corresponding to this `Tensor`. Defined in
  // /// TensorOptions.h.
  // /// We don't have a stable def for that yet sigh
  // TensorOptions options() const {
  //   return ? ? ? ;
  // }

  // IntArrayRef sizes(const TensorBase& t) {
  //   return ? ? ? ;
  // }
  
  // // TypeMeta and caffe2 scalar types are in DefaultDtype.h and
  // // c10/core/ScalarType.h
  // caffe2::TypeMeta dtype() const {
  //   return ? ? ? ;
  // }

  // ScalarType scalar_type() const {
  //   return ? ? ? ;
  // }

  // // I only need the API to support 0 arguments so far
  // bool is_contiguous(
  //     at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
  //   return ? ? ? ;
  // }
};

} // namespace torch::stable
