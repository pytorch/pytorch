#pragma once

// THIS FILE SHOULD BE HEADER ONLY BUT ISN'T ENFORCED
// I only need it for AOTI_TORCH_ERROR_CODE_CHECK
#include <torch/csrc/inductor/aoti_runtime/utils.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>

namespace torch::stable {

using DeviceIndex =
    int8_t; // this is from c10/core/Device.h and can be header only?

inline void delete_tensor_object(AtenTensorHandle ath) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
}

// The torch::stable::Tensor class is a highlevel C++ header-only wrapper around
// the C shim Tensor APIs. There are several goals of this class over
// AtenTensorHandle and RAIIAtenTensorHandle:
// 1. torch::stable::Tensor is a nicer UX much closer to torch::Tensor than the
// C
//    APIs with AtenTensorHandle. Under the hood we still call to these C shim
//    APIs to preserve stability.
// 2. RAIIAtenTensorHandle boils down to a uniq_ptr that forces the user to pass
//    around ownership. This makes it difficult to pass one input into 2
//    different functions, e.g., doing something like c = a(t) + b(t) if a and b
//    both require ownership of t. Here, we use a shared_ptr accompanied with
//    some other tricks to make our contract-based memory management much more
//    possible.
class Tensor {
 private:
  std::shared_ptr<AtenTensorOpaque> ath_;

 public:
  Tensor() = delete;

  // Wrap AtenTensorHandle
  explicit Tensor(AtenTensorHandle ath) : ath_(ath, delete_tensor_object) {}

  // Wrap StableIValue
  explicit Tensor(StableIValue siv)
      : ath_(to<AtenTensorHandle>(siv), delete_tensor_object) {}

  // Copy and move constructors can be default cuz the underlying handle is a
  // shared_ptr
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) noexcept = default;

  // Copy and move assignment operators can be default cuz the underlying handle
  // is a shared_ptr
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  ~Tensor() = default;

  AtenTensorHandle get() const {
    return ath_.get();
  }

  StableIValue get_StableIValue() const {
    AtenTensorHandle handle = ath_.get();

    // The following is our way of incrementing the refcount of the underlying
    // Tensor that we point to. Why do we want this supposedly weird behavior?
    // Because! We expect users to only need a StableIValue when they are trying
    // to pass the Tensor into a stack-based API, e,g.,
    // aoti_torch_call_dispatcher.
    //
    // A stack-based API is one that expects a stack of inputs converted to
    // StableIValues. Our contract with any stack-based API is that the stack
    // has ownership of its Tensor arguments. Since this torch::stable::Tensor
    // object will likely go out of scope by the end of the user extension's
    // local function and will thus delete its reference on the at::Tensor,
    // we create a new AtenTensorHandle for that use case.
    AtenTensorHandle new_ath;
    aoti_torch_new_tensor_handle(handle, &new_ath);

    return from(new_ath);
  }

  void* data_ptr() const {
    void* data_ptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  int64_t stride(int64_t dim) const {
    int64_t stride;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_stride(ath_.get(), dim, &stride));
    return stride;
  }

  template <typename T>
  T* data_ptr() const {
    // the actual implementation directs through mutable_data_ptr but like. how
    // different is that really.
    return static_cast<T*>(this->data_ptr());
  }

  /// Returns a `Tensor`'s device index.
  DeviceIndex get_device() const {
    int32_t device_index;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(ath_.get(), &device_index));
    return static_cast<DeviceIndex>(device_index);
  }

  bool is_cuda() const {
    int32_t device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cuda();
  }

  int64_t size(int64_t dim) const {
    int64_t size;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(ath_.get(), dim, &size));
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
