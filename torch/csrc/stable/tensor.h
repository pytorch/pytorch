#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>
#include <memory>

namespace torch::stable {

using DeviceIndex =
    int8_t; // this is from c10/core/Device.h and can be header only

// The torch::stable::Tensor class is a highlevel C++ wrapper around
// the C shim Tensor APIs. We've modeled this class after TensorBase, as custom
// op kernels only really need to interact with Tensor metadata (think sizes,
// strides, device, dtype). Other functions on Tensor (like empty_like) should
// live like the ATen op that they are and exist outside of this struct.
//
// There are several goals of this class over AtenTensorHandle and
// RAIIAtenTensorHandle:
// 1. torch::stable::Tensor is a nicer UX much closer to torch::Tensor than the
//    C APIs with AtenTensorHandle. Under the hood we still call to these C shim
//    APIs to preserve stability.
// 2. RAIIAtenTensorHandle boils down to a uniq_ptr that forces the user to pass
//    around ownership. This makes it difficult to pass one input into 2
//    different functions, e.g., doing something like c = a(t) + b(t) for
//    stable::Tensor t. Thus, we use a shared_ptr here.
class Tensor {
 private:
  std::shared_ptr<AtenTensorOpaque> ath_;

 public:
  // Construct a stable::Tensor with an uninitialized AtenTensorHandle (ATH)
  // Steals ownership from the ATH
  Tensor() {
    AtenTensorHandle ret;
    TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&ret));
    ath_ = std::shared_ptr<AtenTensorOpaque>(ret, [](AtenTensorHandle ath) {
      TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
    });
  }

  // Construct a stable::Tensor from an AtenTensorHandle (ATH)
  // Steals ownership from the ATH
  explicit Tensor(AtenTensorHandle ath)
      : ath_(ath, [](AtenTensorHandle ath) {
          TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
        }) {}

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

  // Returns a borrowed reference to the AtenTensorHandle
  AtenTensorHandle get() const {
    return ath_.get();
  }

  // =============================================================================
  // C-shimified TensorBase APIs: the below APIs have the same signatures and
  // semantics as their counterparts in TensorBase.h.
  // =============================================================================

  void* data_ptr() const {
    void* data_ptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  int64_t dim() const {
    int64_t dim;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(ath_.get(), &dim));
    return dim;
  }

  int64_t numel() const {
    int64_t numel;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(ath_.get(), &numel));
    return numel;
  }

  // note: this is a subset of the original TensorBase API. It takes no
  // arguments whereas the original API takes in a kwarg of memory format.
  // Here, we assume the default contiguous memory format.
  bool is_contiguous() const {
    bool is_contiguous;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_is_contiguous(ath_.get(), &is_contiguous));
    return is_contiguous;
  }

  int64_t stride(int64_t dim) const {
    int64_t stride;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_stride(ath_.get(), dim, &stride));
    return stride;
  }

  DeviceIndex get_device() const {
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(ath_.get(), &device_index));
    return static_cast<DeviceIndex>(device_index);
  }

  bool is_cuda() const {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cuda();
  }

  int64_t size(int64_t dim) const {
    int64_t size;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(ath_.get(), dim, &size));
    return size;
  }

  bool defined() const {
    bool defined;
    TORCH_ERROR_CODE_CHECK(aoti_torch_is_defined(ath_.get(), &defined));
    return defined;
  }

  // =============================================================================
  // END of C-shimified TensorBase APIs
  // =============================================================================
};

} // namespace torch::stable
