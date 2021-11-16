#pragma once

#include <string>
#include <vector>

#include <torch/csrc/lazy/core/shape.h>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensors/literal.h"
#include "lazy_tensors/span.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes);

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<compiler::BackendDataPtr> data_handles,
    at::ScalarType dest_element_type);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
compiler::BackendDataPtr TensorToDataHandle(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
std::vector<compiler::BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<torch::lazy::BackendDevice>& devices);

// Makes a deep copy of an ATEN tensor.
static inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
static inline at::Tensor CopyTensor(const at::Tensor& ref,
                                    at::ScalarType dest_type,
                                    bool copy = true) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

// Unwraps tensor to target dtype if it's a wrapped number.
static inline at::Tensor UnwrapNumber(const at::Tensor& tensor,
                                      at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}

template <typename T>
at::Scalar MakeIntScalar(T value) {
  return at::Scalar(static_cast<int64_t>(value));
}

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value);

}  // namespace torch_lazy_tensors
