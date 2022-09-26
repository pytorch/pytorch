#pragma once

#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/shape.h>

#include <ATen/FunctionalTensorWrapper.h>

#include <string>
#include <vector>

namespace torch {
namespace lazy {

TORCH_API std::vector<int64_t> ComputeArrayStrides(
    c10::ArrayRef<int64_t> sizes);

TORCH_API std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<BackendDataPtr> data_handles,
    at::ScalarType dest_element_type);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
TORCH_API BackendDataPtr
TensorToDataHandle(const at::Tensor& tensor, const BackendDevice& device);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
TORCH_API std::vector<BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<BackendDevice>& devices);

// Makes a deep copy of an ATEN tensor.
inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
inline at::Tensor CopyTensor(
    const at::Tensor& ref,
    at::ScalarType dest_type,
    bool copy = true) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}

template <typename T, typename S>
T OptionalOr(const c10::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

// Unwraps tensor to target dtype if it's a wrapped number.
inline at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
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
TORCH_API bool IsSpecialScalar(const at::Scalar& value);

// Note: returns a reference instead of a fresh tensor to avoid refcount bumps.
inline const at::Tensor& maybe_unwrap_functional(const at::Tensor& tensor) {
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    return at::functionalization::impl::unsafeGetFunctionalWrapper(tensor)
        ->value();
  } else {
    return tensor;
  }
}

} // namespace lazy
} // namespace torch
