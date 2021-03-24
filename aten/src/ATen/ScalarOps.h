#pragma once

#include <c10/core/Scalar.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>

namespace at {
namespace detail {
// When filling a number to 1-element CPU tensor, we want to skip
// everything but manipulate data ptr directly.
// Ideally this fast pass should be implemented in TensorIterator,
// but we also want to skip compute_types which in not avoidable
// in TensorIterator for now.
Tensor& scalar_fill(Tensor& self, const Scalar& value);
TORCH_API Tensor scalar_tensor_static(const Scalar& s, c10::optional<ScalarType> dtype_opt, c10::optional<Device> device_opt);
} // namespace detail
} // namespace at

// This is in the c10 namespace because we use ADL to find the functions in it.
namespace c10 {

// FIXME: this should be (and was) Scalar::toTensor, but there is currently no way
// to implement this without going through Derived Types (which are not part of core).
inline at::Tensor scalar_to_tensor(const Scalar& s, const Device device = at::kCPU) {
  // This is the fast track we have for CPU scalar tensors.
  if (device == at::kCPU) {
    if (s.isFloatingPoint()) {
      return at::detail::scalar_tensor_static(s, at::kDouble, at::kCPU);
    } else if (s.isComplex()) {
      return at::detail::scalar_tensor_static(s, at::kComplexDouble, at::kCPU);
    } else if (s.isBoolean()) {
      return at::detail::scalar_tensor_static(s, at::kBool, at::kCPU);
    } else {
      AT_ASSERT(s.isIntegral(false));
      return at::detail::scalar_tensor_static(s, at::kLong, at::kCPU);
    }
  }
  if (s.isFloatingPoint()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kDouble));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kComplexDouble));
  } else {
    AT_ASSERT(s.isIntegral(false));
    return at::scalar_tensor(s, at::device(device).dtype(at::kLong));
  }
}

}
